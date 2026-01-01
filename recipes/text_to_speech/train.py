import argparse
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from balacoon_train.config import Config
from balacoon_train.data import create_fold_data_loader
from balacoon_train.tts_alm import TTSALMModel


torch.set_float32_matmul_precision("medium")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank", type=int, default=int(os.environ.get("RANK", 0)), help="Rank of the process"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", 1)),
        help="Total number of processes",
    )

    # Dataset arguments
    parser.add_argument("--train", type=str, required=True, help="Path to train.txt")
    parser.add_argument(
        "--acoustic-tokens", type=str, required=True, help="Path to acoustic tokens directory"
    )
    parser.add_argument("--phonemes", type=str, required=True, help="Path to phonemes directory with alignment info")
    parser.add_argument("--phoneme-mapping", type=str, required=True, help="Path to phoneme mapping")
    parser.add_argument("--dev", type=str, help="Path to dev.txt")

    # Training arguments
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint to resume/load")
    parser.add_argument("--out", type=str, required=True, help="Output directory for training logs")

    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config/train.yaml")
    config = Config.load(config_path)

    # Update config with CLI arguments
    config.data.locations.train_txt = args.train
    config.data.locations.acoustic_tokens_dir = [args.acoustic_tokens]
    config.data.locations.phoneme_dir = [args.phonemes]
    config.data.locations.phoneme_mapping = args.phoneme_mapping
    if args.dev:
        config.data.locations.dev_txt = args.dev
    config.resolve()

    # Create data loader
    # Note: we manually handle data sharding via stride/strides_num
    train_loader = create_fold_data_loader(
        config.data,
        "train",
        shuffle=True,
        stride=args.rank,
        strides_num=args.world_size,
    )

    # Create validation loader if dev is specified
    val_loader = None
    if args.dev:
        # Validation runs on all nodes, so we shard it too
        val_loader = create_fold_data_loader(
            config.data,
            "dev",
            shuffle=False,
            stride=args.rank,
            strides_num=args.world_size,
        )

    # Create model
    # Pass config.model so TTSALMModel uses parameters from the 'model' section of YAML
    model = TTSALMModel(config.model)

    # Initialize Trainer
    # When using custom data sharding (stride/strides_num), we must disable replace_sampler_ddp
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=int(config.model.num_steps // 3),
        save_top_k=-1,
    )
    trainer = L.Trainer(
        default_root_dir=args.out,
        accelerator="gpu",
        devices="auto",
        strategy="ddp" if args.world_size > 1 else "auto",
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback],
        max_steps=config.model.num_steps,
        precision=config.model.precision,
    )

    # Start training
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt
    )


if __name__ == "__main__":
    main()
