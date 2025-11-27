import argparse
import os
import torch
from tqdm import tqdm
from balacoon_train.config import Config
from balacoon_train.data import create_fold_data_loader
from balacoon_train.alm import ALMModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True, help="Path to test.txt")
    parser.add_argument("--id-mapping", type=str, required=True, help="Path to id mapping file")
    parser.add_argument(
        "--acoustic-tokens", type=str, required=True, help="Path to acoustic tokens directory"
    )
    parser.add_argument("--pitch", type=str, required=True, help="Path to pitch directory")
    parser.add_argument("--phonemes", type=str, required=True, help="Path to phonemes directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint to load")
    parser.add_argument("--out", type=str, required=True, help="Output directory predictions")

    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config/predict.yaml")
    config = Config.load(config_path)

    # Update config with CLI arguments
    config.data.locations.test_txt = args.test
    config.data.locations.id_mapping = args.id_mapping
    config.data.locations.acoustic_tokens_dir = [args.acoustic_tokens]
    config.data.locations.pitch_dir = [args.pitch]
    config.data.locations.phoneme_dir = [args.phonemes]
    config.resolve()

    # Create data loader
    test_loader = create_fold_data_loader(
        config.data,
        "test",
        shuffle=False,
        stride=0,
        strides_num=1,
    )

    # Create model
    # We need to pass config.model to load_from_checkpoint because ALMModel
    # doesn't save hyperparameters automatically
    model = ALMModel.load_from_checkpoint(args.ckpt, config=config.model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Run prediction
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        with torch.no_grad():
            model.generate(batch, args.out)


if __name__ == "__main__":
    main()
