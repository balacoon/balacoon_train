
import pytest
from dataclasses import dataclass
from omegaconf import OmegaConf
from balacoon_train.config import Config

@dataclass
class DummyConfig:
    learning_rate: float = 0.001
    batch_size: int = 32

def test_config_init_and_access():
    """Test basic initialization and attribute access."""
    # Initialize with nested dict structure
    conf = Config(OmegaConf.create({"model": {"hidden_dim": 128}}))
    
    assert conf.model.hidden_dim == 128
    # Test dictionary-like access
    assert conf["model"]["hidden_dim"] == 128

def test_config_nested_access():
    """Test nested configuration access."""
    # Assign dicts instead of Config objects
    conf = Config()
    conf.train = {"optimizer": {"lr": 0.01}}
    
    assert conf.train.optimizer.lr == 0.01

def test_config_add_missing():
    """Test adding missing values from dataclass."""
    conf = Config()
    conf.learning_rate = 0.05  # Override default
    
    conf.add_missing(DummyConfig)
    
    assert conf.learning_rate == 0.05
    assert conf.batch_size == 32

def test_config_to_dict():
    """Test converting config to dictionary."""
    conf = Config()
    conf.param1 = 10
    conf.sub = {"param2": 20}
    
    d = conf.to_dict()
    assert d == {"param1": 10, "sub": {"param2": 20}}
