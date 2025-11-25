"""
Copyright 2025 Balacoon

Tests for data container
"""

import pytest
import torch
import numpy as np
from balacoon_train.data.container import Container

def test_container_init():
    """Test container initialization."""
    ids = ["id1", "id2"]
    container = Container(ids)
    assert container.get_ids() == ids
    
    single_container = Container("id1")
    assert single_container.get_id() == "id1"

def test_container_set_get_item():
    """Test setting and getting items in container."""
    container = Container(["id1"])
    data = torch.randn(1, 10)
    container["audio"] = data
    
    assert "audio" in container
    assert torch.equal(container["audio"], data)
    assert torch.equal(container.get("audio"), data)

def test_container_batch_check():
    """Test batch size consistency check."""
    container = Container(["id1", "id2"]) # batch size 2
    
    # Correct batch size
    data_ok = torch.randn(2, 10)
    container["data_ok"] = data_ok
    
    # Incorrect batch size
    data_bad = torch.randn(3, 10)
    with pytest.raises(AssertionError):
        container["data_bad"] = data_bad

def test_container_multiple_items():
    """Test adding multiple items at once."""
    container = Container(["id1"])
    names = ["a", "b"]
    tensors = [torch.tensor([1]), torch.tensor([2])]
    
    container[names] = tensors
    
    assert container["a"].item() == 1
    assert container["b"].item() == 2

def test_container_to_device():
    """Test moving container to device (CPU here)."""
    container = Container(["id1"])
    data = torch.randn(1, 5)
    container["data"] = data
    
    # Move to CPU (which it already is, but exercises the code path)
    container_cpu = container.to(-1)
    
    assert "data" in container_cpu
    assert torch.equal(container_cpu["data"], data)

