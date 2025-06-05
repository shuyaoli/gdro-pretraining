import pytest
import torch
from dynamic_loading_callback import _project_to_simplex

def test_project_to_simplex_basic():
    """Test basic functionality with a simple input vector"""
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    result = _project_to_simplex(input_tensor)
    
    # Check that result is a probability distribution
    assert torch.all(result >= 0)  # All elements should be non-negative
    assert abs(result.sum() - 1.0) < 1e-6  # Sum should be 1
    assert len(result) == len(input_tensor)  # Should maintain same length
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)

def test_project_to_simplex_basic_2():
    """Test basic functionality with a simple input vector"""
    input_tensor = torch.tensor([0.2, 0.3, 0.4])
    result = _project_to_simplex(input_tensor)
    assert torch.allclose(result, torch.tensor([0.7/3, 1.0/3, 1.3/3]), atol=1e-6)
    
    # Check that result is a probability distribution
    assert torch.all(result >= 0)  # All elements should be non-negative
    assert abs(result.sum() - 1.0) < 1e-6  # Sum should be 1
    assert len(result) == len(input_tensor)  # Should maintain same length

def test_project_to_simplex_basic_3():
    """Test basic functionality with a simple input vector"""
    input_tensor = torch.tensor([0.2, 0.3, 0.5])
    result = _project_to_simplex(input_tensor)
    assert torch.allclose(result, torch.tensor([0.2, 0.3, 0.5]), atol=1e-6)

def test_project_to_simplex_basic_2():
    """Test basic functionality with a simple input vector"""
    input_tensor = torch.tensor([1, 0.5])
    result = _project_to_simplex(input_tensor)
    assert torch.allclose(result, torch.tensor([0.75, 0.25]), atol=1e-6)

def test_project_to_simplex_negative():
    """Test with negative values"""
    input_tensor = torch.tensor([-1.0, 2.0, -3.0])
    result = _project_to_simplex(input_tensor)
    
    assert torch.all(result >= 0)
    assert abs(result.sum() - 1.0) < 1e-6
    assert len(result) == len(input_tensor)

def test_project_to_simplex_zeros():
    """Test with zeros"""
    input_tensor = torch.tensor([0.0, 0.0, 0.0])
    result = _project_to_simplex(input_tensor)
    
    assert torch.all(result >= 0)
    assert abs(result.sum() - 1.0) < 1e-6
    assert len(result) == len(input_tensor)
    # Should be uniform distribution
    expected = torch.ones_like(result) / len(result)
    assert torch.allclose(result, expected, atol=1e-6)

def test_project_to_simplex_large_values():
    """Test with large values"""
    input_tensor = torch.tensor([1000.0, 2000.0, 3000.0])
    
    result = _project_to_simplex(input_tensor)
    
    assert torch.all(result >= 0)
    assert abs(result.sum() - 1.0) < 1e-6
    assert len(result) == len(input_tensor)
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)

def test_project_to_simplex_single_element():
    """Test with single element"""
    input_tensor = torch.tensor([5.0])
    result = _project_to_simplex(input_tensor)
    
    assert torch.all(result >= 0)
    assert abs(result.sum() - 1.0) < 1e-6
    assert len(result) == 1
    assert abs(result[0] - 1.0) < 1e-6

def test_project_to_simplex_cuda():
    """Test on CUDA if available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    result = _project_to_simplex(input_tensor)
    
    assert torch.all(result >= 0)
    assert abs(result.sum() - 1.0) < 1e-6
    assert len(result) == len(input_tensor)
    assert result.device.type == 'cuda'

def test_project_to_simplex_preserves_order():
    """Test that the order of elements is preserved"""
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    result = _project_to_simplex(input_tensor)
    
    
    # Check that the relative ordering is preserved
    for i in range(len(input_tensor)-1):
        if input_tensor[i] > input_tensor[i+1]:
            assert result[i] >= result[i+1]
        elif input_tensor[i] < input_tensor[i+1]:
            assert result[i] <= result[i+1] 