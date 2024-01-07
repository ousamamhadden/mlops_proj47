import torch
from src.models.model import MyAwesomeModel
import pytest
    
@pytest.mark.parametrize("X, Y", [((1, 1, 28, 28), (1, 10)), ((10, 1, 28, 28), (10, 10))])
def test_model(X, Y):
    model = MyAwesomeModel()
    model.eval()
    x = torch.randn(X)
    y = model(x)
    assert y.shape == Y, f"The output shape does not equal to {Y} given the input {X}"

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))