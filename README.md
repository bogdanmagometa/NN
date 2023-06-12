# Manual Implementation of Deep Learning algorithms

Project structure:
- `nn/` - folder with implementation of automatic differentiation (AD) of Tensor operations, optimizers and parts of neural networks (linear layer, activation layers).
- `tests/test_tensor.py` - unit tests for automatic differentiation (AD) of Tensor operations.
- `test_optimizers.ipynb` - comparing four implemented optimizers.
- `comparison_with_pytorch.ipynb` - comparison of our optimizers with PyTorch ones.
- `nn_models.ipynb` - training MLPs using our implementation of AD.
- `test_perceptron.ipynb` - training perceptron using our implementation of AD to mimic AND gate.
- `torch_models.ipynb` - implementing the same models as in `nn_models.ipynb` (+ CNN) but in PyTorch.
