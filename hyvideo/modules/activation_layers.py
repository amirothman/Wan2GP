from torch import nn


def get_activation_layer(act_type):
    """Get activation layer

    Args:
        act_type (str): the activation type

    Returns:
        torch.nn.functional: the activation layer

    """
    if act_type == "gelu":
        return lambda: nn.GELU()
    if act_type == "gelu_tanh":
        # Approximate `tanh` requires torch >= 1.13
        return lambda: nn.GELU(approximate="tanh")
    if act_type == "relu":
        return nn.ReLU
    if act_type == "silu":
        return nn.SiLU
    raise ValueError(f"Unknown activation type: {act_type}")
