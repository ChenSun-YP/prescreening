import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def create_dataset(x_data, y_data, tmax=None):
    """
    Creates a TensorDataset with x_t and y_{t-1} as inputs, y_t as targets, for the full time series.
    Args:
        x_data (Tensor): Input spike data [1, 1, tmax].
        y_data (Tensor): Output spike data [1, 1, tmax].
        ff_kernel (Tensor): Feedforward kernel tensor.
        fb_kernel (Tensor): Feedback kernel tensor.
        axy_kernel (Tensor): STDP axy kernel tensor.
        ixy_kernel (Tensor): Induction ixy kernel tensor.
        tmax (int, optional): Total time steps.
    Returns:
        TensorDataset: Dataset with inputs [1, 2, usable_length] and targets [1, usable_length].
    """
    x_data = x_data.clone().detach().float().squeeze()
    y_data = y_data.clone().detach().float().squeeze()

    if tmax is None:
        tmax = x_data.size(0)

    y_history = torch.cat(
        (torch.zeros(1), y_data[:-1])
    )  # y_t-1 shift  to the left by 1
    y_target = y_data  # y_t

    # Stack x_t and y_{t-1} as input channels
    inputs = torch.stack((x_data, y_history), dim=0).unsqueeze(0)
    labels = y_target.unsqueeze(0)

    indices = torch.tensor([[0, tmax]])

    return TensorDataset(inputs, labels, indices), 0, x_data, y_target
