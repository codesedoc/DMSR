import torch

LOSS_FN_NAME2CLASS = {
    'mlsm': torch.nn.MultiLabelSoftMarginLoss,
    'mse': torch.nn.MSELoss,
    'ce': torch.nn.CrossEntropyLoss
}
