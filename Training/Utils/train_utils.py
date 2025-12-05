import torch

def accuracy(Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare