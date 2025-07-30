from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd).apply(x)


class Classifier(nn.Module):
    def __init__(self, input_dim = 768, num_classes=31):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.Linear(input_dim//2, input_dim//4),
            nn.Linear(input_dim//4, input_dim//8),
            nn.Linear(input_dim//8, num_classes)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x