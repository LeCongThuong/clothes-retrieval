import torch
import torch.nn as nn
from arcface import ArcMarginProduct


class HardminingLoss(nn.Module):

    def __init__(self, in_features, out_features, anpha=1.5, beta=1.1, A=35, B=0.75, s=30.0, m=0.50, easy_margin=False):
        super(HardminingLoss, self).__init__()
        self.anpha = anpha
        self.beta = beta
        self.A = A
        self.B = B
        self.arcface = ArcMarginProduct(in_features, out_features, s, m, easy_margin)

    def forward(self, input, label):
        out_arcface = self.arcface(input, label)
        x = self.beta * out_arcface
        y = self.A * (x - self.B)
        z = torch.sigmoid(y)
        out_hardmining = self.anpha * x * z
        return out_hardmining.mean()



