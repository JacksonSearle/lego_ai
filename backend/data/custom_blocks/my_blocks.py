import torch
import torch.nn as nn

from lego.sdk import Param, lego_block, make_activation


@lego_block(
    name="My Block",
    color="#22d3ee",
    description="What this block does.",
    params=[
        Param("width", "int", 128, "Width", "Number of output features.", min=1),
        Param("activation", "activation", "gelu", "Activation", "Non-linearity to apply."),
    ],
)
class MyBlock(nn.Module):
    def __init__(self, in_shape, width=128, activation="gelu"):
        super().__init__()
        self.fc = nn.Linear(in_shape[-1], width)
        self.act = make_activation(activation)

    def forward(self, x):
        return self.act(self.fc(x))
