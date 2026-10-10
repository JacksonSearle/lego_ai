"""My stage: a fused block.

Made in the Builder by fusing 3 blocks into one. It builds exactly those blocks in order.
You can edit PARTS (same format as a model's block list), the name or the color, then Save.
"""
from lego.sdk import compose, lego_block

PARTS = [
    {
        "type": "drawn_activation",
        "params": {
            "points": [
                -3.0,
                -2.5,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0
            ]
        },
        "repeat": 1
    },
    {
        "type": "linear",
        "params": {
            "out_features": 128,
            "activation": "none",
            "bias": True
        },
        "repeat": 1
    },
    {
        "type": "classifier",
        "params": {
            "pool": "auto",
            "dropout": 0.0
        },
        "repeat": 1
    }
]


@lego_block(
    name='My stage',
    color='#7c3aed',
    description='Fused block: Drawn Activation → Linear → Output Head.',
    tags=['general'],
    parts=PARTS,
)
def fused_my_stage(in_shape, ctx=None):
    return compose(in_shape, ctx, PARTS)
