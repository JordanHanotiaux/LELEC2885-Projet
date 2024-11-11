from torch import (
    Tensor,
)
from torch.nn import (
    Module,
    PReLU,
    Conv2d,
    Sequential,
)
from DepthSeparableConv import DpthSpConv



IN_CH: int = 4
OUT_CH: int = 2
GROWING_FACTOR: int = 2
ACTIVATION: type[Module] = PReLU
KERNEL_SIZE: tuple[int, int] = (3, 3)



class InvResBtlnk(Module):
    """
    Inverted Residual Bottleneck module
    _summary_

    Args:
        Module (_type_): _description_
    """


    def __init__(
        self,
        in_ch: int,
        growing_factor: int = GROWING_FACTOR,
        activation: type[Module] = ACTIVATION,
        ker_sz: tuple[int, int] = KERNEL_SIZE,
    ) \
        -> None:

        super().__init__()
        self.in_ch: int = in_ch
        self.growing_factor: int = growing_factor

        inside_ch: int = in_ch * growing_factor
        self.up_filtering: Sequential = Sequential(
            Conv2d(
                in_channels=in_ch,
                out_channels=inside_ch,
                kernel_size=(1, 1),
                padding='same'
            ),
            activation()
        )

        self.depth_sep_conv: DpthSpConv = DpthSpConv(
            in_ch=inside_ch,
            out_ch=in_ch,
            ker_sz=ker_sz,
            activation=activation
        )


    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = self.up_filtering(x)
        z = self.depth_sep_conv(z)
        return z + x



if __name__ == "__main__":
    irb: InvResBtlnk = InvResBtlnk(IN_CH, OUT_CH)
    print(irb)
    from torch import rand
    x: Tensor = rand(2, IN_CH, 16, 16)
    res: Tensor = irb(x)
    print(x.shape, '->', res.shape)
