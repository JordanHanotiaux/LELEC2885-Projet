from torch import (
    Tensor,
    cat,
)
from torch.nn import (
    Module,
    Conv2d,
    PReLU,
    ConvTranspose2d,
    Identity,
)
from Networks.Architectures.InvResBottleneck import *



IN_CH: int = 1
GROWING_FACTOR: int = 2
ACTIVATION: type[Module] = PReLU
KERNEL_SIZE: tuple[int, int] = (3, 3)



class UNetBlock(Module):
    """
    _summary_

    Args:
        Module (_type_): _description_
    """


    def __init__(
        self,
        in_ch: int,
        inside_ch: int,
        inner_block: Module = Identity(),
        growing_factor: int = GROWING_FACTOR,
        activation: type[Module] = ACTIVATION,
        kernel_size: tuple[int, int] = KERNEL_SIZE,
    ) -> \
        None:

        super().__init__()
        self.in_ch: int = in_ch

        self.encoder: InvResBtlnk = InvResBtlnk(
            in_ch,
            growing_factor,
            activation,
            kernel_size
        )

        self.down_scaling: Conv2d = Conv2d(
            in_channels=in_ch,
            out_channels=inside_ch,
            kernel_size=(2, 2),
            stride=(2, 2),
        )

        self.inner_block: Module = inner_block
        
        self.up_scaling: ConvTranspose2d = ConvTranspose2d(
            in_channels=inside_ch,
            out_channels=in_ch,
            kernel_size=(2, 2),
            stride=(2, 2)
        )

        self.cat_mixer: Conv2d = Conv2d(
            in_channels=2*in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding='same'
        )

        self.decoder: InvResBtlnk = InvResBtlnk(
            in_ch,
            growing_factor,
            activation,
            kernel_size
        )


    def forward(self, x: Tensor) -> Tensor:
        a: Tensor = self.encoder(x)
        b: Tensor = self.down_scaling(a)
        c: Tensor = self.inner_block(b)
        d: Tensor = self.up_scaling(c)
        e: Tensor = self.cat_mixer(
            cat(
                (a, d),
                dim=1
            )
        )
        f: Tensor = self.decoder(e)
        # for t in (a, b, c, d, e, f):
        #     print(f'{t.shape =}')

        return f



if __name__ == "__main__":
    unet: UNetBlock = UNetBlock(
        in_ch=IN_CH,
        inside_ch=2*IN_CH
    )
    print(unet)
    from torch import rand
    x: Tensor = rand(16, IN_CH, 4, 4)
    unet(x)
