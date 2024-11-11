from torch import (
    Tensor,
)
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    PReLU,
    BatchNorm2d,
)



class DpthSpConv(Sequential):
    """_summary_

    Args:
        Sequential (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ker_sz: tuple[int, int] = (3, 3),
        activation: type[Module] = PReLU,
    ) -> \
        None:

        super().__init__()
        self.in_ch: int = in_ch
        self.out_ch: int = out_ch

        self.conv1: Conv2d = Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ker_sz,
            stride=(1, 1),
            padding='same',
            groups=in_ch
        )

        self.activation: Module = activation()

        self.conv2: Conv2d = Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )
        
        self.batch_norm: BatchNorm2d = BatchNorm2d(
            num_features=out_ch
        )



if __name__ == '__main__':
    in_ch: int = 4
    out_ch: int = 2
    depth_sep_conv: DpthSpConv = DpthSpConv(
        in_ch,
        out_ch
    )
    print(depth_sep_conv)
    from torch import rand
    x: Tensor = rand(2, in_ch, 2, 2)
    res: Tensor = depth_sep_conv(x)
    print(x.shape, '->', res.shape)
