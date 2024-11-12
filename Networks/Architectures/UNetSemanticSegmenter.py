from torch import (
    Tensor,
)
from torch.nn import (
    Module,
    Conv2d,
    Sequential,
    PReLU,
    BatchNorm2d,
    Sigmoid,
    Identity,
)
from Networks.Architectures.UNetBlock import *



class UNetSmnticSgmntr(Sequential):
    """
    __summary__

    Args:
        Module (_type_): _description_
    """


    def __init__(
        self,
        in_ch: int,
        inside_ch: int,
        depth: int,
        # stem = None,
        # body = None,
        # head = None,
        threshold: float,
    ) -> \
        None:

        super().__init__()
        self.depth: int = depth
        self.stem: Sequential = Sequential(
            Conv2d(
                in_channels=in_ch,
                out_channels=inside_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding='same'
            ),
            PReLU(),
            Conv2d(
                in_channels=inside_ch,
                out_channels=inside_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding='same'
            ),
            PReLU()
        )

        self.body: Module = Identity()
        for i in range(depth):
            self.body = UNetBlock(
                in_ch=inside_ch*2**(depth-i-1),
                inside_ch=inside_ch*2**(depth-i),
                inner_block=self.body
            )

        self.head: Sequential = Sequential(
            Conv2d(
                in_channels=inside_ch,
                out_channels=in_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding='same'
            ),
            PReLU(),
            Conv2d(
                in_channels=in_ch,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding='same'
            ),
            PReLU(),
            BatchNorm2d(num_features=1),
            Sigmoid()
        )

        self.threshold: float = threshold


    def pred(self, x: Tensor) -> Tensor:
        return (self(x) > self.threshold).long()



if __name__ == '__main__':
    from torch import rand 
    x: Tensor = rand(1, 2, 4, 4)
    unet_segmenter: UNetSmnticSgmntr = UNetSmnticSgmntr(
        in_ch=2,
        inside_ch=8,
        depth=2,
        threshold=0.5
    )
    print(unet_segmenter)
    print(unet_segmenter(x))
    print(unet_segmenter.pred(x))
