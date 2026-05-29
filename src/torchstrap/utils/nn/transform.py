from beartype.typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class TensorTransform(Protocol):
    def __call__(self, input: torch.Tensor) -> torch.Tensor: ...


class Transform(torch.nn.Module):
    def __init__(self, transform: TensorTransform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform(x)


class OneHotEncode(torch.nn.Module):
    def __init__(self, num_classes=-1):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.as_tensor(
            torch.nn.functional.one_hot(
                torch.as_tensor(x.squeeze_(), dtype=torch.long),
                num_classes=self.num_classes,
            ),
            dtype=torch.float32,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(one-hot encodes labels)"


class Normalize(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, stddev: torch.Tensor, inplace: bool = True):
        super().__init__()
        self.register_buffer(name="mean", tensor=mean)
        self.register_buffer(name="stddev", tensor=stddev)
        self.inplace = inplace

    def forward(self, x):
        # print(f"{self.__repr__()}, {x[0]}")
        if self.inplace:
            return x.sub_(self.mean).div_(self.stddev)
        else:
            return x.sub(self.mean).div(self.stddev)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.stddev})"
