from collections.abc import Sequence, Iterable

from beartype.typing import Protocol, Annotated
from beartype.vale import IsAttr, IsEqual

import torch

Scalar = Annotated[
    torch.Tensor, 
    IsAttr["ndim", IsEqual[0]] &
    IsAttr["dtype", IsEqual[torch.float32]]
]

Vector = Annotated[
    torch.Tensor, 
    IsAttr["ndim", IsEqual[0] | IsEqual[1]] &
    IsAttr["dtype", IsEqual[torch.float32]]
]

FloatScalarLike = float | Vector | Sequence[float | Scalar]

BoolVector = Annotated[
    torch.Tensor,
    IsAttr["ndim", IsEqual[0] | IsEqual[1]] &
    IsAttr["dtype", IsEqual[torch.bool]]
]

class LoaderT(
    Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    Protocol
):
    ...
