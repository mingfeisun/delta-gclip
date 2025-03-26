"""
This file provides a basic implementation of the gradient-clipping
algorithm as a pytorch optimizer.
"""

from typing import Any, Dict, Iterable

import torch

from dGClip import dGClip

class GClip(dGClip):
    """Class for the Gradient Clipping optimizer"""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]],
        lr: float,
        gamma: float = 0.1,
        weight_decay: float = 0
    ) -> None:
        super().__init__(params, lr, gamma, delta=0, weight_decay=weight_decay)