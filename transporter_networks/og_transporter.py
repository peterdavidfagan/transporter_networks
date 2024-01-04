"""Implementation of original transporter model."""

import dataclasses
import warnings
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import einops as e
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct
from flax.linen import initializers
from flax.training import train_state
from jax import random


from hydra.utils import call, instantiate


class TransporterPick(nn.Module):
    """Transporter pick module."""
    
    config: dict

    @nn.compact
    def __call__(self, rgbd, train=False):
        """Forward pass."""
        raise NotImplementedError


class TransporterPlace(nn.Module):
    """Transporter place module."""
    
    config: dict

    @nn.compact
    def __call__(self, x: chex.Array, y: chex.Array) -> chex.Array:
        """Forward pass."""
        raise NotImplementedError


@struct.dataclass
class Transporter:
    """Transporter model."""
    pick_model: trainstate.TrainState
    place_model: trainstate.TrainState
