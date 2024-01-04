"""Implementation of a goal-conditioned transporter."""

import dataclasses
import warnings
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import einops as e
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct
from flax.linen import initializers
from flax.training import train_state
from jax import random


import hydra
from hydra import compose, initialize
from hydra.utils import call, instantiate

from transporter import Transporter

@struct.dataclass
class GoalConditionedTransporter:
    """Goal-conditioned Transporter."""
    pick_model: train_state.TrainState
    place_model_query: train_state.TrainState
    place_model_key: train_state.TrainState
    goal_model: train_state.TrainState
