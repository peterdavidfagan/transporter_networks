"""Implementation of original transporter model."""

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


class TransporterPick(nn.Module):
    """Transporter pick module."""
    
    config: dict

    @nn.compact
    def __call__(self, rgbd, train=False):
        """Forward pass."""
        x = rgbd
        
        # input conv projection
        x = instantiate(self.config["input_projection"]["conv"])(x)
        x = call(self.config["input_projection"]["pool"])(x)

        for block in self.config["blocks"]:
            # intermittently there are upsampling layers
            if block.name=="upsample":
                B, H, W, C = x.shape
                x = jax.image.resize(x, shape=(B, H*2, W*2, C), method="bilinear")
            # if not upsampling layer then we have a resnet block
            else:
                # original implementation has differing dimensions for layers within a block
                # in future will attempt to use more traditional resnet block
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                
                residual = x
                for i in range(block.num_blocks):
                    x = instantiate(block.resnet_block.norm)(x)
                    x = call(block.resnet_block.activation)(x)
                    if i==0:
                        x = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(x)
                    elif i==mid_idx:
                        x = instantiate(block.resnet_block.conv, strides=[1,1])(x) 
                    else:
                        x = instantiate(block.resnet_block.conv,kernel_size=[1,1], strides=[1,1], padding="VALID")(x)
                
                if residual.shape != x.shape:
                    residual = instantiate(block.resnet_block.conv)(residual)

                x = residual + x
        
        # take softmax over the channels
        x = jax.nn.softmax(x, axis=-1)
        x = e.rearrange(x, "b h w c -> b h (w c)")
        return x

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
    pick_model: train_state.TrainState
    place_model: train_state.TrainState


if __name__ == "__main__":
    # read network config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="./config", job_name="default_config")
    TRANSPORTER_CONFIG = compose(config_name="og_transporter")

    # generate dummy rgbd image
    rgbd = jnp.ones((1, 480, 640, 4))
    print("Input shape: ", rgbd.shape)

    # try to instantiate model
    pick_model = TransporterPick(config=TRANSPORTER_CONFIG["pick"])
    key = random.PRNGKey(0)
    params = pick_model.init(key, rgbd)
    dummy_output = pick_model.apply(params, rgbd, train=True)
    dummy_output = pick_model.apply(params, rgbd, train=False)
    print("Output shape: ", dummy_output.shape)

