"""A ResNet Implementation."""

import einops as e
import chex
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.linen import initializers

from hydra.utils import instantiate

class ResNetV2Block(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):
        # start with convolution projection
        x = instantiate(self.config.input_projection.conv)(x)
        x = call(self.config.input_projection.pool)(x)

        # resnetv2block
        residual = x
        
        for _ in range(self.config.num_blocks):
            x = instantiate(self.config.resnet_block.norm)(x)
            x = call(self.config.resnet_block.activation)(x)
            x = instantiate(self.config.resnet_block.conv)(x)

        if residual.shape != x.shape:
            residual = instantiate(self.config.resnet_block.conv)(residual)
  
        x = x+residual
        
        #flatten output
        x = jnp.reshape(x, (*x.shape[:3], -1))
        x = instantiate(self.config.output_projection.dense)(x)

        return x

