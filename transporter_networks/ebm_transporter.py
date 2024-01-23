"Implementation of original transporter model."""

import dataclasses
import warnings
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import einops as e
from clu import metrics
import chex
import optax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
import flax.struct as struct
from flax.linen import initializers
from flax.training import train_state
from jax import random


import hydra
from hydra import compose, initialize
from hydra.utils import call, instantiate

class TransporterEBMPlaceNetwork(nn.Module):
    """
    Transporter network place module.

    In order to understand the network structure, please refer to the model config file README for a model card.
    """
    
    config: dict

    @nn.compact
    def __call__(self, pick_rgbd, place_rgbd, train=False):
        """Forward pass."""
        
        # first generate features from pick
        pick_rgbd_features = instantiate(self.config["query"]["input_projection"]["conv"])(pick_rgbd)
        pick_rgbd_features = call(self.config["query"]["input_projection"]["pool"])(pick_rgbd_features)
        for idx, block in enumerate(self.config["query"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = pick_rgbd_features.shape
                pick_rgbd_features = jax.image.resize(pick_rgbd_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = pick_rgbd_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        pick_rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(pick_rgbd_features)
                    elif i==mid_idx:
                        pick_rgbd_features = instantiate(block.resnet_block.conv, strides=[1,1])(pick_rgbd_features)
                    else:
                        pick_rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(pick_rgbd_features)
                    if self.config.use_batchnorm:
                        pick_rgbd_features = instantiate(block.resnet_block.norm, use_running_average=not train)(pick_rgbd_features)
                    
                    if i==block.num_blocks-1:
                        if residual.shape != pick_rgbd_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        pick_rgbd_features += residual
                    
                    if idx != len(self.config["query"]["blocks"])-1:
                        pick_rgbd_features = call(block.resnet_block.activation)(pick_rgbd_features)
        
        # second generate features for rgbd crop
        place_rgbd_features = instantiate(self.config["key"]["input_projection"]["conv"])(place_rgbd)
        place_rgbd_features = call(self.config["key"]["input_projection"]["pool"])(place_rgbd_features)
        for idx, block in enumerate(self.config["key"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = place_rgbd_features.shape
                place_rgbd_features = jax.image.resize(place_rgbd_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = place_rgbd_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        place_rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(place_rgbd_features)
                    elif i==mid_idx:
                        place_rgbd_features = instantiate(block.resnet_block.conv, strides=[1,1])(place_rgbd_features)
                    else:
                        place_rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(place_rgbd_features)
                    
                    if self.config.use_batchnorm:
                        place_rgbd_features = instantiate(block.resnet_block.norm, use_running_average=not train)(place_rgbd_features)
                    if i==block.num_blocks-1:
                        if residual.shape != place_rgbd_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        place_rgbd_features += residual

                    if idx != len(self.config["key"]["blocks"])-1:
                        place_rgbd_features = call(block.resnet_block.activation)(place_rgbd_features)
        

        q_vals = pick_rgbd_features * place_rgbd_features
        q_vals = jnp.sum(q_vals, axis=-1, keepdims=False)
        q_vals = e.rearrange(q_vals, "b h w -> b (h w)") # flatten spatial dims
        
        # normalize q_vals before softmax
        q_vals -= jnp.mean(q_vals, axis=-1, keepdims=True)
        q_vals /= jnp.std(q_vals, axis=-1, keepdims=True)
        q_vals = jax.nn.softmax(q_vals, axis=-1)
        
        return q_vals

