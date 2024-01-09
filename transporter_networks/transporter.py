"""Implementation of original transporter model."""

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
import flax.linen as nn
import flax.struct as struct
from flax.linen import initializers
from flax.training import train_state
from jax import random


import hydra
from hydra import compose, initialize
from hydra.utils import call, instantiate


class TransporterNetwork(nn.Module):
    """
    Transporter network module.

    In order to understand the network structure, please refer to the model config file README for a model card.
    """
    
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
        
        if self.config.output_softmax:
            x = jax.nn.softmax(x, axis=-1)
            x = e.rearrange(x, "b h w c -> b h (w c)") # squeeze last dim

        return x


@struct.dataclass
class Transporter:
    """Transporter model."""
    pick_model_state: train_state.TrainState
    place_model_query_state: train_state.TrainState
    place_model_key_state: train_state.TrainState

@struct.dataclass
class TransporterMetrics(metrics.Collection):
    """Transporter Training Metrics."""
    loss: metrics.Average.from_output("loss")

class TransporterTrainState(train_state.TrainState):
    """Transporter training state."""
    metrics: TransporterMetrics


def create_transporter_train_state(
        rgbd,
        model,
        model_key,
        optimizer,
        ):
    """Create initial training state."""
    variables = model.init(
        {
            "params": model_key,
        },
        rgbd,
            )
    params = variables["params"]

    return TransporterTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics=TransporterMetrics.empty(),
        )

def pick_train_step(
        state,
        rgbd, 
        target_pixel_ids,
        ):
    """Train step for pick model."""

    def compute_pick_loss(params):
        """Compute pick loss."""
        q_vals = state.apply_fn({"params": params}, rgbd).reshape(rgbd.shape[0], -1)
        target = jax.nn.one_hot(target_pixel_ids, num_classes=q_vals.shape[-1])
        loss = optax.softmax_cross_entropy(logits=q_vals, labels=target).mean()
        return loss

    # compute gradients
    grad_fn = jax.grad(compute_pick_loss)
    grads = grad_fn(state.params)

    # update state
    state = state.apply_gradients(grads=grads)

    # TODO: update metrics

    return state

def place_train_step(
        query_state,
        key_state,
        rgbd,
        rgbd_crop,
        target_pixel_ids,
        ):

    def compute_place_loss(query_params, key_params):
        """Compute place loss."""
        query_q_vals = query_state.apply_fn({"params": query_params}, rgbd)
        key_q_vals = key_state.apply_fn({"params": key_params}, rgbd_crop)
        jax.debug.print("query_q_vals: {shape}", shape=query_q_vals.shape)
        jax.debug.print("key_q_vals: {shape}", shape=key_q_vals.shape)

        # convolve key_q_vals with query_q_vals
        #query_q_vals = e.rearrange(query_q_vals, "b h w c -> b c h w")
        dn = jax.lax.conv_dimension_numbers(query_q_vals.shape, key_q_vals.shape, ('NHWC', 'OHWI', 'NHWC'))
        q_vals = jax.lax.conv_general_dilated(
            query_q_vals,
            key_q_vals,
            (1, 1),
            "SAME",
            (1, 1),
            (1, 1),
            dn)
        jax.debug.print("q_vals: {shape}", shape=q_vals.shape)
        # for now take mean over channels
        q_vals = q_vals.mean(axis=-1)
        jax.debug.print("q_vals: {shape}", shape=q_vals.shape)
        q_vals = e.rearrange(q_vals, "b h w -> b (h w)")
        
        target = jax.nn.one_hot(target_pixel_ids, num_classes=q_vals.shape[-1])
        jax.debug.print("target: {shape}", shape=target.shape)
        loss = optax.softmax_cross_entropy(logits=q_vals, labels=target).mean()
        return loss

    # compute gradients
    grad_fn = jax.grad(compute_place_loss, argnums=(0, 1))
    query_grads, key_grads = grad_fn(query_state.params, key_state.params)

    # update state
    query_state = query_state.apply_gradients(grads=query_grads)
    key_state = key_state.apply_gradients(grads=key_grads)

    return query_state, key_state

if __name__ == "__main__":
    # read network config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="./config", job_name="default_config")
    TRANSPORTER_CONFIG = compose(config_name="og_transporter")

    # generate dummy rgbd image
    rgbd = jnp.ones((1, 480, 640, 4))
    print("Input shape: ", rgbd.shape)

    # try to instantiate pick model
    pick_model = TransporterNetwork(config=TRANSPORTER_CONFIG["pick"])
    key = random.PRNGKey(0)
    params = pick_model.init(key, rgbd)
    dummy_output = pick_model.apply(params, rgbd, train=False)
    print("Pick output shape: ", dummy_output.shape)

    # try to instantiate place model
    place_model = TransporterNetwork(config=TRANSPORTER_CONFIG["place"]["query"])
    params = place_model.init(key, rgbd)
    dummy_output = place_model.apply(params, rgbd, train=False)
    print("Place output shape: ", dummy_output.shape)
    

    # try to instantiate train state for all models
    key1, key2, key3 = random.split(key, 3)
    optimizer1 = optax.adam(1e-3)
    optimizer2 = optax.adam(1e-3)
    optimizer3 = optax.adam(1e-3)
    
    pick_model_state = create_transporter_train_state(
        rgbd,
        pick_model,
        key1,
        optimizer1,
        )
    
    place_model_query_state = create_transporter_train_state(
        rgbd,
        place_model,
        key2,
        optimizer2,
        )
    
    place_model_key_state = create_transporter_train_state(
        rgbd,
        place_model,
        key3,
        optimizer3,
        )
    
    transporter_model = Transporter(
        pick_model=pick_model_state,
        place_model_query=place_model_query_state,
        place_model_key=place_model_key_state,
        )
