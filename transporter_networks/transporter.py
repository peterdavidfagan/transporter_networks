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

        for idx, block in enumerate(self.config["blocks"]):
            # either apply upsampling block or resnet block
            if block.name=="upsample":
                B, H, W, C = x.shape
                x = jax.image.resize(x, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = x
                # original implementation has differing dimensions for layers within a given resnet block
                # for instance only the middle conv layer has a kernel size different from (1,1)
                # similarly stride is (1,1) except for first conv layer
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                
                for i in range(block.num_blocks):
                    
                    # apply convolution with different params depending on position in block
                    if i==0:
                        x = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(x)
                    elif i==mid_idx:
                        x = instantiate(block.resnet_block.conv, strides=[1,1])(x)
                    else:
                        x = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(x)
                    
                    # apply norm
                    # x = instantiate(block.resnet_block.norm, use_running_average=not train)(x)
                    
                    # if last layer in block add residual
                    if i==block.num_blocks-1:
                        if residual.shape != x.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        x += residual
                    
                    if idx != len(self.config["blocks"])-1:
                        x = call(block.resnet_block.activation)(x)

        # compute softmax over image output
        if self.config.output_softmax:
            q_vals = e.rearrange(x, "b h w c -> b (h w c)") # flatten spatial dims
            
            # normalize q_vals before softmax
            q_vals -= jnp.mean(q_vals, axis=-1, keepdims=True)
            q_vals /= jnp.std(q_vals, axis=-1, keepdims=True)
            
            x = jax.nn.softmax(q_vals, axis=-1)

        return x


class TransporterPlaceNetwork(nn.Module):
    """
    Transporter network place module.

    In order to understand the network structure, please refer to the model config file README for a model card.
    """
    
    config: dict

    @nn.compact
    def __call__(self, rgbd, rgbd_crop, train=False):
        """Forward pass."""
        
        # first generate features for rgbd image
        rgbd_features = instantiate(self.config["query"]["input_projection"]["conv"])(rgbd)
        rgbd_features = call(self.config["query"]["input_projection"]["pool"])(rgbd_features)
        for idx, block in enumerate(self.config["query"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = rgbd_features.shape
                rgbd_features = jax.image.resize(rgbd_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = rgbd_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(rgbd_features)
                    elif i==mid_idx:
                        rgbd_features = instantiate(block.resnet_block.conv, strides=[1,1])(rgbd_features)
                    else:
                        rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(rgbd_features)
                    if self.config.use_batchnorm:
                        rgbd_features = instantiate(block.resnet_block.norm, use_running_average=not train)(rgbd_features)
                    
                    if i==block.num_blocks-1:
                        if residual.shape != rgbd_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        rgbd_features += residual
                    
                    if idx != len(self.config["query"]["blocks"])-1:
                        rgbd_features = call(block.resnet_block.activation)(rgbd_features)
        
        # second generate features for rgbd crop
        rgbd_crop_features = instantiate(self.config["key"]["input_projection"]["conv"])(rgbd_crop)
        rgbd_crop_features = call(self.config["key"]["input_projection"]["pool"])(rgbd_crop_features)
        for idx, block in enumerate(self.config["key"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = rgbd_crop_features.shape
                rgbd_crop_features = jax.image.resize(rgbd_crop_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = rgbd_crop_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        rgbd_crop_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(rgbd_crop_features)
                    elif i==mid_idx:
                        rgbd_crop_features = instantiate(block.resnet_block.conv, strides=[1,1])(rgbd_crop_features)
                    else:
                        rgbd_crop_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(rgbd_crop_features)
                    
                    if self.config.use_batchnorm:
                        rgbd_crop_features = instantiate(block.resnet_block.norm, use_running_average=not train)(rgbd_crop_features)
                    if i==block.num_blocks-1:
                        if residual.shape != rgbd_crop_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        rgbd_crop_features += residual

                    if idx != len(self.config["key"]["blocks"])-1:
                        rgbd_crop_features = call(block.resnet_block.activation)(rgbd_crop_features)
        

        rgbd_features = e.rearrange(rgbd_features, "b h w c -> b 1 h w c")
        rgbd_crop_features = e.rearrange(rgbd_crop_features, "b h w c -> b 1 h w c")
        dn = jax.lax.conv_dimension_numbers(rgbd_features.shape[1:], rgbd_crop_features.shape[1:], ('NHWC', 'OHWI', 'NHWC'))
        q_vals = jax.vmap(jax.lax.conv_general_dilated, (0, 0, None, None, None, None, None), 0)(
            rgbd_features,
            rgbd_crop_features,
            (1, 1),
            "SAME",
            None,
            None,
            dn)
        q_vals = e.rearrange(q_vals, "b d h w c -> b (d h w c)") # flatten spatial dims
        
        # normalize q_vals before softmax
        q_vals -= jnp.mean(q_vals, axis=-1, keepdims=True)
        q_vals /= jnp.std(q_vals, axis=-1, keepdims=True)
        q_vals = jax.nn.softmax(q_vals, axis=-1)
        
        return q_vals


@struct.dataclass
class Transporter:
    """Transporter model."""
    pick_model_state: train_state.TrainState
    place_model_state: train_state.TrainState

@struct.dataclass
class TransporterMetrics(metrics.Collection):
    """Transporter Training Metrics."""
    loss: metrics.Average.from_output("loss")
    success_rate: metrics.Average.from_output("success_rate")

class TransporterTrainState(train_state.TrainState):
    """Transporter training state."""
    batch_stats: Any
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
        train=False,
            )
    params = variables["params"]
    #batch_stats = variables["batch_stats"]

    return TransporterTrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=None,
        tx=optimizer,
        metrics=TransporterMetrics.empty(),
        )

def create_transporter_place_train_state(
        rgbd,
        rgbd_crop,
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
        rgbd_crop,
        train=False,
            )
    params = variables["params"]
    #batch_stats = variables["batch_stats"]

    return TransporterTrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=None,#batch_stats,
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
        q_vals = state.apply_fn({"params": params},
                rgbd,
                train=True,
                #mutable=["batch_stats"],
                )
        target = jax.nn.one_hot(target_pixel_ids, num_classes=q_vals.shape[-1])
        loss = -jnp.sum(jnp.multiply(target, jnp.log(q_vals+1e-8)), axis=-1).mean() # add near zero to avoid log(0)
        predicted_idx = jnp.argmax(q_vals, axis=-1)
        success_rate = jnp.sum(predicted_idx == target_pixel_ids) / target_pixel_ids.size
        
        return loss, success_rate

    # compute and apply gradients
    grad_fn = jax.value_and_grad(compute_pick_loss, has_aux=True)
    (loss, success_rate), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # update batch stats
    #state = state.replace(batch_stats=updates["batch_stats"])

    # update metrics
    metric_updates = state.metrics.single_from_model_output(loss=loss, success_rate=success_rate)
    state = state.replace(metrics = state.metrics.merge(metric_updates))

    return state, loss, success_rate

def place_train_step(
        state,
        rgbd,
        rgbd_crop,
        target_pixel_ids,
        ):

    def compute_place_loss(params):
        """Compute place loss."""
        q_vals = state.apply_fn({"params": params},# "batch_stats": state.batch_stats}, 
                rgbd,
                rgbd_crop,
                train=True,
                #mutable=["batch_stats"],
                )
        
        # compute softmax over image output
        target = jax.nn.one_hot(target_pixel_ids, num_classes=q_vals.shape[-1])
        loss = -jnp.sum(jnp.multiply(target, jnp.log(q_vals+1e-8)), axis=-1).mean() # add near zero to avoid log(0)
        predicted_idx = jnp.argmax(q_vals, axis=-1)
        success_rate = jnp.sum(predicted_idx == target_pixel_ids) / target_pixel_ids.size
        
        return loss , success_rate

    # compute and apply gradients
    grad_fn = jax.value_and_grad(compute_place_loss, has_aux=True)
    (loss, success_rate), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # update batch stats
    #state = state.replace(batch_stats=updates["batch_stats"])

    # update metrics (TODO: consider merging place components, currently storing metrics on query state)
    metric_updates = state.metrics.single_from_model_output(loss=loss, success_rate=success_rate)
    state = state.replace(metrics = state.metrics.merge(metric_updates))

    return state, loss, success_rate

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
