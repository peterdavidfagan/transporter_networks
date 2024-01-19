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


class TransporterGoalNetwork(nn.Module):
    """
    Transporter network place module.

    In order to understand the network structure, please refer to the model config file README for a model card.
    """
    
    config: dict

    @nn.compact
    def __call__(self, rgbd, goal_rgbd, crop_idx, train=False):
        """Forward pass."""
        
        # first generate query features
        query_features= instantiate(self.config["query"]["input_projection"]["conv"])(rgbd)
        query_features= call(self.config["query"]["input_projection"]["pool"])(rgbd_features)
        for idx, block in enumerate(self.config["query"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = rgbd_features.shape
                query_features= jax.image.resize(rgbd_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = rgbd_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        query_features= instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(rgbd_features)
                    elif i==mid_idx:
                        query_features= instantiate(block.resnet_block.conv, strides=[1,1])(rgbd_features)
                    else:
                        query_features= instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(rgbd_features)
                    if self.config.use_batchnorm:
                        query_features= instantiate(block.resnet_block.norm, use_running_average=not train)(rgbd_features)
                    
                    if i==block.num_blocks-1:
                        if residual.shape != rgbd_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        query_features+= residual
                    
                    if idx != len(self.config["query"]["blocks"])-1:
                        query_features= call(block.resnet_block.activation)(rgbd_features)
        
        # second generate features for goal image
        goal_rgbd_features = instantiate(self.config["goal"]["input_projection"]["conv"])(goal_rgbd)
        goal_rgbd_features = call(self.config["goal"]["input_projection"]["pool"])(goal_rgbd_features)
        for idx, block in enumerate(self.config["goal"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = goal_rgbd_features.shape
                goal_rgbd_features = jax.image.resize(goal_rgbd_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = goal_rgbd_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        goal_rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(goal_rgbd_features)
                    elif i==mid_idx:
                        goal_rgbd_features = instantiate(block.resnet_block.conv, strides=[1,1])(goal_rgbd_features)
                    else:
                        goal_rgbd_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(goal_rgbd_features)
                    if self.config.use_batchnorm:
                        goal_rgbd_features = instantiate(block.resnet_block.norm, use_running_average=not train)(goal_rgbd_features)
                    
                    if i==block.num_blocks-1:
                        if residual.shape != goal_rgbd_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        goal_rgbd_features += residual

                    if idx != len(self.config["key"]["blocks"])-1:
                        goal_rgbd_features = call(block.resnet_block.activation)(goal_rgbd_features)

        # third generate key features
        key_features = instantiate(self.config["key"]["input_projection"]["conv"])(rgbd_crop)
        key_features = call(self.config["key"]["input_projection"]["pool"])(key_features)
        for idx, block in enumerate(self.config["key"]["blocks"]):
            if block.name=="upsample":
                B, H, W, C = key_features.shape
                key_features = jax.image.resize(key_features, shape=(B, H*2, W*2, C), method="bilinear")
            else:
                residual = key_features
                mid_idx = np.median([x for x in range(block.num_blocks)]) # get index of middle block
                for i in range(block.num_blocks):
                    if i==0:
                        key_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], padding="VALID")(key_features)
                    elif i==mid_idx:
                        key_features = instantiate(block.resnet_block.conv, strides=[1,1])(key_features)
                    else:
                        key_features = instantiate(block.resnet_block.conv, kernel_size=[1,1], strides=[1,1], padding="VALID")(key_features)
                    
                    if self.config.use_batchnorm:
                        key_features = instantiate(block.resnet_block.norm, use_running_average=not train)(key_features)
                    if i==block.num_blocks-1:
                        if residual.shape != key_features.shape: # check if residual needs to be projected
                            residual = instantiate(block.resnet_block.conv)(residual)
                        key_features += residual

                    if idx != len(self.config["key"]["blocks"])-1:
                        key_features = call(block.resnet_block.activation)(key_features)
        

        # condition on goal
        query_features = query_features * goal_rgbd_features
        key_features = key_features * goal_rgbd_features
        
        # crop query feature map
        kernel = jax.lax.dynamic_slice(
                query_features, 
                (0, crop_idx[0], crop_idx[1], 0), 
                (query_features.shape[0], crop_size[0], crop_size[1], query_features.shape[-1]),
                )

        kernel = e.rearrange(kernel, "b h w c -> b 1 h w c")
        key_features = e.rearrange(key_features, "b h w c -> b 1 h w c")
        dn = jax.lax.conv_dimension_numbers(key_features.shape[1:], kernel.shape[1:], ('NHWC', 'OHWI', 'NHWC'))
        q_vals = jax.vmap(jax.lax.conv_general_dilated, (0, 0, None, None, None, None, None), 0)(
            key_features,
            kernel,
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
class GoalTransporter:
    """Transporter model."""
    pick_model_state: train_state.TrainState
    place_model_state: train_state.TrainState

def create_goal_transporter_place_train_state(
        rgbd,
        goal,
        crop_idx,
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
        goal,
        crop_idx,
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

def place_train_step(
        state,
        rgbd,
        goal,
        crop_idx,
        target_pixel_ids,
        ):

    def compute_place_loss(params):
        """Compute place loss."""
        q_vals = state.apply_fn({"params": params},# "batch_stats": state.batch_stats}, 
                rgbd,
                goal,
                crop_idx,
                train=True,
                #mutable=["batch_stats"],
                )
        
        # compute softmax over image output
        target = jax.nn.one_hot(target_pixel_ids, num_classes=q_vals.shape[-1])
        loss = -jnp.sum(jnp.multiply(target, jnp.log(q_vals+1e-8)), axis=-1).mean() # add near zero to avoid log(0)
        
        return loss 

    # compute and apply gradients
    grad_fn = jax.value_and_grad(compute_place_loss, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # update batch stats
    #state = state.replace(batch_stats=updates["batch_stats"])

    # update metrics (TODO: consider merging place components, currently storing metrics on query state)
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    state = state.replace(metrics = state.metrics.merge(metric_updates))

    return state, loss

if __name__ == "__main__":
    # read network config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="./config", job_name="default_config")
    TRANSPORTER_CONFIG = compose(config_name="og_transporter")

    # generate dummy rgbd image
    rgbd = jnp.ones((1, 480, 640, 4))
    goal = jnp.ones((1, 480, 640, 4))
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
