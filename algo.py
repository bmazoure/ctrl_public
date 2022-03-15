from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.random import PRNGKey
from flax import core
from flax import struct
import optax
import time
"""
Inspired by code from Flax: https://github.com/google/flax/blob/main/examples/ppo/ppo_lib.py
"""


def compute_distance(A):
    similarity = jnp.dot(A, A.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = jnp.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    # inv_square_mag[jnp.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = jnp.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag


@partial(jax.jit, static_argnames=("apply_fn", "policy_fn", "sample"))
def select_action(
    params,
    apply_fn,
    policy_fn,
    state: jnp.ndarray,
    rng: PRNGKey,
    sample: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKey]:
    value, pi = apply_fn(params,
                         state,
                         jnp.zeros(len(state)),
                         method=policy_fn)

    if sample:
        rng, key = jax.random.split(rng)
        action = pi.sample(seed=key)
    else:
        action = pi.mode()

    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], rng


def get_transition(
    train_state: TrainState,
    policy_fn,
    env,
    state,
    batch,
    rng: PRNGKey,
):
    action, log_pi, value, new_key = select_action(train_state.params,
                                                   train_state.apply_fn,
                                                   policy_fn,
                                                   state.astype(jnp.float32) /
                                                   255.,
                                                   rng,
                                                   sample=True)
    next_state, reward, done, _ = env.step(action)
    batch.append(state, action, reward, done, np.array(log_pi),
                 np.array(value))
    return train_state, next_state, batch, new_key


@partial(jax.jit)
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def extract_windows_vectorized(array, window_size):
    max_timestep = array.shape[0] - window_size
    sub_windows = (0 + np.expand_dims(np.arange(window_size), 0) +
                   np.expand_dims(np.arange(max_timestep + 1), 0).T)

    return array[sub_windows]


def state_update(online_state,
                 target_state,
                 tau: float = 1.):
    """ Update key weights as tau * online + (1-tau) * target
    """
    fc_pi = online_state.params['params']['fc_pi'].copy(add_or_replace={})
    fc_v = online_state.params['params']['fc_v'].copy(add_or_replace={})
    protos = online_state.params['params']['protos'].copy(add_or_replace={})
    weights = target_update(online_state.params['params'], target_state.params['params'], tau)
    
    weights = weights.copy(add_or_replace={"fc_pi":fc_pi,
                                            "fc_v":fc_v,
                                            "protos":protos})
    
    new_params = target_state.params.copy(
        add_or_replace={
            'params':
            weights
        })

    target_state = target_state.replace(params=new_params)
    return target_state


def target_update(online, target, tau: float):
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), online, target)

    return new_target_params


def l2_normalize(A, axis=-1, eps=1e-8):
    return A * jax.lax.rsqrt((A * A).sum(axis=axis, keepdims=True) + eps)


def group_by(data, var):
    x, g = var
    x_grouped, group_cnts = data
    # append entries into specified group
    x_grouped = jax.ops.index_add(x_grouped, (g, group_cnts[g]), x)
    # track how many entries appended into each group
    group_cnts = jax.ops.index_add(group_cnts, g, 1)
    return (x_grouped, group_cnts), 0  # '0' is just a dummy value


def cos_loss(p, z):
    z = jax.lax.stop_gradient(z)
    p = l2_normalize(p, axis=1)
    z = l2_normalize(z, axis=1)
    dist = 2 - 2 * jnp.sum(p * z, axis=1)
    return dist


def sinkhorn(scores, temp=0.1, k=3):
    Q = scores / temp
    Q -= jnp.max(Q)

    Q = jnp.transpose(jnp.exp(Q))
    Q /= jnp.sum(Q)

    r = jnp.ones(jnp.shape(Q)[0]) / Q.shape[0]
    c = jnp.ones(jnp.shape(Q)[1]) / Q.shape[1]

    for _ in jnp.arange(k):
        u = jnp.sum(Q, axis=1)
        u = r / u
        Q *= jnp.expand_dims(u, axis=1)
        Q *= jnp.expand_dims((c / jnp.sum(Q, axis=0)), axis=0)
    Q = Q / jnp.sum(Q, axis=0, keepdims=True)
    return jnp.transpose(Q)


def loss_cluster(params_embedding, apply_fn_embedding, params_target, apply_fn_target, cluster_fn, cluster_target_fn, protos_fn,
                 temp, k, myow_k, num_clusters, myow_reg, rng, state, action,
                 reward):
    state = state.astype(jnp.float32) / 255.

    v_clust, w_clust, v_pred, w_pred = apply_fn_embedding(params_embedding,
                                                          state=state,
                                                          action=action,
                                                          reward=reward,
                                                          method=cluster_fn)

    v_clust_target, w_clust_target, v_pred_target, w_pred_target = apply_fn_target(params_target,
                                                          state=state,
                                                          action=action,
                                                          reward=reward,
                                                          method=cluster_target_fn)

    protos = apply_fn_embedding(params_embedding,
                                jnp.eye(v_clust.shape[1]),
                                method=protos_fn)

    v_clust_target = l2_normalize(v_clust_target, axis=-1)
    # w_clust_target = l2_normalize(w_clust_target, axis=-1)

    v_clust = l2_normalize(v_clust, axis=-1)
    # w_clust = l2_normalize(w_clust, axis=-1)
    protos = l2_normalize(protos, axis=-1)

    scores_v = v_clust @ protos
    log_p = nn.log_softmax(scores_v / temp, axis=1)


    scores_v_target = v_clust_target @ protos
    # scores_w_target = w_clust_target @ protos
    q_target = sinkhorn(scores_v_target, temp=temp, k=k)
    # q_target = jax.lax.stop_gradient(q_target)
    proto_loss = -jnp.mean(jnp.sum(q_target * log_p, axis=1))

    # MYOW loss
    dist = compute_distance(jnp.transpose(protos))
    vals, indx = jax.lax.top_k(-dist, myow_k + 1)
    cluster_idx = jnp.argmax(scores_w_target, 1)
    X_grouped = jnp.zeros((num_clusters, myow_k + 1))
    group_cnts = jnp.zeros((num_clusters, ), np.int32)

    (cluster_membership_list, group_cnts), _ = jax.lax.scan(
        group_by,
        (X_grouped, group_cnts),  # initial state
        (jnp.arange(len(cluster_idx)), cluster_idx))  # data to loop over

    myow_loss = 0.
    for k_idx in range(myow_k):
        nearby_cluster_idx = jnp.take_along_axis(indx[:, 0 + 1],
                                                 cluster_idx,
                                                 axis=0)
        rng, key = jax.random.split(rng)
        idx = jax.random.randint(key=key,
                                 shape=(cluster_membership_list.shape[0], 1),
                                 minval=0,
                                 maxval=myow_k + 1)
        nearby_vec_idx = jnp.expand_dims(
            jnp.take_along_axis(jnp.take_along_axis(cluster_membership_list,
                                                    idx,
                                                    axis=1)[:, 0],
                                nearby_cluster_idx,
                                axis=0), 1)
        nearby_vec = jnp.take_along_axis(w_pred_target, nearby_vec_idx, axis=0)
        myow_loss += cos_loss(v_pred, nearby_vec).mean()
    myow_loss *= myow_reg
    joint_loss = proto_loss + myow_loss
    return joint_loss, (q_target, proto_loss, myow_loss)

@partial(jax.jit,
         static_argnames=("apply_fn", "policy_fn", "clip_eps", "critic_coeff", "entropy_coeff"))
def loss_critic(params_model: flax.core.frozen_dict.FrozenDict,
                apply_fn: Callable[..., Any], policy_fn: Callable,
                state: jnp.ndarray, target: jnp.ndarray,
                value_old: jnp.ndarray, log_pi_old: jnp.ndarray,
                gae: jnp.ndarray, action: jnp.ndarray, clip_eps: float,
                critic_coeff: float, entropy_coeff: float) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.

    value_pred = apply_fn(params_model, state, method=policy_fn)
    value_pred = value_pred[:, 0]

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    return value_loss, (value_pred.mean())

@partial(jax.jit,
         static_argnames=("apply_fn", "policy_fn", "clip_eps", "critic_coeff", "entropy_coeff"))
def loss_actor_and_gae(params_model: flax.core.frozen_dict.FrozenDict,
                       apply_fn: Callable[..., Any], policy_fn: Callable,
                       state: jnp.ndarray, target: jnp.ndarray,
                       value_old: jnp.ndarray, log_pi_old: jnp.ndarray,
                       gae: jnp.ndarray, action: jnp.ndarray, clip_eps: float,
                       critic_coeff: float,
                       entropy_coeff: float) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.
    gae = (gae - gae.mean()) / (gae.std() + 1e-5)

    adv_pred, pi = apply_fn(params_model,
                            state,
                            action[:, 0],
                            method=policy_fn)
    adv_pred = adv_pred.squeeze(1)
    adv_loss = jnp.square(adv_pred - gae).mean()

    log_prob = pi.log_prob(action[:, 0])
    ratio = jnp.exp(log_prob - log_pi_old)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

    ent = pi.entropy().mean()

    total_loss = loss_actor + 0.25 * adv_loss - entropy_coeff * ent

    return total_loss, (adv_loss, loss_actor, ent, target.mean(), gae.mean())


def loss_actor_and_critic(params_model: flax.core.frozen_dict.FrozenDict,
                          apply_fn: Callable[..., Any], policy_fn: Callable,
                          state: jnp.ndarray, target: jnp.ndarray,
                          value_old: jnp.ndarray, log_pi_old: jnp.ndarray,
                          gae: jnp.ndarray, action: jnp.ndarray,
                          clip_eps: float, critic_coeff: float,
                          entropy_coeff: float) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.

    value_pred, pi = apply_fn(params_model, state, method=policy_fn)
    value_pred = value_pred[:, 0]

    log_prob = pi.log_prob(action[:, 0])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    ent = pi.entropy().mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * ent

    return total_loss, (value_loss, loss_actor, ent, value_pred.mean(),
                        target.mean(), gae.mean())


@partial(jax.jit,
         static_argnames=("policy_fn", "num_envs", "n_steps", "n_minibatch", "epoch_ppo", "clip_eps", "entropy_coeff", "critic_coeff"))
def update_ppo(train_state: TrainState, policy_fn, batch: Tuple, num_envs: int,
               n_steps: int, n_minibatch: int, epoch_ppo: int, clip_eps: float,
               entropy_coeff: float, critic_coeff: float, rng: PRNGKey):

    state, action, reward, log_pi_old, value, target, gae = batch

    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(num_envs * n_steps)
    idxes_policy = []
    for _ in range(epoch_ppo):
        rng, key = jax.random.split(rng)
        idxes = jax.random.permutation(rng, idxes)
        idxes_policy.append(idxes)
    idxes_policy = jnp.array(idxes_policy).reshape(-1, size_minibatch)

    avg_metrics_dict = defaultdict(int)

    state = flatten_dims(state)
    action = flatten_dims(action).reshape(-1, 1)
    log_pi_old = flatten_dims(log_pi_old)
    value = flatten_dims(value)
    target = flatten_dims(target)
    gae = flatten_dims(gae)

    def scan_policy(train_state, idx):
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    policy_fn=policy_fn,
                                    state=state[idx],
                                    target=target[idx],
                                    value_old=value[idx],
                                    log_pi_old=log_pi_old[idx],
                                    gae=gae[idx],
                                    action=action[idx],
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff)

        train_state = train_state.apply_gradients(grads=grads, opt_idx=0)
        return train_state, total_loss
    train_state, total_loss = jax.lax.scan(scan_policy, train_state, idxes_policy)
    total_loss, (value_loss, loss_actor, ent, value_pred, target_val, gae_val) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['value_loss'] += value_loss.mean()
    avg_metrics_dict['loss_actor'] += loss_actor.mean()
    avg_metrics_dict['ent'] += ent.mean()
    avg_metrics_dict['value_pred'] += value_pred.mean()
    avg_metrics_dict['target_val'] += target_val.mean()
    avg_metrics_dict['gae_val'] += gae_val.mean()

    return avg_metrics_dict, train_state, rng


@partial(jax.jit,
         static_argnames=("policy_fn", "step","num_envs", "n_steps", "n_minibatch", "epoch_policy", "epoch_value", "clip_eps", "entropy_coeff", "critic_coeff"))
def update_daac(train_state: TrainState, policy_fn, batch: Tuple, step: int,
                num_envs: int, n_steps: int, n_minibatch: int,
                epoch_policy: int, epoch_value: int, clip_eps: float,
                entropy_coeff: float, critic_coeff: float, rng: PRNGKey):

    state, action, reward, log_pi_old, value, target, gae = batch

    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(num_envs * n_steps)
    idxes_policy = []
    for _ in range(epoch_policy):
        rng, key = jax.random.split(rng)
        idxes = jax.random.permutation(rng, idxes)
        idxes_policy.append(idxes)
    idxes_policy = jnp.array(idxes_policy).reshape(-1, size_minibatch)

    idxes_value = []
    for _ in range(epoch_value):
        rng, key = jax.random.split(rng)
        idxes = jax.random.permutation(rng, idxes)
        idxes_value.append(idxes)
    idxes_value = jnp.array(idxes_value).reshape(-1, size_minibatch)

    avg_metrics_dict = defaultdict(int)

    state = flatten_dims(state)
    action = flatten_dims(action).reshape(-1, 1)
    log_pi_old = flatten_dims(log_pi_old)
    value = flatten_dims(value)
    target = flatten_dims(target)
    gae = flatten_dims(gae)

    def scan_policy(train_state, idx):
        grad_fn = jax.value_and_grad(loss_actor_and_gae, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    policy_fn=policy_fn,
                                    state=state[idx],
                                    target=target[idx],
                                    value_old=value[idx],
                                    log_pi_old=log_pi_old[idx],
                                    gae=gae[idx],
                                    action=action[idx],
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff)

        train_state = train_state.apply_gradients(grads=grads, opt_idx=0)
        return train_state, total_loss
    train_state, total_loss = jax.lax.scan(scan_policy, train_state, idxes_policy)
    total_loss, (adv_loss, loss_actor, ent, target_val, gae_val) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['adv_loss'] += adv_loss.mean()
    avg_metrics_dict['loss_actor'] += loss_actor.mean()
    avg_metrics_dict['ent'] += ent.mean()
    avg_metrics_dict['target_val'] += target_val.mean()
    avg_metrics_dict['gae_val'] += gae_val.mean()
    
    def scan_value(train_state, idx):
        grad_fn = jax.value_and_grad(loss_critic, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    policy_fn=policy_fn,
                                    state=state[idx],
                                    target=target[idx],
                                    value_old=value[idx],
                                    log_pi_old=log_pi_old[idx],
                                    gae=gae[idx],
                                    action=action[idx],
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff)

        train_state = train_state.apply_gradients(grads=grads, opt_idx=2)
        return train_state, total_loss
    train_state, total_loss = jax.lax.scan(scan_value, train_state, idxes_value)
    total_loss, (value_pred) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['adv_loss'] += value_pred.mean()

    return avg_metrics_dict, train_state, rng


@partial(jax.jit,
         static_argnames=("cluster_fn", "cluster_target_fn", "protos_fn", "num_envs", "n_minibatch", "epoch_cluster", "temp", "k", "myow_k",
                          "num_clusters", "cluster_len", "myow_reg"))
def update_cluster(train_state: TrainState, target_state: TrainState, cluster_fn, cluster_target_fn, protos_fn,
                   batch: Tuple, num_envs: int, n_minibatch: int,
                   epoch_cluster: int, temp: float, k: int, myow_k: int,
                   num_clusters: int, cluster_len: int, myow_reg: float,
                   rng: PRNGKey):
    state, action, reward, _, _, _, _ = batch

    n_steps = state.shape[0]

    # CTRL has large memory footprint bc sliding window so take 1/8 of them
    size_batch = num_envs * (n_steps-cluster_len+1) // 2
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(size_batch)
    idxes_cluster = jax.random.permutation(rng, idxes).reshape(-1, size_minibatch)
    
    img_shape = state.shape[2:]

    avg_metrics_dict = defaultdict(int)

    state = extract_windows_vectorized(state, cluster_len).transpose(0, 2, 1, 3, 4, 5).reshape(-1, cluster_len, *img_shape)
    action = extract_windows_vectorized(action, cluster_len).transpose(0, 2, 1).reshape(-1, cluster_len)

    reward = extract_windows_vectorized(reward, cluster_len).transpose(0, 2, 1).reshape(-1, cluster_len)
    
    def scan_cluster(train_state, idx):
        grad_fn = jax.value_and_grad(loss_cluster, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                        train_state.apply_fn,
                                        target_state.params,
                                        target_state.apply_fn,
                                        cluster_fn,
                                        cluster_target_fn,
                                        protos_fn,
                                        temp=temp,
                                        k=k,
                                        myow_k=myow_k,
                                        num_clusters=num_clusters,
                                        myow_reg=myow_reg,
                                        rng=rng,
                                        state=state[idx],
                                        action=action[idx],
                                        reward=reward[idx])

        train_state = train_state.apply_gradients(grads=grads, opt_idx=1)
        return train_state, total_loss
    train_state, total_loss = jax.lax.scan(scan_cluster, train_state, idxes_cluster)
    total_loss, (soft_scores, proto_loss, myow_loss) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['proto_loss'] += proto_loss.mean()
    avg_metrics_dict['myow_loss'] += myow_loss.mean()

    return avg_metrics_dict, train_state, rng


class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

  Synopsis::

      state = TrainState.create(
          apply_fn=model.apply,
          params=variables['params'],
          tx=tx)
      grad_fn = jax.grad(make_loss_fn(state.apply_fn))
      for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

  Note that you can easily extend this dataclass by subclassing it for storing
  additional data (e.g. additional variable collections).

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: Sequence[optax.GradientTransformation] = struct.field(
        pytree_node=False)
    opt_state: [optax.OptState]

    def apply_gradients(self, *, grads, opt_idx, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx[opt_idx].update(
            grads, self.opt_state[opt_idx], self.params)
        new_params = optax.apply_updates(self.params, updates)
        twin_opt_state = []
        for i in range(len(self.opt_state)):
            if i == opt_idx:
                twin_opt_state.append(new_opt_state)
            else:
                twin_opt_state.append(self.opt_state[i])
        twin_opt_state = tuple(twin_opt_state)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=twin_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        print(f"JAX BACKEND: {jax.default_backend()}")
        opt_state = tuple([tx_.init(params) for tx_ in tx])
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
