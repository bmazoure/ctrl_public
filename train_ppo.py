import os
from collections import deque

import jax.numpy as jnp
import numpy as np
import optax
import wandb
import time
from absl import app, flags
from flax.training.train_state import TrainState
from algo import TrainState
# from flax.training import checkpoints
from jax.random import PRNGKey

from algo import get_transition, select_action, update_ppo, update_daac, update_cluster, state_update
from buffer import Batch
from models import CTRLModel, CTRLDAACModel
from vec_env import ProcgenVecEnvCustom


def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)


"""
To run:
JAX_OMNISTAGING=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python train_ppo.py --wandb_mode=disabled
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of Procgen envs.")
flags.DEFINE_integer("train_steps", 25_000_000, "Number of train frames.")
flags.DEFINE_enum("algo", "ppo_ctrl",
                  ['ppo_ctrl', 'daac_ctrl', 'daac', 'ppo'], "Algorithm name.")
# PPO
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("gamma", 0.999, "Gamma")
flags.DEFINE_integer("n_steps", 256, "GAE n-steps")
flags.DEFINE_integer("n_minibatch", 8, "Number of PPO minibatches")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
flags.DEFINE_integer("epoch_ppo", 3, "Number of PPO epochs on a single batch")
flags.DEFINE_float("clip_eps", 0.2, "Clipping range")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")
flags.DEFINE_float("entropy_coeff", 0.01, "Entropy loss coefficient")
flags.DEFINE_float("critic_coeff", 0.5, "Value loss coefficient")
# CTRL
flags.DEFINE_float("lr_ctrl", 1e-4, "CTRL learning rate")
flags.DEFINE_string("embedding_type", "concat", "Type (concat, attention)")
flags.DEFINE_integer("n_att_heads", 2, "Number of attention heads")
flags.DEFINE_float("temp", 0.1, "Clustering temperature")
flags.DEFINE_integer("k", 1, "Clustering sub-iterations")
flags.DEFINE_integer("cluster_len", 10, "Cluster length (timesteps)")
flags.DEFINE_integer("num_clusters", 200, "Number of prototypes")
flags.DEFINE_integer("epoch_ctrl", 1, "Number of CTRL epochs")
flags.DEFINE_integer("n_minibatch_ctrl", 8, "Number of CTRL minibatches")
flags.DEFINE_integer("myow_k", 1, "MYOW k-NN")
flags.DEFINE_float("myow_reg", 1, "MYOW coeff wrt proto loss")
flags.DEFINE_float("ema_ctrl", 0.95, "EMA for target network for SK and MYOW targets")
# Logging
flags.DEFINE_integer("checkpoint_interval", 25 * 999424,
                     "Chcekpoint frequency (about 1M)")
flags.DEFINE_string("model_dir", "model_weights", "Model weights dir")
flags.DEFINE_string("run_id", "jax_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_string("wandb_mode", "disabled",
                    "W&B logging (disabled, online, offline)")
flags.DEFINE_string("wandb_key", "02e3820b69de1b1fcc645edcfc3dd5c5079839a1",
                    "W&B key")
flags.DEFINE_string("wandb_entity", "ssl_rl",
                    "W&B entity (username or team name)")
flags.DEFINE_string("wandb_project", "jax_ctrl", "W&B project name")


def main(argv):
    if FLAGS.seed == -1:
        seed = np.random.randint(100000000)
    else:
        seed = FLAGS.seed
    np.random.seed(seed)
    key = PRNGKey(seed)

    # Clip rewards for training
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                              num_levels=200,
                              mode='easy',
                              start_level=0,
                              paint_vel_info=False,
                              num_envs=FLAGS.num_envs,
                              normalize_rewards=True)
    # Report unclipped rewards for test
    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                      num_levels=200,
                                      mode='easy',
                                      start_level=0,
                                      paint_vel_info=False,
                                      num_envs=FLAGS.num_envs,
                                      normalize_rewards=True)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                       num_levels=0,
                                       mode='easy',
                                       start_level=0,
                                       paint_vel_info=False,
                                       num_envs=FLAGS.num_envs,
                                       normalize_rewards=True)

    os.environ["WANDB_API_KEY"] = FLAGS.wandb_key
    group_name = "%s_%s_%s" % (FLAGS.algo, FLAGS.env_name, FLAGS.run_id)
    name = "%s_%s_%s_%d" % (FLAGS.algo, FLAGS.env_name, FLAGS.run_id,
                            np.random.randint(100000000))

    wandb.init(project=FLAGS.wandb_project,
               entity=FLAGS.wandb_entity,
               config=FLAGS,
               group=group_name,
               name=name,
               sync_tensorboard=False,
               mode=FLAGS.wandb_mode)

    if "ppo" in FLAGS.algo:
        model = CTRLModel(dims=(256, 256),
                          n_cluster=FLAGS.num_clusters,
                          n_actions=env.action_space.n,
                          n_att_heads=FLAGS.n_att_heads,
                          embedding_type=FLAGS.embedding_type)
        model_target = CTRLModel(dims=(256, 256),
                          n_cluster=FLAGS.num_clusters,
                          n_actions=env.action_space.n,
                          n_att_heads=FLAGS.n_att_heads,
                          embedding_type=FLAGS.embedding_type)
    elif "daac" in FLAGS.algo:
        model = CTRLDAACModel(dims=(256, 256),
                              n_cluster=FLAGS.num_clusters,
                              n_actions=env.action_space.n,
                              n_att_heads=FLAGS.n_att_heads,
                              embedding_type=FLAGS.embedding_type)
        model_target = CTRLDAACModel(dims=(256, 256),
                              n_cluster=FLAGS.num_clusters,
                              n_actions=env.action_space.n,
                              n_att_heads=FLAGS.n_att_heads,
                              embedding_type=FLAGS.embedding_type)

    fake_args_cluster = (jnp.zeros(
        (1, FLAGS.cluster_len, 64, 64, 3)), jnp.zeros((1, FLAGS.cluster_len)))
    params_model = model.init(key,
                              state=fake_args_cluster[0],
                              action=fake_args_cluster[1],
                              reward=fake_args_cluster[1])
    params_model_target = model_target.init(key,
                              state=fake_args_cluster[0],
                              action=fake_args_cluster[1],
                              reward=fake_args_cluster[1])

    tx_ppo = optax.chain(optax.clip_by_global_norm(FLAGS.max_grad_norm),
                         optax.adam(FLAGS.lr, eps=1e-5))
    if "daac" in FLAGS.algo:
        tx_value = optax.chain(optax.clip_by_global_norm(FLAGS.max_grad_norm),
                               optax.adam(FLAGS.lr, eps=1e-5))
    tx_cluster = optax.chain(optax.clip_by_global_norm(FLAGS.max_grad_norm),
                             optax.adam(FLAGS.lr_ctrl, eps=1e-5))
    tx_target = optax.chain(optax.clip_by_global_norm(FLAGS.max_grad_norm),
                         optax.adam(FLAGS.lr, eps=1e-5))

    train_state = TrainState.create(apply_fn=model.apply,
                                    params=params_model,
                                    tx=(tx_ppo, tx_cluster,
                                        tx_value) if "daac" in FLAGS.algo else
                                    (tx_ppo, tx_cluster))
    train_state_target = TrainState.create(apply_fn=model_target.apply,
                                    params=params_model_target,
                                    tx=(tx_target,))

    batch = Batch(discount=FLAGS.gamma,
                  gae_lambda=FLAGS.gae_lambda,
                  n_steps=FLAGS.n_steps + 1,
                  num_envs=FLAGS.num_envs,
                  state_space=env.observation_space)

    state = env.reset()
    state_id = env_test_ID.reset()
    state_ood = env_test_OOD.reset()

    epinfo_buf_id = deque(maxlen=100)
    epinfo_buf_ood = deque(maxlen=100)

    returns_ood_acc = []

    for step in range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1)):
        train_state, state, batch, key = get_transition(
            train_state, model.ac, env, state, batch, key)

        action_id, _, _, key = select_action(train_state.params,
                                             train_state.apply_fn,
                                             model.ac,
                                             state_id.astype(jnp.float32) /
                                             255.,
                                             key,
                                             sample=True)
        state_id, _, _, infos_id = env_test_ID.step(action_id)

        action_ood, _, _, key = select_action(train_state.params,
                                              train_state.apply_fn,
                                              model.ac,
                                              state_ood.astype(jnp.float32) /
                                              255.,
                                              key,
                                              sample=True)
        state_ood, _, _, infos_ood = env_test_OOD.step(action_ood)

        for info in infos_id:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_id.append(maybe_epinfo)

        for info in infos_ood:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_ood.append(maybe_epinfo)

        if step % (FLAGS.n_steps + 1) == 0:
            data = batch.get()

            if "ppo" in FLAGS.algo:
                start_time = time.time()
                metric_dict, train_state, key = update_ppo(
                    train_state, model.ac, data, FLAGS.num_envs, FLAGS.n_steps,
                    FLAGS.n_minibatch, FLAGS.epoch_ppo, FLAGS.clip_eps,
                    FLAGS.entropy_coeff, FLAGS.critic_coeff, key)
                print('PPO took %f seconds' % (time.time() - start_time))
            elif "daac" in FLAGS.algo:
                start_time = time.time()
                metric_dict, train_state, key = update_daac(
                    train_state, model.ac, data, step, FLAGS.num_envs, FLAGS.n_steps,
                    FLAGS.n_minibatch, 1, 9, FLAGS.clip_eps,
                    FLAGS.entropy_coeff, FLAGS.critic_coeff, key)
                print('DAAC took %f seconds' % (time.time() - start_time))

            if "ctrl" in FLAGS.algo:
                start_time = time.time()
                cluster_metric_dict, train_state, key = update_cluster(
                    train_state, train_state_target, model.cluster, model_target.cluster, model.protos_fn, data,
                    FLAGS.num_envs, FLAGS.n_minibatch_ctrl, FLAGS.epoch_ctrl,
                    FLAGS.temp, FLAGS.k, FLAGS.myow_k, FLAGS.num_clusters,
                    FLAGS.cluster_len, FLAGS.myow_reg, key)

                train_state_target = state_update(train_state, train_state_target, tau=FLAGS.ema_ctrl)
                print('CTRL took %f seconds' % (time.time() - start_time))

                renamed_dict = {}
                for k, v in cluster_metric_dict.items():
                    renamed_dict["%s/%s" % (FLAGS.env_name, k)] = v
                wandb.log(renamed_dict, step=FLAGS.num_envs * step)

            batch.reset()

            renamed_dict = {}
            for k, v in metric_dict.items():
                renamed_dict["%s/%s" % (FLAGS.env_name, k)] = v
            wandb.log(renamed_dict, step=FLAGS.num_envs * step)

            wandb.log({
                "%s/ep_return_200" % (FLAGS.env_name):
                safe_mean([info['r'] for info in epinfo_buf_id]),
                "%s/step" % (FLAGS.env_name):
                FLAGS.num_envs * step
            })
            wandb.log({
                "%s/ep_return_all" % (FLAGS.env_name):
                safe_mean([info['r'] for info in epinfo_buf_ood]),
                "%s/step" % (FLAGS.env_name):
                FLAGS.num_envs * step
            })

            # Log every ~ 0.5M frames
            if step % 7967 == 0:
                returns_ood_acc.append(
                    safe_mean([info['r'] for info in epinfo_buf_ood]))

            print('[%d]\tEprew200: %.3f\tEprew0: %.3f' %
                  (FLAGS.num_envs * step,
                   safe_mean([info['r'] for info in epinfo_buf_id]),
                   safe_mean([info['r'] for info in epinfo_buf_ood])))
    return (returns_ood_acc, safe_mean([info['r'] for info in epinfo_buf_ood]) )


if __name__ == '__main__':
    app.run(main)
    quit()
