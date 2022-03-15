from typing import Any, Optional, Tuple, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def default_conv_init():
    return nn.initializers.glorot_uniform()

def default_relu_init():
    return nn.initializers.orthogonal(jnp.sqrt(2))

def default_linear_init():
    return nn.initializers.orthogonal(1.)

def default_logits_init():
    return nn.initializers.orthogonal(0.01)


class ResidualBlock(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_2')(y)

        return y + x


class Impala(nn.Module):
    """IMPALA architecture."""
    prefix: str

    @nn.compact
    def __call__(self, x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2),
                                                        (32, 2)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           kernel_init=default_conv_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j))
                out = block(out)

        out = out.reshape(out.shape[0], -1)
        out = nn.relu(out)
        out = nn.Dense(256,
                       kernel_init=default_relu_init(),
                       name=self.prefix + '/representation')(out)
        out = nn.relu(out)
        return out

class MLP(nn.Module):
  dims: Sequence[int]
  prefix: str

  @nn.compact
  def __call__(self, x):
    init = default_relu_init
    for i, dim in enumerate(self.dims):
        if i == len(self.dims) - 1:
            init = default_linear_init
        x = nn.Dense(dim,
                    kernel_init=init(),
                    name=self.prefix+'/%d' % i)(x)
        if i < len(self.dims) - 1:
            x = nn.relu(x)
    return x

class CTRLModel(nn.Module):
    """Critic+Actor+Cluster."""
    n_actions: int
    dims: Sequence[int]
    n_cluster: int
    embedding_type: str
    n_att_heads: int
    

    def setup(self):
        self.encoder = Impala(prefix='')

        self.fc_v = nn.Dense(1,
                     kernel_init=default_linear_init(),
                     name='fc_v')

        self.fc_pi = nn.Dense(self.n_actions,
                          kernel_init=default_logits_init(),
                          name='fc_pi')

        self.action_mlp = MLP(dims=list(self.dims[:-1])+[self.dims[-1]*2], prefix='action_embedding')
        self.reward_mlp = MLP(dims=list(self.dims[:-1])+[self.dims[-1]*2], prefix='reward_embedding')

        self.attn = nn.SelfAttention(num_heads=self.n_att_heads,
                                 qkv_features=self.dims[-1],
                                 out_features=self.dims[-1])
        self.concat = nn.Dense(self.dims[-1], kernel_init=default_linear_init(), name='concat')

        self.v_clust_mlp = MLP(dims=self.dims, prefix='v_clust_mlp')
        self.w_clust_mlp = MLP(dims=self.dims, prefix='w_clust_mlp')

        self.v_pred_mlp = MLP(dims=self.dims, prefix='v_pred_mlp')
        self.w_pred_mlp = MLP(dims=self.dims, prefix='w_pred_mlp')

        self.protos = nn.Dense(self.n_cluster,
                          kernel_init=default_linear_init(),
                          name='protos')

    @nn.compact
    def __call__(self, state, action, reward):
        value, pi = self.ac(state[0])
        v_clust, w_clust, v_pred, w_pred = self.cluster(state, action, reward)
        Q = self.protos(v_clust)

        return (value, pi), (v_clust, w_clust, v_pred, w_pred), Q

    def protos_fn(self, x):
        return self.protos(x)

    def ac(self, state, dummy=None):
        # Features
        z = self.encoder(state)
        # Linear critic
        v = self.fc_v(z)
        # Linear policy logits
        logits = self.fc_pi(z)
        pi = tfd.Categorical(logits=logits)
        return v, pi
    

    def cluster(self, state, action, reward):
        """
        state: n_batch x n_timesteps x H x W x C
        action: n_batch x n_timesteps
        reward: n_batch x n_timesteps
        """
        img_shape = state.shape[2:]
        batch_shape = state.shape[:2]

        # z_state: n_batch x n_timesteps x n_hidden
        z_state = self.encoder(state.reshape(-1,*img_shape)).reshape(*batch_shape, -1)

        # z_action: n_batch x n_timesteps x n_hidden
        z_action = jax.nn.one_hot(action.reshape(-1), self.n_actions)
        z_action = self.action_mlp(z_action)
        gamma_a, beta_a = z_action.reshape(*batch_shape, -1).split(2, axis=-1)

        if self.embedding_type == "concat":
            z = ((1 + gamma_a) * z_state + beta_a).reshape(state.shape[0], -1)
            z = self.concat(z)
        elif self.embedding_type == "attention":
            # z_reward: n_batch x n_timesteps x n_hidden
            # z_reward = self.reward_mlp(reward.reshape(-1, 1))
            # gamma_r, beta_r = z_reward.reshape(*batch_shape, -1).split(2, axis=-1)

            z = ((1 + gamma_a) * z_state + beta_a) # + ((1 + gamma_r) * z_state + beta_r)
            z = z.reshape(state.shape[0], -1)
            z = self.attn(z)

        v_clust = self.v_clust_mlp(z)
        w_clust = self.w_clust_mlp(v_clust)

        v_pred = self.v_pred_mlp(z)
        w_pred = self.w_pred_mlp(v_pred)

        return v_clust, w_clust, v_pred, w_pred

class CTRLDAACModel(nn.Module):
    """Critic+Actor+Cluster."""
    n_actions: int
    dims: Sequence[int]
    n_cluster: int
    embedding_type: str
    n_att_heads: int

    def setup(self):
        self.encoder_policy = Impala(prefix='')
        self.encoder_value = Impala(prefix='')

        self.fc_v = nn.Dense(1,
                     kernel_init=default_linear_init(),
                     name='fc_v')

        self.fc_pi = nn.Dense(self.n_actions,
                          kernel_init=default_logits_init(),
                          name='fc_pi')

        self.fc_adv = nn.Dense(1,
                          kernel_init=default_linear_init(),
                          name='fc_adv')

        self.action_mlp = MLP(dims=list(self.dims[:-1])+[self.dims[-1]*2], prefix='action_embedding')

        self.attn = nn.SelfAttention(num_heads=self.n_att_heads,
                                 qkv_features=self.dims[-1],
                                 out_features=self.dims[-1],
                                 name='attention')
        self.concat = nn.Dense(self.dims[-1], kernel_init=default_linear_init(), name='concat')

        self.v_clust_mlp = MLP(dims=self.dims, prefix='v_clust_mlp')
        self.w_clust_mlp = MLP(dims=self.dims, prefix='w_clust_mlp')

        self.v_pred_mlp = MLP(dims=self.dims, prefix='v_pred_mlp')
        self.w_pred_mlp = MLP(dims=self.dims, prefix='w_pred_mlp')

        self.protos = nn.Dense(self.n_cluster,
                          kernel_init=default_linear_init(),
                          name='protos')

    @nn.compact
    def __call__(self, state, action, reward):
        adv, pi = self.ac(state[0], action[0])
        value = self.ac(state[0], None)
        v_clust, w_clust, v_pred, w_pred = self.cluster(state, action, reward)
        Q = self.protos(v_clust)

        return (value, pi), (v_clust, w_clust, v_pred, w_pred), Q

    def protos_fn(self, x):
        return self.protos(x)

    def ac(self, state, action=None):
        if action is not None:
            action = jax.nn.one_hot(action, num_classes=15)
            # Features
            z_pi = self.encoder_policy(state)
            # z_pi = jax.lax.stop_gradient(z_pi)
            # Linear policy logits
            logits = self.fc_pi(z_pi)
            pi = tfd.Categorical(logits=logits)
            z_pi = jnp.concatenate([z_pi, action], axis=1)
            adv = self.fc_adv(z_pi)
            
            return adv, pi
        else:
            z_v = self.encoder_value(state)
            # Linear critic
            v = self.fc_v(z_v)
            return v


    def cluster(self, state, action, reward):
        """
        state: n_batch x n_timesteps x H x W x C
        action: n_batch x n_timesteps
        reward: n_batch x n_timesteps
        """
        img_shape = state.shape[2:]
        batch_shape = state.shape[:2]

        # z_state: n_batch x n_timesteps x n_hidden
        z_state = self.encoder_policy(state.reshape(-1,*img_shape)).reshape(*batch_shape, -1)

        # z_action: n_batch x n_timesteps x n_hidden
        z_action = jax.nn.one_hot(action.reshape(-1), self.n_actions)
        z_action = self.action_mlp(z_action)
        gamma_a, beta_a = z_action.reshape(*batch_shape, -1).split(2, axis=-1)

        if self.embedding_type == "concat":
            z = ((1 + gamma_a) * z_state + beta_a).reshape(state.shape[0], -1)
            z = self.concat(z)
        elif self.embedding_type == "attention":      

            z = ((1 + gamma_a) * z_state + beta_a) #+ 0.5 * ((1 + gamma_r) * z_state + beta_r)
            z = z.reshape(state.shape[0], -1)
            z = self.attn(z)

        v_clust = self.v_clust_mlp(z)
        w_clust = self.w_clust_mlp(v_clust)

        v_pred = self.v_pred_mlp(z)
        w_pred = self.w_pred_mlp(v_pred)

        return v_clust, w_clust, v_pred, w_pred