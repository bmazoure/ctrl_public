# Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL

Code for Cross-Trajectory Representation Learning (ICLR 2022).

## Prerequisites & installation

The only packages needed for the repo are jax, jaxlib, flax and optax.

Jax for CUDA can be installed as follows:

```
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

To install all requirements all at once, run

```
pip install -r requirements.txt
```

## Use [Training]

After installing Jax, simply run

```
python train_ppo.py --env_name="bigfish" --seed=31241 --train_steps=25_000_000 --algo=ppo_ctrl
```

to train PPO+CTRL on bigfish on 25M frames.

To cite:
```
@article{mazoure2022cross,
  title={Cross-Trajectory Representation learning for zero-shot generalization in RL},
  author={Mazoure, Bogdan and Ahmed, Ahmed M and MacAlpine, Patrick and Hjelm, R Devon and Kolobov, Andrey},
  journal={International Conference on Learning Representations},
  year={2022}
}
```