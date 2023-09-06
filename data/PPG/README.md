
# Generating Demonstration from PPG

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from [`environment.yml`](environment.yml) manually.

```
conda env update --name ppg --file ./cy_env.yaml
conda activate ppg
cd procgen_highres
pip install -e .
cd ..
pip install -e phasic-policy-gradient
```

## Generate Demonstrations with High resolution (512 X 512)
### WARNING) First, train PPG policy following the guide below. Then, use the following code for generating demonstrations.
```python
python generate_procgen_dataset.py --model_path {path of saved model file} --num_demonstrations {number of demonstrations} --env_name {procgen env name} --num_levels {number of levels} --start_level {level to start} --distribution_mode {easy/hard}
```


# PPG
## Reproduce and Visualize Results

PPG with default hyperparameters (results/ppg-runN):

```
mpiexec -np 4 python -m phasic_policy_gradient.train
python -m phasic_policy_gradient.graph --experiment_name ppg
```

PPO baseline (results/ppo-runN):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared
python -m phasic_policy_gradient.graph --experiment_name ppo
```

PPG, varying E_pi (results/e-pi-N):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi N
python -m phasic_policy_gradient.graph --experiment_name e_pi
```

PPG, varying E_aux (results/e-aux-N):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_aux_epochs N
python -m phasic_policy_gradient.graph --experiment_name e_aux
```

PPG, varying N_pi (results/n-pi-N):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --n_pi N
python -m phasic_policy_gradient.graph --experiment_name n_pi
```

PPG, using L_KL instead of L_clip (results/ppgkl-runN):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --clip_param 0 --kl_penalty 1
python -m phasic_policy_gradient.graph --experiment_name ppgkl
```

PPG, single network variant (results/ppgsingle-runN):

```
mpiexec -np 4 python -m phasic_policy_gradient.train --arch detach
python -m phasic_policy_gradient.graph --experiment_name ppg_single_network
```

Pass `--normalize_and_reduce` to compute and visualize the mean normalized return with `phasic_policy_gradient.graph`.

# Citation

Please cite using the following bibtex entry:

```
@article{cobbe2020ppg,
  title={Phasic Policy Gradient},
  author={Cobbe, Karl and Hilton, Jacob and Klimov, Oleg and Schulman, John},
  journal={arXiv preprint arXiv:2009.04416},
  year={2020}
}
```
