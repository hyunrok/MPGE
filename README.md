## Multi-Policy Grounding and Ensemble Policy Learning (MPGE)
Code release for MPGE.

### Installation

##### Prerequisites
- Python(>=3.7)
- stable-baseline. 
    - RL algorrithms are based on [stable-baseline framework](https://github.com/hill-a/stable-baselines) (for tensorflow)
    - Please follow [this](https://stable-baselines.readthedocs.io/en/master/guide/install.html) to install stable-baseline  
- Mujoco, mujoco-py:
    - We need Mujoco and license to run it. Please refer the official site (http://www.mujoco.org/).
    - Also, install [mujoco-py](https://github.com/openai/mujoco-py)
- yaml

### Preparing rollout policies 
Trained policies are stored in the following location

**data/models/initial_policies**

[Multi-seed policies] Run "train_initial_policy_MS.py" by modifying codes according to target environments.

[Entropy regularized policies] Run "train_initial_policy_ER.py" by modifying codes according to target environments. 

### Training *MPG*:
"run_multi_policy_grounding.py" is for run MPG. Following shell script can be used to run MPG with non-ensemble policy learning experiments.

``./run_experiments_MPG.sh``

Default setting is running MPG-MS for BrokenHalfCheetah-v2 task, using 5 rollout policies, 2000 transition samples per each policy and run seed 1 to 5 at once.

"ensemble_policy_learning.py" is for run ensemble policy learning method based on MPG results. It will read saved "action_transformer_policy1.pkl" files stored in "data/models/grounding" folder. Following shell script can be used to run ensemble policy learning experiments.

``./run_experiments_Ensemble.sh``

Default setting is using 5 MPG results obtained by default "run_experiments_MPG.sh".

For more experiments with various environments, number of target transition samples, number of rollout policies, and number of action transformation policies, please refer shell script and run files.

### Details of project structure

**data/models/grounding**: folder for storing results \
**data/models/real_traj_MS** and **data/models/real_traj_ER**: folder for target transition samples \
**grounding_codes/envs**: files for defining target environments \
**grounding_codes/stable_baselines_moidifications**: list of files that modifies original 'stable-baseline' repository\
**grounding_codes/atp_envs.py**: define 1. augmented MDP for grounding, and 2. grounded environment \
**grounding_codes/discriminators.py**: discriminative classifier for using in grounding process \
**grounding_codes/generate_target_traj.py**: collect samples from target environment \
**grounding_codes/TRPOGAIfO_multi_policy.py**: define action transformation policy that will be learned with MPG\