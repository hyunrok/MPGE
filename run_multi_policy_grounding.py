import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import glob

from collections import OrderedDict

from grounding_codes.TRPOGAIfO_multi_policy import TRPOGAIfO_multi_pol
from grounding_codes.atp_envs import *
from grounding_codes.generate_target_traj import generate_target_traj
from grounding_codes.stable_baselines_modifications.MlpPolicy_multi_src import MlpPolicy_multi_src


import random
import yaml

def collect_demo(args, generate_demo = False, save_path = None, rollout_policies_paths = None):
    """
    Collect target transition samples (s,a,s')

    :param args: (list) argument values of experiments
    :param generate_demo: (bool) if false, read the location of stored data
    :param save_path: (str) path to save data
    :param rollout_policies_paths: (str) path to read rollout_policies
    """
    if generate_demo:
        data_save_dir = []
        for k in range(args.num_src):
            idx = (args.expt_number + k - 1) % len(rollout_policies_paths)
            if args.n_episodes is not None:
                data_save_dir_i = os.path.join(save_path,
                                               "{}_episodes_{}_seed{}".format(args.trg_env, args.n_episodes,
                                                                              idx))
            else:
                data_save_dir_i = os.path.join(save_path,
                                               "{}_transitions_{}_seed{}".format(args.trg_env, args.n_transitions,
                                                                                 idx))
            real_env = gym.make(args.trg_env)
            real_env.seed(args.expt_number + 100)
            generate_target_traj(rollout_policies_paths[idx], real_env, data_save_dir_i,
                                 args.n_episodes,
                                 args.n_transitions, seed = args.expt_number - 1, deterministic=args.deter_rollout)
            data_save_dir_i = data_save_dir_i + ".npz"
            data_save_dir.append(data_save_dir_i)
    else:
        data_save_dir = []
        for k in range(args.num_src):
            idx = (args.expt_number + k - 1) % len(rollout_policies_paths)
            if args.n_episodes is not None:
                data_save_dir_i = os.path.join(save_path,
                                               "{}_episodes_{}_seed{}".format(args.trg_env, args.n_episodes,
                                                                              idx))
            else:
                data_save_dir_i = os.path.join(save_path,
                                               "{}_transitions_{}_seed{}".format(args.trg_env, args.n_transitions,
                                                                                 idx))
            data_save_dir_i = data_save_dir_i + ".npz"
            data_save_dir.append(data_save_dir_i)

    for i in range(len(data_save_dir)):
        print(data_save_dir[i])
        assert os.path.isfile(data_save_dir[i])
        print("Loading Demo Data: {}".format(data_save_dir[i]))

    return data_save_dir

def main():
    parser = argparse.ArgumentParser(description='Multi-source Grounding')
    parser.add_argument('--src_env', default='HalfCheetah-v2', help="Name of source environment registered in Mujoco")
    parser.add_argument('--trg_env', default='HalfCheetahBroken-v2', help="Name of target environment")
    parser.add_argument('--demo_sub_dir', default='BrokenCheetah', help="Subdirectory for demonstration")
    parser.add_argument('--rollout_set', default='MS', help='Types of rollout policy set')
    parser.add_argument('--training_steps_atp', default=int(2e7), type=int, help="Total time steps to learn action transformation policy")
    parser.add_argument('--training_steps_policy', default=int(3e6), type=int, help="Total time steps to learn agent policy")
    parser.add_argument('--expt_number', default=1, type=int, help="Experiment number used for random seed")
    parser.add_argument('--determinisitc_atp', action='store_true', help="Stochasticity of action transformation policy when defining grounded environment")
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--n-episodes', help='Number of expert episodes', type=int, default=None)
    parser.add_argument('--n-transitions', help='Number of expert transitions', type=int, default=2000)
    parser.add_argument('--single_pol', action='store_true', help="Set to use single source policy")
    parser.add_argument('--num_src', default=5, type=int, help="Number of rollout policies")
    parser.add_argument('--collect_demo', action='store_true', help="Set to collect new samples from target environment")
    parser.add_argument('--deter_rollout', action='store_true',help="Set to deploy rollout policy deterministically")

    # Hyperparameters related to output log
    parser.add_argument('--namespace', default="CODE_release_test", type=str, help="namespace for the experiments")
    parser.add_argument('--tensorboard', action='store_true', help="visualize training in tensorboard")
    parser.add_argument('--eval', action='store_true', help="Set to true to evaluate the agent policy in the real environment, after training in grounded environment")
    parser.add_argument('--plot', action='store_true', help="Visualize the action transformer policy")

    parser.add_argument(
        '--task',
        type=str,
        default='trpo-gaifo')

    args = parser.parse_args()
    if 'Broken' in args.trg_env:
        true_transformation = 'Broken'
    elif 'PositiveSkew' in args.trg_env:
        true_transformation = np.array([1.5])
    else:
        true_transformation = None

    # set the seeds here for experiments
    random.seed(args.expt_number)
    np.random.seed(args.expt_number)

    expt_type = 'sim2sim' if args.src_env == args.trg_env else 'sim2real'
    expt_label = args.trg_env + '_' + args.namespace + '_num_src_' + str(args.num_src) + '_' + expt_type + '_seed' + str(args.expt_number)

    # create the experiment folder
    base_path = 'data/models/grounding/'
    expt_path = base_path + expt_label
    tensorboard_log = 'data/TBlogs/' + expt_label if args.tensorboard else None

    if not os.path.exists(expt_path):
        os.makedirs(expt_path)

    with open('data/grounding.yml', 'r') as f:
        hyperparams_dict = yaml.load(f)
        hyperparams = hyperparams_dict[args.src_env]
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    with open(os.path.join(expt_path, 'config.yml'), 'w') as f:
        yaml.dump({**args.__dict__, **saved_hyperparams}, f)

    # Load src policies

    load_paths = "data/models/initial_policies/" + args.rollout_set + "/" + args.src_env + "/"
    if args.rollout_set == 'ER':
        load_paths = load_paths + '/seed' + str(args.expt_number) + "/"
    initial_rollout_policies_paths = sorted(glob.glob(load_paths + '*'))
    initial_rollout_policies = []
    num_src = args.num_src
    if num_src > 1 and args.single_pol:
        raise ValueError('Mismatch in arguments for single-policy grounding')
    for k in range(num_src):
        idx = (args.expt_number + k - 1) % len(initial_rollout_policies_paths)
        print(initial_rollout_policies_paths[idx])
        initial_rollout_policies.append(
            SAC.load(initial_rollout_policies_paths[idx], seed=args.expt_number))

    target_policy = SAC.load(initial_rollout_policies_paths[args.expt_number - 1], seed=args.expt_number)

    # Define action transformer policy
    atp_env = ATPEnv_Multiple(env=gym.make(args.src_env),
                              rollout_policy_list=initial_rollout_policies,
                              seed=args.expt_number,
                              deter_rollout=args.deter_rollout)

    # Additional configuration for TRPOGAIfO
    config = {'shaping_mode': args.task,
              'num_src': num_src,
              'rew_discriminator': False}

    # Collect target transition samples
    if args.rollout_set == 'ER':
        save_path = 'data/real_traj_ER'
    elif args.rollout_set == 'MS':
        save_path = 'data/real_traj_MS'
    if args.demo_sub_dir is not None:
        save_path = os.path.join(save_path, args.demo_sub_dir)
        if args.rollout_set == 'ER':
            save_path = os.path.join(save_path, 'seed' + str(args.expt_number))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    config['expert_data_path'] = collect_demo(args,
                                              save_path = save_path,
                                              generate_demo = args.collect_demo,
                                              rollout_policies_paths = initial_rollout_policies_paths,)

    atp = TRPOGAIfO_multi_pol(policy=MlpPolicy_multi_src,
                              env=atp_env,
                              tensorboard_log=tensorboard_log,
                              verbose=args.verbose,
                              config=config,
                              policy_kwargs={'num_src': num_src},
                              seed=args.expt_number,
                              **hyperparams)

    if args.verbose > 0:
        with open(expt_path + "/demonstration_info.txt", "a") as f:
            for i in range(num_src):
                loaded_data = atp.expert_dataset[i]
                print("Demonstration idx: {}".format(i), file=f)
                print("Total trajectories: {}".format(loaded_data.num_traj), file=f)
                print("Total transitions: {}".format(loaded_data.num_transition), file=f)
                print("Total trajectories for limited transitions: {}".format(len(loaded_data.returns)), file=f)
                print("Average returns: {}".format(loaded_data.avg_ret), file=f)
                print("Std for returns: {}".format(loaded_data.std_ret), file=f)
                print("", file=f)

    # ground the environment
    grounding_step = 1
    print('################# START GROUNDING #################')
    grnd_env_plot = GroundedEnv(env=gym.make(args.src_env),
                                action_tf_policy=atp,
                                use_deterministic=True,
                                )
    grnd_env_plot.test_grounded_environment(expt_path=expt_path + '/initial', true_transformation=true_transformation)

    # Update action transformer policy
    cb = TestGroundedCallback(plot_freq=100000, save_path=expt_path, name_prefix=str(grounding_step),
                              verbose=args.verbose, true_transformation=true_transformation)
    atp.learn(total_timesteps=args.training_steps_atp,
              reset_num_timesteps=True,
              callback=cb)

    # Save action transformer policy
    print('##### SAVING ACTION TRANSFORMER POLICY #####')
    atp.save(expt_path + '/action_transformer_policy' + str(grounding_step) + '.pkl')

    ### Update rollout policy
    print('################# START POLICY LEARNING #################')
    # Set grounded environment
    src_env = gym.make(args.src_env)
    grnd_env = GroundedEnv(env=src_env,
                           action_tf_policy=atp,
                           use_deterministic=args.determinisitc_atp,
                           )
    grnd_env.seed(args.expt_number)
    target_policy.set_env(grnd_env)

    # Test grounded environment
    if args.plot:
        # action transformer plot
        grnd_env.test_grounded_environment(expt_path=expt_path + '/plot_' + str(grounding_step) + '.png',
                                           true_transformation=true_transformation)
        with open(expt_path + "/rollout_selections.txt", "a") as f:
            print(atp_env.num_selection, file=f)

    # Learn rollout policy
    src_env_eval = gym.make(args.src_env)
    grnd_env_eval = GroundedEnv(env=src_env_eval,
                                action_tf_policy=atp,
                                use_deterministic=args.determinisitc_atp,
                                )
    grnd_env_eval.seed(args.expt_number)
    trg_env_eval = gym.make(args.trg_env)
    trg_env_eval.seed(args.expt_number)
    cb = EvalTrgCallback(eval_freq=100000, save_path=expt_path,
                         grnd_env=grnd_env_eval, trg_env=trg_env_eval,
                         name_prefix=str(grounding_step), verbose=args.verbose)
    target_policy.learn(total_timesteps=args.training_steps_policy, callback=cb)
    target_policy.save(expt_path + '/target_policy_' + str(grounding_step - 1) + '.pkl')

    if args.eval:
        print("################# START EVALUATION #################")
        test_env = gym.make(args.trg_env)

        try:
            val = evaluate_policy_on_env(test_env,
                                         target_policy,
                                         render=False,
                                         iters=50,
                                         deterministic=True,
                                         )

            with open(expt_path + "/output.txt", "a") as txt_file:
                print(val, file=txt_file)

            val = evaluate_policy_on_env(test_env,
                                         target_policy,
                                         render=False,
                                         iters=50,
                                         deterministic=False,
                                         )

            with open(expt_path + "/stochastic_output.txt", "a") as txt_file:
                print(val, file=txt_file)
            print(expt_path)
        except Exception as e:
            print(e)

    os._exit(0)

if __name__ == '__main__':
    main()
    os._exit(0)