import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import glob

from grounding_codes.TRPOGAIfO_multi_policy import TRPOGAIfO_multi_pol
from grounding_codes.atp_envs import *

import random
import yaml

def main():
    parser = argparse.ArgumentParser(description='Ensemble policy learning')

    parser.add_argument('--src_env', default='HalfCheetah-v2', help="Name of source environment registered in Mujoco")
    parser.add_argument('--trg_env', default='HalfCheetahBroken-v2', help="Name of target environment")
    parser.add_argument('--rollout_set', default='MS', help='Types of rollout policy set')
    parser.add_argument('--training_steps_policy', default=int(1e4), type=int, help="")

    parser.add_argument('--expt_number', default=1, type=int, help="Expertiment number used for random seed")
    parser.add_argument('--determinisitc_atp', action='store_true', help="Stochasticity of action transformation policy when defining grounded environment")
    parser.add_argument('--num_atps', default=1, type=int, help="Number of action transformation policies (N)")
    parser.add_argument('--num_src', default=5, type=int, help="Number of rollout policies has been used for action transformation policy (K)")
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)

    # Hyperparameters related to output log
    parser.add_argument('--namespace', default="TEST", type=str, help="Namespace for the experiments. Determine path to read saved action transformation policies")
    parser.add_argument('--eval', action='store_true', help="set to true to evaluate the agent policy in the real environment, after training in grounded environment")


    args = parser.parse_args()

    # set the seeds here for experiments
    random.seed(args.expt_number)
    np.random.seed(args.expt_number)

    expt_type = 'sim2sim' if args.src_env == args.trg_env else 'sim2real'
    expt_label = args.trg_env + '_' + args.namespace + '_num_src_' + str(args.num_src) + '_' + expt_type + '_seed' + str(args.expt_number)

    # create the experiment folder
    base_path = 'data/models/grounding/'
    expt_path = base_path + expt_label

    grounding_step = 1 # len(glob.glob(expt_path + '/action_transformer_policy*.pkl'))

    with open(os.path.join(expt_path, 'config_pol_learning.yml'), 'w') as f:
        yaml.dump({**args.__dict__}, f)

    # Load src policies
    if grounding_step == 1:
        load_paths = "data/models/initial_policies/" + args.rollout_set + "/" + args.src_env + "/"
        if args.rollout_set == 'ER':
            load_paths = load_paths + '/seed' + str(args.expt_number) + "/"
        policy_paths = sorted(glob.glob(load_paths + '*'))
        target_policy = SAC.load(policy_paths[args.expt_number - 1],
                                 seed=args.expt_number)

    ### Update rollout policy
    # Load ATP
    # Update action transformer policy
    atp_policies = []
    for k in range(args.num_atps):
        idx = (args.expt_number - k)
        if idx<=0: idx += len(policy_paths)
        prev_label = args.trg_env + '_' + args.namespace + '_num_src_' + str(args.num_src) + '_' + expt_type + '_seed' + str(idx)
        prev_path = base_path + prev_label
        policy_path = prev_path + '/action_transformer_policy' + str(grounding_step) + '.pkl'
        config = {'expert_data_path': None, 'num_src': args.num_atps}
        atp = TRPOGAIfO_multi_pol.load(load_path=policy_path,
                                       seed=args.expt_number,
                                       config=config,
                                       )
        atp_policies.append(atp)

    # Set grounded environment
    src_env = gym.make(args.src_env)
    grnd_env = GroundedEnv(env=src_env,
                           action_tf_policy=atp_policies,
                           use_deterministic=args.determinisitc_atp,
                           )
    grnd_env.seed(args.expt_number)
    target_policy.set_env(grnd_env)

    # Learn rollout policy
    src_env_eval = gym.make(args.src_env)
    grnd_env_eval = GroundedEnv(env=src_env_eval,
                                action_tf_policy=atp_policies,
                                use_deterministic=args.determinisitc_atp,
                                )
    grnd_env_eval.seed(args.expt_number)
    trg_env_eval = gym.make(args.trg_env)
    trg_env_eval.seed(args.expt_number)
    cb = EvalTrgCallback(eval_freq=100000, save_path=expt_path,
                         grnd_env=grnd_env_eval, trg_env=trg_env_eval,
                         name_prefix=str(grounding_step) + '_multi_sim', verbose=args.verbose)
    target_policy.learn(total_timesteps=args.training_steps_policy, callback=cb)
    target_policy.save(expt_path + '/target_policy_multi_sim' + str(grounding_step - 1) + '.pkl')

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

            with open(expt_path + "/output_multi_sim.txt", "a") as txt_file:
                print(val, file=txt_file)

            val = evaluate_policy_on_env(test_env,
                                         target_policy,
                                         render=False,
                                         iters=50,
                                         deterministic=False,
                                         )

            with open(expt_path + "/stochastic_output_multi_sim.txt", "a") as txt_file:
                print(val, file=txt_file)
            print(expt_path)
        except Exception as e:
            print(e)

    os._exit(0)

if __name__ == '__main__':
    main()
    os._exit(0)