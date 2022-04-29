import os
import gym
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import SAC
from stable_baselines.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import numpy as np
import yaml
import glob

ENV_NAME = 'Hopper-v2'
PARAMS_ENV = 'Hopper-v2' # HalfCheetah, Hopper, InvertedPendulum
TIME_STEPS = 1000 # 3000000, 2000000, 60000

SEED = 1
COEFFICIENT = 0.01
SAVE_INTERMEDIATE = True
SAVE_FREQ = 100 # 10000

PREV_POLICIES_FOLDER = "data/models/initial_policies/ER/"+ ENV_NAME +"/seed"+str(SEED)+"/" # None
NUM_PREV = len(glob.glob(PREV_POLICIES_FOLDER + '*'))
INDICATOR = 'seed'+str(SEED)+'_seq' + str(NUM_PREV)

intermediate_model_path= "data/models/SAC_initial_policy_steps_" + ENV_NAME + "_" + str(TIME_STEPS) + "_" +  INDICATOR + "_.pkl"
final_policy_path = "data/models/initial_policies/ER/"+ ENV_NAME +"/seed"+str(SEED)+"/SAC_initial_policy_steps_" + ENV_NAME + "_" + str(TIME_STEPS) + "_" +  INDICATOR + "_.pkl"

def evaluate_policy_on_env(env,
                           model,
                           iters=1,
                           deterministic=True
                           ):
    return_list = []
    for i in range(iters):
        return_val = 0
        done = False
        obs = env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = env.step(action)
            return_val+=rewards

        if not i%15: print('Iteration ', i, ' done.')
        return_list.append(return_val)
    print('***** STATS FOR THIS RUN *****')
    print('MEAN : ', np.mean(return_list))
    print('STD : ', np.std(return_list))
    print('******************************')
    return np.mean(return_list), np.std(return_list)


def train_initial_policy(
        env_name=ENV_NAME,
        time_steps=TIME_STEPS):
    env = gym.make(env_name)
    env.seed(SEED)

    # Environment to use entropy regularizer
    if PREV_POLICIES_FOLDER is not None:
        prev_policies_path = sorted(glob.glob(PREV_POLICIES_FOLDER + '*'))
        prev_policies = []
        for k in range(NUM_PREV):
            print(prev_policies_path[k])
            prev_policies.append(SAC.load(prev_policies_path[k], seed=SEED))

        env = entropy_wrapper(env, prev_policies, coefficient = COEFFICIENT)

    with open('data/policy_learning.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = args['SAC'][PARAMS_ENV]
    print('~~ Loaded args file ~~')

    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               feature_extraction="mlp", layers=[256, 256])

    model = SAC(CustomPolicy, env,
                verbose=1,
                tensorboard_log='data/TBlogs/initial_policy_training',
                batch_size=args['batch_size'],
                buffer_size=args['buffer_size'],
                ent_coef=args['ent_coef'],
                learning_starts=args['learning_starts'],
                learning_rate=args['learning_rate'],
                train_freq=args['train_freq'],
                seed=SEED,
                )

    if SAVE_INTERMEDIATE:
        check_callback = CheckpointCallback(save_freq=SAVE_FREQ,
                                            save_path=intermediate_model_path[:-4],
                                            name_prefix=ENV_NAME + '_' + str(SEED),
                                            verbose=1,
                                            )
        eval_env = gym.make(ENV_NAME)
        eval_env.seed(SEED)
        eval_callback = EvalCallback(eval_env,
                                     n_eval_episodes=10,
                                     eval_freq=SAVE_FREQ,
                                     log_path=intermediate_model_path[:-4],
                                     deterministic=False,
                                     render=False,
                                     verbose=1)
        callbacks = CallbackList([check_callback, eval_callback])
        if PREV_POLICIES_FOLDER is not None:
            os.makedirs(os.path.join(intermediate_model_path[:-4], 'prev_ent_reg'))
            env_eval_ent = gym.make(env_name)
            env_eval_ent.seed(SEED)
            env_eval_ent = entropy_wrapper(env_eval_ent, prev_policies, coefficient=COEFFICIENT)
            eval_callback_KL = EvalCallback(env_eval_ent,
                                         n_eval_episodes=10,
                                         eval_freq=SAVE_FREQ,
                                         log_path=os.path.join(intermediate_model_path[:-4], 'prev_ent_reg'),
                                         deterministic=False,
                                         render=False,
                                         verbose=1)
            callbacks = CallbackList([check_callback, eval_callback, eval_callback_KL])

        model.learn(total_timesteps=time_steps,
                    tb_log_name=intermediate_model_path.split('/')[-1],
                    log_interval=10,
                    callback=callbacks)
        model.save(final_policy_path)
        npzfile = np.load(intermediate_model_path[:-4] + '/evaluations.npz')
        average_rewards = np.mean(npzfile['results'], axis=1)[:, 0]
        with open(intermediate_model_path[:-4] + "/eval_results.txt", "a") as f:
            for i in range(np.shape(average_rewards)[0]):
                f.write("{}, {}\n".format(npzfile['timesteps'][i], average_rewards[i]))
        if PREV_POLICIES_FOLDER is not None:
            npzfile = np.load(os.path.join(intermediate_model_path[:-4], 'prev_ent_reg') + '/evaluations.npz')
            average_rewards = np.mean(npzfile['results'], axis=1)[:, 0]
            with open(intermediate_model_path[:-4] + "/eval_results_prev_ent_reg.txt", "a") as f:
                for i in range(np.shape(average_rewards)[0]):
                    f.write("{}, {}\n".format(npzfile['timesteps'][i], average_rewards[i]))
        env_last_eval = gym.make(env_name)
        env_last_eval.seed(SEED)
        evaluate_policy_on_env(env_last_eval, model, iters=50)
    else:
        model.learn(total_timesteps=time_steps,
                    tb_log_name=final_policy_path.split('/')[-1],
                    log_interval=10,)
        model.save(final_policy_path)
        env_last_eval = gym.make(env_name)
        env_last_eval.seed(SEED)
        evaluate_policy_on_env(env_last_eval, model, iters=50)

    print('Done :: ', final_policy_path)
    exit()

class entropy_wrapper(gym.Wrapper):
    def __init__(self, env, prev_policies, coefficient):
        super(entropy_wrapper, self).__init__(env)
        self.prev_policies = prev_policies
        self.coefficient = coefficient

    def reset(self, **kwargs):
        self.observation = super(entropy_wrapper, self).reset()
        return self.observation

    def step(self, action):
        policy_entropy = []
        for i in range(len(self.prev_policies)):
            entropy_i = self.prev_policies[i].sess.run(self.prev_policies[i].entropy,
                                                       {self.prev_policies[i].observations_ph: [self.observation]})
            policy_entropy.append(entropy_i)

        self.observation, reward, done, info = self.env.step(action)

        reward += self.coefficient * np.average(policy_entropy) # Give bonus if entropy of prev policies are large

        return self.observation, reward, done, info

if __name__ == '__main__':
    train_initial_policy()