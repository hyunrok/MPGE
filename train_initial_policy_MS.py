import gym
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import SAC
from stable_baselines.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import numpy as np
import yaml

ENV_NAME = 'Hopper-v2'
PARAMS_ENV = 'Hopper-v2'
TIME_STEPS = 3000000
INDICATOR = 'seed1'
SEED = 1
SAVE_INTERMEDIATE = True
SAVE_FREQ = 10000
intermediate_model_path= "data/models/SAC_initial_policy_steps_" + ENV_NAME + "_" + str(TIME_STEPS) + "_" +  INDICATOR + "_.pkl"
final_policy_path = "data/models/initial_policies/MS/"+ ENV_NAME +"/SAC_initial_policy_steps_" + ENV_NAME + "_" + str(TIME_STEPS) + "_" +  INDICATOR + "_.pkl"

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

    with open('data/policy_learning.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = args['SAC'][PARAMS_ENV]

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
        model.learn(total_timesteps=time_steps,
                    tb_log_name=final_policy_path.split('/')[-1],
                    log_interval=10,
                    callback=callbacks)
        model.save(final_policy_path)
        npzfile = np.load(intermediate_model_path[:-4] + '/evaluations.npz')
        average_rewards = np.mean(npzfile['results'], axis=1)[:, 0]
        with open(intermediate_model_path[:-4] + "/eval_results.txt", "a") as f:
            for i in range(np.shape(average_rewards)[0]):
                f.write("{}, {}\n".format(npzfile['timesteps'][i], average_rewards[i]))
        evaluate_policy_on_env(env, model, iters=50)
    else:
        model.learn(total_timesteps=time_steps,
                    tb_log_name=final_policy_path.split('/')[-1],
                    log_interval=10,)
        model.save(final_policy_path)
        evaluate_policy_on_env(env, model, iters=50)

    print('Done :: ', final_policy_path)
    exit()

if __name__ == '__main__':
    train_initial_policy()
