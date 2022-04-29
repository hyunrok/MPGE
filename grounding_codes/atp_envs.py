import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mujoco_py
from stable_baselines import SAC
import os
from stable_baselines.common.callbacks import BaseCallback

def evaluate_policy_on_env(env,
                           model,
                           render=True,
                           iters=1,
                           deterministic=False,
                           constrained=False,
                           return_raw_list=False,
                           ):
    return_list = []
    cost_list = []
    for i in range(iters):
        return_val = 0
        cost_val = 0
        done = False
        obs = env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = env.step(action)
            return_val+=rewards
            if constrained:
              cost_val += info.get('cost', 0)
            else:
              cost_val += 0
            if render:
                env.render()

        if not i%15: print('Iteration ', i, ' done.')
        return_list.append(return_val)
        cost_list.append(cost_val)
    print('***** STATS FOR THIS RUN *****')
    print('MEAN : ', np.mean(return_list))
    print('STD : ', np.std(return_list))
    if return_raw_list:
        return return_list
    else:
        return np.mean(return_list), np.std(return_list)/np.sqrt(len(return_list))

def set_rollout_policy(rollout_policy_path, seed=None):
    return SAC.load(rollout_policy_path, seed = seed)

class TestGroundedCallback(BaseCallback):
    """
    Callback for testing grounded environment every `plot_freq` steps

    :param plot_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    :param verbose: (int) Verbose mode (0: no output, 1: INFO)
    :param true_transformation: (str) Set optimal action transformation to be compared with
    """
    def __init__(self, plot_freq: int, save_path: str, name_prefix='grounded', verbose=0, true_transformation = None):
        super(TestGroundedCallback, self).__init__(verbose)
        self.plot_freq = plot_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.true_transformation = true_transformation

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            if self.n_calls % (self.plot_freq*10) == 0:
                self.model.save(self.save_path + '/action_transformer_policy' + str(self.num_timesteps) + '.pkl')
            # Create grnd env
            grnd_env_plot = GroundedEnv(env=gym.make(self.model.env.spec.id),
                                   action_tf_policy=self.model,
                                   use_deterministic=True,
                                   )
            # Test
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            if self.true_transformation is not None:
                opt_gap = grnd_env_plot.test_grounded_environment(expt_path=path, true_transformation=self.true_transformation)
                with open(self.save_path + "/opt_gap_"+ self.name_prefix +".txt", "a") as f:
                    f.write("{}, {}\n".format(self.num_timesteps, opt_gap))
                    f.close()
            else:
                grnd_env_plot.test_grounded_environment(expt_path=path)
            del grnd_env_plot
            if self.verbose > 1:
                print("Testing grounded environment")
        return True

class EvalTrgCallback(BaseCallback):
    """
    Callback for evaluation at target and grounded environment every `eval_freq` steps

    :param eval_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    :param grnd_env: (GroundedEnv)
    :param trg_env: (gym environment)
    :param verbose: (int) Verbose mode (0: no output, 1: INFO)
    """
    def __init__(self, eval_freq: int, save_path: str, grnd_env, trg_env, name_prefix='grounded', verbose=0):
        super(EvalTrgCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.grnd_env = grnd_env
        self.trg_env = trg_env
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluation at grnd
            if self.verbose > 1:
                print("Evaluation at grounded environment...")
            val = evaluate_policy_on_env(self.grnd_env,
                                         self.model,
                                         render=False,
                                         iters=5,
                                         deterministic=False,
                                         )
            with open(self.save_path + "/eval_at_grnd_"+ self.name_prefix +".txt", "a") as txt_file:
                txt_file.write("{}, {}, {}\n".format(self.num_timesteps, val[0], val[1]))
                txt_file.close()

            # Evaluation at trg
            if self.verbose > 1:
                print("Evaluation at target environment...")
            val = evaluate_policy_on_env(self.trg_env,
                                         self.model,
                                         render=False,
                                         iters=5,
                                         deterministic=False,
                                         )
            with open(self.save_path + "/eval_at_trg_"+ self.name_prefix +".txt", "a") as txt_file:
                txt_file.write("{}, {}, {}\n".format(self.num_timesteps, val[0], val[1]))
                txt_file.close()
        return True

class ATPEnv_Multiple(gym.Wrapper):
    """
    Defines augmented MDP for learning action transformation policy
    """
    def __init__(self,
                 env,
                 rollout_policy_list,
                 seed=None,
                 dynamic_prob=False,
                 deter_rollout=False,
                 ):
        super(ATPEnv_Multiple, self).__init__(env)
        env.seed(seed)
        self.rollout_policy_list = rollout_policy_list
        self.deter_rollout = deter_rollout

        # Set range of transformed action
        low = np.concatenate((self.env.observation_space.low,
                              self.env.action_space.low,
                              np.zeros(len(self.rollout_policy_list)))
                             )
        high = np.concatenate((self.env.observation_space.high,
                               self.env.action_space.high,
                               np.zeros(len(self.rollout_policy_list)))
                              )

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.env_max_act = (self.env.action_space.high - self.env.action_space.low) / 2

        max_act = (self.env.action_space.high - self.env.action_space.low)
        self.action_space = spaces.Box(-max_act, max_act, dtype=np.float32)

        # These are set when reset() is called
        self.latest_obs = None
        self.latest_act = None

        self.rollout_dist = np.ones(len(self.rollout_policy_list)) * 1/len(self.rollout_policy_list)
        self.num_selection = np.zeros(len(self.rollout_policy_list))
        self.dynamic_prob = dynamic_prob
        if dynamic_prob:
            self.rew_list = []
            self.rew_threshold = self.spec.reward_threshold

    def reset(self, **kwargs):
        """Reset function for the wrapped environment"""
        self.latest_obs = self.env.reset(**kwargs)

        if self.dynamic_prob and len(self.rew_list) > 0:
            c_returns = np.sum(self.rew_list)
            self.rew_list = []
            idx_to_update = int(self.num_selection[self.idx] % np.shape(self.epi_returns_g)[1])
            self.epi_returns_g[self.idx, idx_to_update] = c_returns

            avg_epi_return_g = np.average(self.epi_returns_g, axis=1)
            gap = np.abs(self.epi_returns_d - avg_epi_return_g)
            prob_dist = np.exp(gap/self.rew_threshold)
            self.rollout_dist = prob_dist / np.sum(prob_dist)
            if np.sum(self.num_selection)%1000==0:
                print(self.rollout_dist)

        # Choose rollout policy
        self.idx = np.random.choice(len(self.rollout_policy_list), p = self.rollout_dist)
        self.rollout_policy = self.rollout_policy_list[self.idx]
        self.idx_obs = np.eye(len(self.rollout_policy_list))[self.idx]
        self.num_selection[self.idx] += 1

        self.latest_act, _ = self.rollout_policy.predict(self.latest_obs, deterministic=self.deter_rollout)

        # Return (s,a)
        # return np.append(self.latest_obs, self.latest_act)
        return np.concatenate((self.latest_obs, self.latest_act, self.idx_obs))

    def init_selection_prob(self, epi_returns_d, epi_returns_g):
        self.epi_returns_d = epi_returns_d
        self.epi_returns_g = epi_returns_g

    def step(self, action):
        """
        Step function for the wrapped environment
        """
        # input action is the delta transformed action for this Environment
        transformed_action = action + self.latest_act
        transformed_action = np.clip(transformed_action, -self.env_max_act, self.env_max_act)

        sim_next_state, sim_rew, sim_done, info = self.env.step(transformed_action)
        if self.dynamic_prob:
            self.rew_list.append(sim_rew)

        info['transformed_action'] = transformed_action

        # get target policy action
        target_policy_action, _ = self.rollout_policy.predict(sim_next_state, deterministic=self.deter_rollout)

        # concat_sa = np.append(sim_next_state, target_policy_action)
        concat_sa = np.concatenate((sim_next_state, target_policy_action, self.idx_obs))

        self.latest_obs = sim_next_state
        self.latest_act = target_policy_action

        return concat_sa, sim_rew, sim_done, info

    def close(self):
        self.env.close()

class GroundedEnv(gym.ActionWrapper):
    """
    Defines the grounded environment
    """
    def __init__(self,
                 env,
                 action_tf_policy,
                 use_deterministic=True,
                 ):
        super(GroundedEnv, self).__init__(env)
        if isinstance(action_tf_policy, list):
            self.num_simul = len(action_tf_policy)
            idx = np.random.randint(0,self.num_simul)
            self.atp_list = action_tf_policy
            self.action_tf_policy = self.atp_list[idx]
        else:
            self.num_simul = 1
            self.action_tf_policy = action_tf_policy

        self.transformed_action_list = []
        self.raw_actions_list = []

        self.latest_obs = None
        self.time_step_counter = 0
        self.high = env.action_space.high
        self.low = env.action_space.low

        max_act = (self.high - self.low)
        self.transformed_action_space = spaces.Box(-max_act, max_act, dtype=np.float32)
        self.use_deterministic = use_deterministic

    def reset(self, **kwargs):
        self.latest_obs = self.env.reset(**kwargs)
        self.time_step_counter = 0
        if self.num_simul > 1:
            idx = np.random.randint(0, self.num_simul)
            self.action_tf_policy = self.atp_list[idx]
        return self.latest_obs

    def step(self, action):
        self.time_step_counter += 1

        concat_sa = np.append(self.latest_obs, action)

        delta_transformed_action, _ = self.action_tf_policy.predict(concat_sa, deterministic=self.use_deterministic)
        transformed_action = action + delta_transformed_action
        transformed_action = np.clip(transformed_action, self.low, self.high)

        self.latest_obs, rew, done, info = self.env.step(transformed_action)

        info['transformed_action'] = transformed_action
        if self.time_step_counter <= 1e4:
            self.transformed_action_list.append(transformed_action)
            self.raw_actions_list.append(action)

        # change the reward to be a function of the input action and
        # not the transformed action
        if 'Hopper' in self.env.unwrapped.spec.id:
            rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
        elif 'HalfCheetah' in self.env.unwrapped.spec.id:
            rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()
        return self.latest_obs, rew, done, info

    def reset_saved_actions(self):
        self.transformed_action_list = []
        self.raw_actions_list = []

    def plot_action_transformation(
            self,
            expt_path=None,
            show_plot=False,
            max_points=3000,
            true_transformation=None):
        """Graphs transformed actions vs input actions"""
        num_action_space = self.env.action_space.shape[0]
        action_low = self.env.action_space.low[0]
        action_high = self.env.action_space.high[0]

        self.raw_actions_list = np.asarray(self.raw_actions_list)
        self.transformed_action_list = np.asarray(self.transformed_action_list)

        opt_gap = None
        if true_transformation is not None:
            if true_transformation == 'Broken':
                true_array = np.copy(self.raw_actions_list)
                true_array[:, 0] = np.zeros_like(true_array[:, 0])
                opt_gap = np.mean(np.linalg.norm(true_array - self.transformed_action_list, axis=1))
            else:
                true_array = np.ones_like(self.transformed_action_list) * true_transformation
                opt_gap = np.mean(np.linalg.norm(true_array - (self.transformed_action_list - self.raw_actions_list), axis=1))

        mean_delta = np.mean(np.abs(self.raw_actions_list - self.transformed_action_list))
        max_delta = np.max(np.abs(self.raw_actions_list - self.transformed_action_list))
        print("Mean delta transformed_action: ", mean_delta)
        print("Max:", max_delta)
        # Reduce sample size
        index = np.random.choice(np.shape(self.raw_actions_list)[0], max_points, replace=False)
        self.raw_actions_list = self.raw_actions_list[index]
        self.transformed_action_list = self.transformed_action_list[index]

        colors = ['go', 'bo', 'ro', 'mo', 'yo', 'ko', 'go', 'bo', 'ro', 'mo', 'yo', 'ko']

        if num_action_space > len(colors):
            print("Unsupported Action space shape.")
            return

        # plotting the data points starts here
        fig = plt.figure(figsize=(int(10*num_action_space), 8))
        plt.rcParams['font.size'] = '24'
        for act_num in range(num_action_space):
            ax = fig.add_subplot(1, num_action_space, act_num+1)
            ax.plot(self.raw_actions_list[:, act_num], self.transformed_action_list[:, act_num], colors[act_num], alpha=1, markersize = 2)
            if true_transformation is not None:
                if true_transformation == 'Broken':
                    if act_num == 0:
                        ax.plot([action_low, action_high], [0, 0], 'k-')
                    else:
                        ax.plot([action_low, action_high], [action_low, action_high], 'k-')
                else:
                    ax.plot([action_low, action_high], [-1.5, 4.5], 'k-')
            ax.title.set_text('Action Dimension '+ str(act_num+1))
            if act_num == 0:
                ax.set_ylabel('Transformed action')
            ax.set(xlabel = 'Original action', xlim=[action_low, action_high], ylim=[action_low, action_high])

        plt.savefig(expt_path)
        if show_plot: plt.show()
        plt.close()

        return mean_delta, max_delta, opt_gap

    def test_grounded_environment(self,
                                  expt_path,
                                  target_policy=None,
                                  random=True,
                                  true_transformation=None,
                                  num_steps=2048,
                                  deter_target=False):
        """Tests the grounded environment for action transformation"""
        print("TESTING GROUNDED ENVIRONMENT")
        self.reset_saved_actions()
        obs = self.reset()
        time_step_count = 0
        for _ in range(num_steps):
            time_step_count += 1
            if not random:
                action, _ = target_policy.predict(obs, deterministic=deter_target)
            else:
                action = self.action_space.sample()
            obs, _, done, _ = self.step(action)
            if done:
                obs = self.reset()
                done = False

        _, _, opt_gap = self.plot_action_transformation(expt_path=expt_path, max_points=num_steps, true_transformation=true_transformation)
        self.reset_saved_actions()

        return opt_gap

    def close(self):
        self.env.close()