import numpy as np

from gym import spaces

from grounding_codes.atp_envs import set_rollout_policy

def generate_target_traj(rollout_policy_path, env, save_path=None, n_episodes=None, n_transitions=10000, seed=None, deterministic=False):
    """
    Function for collecting samples from environment

    :param rollout_policy_path: (str) path to read rollout policy
    :param env: (gym.env) target environment
    :param save_path: (str) path to store data
    :param n_episodes: (int) number of episodes to collect data
    :param n_transitions: (int) number of transitions to collect data
    :param seed: (int) random seed
    :param deterministic: (bool) whether to use deterministic or probabilistic rollout policy
    """

    rollout_policy = set_rollout_policy(rollout_policy_path, seed=seed)

    assert n_episodes == None or n_transitions == None

    if n_episodes == None:
        print("Generate traget trajectory up to " + str(n_transitions) + " transition samples")
    elif n_transitions == None:
        print("Generate traget trajectory up to " + str(n_episodes) + " episodes")

    actions = []
    observations = []
    next_observations = []
    rewards = []
    if n_episodes is not None:
        episode_returns = np.zeros((n_episodes,))
    else:
        episode_returns = []
    episode_starts = []

    tmp_actions = []
    tmp_observations = []
    tmp_next_observations = []
    tmp_rewards = []
    tmp_episode_starts = []
    reward_sum = 0.0

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None

    ## SET MAXIMUM transitions
    if n_transitions is not None:
        while True:
            action, state = rollout_policy.predict(obs, state=state, mask=mask, deterministic=deterministic)
            tmp_observations.append(np.append(obs, action))
            tmp_next_observations.append(np.append(obs, action)) # First element will be deleted

            obs, reward, done, infos = env.step(action)

            tmp_actions.append(action)
            tmp_rewards.append(reward)
            tmp_episode_starts.append(done)
            reward_sum += reward

            idx += 1

            if done:
                if idx <= n_transitions:
                    print("Find expert trajectory with reward {:.2f}".format(reward_sum))
                    episode_returns.append(reward_sum)
                    actions += tmp_actions
                    rewards += tmp_rewards
                    observations += tmp_observations
                    tmp_next_observations = tmp_next_observations[1:]
                    # Add dummy action in the last observation
                    if isinstance(env.action_space, spaces.Box):
                        dummy_action = np.zeros(env.action_space.shape)
                        tmp_next_observations.append(np.append(obs,dummy_action))
                    elif isinstance(env.action_space, spaces.Discrete):
                        dummy_action = 0
                        tmp_next_observations.append(np.append(obs, dummy_action))
                    next_observations += tmp_next_observations
                    episode_starts += tmp_episode_starts
                    ep_idx += 1

                    reward_sum = 0.0
                    del tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts
                    tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts  = [], [], [], [], []

                    obs = env.reset()
                    # Reset the state in case of a recurrent policy
                    state = None
                else:
                    break
            elif idx == n_transitions:
                print("Find partial trajectory with reward {:.2f}".format(reward_sum))
                episode_returns.append(reward_sum)
                actions += tmp_actions
                rewards += tmp_rewards
                observations += tmp_observations
                tmp_next_observations = tmp_next_observations[1:]
                # Add dummy action in the last observation
                if isinstance(env.action_space, spaces.Box):
                    dummy_action = np.zeros(env.action_space.shape)
                    tmp_next_observations.append(np.append(obs, dummy_action))
                elif isinstance(env.action_space, spaces.Discrete):
                    dummy_action = 0
                    tmp_next_observations.append(np.append(obs, dummy_action))
                next_observations += tmp_next_observations
                episode_starts += tmp_episode_starts
                ep_idx += 1

                reward_sum = 0.0
                del tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts
                tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts = [], [], [], [], []
                break
        episode_returns = np.array(episode_returns)
    else: # SET MAXIMUM number of EPISODES
        while ep_idx < n_episodes:
            # tmp_observations.append(obs)
            action, state = rollout_policy.predict(obs, state=state, mask=mask, deterministic=deterministic)
            tmp_observations.append(np.append(obs, action))
            tmp_next_observations.append(np.append(obs, action)) # Will be modified

            obs, reward, done, infos = env.step(action)

            tmp_actions.append(action)
            tmp_rewards.append(reward)
            tmp_episode_starts.append(done)
            reward_sum += reward

            idx += 1

            if done:
                print("Find expert trajectory with reward {:.2f}".format(reward_sum))
                episode_returns[ep_idx] = reward_sum
                actions += tmp_actions
                rewards += tmp_rewards
                observations += tmp_observations
                tmp_next_observations = tmp_next_observations[1:]
                # Add dummy action in the last observation
                if isinstance(env.action_space, spaces.Box):
                    dummy_action = np.zeros(env.action_space.shape)
                    tmp_next_observations.append(np.append(obs,dummy_action))
                elif isinstance(env.action_space, spaces.Discrete):
                    dummy_action = 0
                    tmp_next_observations.append(np.append(obs, dummy_action))
                next_observations += tmp_next_observations
                episode_starts += tmp_episode_starts
                ep_idx += 1

                reward_sum = 0.0
                del tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts
                tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts  = [], [], [], [], []

                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None

    if isinstance(env.observation_space, spaces.Box):
        shape = tuple(map(sum, zip(env.observation_space.shape, env.action_space.shape)))
        observations = np.concatenate(observations).reshape((-1,) + shape)
        next_observations = np.concatenate(next_observations).reshape((-1,) + shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        raise NotImplementedError("Discrete observation space is not supported yet")
        # observations = np.array(observations).reshape((-1, 1))
        # next_observations = np.array(next_observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        raise NotImplementedError("Discrete action space is not supported yet")
        # actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions) and len(next_observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'next_obs': next_observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict