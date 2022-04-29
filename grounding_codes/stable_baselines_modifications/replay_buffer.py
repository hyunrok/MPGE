import random

import numpy as np

class ReplayBuffer(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        # return np.array(obses_t), np.array(actions), np.array(rewards).reshape(-1,1), np.array(obses_tp1), np.array(dones).reshape(-1,1)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obs, acs, rewards, next_obs, dones = self._encode_sample(idxes)
        return obs, acs, rewards.reshape(-1, 1), next_obs, dones.reshape(-1, 1)

    def initialize_teacher_buffer_idx(self, expert_dataset, idx_obs):
        demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = expert_dataset.get_transitions()

        episode_lengths = np.where(demo_dones == 1)[0]
        n_samples = len(demo_obs)
        # get episode_score for each demo sample, either 0 or episode-reward
        episode_idx, n_episodes = 0, len(demo_episode_scores)
        for idx in range(n_samples - 1):
            episode_score = demo_episode_scores[episode_idx]
            episode_length = episode_lengths[episode_idx]

            if demo_dones[idx + 1] == 1:
                print('{}-th sample, episode_score for demonstration tarjectory: {}'.format(idx, episode_score))
                episode_idx += 1
                assert episode_length - idx >= 0
            # scaled_demo_action = scale_action(self.action_space, demo_actions[idx])
            self.add(np.append(demo_obs[idx], idx_obs), demo_actions[idx], demo_rewards[idx], np.append(demo_next_obs[idx], idx_obs), float(demo_dones[idx]))

            if idx % 1000 == 0:
                print("Adding demonstration to the replay buffer, processing {} %  ..".format(
                    float(idx + 1) * 100 / n_samples))
        ### add last sample to buffer
        # scaled_demo_action = scale_action(self.action_space, demo_actions[-1])
        self.add(np.append(demo_obs[-1], idx_obs), demo_actions[-1], demo_rewards[-1], np.append(demo_next_obs[-1], idx_obs), float(demo_dones[-1]))

    # def initialize_teacher_buffer_multi_src(self, expert_dataset):
    #     for i in range(len(expert_dataset)):
    #         demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = expert_dataset[i].get_transitions()
    #         idx_obs = np.eye(len(expert_dataset))[i]
    #
    #         episode_lengths = np.where(demo_dones == 1)[0]
    #         n_samples = len(demo_obs)
    #         # get episode_score for each demo sample, either 0 or episode-reward
    #         episode_idx, n_episodes = 0, len(demo_episode_scores)
    #         for idx in range(n_samples - 1):
    #             episode_score = demo_episode_scores[episode_idx]
    #             episode_length = episode_lengths[episode_idx]
    #
    #             if demo_dones[idx + 1] == 1:
    #                 print('{}-th sample, episode_score for demonstration tarjectory: {}'.format(idx, episode_score))
    #                 episode_idx += 1
    #                 assert episode_length - idx >= 0
    #             # scaled_demo_action = scale_action(self.action_space, demo_actions[idx])
    #             # self.add(demo_obs[idx], demo_actions[idx], demo_rewards[idx], demo_next_obs[idx], float(demo_dones[idx]))
    #             self.add(np.append(demo_obs[idx], idx_obs), demo_actions[idx], demo_rewards[idx], np.append(demo_next_obs[idx], idx_obs), float(demo_dones[idx]))
    #
    #             if idx % 1000 == 0:
    #                 print("Adding demonstration to the replay buffer, processing {} %  ..".format(
    #                     float(idx + 1) * 100 / n_samples))
    #         ### add last sample to buffer
    #         # scaled_demo_action = scale_action(self.action_space, demo_actions[-1])
    #         # self.add(demo_obs[-1], demo_actions[-1], demo_rewards[-1], demo_next_obs[-1], float(demo_dones[-1]))
    #         self.add(np.append(demo_obs[-1], idx_obs), demo_actions[-1], demo_rewards[-1], np.append(demo_next_obs[-1], idx_obs), float(demo_dones[-1]))