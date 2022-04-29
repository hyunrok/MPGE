"""
EXTENSION from OPOLO repository https://github.com/illidanlab/opolo-code
"""


import time
from contextlib import contextmanager
from collections import deque

import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common import explained_variance, zipsame, dataset, colorize, ActorCriticRLModel, \
    SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.policies import ActorCriticPolicy
from grounding_codes.stable_baselines_modifications.TRPOGAIfO_utils import SegmentGenerator, add_vtarg_and_adv, flatten_lists
from grounding_codes.discriminators import GAIfODiscriminator_sas
from grounding_codes.stable_baselines_modifications.dataset import ExpertDataset
from grounding_codes.stable_baselines_modifications.replay_buffer import ReplayBuffer


class TRPOGAIfO_multi_pol(ActorCriticRLModel):
    """
    GAIfO implementation based on the Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)
    extending for using multiple policies at once

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=1, config={}):
        super(TRPOGAIfO_multi_pol, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.using_gail = True
        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # GAIL Params
        # self.hidden_size_adversary = 100
        # self.adversary_entcoeff = 1e-3
        self.expert_dataset = None
        self.g_step = 1
        self.d_step = 1
        self.d_stepsize = 3e-4

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.allmean = None
        self.nworkers = None
        self.rank = None
        self.reward_giver = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.params = None
        self.summary = None
        self.config = config

        ## Customized parameters
        self.demo_buffer_size = int(1e5)
        # self.d_gradient_steps = self.d_step # 1
        # self.d_batch_size = self.timesteps_per_batch // self.d_step
        # self.d_learning_rate = self.d_stepsize

        self.d_batch_size = 256
        if timesteps_per_batch < self.d_batch_size:
            self.d_batch_size = timesteps_per_batch
        self.d_gradient_steps = timesteps_per_batch // self.d_batch_size
        self.d_learning_rate = 3e-4

        if _init_setup_model:
            self.setup_model(config)

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self, config):
        self.num_src = config.get('num_src')
        self.dummy_obs = np.zeros(self.num_src)

        # prevent import loops
        self.expert_data_path = config.get('expert_data_path', None)
        if self.expert_data_path is not None:
            self.expert_dataset = []
            for i in range(len(self.expert_data_path)):
                self.expert_dataset.append(
                    ExpertDataset(expert_path=self.expert_data_path[i], ob_flatten=False))
                print('-' * 20 + "expert_data_path: {}".format(self.expert_data_path[i]))
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the TRPO model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            np.set_printoptions(precision=3)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                if self.using_gail:
                    print("Using sas discriminator")
                    self.discriminator = GAIfODiscriminator_sas(
                        self.observation_space,
                        self.action_space,
                        hidden_size=256,
                        entcoeff=0.001,
                        gradcoeff=10,
                        normalize=True,
                        idx_dim=self.num_src,
                    )

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target avantage function (if applicable)
                    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

                    observation = self.policy_pi.obs_ph
                    action = self.policy_pi.pdtype.sample_placeholder([None])

                    kloldnew = old_policy.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    entbonus = self.entcoeff * meanent

                    vferr = tf.reduce_mean(tf.square(self.policy_pi.value_flat - ret))

                    # advantage * pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action) -
                                   old_policy.proba_distribution.logp(action))
                    surrgain = tf.reduce_mean(ratio * atarg)

                    optimgain = surrgain + entbonus
                    losses = [optimgain, meankl, entbonus, surrgain, meanent]
                    self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

                    dist = meankl

                    all_var_list = tf_util.get_trainable_vars("model")
                    var_list = [v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name]
                    vf_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name]

                    self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                    self.set_from_flat = tf_util.SetFromFlat(var_list, sess=self.sess)

                    klgrads = tf.gradients(dist, var_list)
                    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
                    shapes = [var.get_shape().as_list() for var in var_list]
                    start = 0
                    tangents = []
                    for shape in shapes:
                        var_size = tf_util.intprod(shape)
                        tangents.append(tf.reshape(flat_tangent[start: start + var_size], shape))
                        start += var_size
                    gvp = tf.add_n([tf.reduce_sum(grad * tangent)
                                    for (grad, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
                    # Fisher vector products
                    fvp = tf_util.flatgrad(gvp, var_list)
                    policy_summary = []
                    policy_summary.append(tf.summary.scalar('entropy_loss', meanent))
                    policy_summary.append(tf.summary.scalar('policy_gradient_loss', optimgain))
                    policy_summary.append(tf.summary.scalar('value_function_loss', surrgain))
                    policy_summary.append(tf.summary.scalar('approximate_kullback-leibler', meankl))
                    policy_summary.append(tf.summary.scalar('loss', optimgain + meankl + entbonus + surrgain + meanent))

                    self.assign_old_eq_new = \
                        tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                          zipsame(tf_util.get_globals_vars("oldpi"),
                                                                  tf_util.get_globals_vars("model"))])
                    self.compute_losses = tf_util.function([observation, old_policy.obs_ph, action, atarg], losses)
                    self.compute_fvp = tf_util.function([flat_tangent, observation, old_policy.obs_ph, action, atarg],
                                                        fvp)
                    self.compute_vflossandgrad = tf_util.function([observation, old_policy.obs_ph, ret],
                                                                  tf_util.flatgrad(vferr, vf_var_list))

                    @contextmanager
                    def timed(msg):
                        if self.rank == 0 and self.verbose >= 1:
                            print(colorize(msg, color='magenta'))
                            start_time = time.time()
                            yield
                            print(colorize("done in {:.3f} seconds".format((time.time() - start_time)),
                                           color='magenta'))
                        else:
                            yield

                    def allmean(arr, nworkers):
                        assert isinstance(arr, np.ndarray)
                        out = np.empty_like(arr)
                        MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
                        out /= nworkers
                        return out

                    tf_util.initialize(sess=self.sess)

                    th_init = self.get_flat()
                    MPI.COMM_WORLD.Bcast(th_init, root=0)
                    self.set_from_flat(th_init)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                    if self.using_gail:
                        # self.d_adam = MpiAdam(self.reward_giver.get_trainable_variables(), sess=self.sess)
                        self.d_adam = MpiAdam(self.discriminator.get_trainable_variables(), sess=self.sess)
                        self.d_adam.sync()
                    self.vfadam.sync()

                with tf.variable_scope("input_info", reuse=False):
                    policy_summary.append(tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret)))
                    policy_summary.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.vf_stepsize)))
                    policy_summary.append(tf.summary.scalar('advantage', tf.reduce_mean(atarg)))
                    policy_summary.append(tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl)))

                    if self.full_tensorboard_log:
                        policy_summary.append(tf.summary.histogram('discounted_rewards', ret))
                        policy_summary.append(tf.summary.histogram('learning_rate', self.vf_stepsize))
                        policy_summary.append(tf.summary.histogram('advantage', atarg))
                        policy_summary.append(tf.summary.histogram('kl_clip_range', self.max_kl))
                        if tf_util.is_image(self.observation_space):
                            policy_summary.append(tf.summary.image('observation', observation))
                        else:
                            policy_summary.append(tf.summary.histogram('observation', observation))

                self.timed = timed
                self.allmean = allmean

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                self.params = tf_util.get_trainable_vars("model") + tf_util.get_trainable_vars("oldpi")
                if self.using_gail:
                    self.params.extend(self.discriminator.get_trainable_variables())

                self.summary = tf.summary.merge(policy_summary)

                self.compute_lossandgrad = \
                    tf_util.function([observation, old_policy.obs_ph, action, atarg, ret],
                                     [self.summary, tf_util.flatgrad(optimgain, var_list)] + losses)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="TRPO",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            with self.sess.as_default():
                callback.on_training_start(locals(), globals())

                seg_generator = SegmentGenerator(
                    self.policy_pi,
                    self.env,
                    self.timesteps_per_batch,
                    self.discriminator,
                    sess=self.sess,
                    config=self.config,
                    callback=callback
                )
                seg_gen = seg_generator.traj_segment_generator(gail=self.using_gail)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
                reward_buffer = deque(maxlen=40)  # rolling buffer for episode rewards

                true_reward_buffer = None
                if self.using_gail:
                    true_reward_buffer = deque(maxlen=40)

                    # Initialize demonstration buffer
                    batchsize = self.timesteps_per_batch // self.d_step
                    self.teacher_buffer = []
                    for i in range(self.num_src):
                        idx_obs = np.eye(self.num_src)[i]
                        teacher_buffer_i = ReplayBuffer(self.demo_buffer_size)
                        teacher_buffer_i.initialize_teacher_buffer_idx(self.expert_dataset[i], idx_obs)
                        self.teacher_buffer.append(teacher_buffer_i)

                    #  Stats not used for now
                    # TODO: replace with normal tb logging
                    #  g_loss_stats = Stats(loss_names)
                    #  d_loss_stats = Stats(reward_giver.loss_name)
                    #  ep_stats = Stats(["True_rewards", "Rewards", "Episode_length"])

                while True:
                    # if callback is not None:
                    #     # Only stop training if return value is False, not when it is None. This is for backwards
                    #     # compatibility with callbacks that have no return statement.
                    #     if callback(locals(), globals()) is False:
                    #         break
                    # if total_timesteps and timesteps_so_far >= total_timesteps:
                    #     break

                    if timesteps_so_far >= total_timesteps:
                        break

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    def fisher_vector_product(vec):
                        return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess),
                                            self.nworkers) + self.cg_damping * vec

                    # ------------------ Update G ------------------
                    logger.log("Optimizing Policy...")
                    # g_step = 1 when not using GAIL
                    mean_losses = None
                    vpredbefore = None
                    tdlamret = None
                    observation = None
                    action = None
                    seg = None
                    for k in range(self.g_step):
                        with self.timed("sampling"):
                            seg = seg_gen.__next__()
                        add_vtarg_and_adv(seg, self.gamma, self.lam)
                        observation, action = seg["observations"], seg["actions"]
                        atarg, tdlamret = seg["adv"], seg["tdlamret"]

                        vpredbefore = seg["vpred"]  # predicted value function before update
                        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                        # true_rew is the reward without discount
                        if writer is not None:
                            total_episode_reward_logger(self.episode_reward,
                                                        seg["true_rewards"].reshape(
                                                            (self.n_envs, -1)),
                                                        seg["dones"].reshape((self.n_envs, -1)),
                                                        writer, self.num_timesteps)

                        args = seg["observations"], seg["observations"], seg["actions"], atarg
                        # Subsampling: see p40-42 of John Schulman thesis
                        # http://joschu.net/docs/thesis.pdf
                        fvpargs = [arr[::5] for arr in args]

                        self.assign_old_eq_new(sess=self.sess)

                        with self.timed("computegrad"):
                            steps = self.num_timesteps + (k + 1) * (seg["total_timestep"] / self.g_step)
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata() if self.full_tensorboard_log else None
                            # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                            if writer is not None:
                                summary, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                      options=run_options,
                                                                                      run_metadata=run_metadata)
                                if self.full_tensorboard_log:
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                options=run_options,
                                                                                run_metadata=run_metadata)

                        lossbefore = self.allmean(np.array(lossbefore), self.nworkers)
                        grad = self.allmean(grad, self.nworkers)
                        if np.allclose(grad, 0):
                            logger.log("Got zero gradient. not updating")
                        else:
                            with self.timed("conjugate_gradient"):
                                stepdir = conjugate_gradient(fisher_vector_product, grad, cg_iters=self.cg_iters,
                                                             verbose=self.rank == 0 and self.verbose >= 1)
                            assert np.isfinite(stepdir).all()
                            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                            # abs(shs) to avoid taking square root of negative values
                            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
                            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                            fullstep = stepdir / lagrange_multiplier
                            expectedimprove = grad.dot(fullstep)
                            surrbefore = lossbefore[0]
                            stepsize = 1.0
                            thbefore = self.get_flat()
                            for _ in range(10):
                                thnew = thbefore + fullstep * stepsize
                                self.set_from_flat(thnew)
                                mean_losses = surr, kl_loss, *_ = self.allmean(
                                    np.array(self.compute_losses(*args, sess=self.sess)), self.nworkers)
                                improve = surr - surrbefore
                                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                                if not np.isfinite(mean_losses).all():
                                    logger.log("Got non-finite value of losses -- bad!")
                                elif kl_loss > self.max_kl * 1.5:
                                    logger.log("violated KL constraint. shrinking step.")
                                elif improve < 0:
                                    logger.log("surrogate didn't improve. shrinking step.")
                                else:
                                    logger.log("Stepsize OK!")
                                    break
                                stepsize *= .5
                            else:
                                logger.log("couldn't compute a good step")
                                self.set_from_flat(thbefore)
                            if self.nworkers > 1 and iters_so_far % 20 == 0:
                                # list of tuples
                                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))
                                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                            for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                                logger.record_tabular(loss_name, loss_val)

                        with self.timed("vf"):
                            for _ in range(self.vf_iters):
                                # NOTE: for recurrent policies, use shuffle=False?
                                for (mbob, mbret) in dataset.iterbatches((seg["observations"], seg["tdlamret"]),
                                                                         include_final_partial_batch=False,
                                                                         batch_size=128,
                                                                         shuffle=True):
                                    grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, sess=self.sess),
                                                        self.nworkers)
                                    self.vfadam.update(grad, self.vf_stepsize)

                    logger.record_tabular("explained_variance_tdlam_before",
                                          explained_variance(vpredbefore, tdlamret))

                    if self.using_gail:
                        ## # ------------------ Update D ------------------
                        self.discriminator.train_onpolicy_discriminator(
                            writer, logger, self.d_gradient_steps, self.d_learning_rate, self.d_batch_size,
                            self.teacher_buffer, seg, self.num_timesteps, self.sess, self.d_adam, self.nworkers)

                        # lr: lengths and rewards
                        lr_local = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
                        list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                        lens, rews, true_rets = map(flatten_lists, zip(*list_lr_pairs))
                        true_reward_buffer.extend(true_rets)
                    else:
                        # lr: lengths and rewards
                        lr_local = (seg["ep_lens"], seg["ep_rets"])  # local values
                        list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                        lens, rews = map(flatten_lists, zip(*list_lr_pairs))
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)

                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                    if self.using_gail:
                        logger.record_tabular("EpTrueRewMean", np.mean(true_reward_buffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1

                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)

                    if self.verbose >= 1 and self.rank == 0:
                        logger.dump_tabular()

        callback.on_training_end()
        return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if len(np.shape(observation))> 1:
            observation = np.concatenate((observation, np.repeat([self.dummy_obs], np.shape(observation)[0], axis=0)), axis=1)
        else:
            observation = np.append(observation, self.dummy_obs)

        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, states, _ = self.step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def set_expert_Data(self, expert_data_path):
        # prevent import loops
        self.expert_data_path = expert_data_path
        self.expert_dataset = []
        for i in range(len(self.expert_data_path)):
            self.expert_dataset.append(
                ExpertDataset(expert_path=self.expert_data_path[i], ob_flatten=False))
            print('-' * 20 + "expert_data_path: {}".format(self.expert_data_path[i]))

    def save(self, save_path, cloudpickle=False):
        # if self.using_gail and self.expert_dataset is not None:
        #     # Exit processes to pickle the dataset
        #     self.expert_dataset.prepare_pickling()
        data = {
            "gamma": self.gamma,
            "timesteps_per_batch": self.timesteps_per_batch,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "lam": self.lam,
            "entcoeff": self.entcoeff,
            "cg_damping": self.cg_damping,
            "vf_stepsize": self.vf_stepsize,
            "vf_iters": self.vf_iters,
            # "hidden_size_adversary": self.hidden_size_adversary,
            # "adversary_entcoeff": self.adversary_entcoeff,
            # "expert_dataset": self.expert_dataset,
            "g_step": self.g_step,
            "d_step": self.d_step,
            "d_stepsize": self.d_stepsize,
            "using_gail": self.using_gail,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, config={}, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        # model.setup_model()
        if len(config) == 0:
            model.setup_model()
        else:
            model.setup_model(config)

        model.load_parameters(params)

        return model