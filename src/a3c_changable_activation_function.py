from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, StateActionPredictor, StatePredictor
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
import pandas as pd
from distutils.dir_util import copy_tree
import math
import random

from constants import constants
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logDirCounter = 1000000

nowDistance = 0
nowMaxDistance = 800
nowMaxDistanceCounter = 0
realMaxDistance = 0
lastDistance = 0
normalizationParameter = 40.0
nowY = 0
lastY = 0
isflying = False
startFlyingIdx = -1
minClip = -0.1
flyingMaxClip = 0.1

fixed_level = 2
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.1  # e-greedy threshold end value
EPS_DECAY = 500000  # e-greedy threshold decay
EPS_threshold = 1
EPS_step = 0

def discount(x, gamma):
    """
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0, clip=False):
    """
    Given a rollout, compute its returns and the advantage.
    """
    # collecting transitions
    if rollout.unsup:
        batch_si = np.asarray(rollout.states + [rollout.end_state])
    else:
        batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(rollout.bonuses + [0])
        """
        if(len(rollout.bonuses) > 0):
            rewards_plus_v += np.asarray(rollout.bonuses + [-rollout.bonuses[-1]])
        else:
            rewards_plus_v += np.asarray(rollout.bonuses + [0])
        """
    if clip:
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # value network target

    # collecting target for policy network
    rewards = np.asarray(rollout.rewards)
    if rollout.unsup:
        rewards += np.asarray(rollout.bonuses)
    if clip:
        rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, unsup=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.unsup = unsup
        if self.unsup:
            self.bonuses = [] # curiousity
            self.end_state = None


    def add(self, state, action, reward, value, terminal, features,
                bonus=None, end_state=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        if self.unsup:
            self.bonuses += [bonus]
            self.end_state = end_state

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if self.unsup:
            self.bonuses.extend(other.bonuses)
            self.end_state = other.end_state

class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, policy, num_local_steps, visualise, predictor, envWrap,
                    noReward, task):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)  # ideally, should be 1. Mostly doesn't matter in our case.
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.predictor = predictor
        self.envWrap = envWrap
        self.noReward = noReward
        self.task = task

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps,
                                        self.summary_writer, self.visualise, self.predictor,
                                        self.envWrap, self.noReward, self.task)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, render, predictor,
                envWrap, noReward, task):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()  # reset lstm memory
    length = 0
    rewards = 0
    values = 0
    if predictor is not None:
        ep_bonus = 0
        life_bonus = 0

    while True:
        terminal_end = False
        rollout = PartialRollout(predictor is not None)

        for _ in range(num_local_steps):
            # run policy
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]
            #action, value_, all_action, features = fetched[0], fetched[1], fetched[2], fetched[3:]
            

            # epsilon greedy
            global EPS_step, EPS_threshold, EPS_END, EPS_START
            EPS_step = policy.global_step.eval()
            EPS_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * EPS_step / EPS_DECAY)
            sample = random.random()

            if(sample < EPS_threshold):
                randomAction = np.zeros(action.shape[0])
                randomAction[random.randint(0, action.shape[0] - 1)] = 1
                #second_largest_index = np.argsort(all_action[0])[-2]
                #randomAction[second_largest_index] = 1
                action = randomAction
            

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = action.argmax()
            state, reward, terminal, info = env.step(stepAct)
            global nowDistance, lastDistance, nowMaxDistance, nowMaxDistanceCounter, realMaxDistance, normalizationParameter
            global nowY, lastY, isflying, startFlyingIdx

            # x axis position
            lastDistance = nowDistance
            nowDistance = info['distance']
            if(nowDistance > realMaxDistance):
                realMaxDistance = nowDistance
            
            # y axis position
            lastY = nowY
            nowY = info['curr_y_position']
            if(lastY != nowY and not isflying):
                isflying = True
                startFlyingIdx = length
            elif(lastY == nowY):

                # if the agent "was" flying, and landing now, then give it huge reward!
                if(isflying):
                    if(startFlyingIdx < len(rollout.bonuses)):
                        tempPreRollout = rollout.bonuses[:startFlyingIdx]
                        tempNextRollout = [b + 1 for b in rollout.bonuses[startFlyingIdx:]]
                        rollout.bonuses = tempPreRollout + tempNextRollout
                        #b = a[:5]
                        #b += [i + 1 if i >= 5 else 0 for i in a[5:]]
                    else:
                        tempNextRollout = [b + 1 for b in rollout.bonuses]

                startFlyingIdx = -1
                isflying = False
            
            if noReward:
                reward = 0.
            """
            if render:
                env.render()
            """

            #rollout.add(self, state, action, reward, value, terminal, features, bonus=None, end_state=None)
            curr_tuple = [last_state, action, reward, value_, terminal, last_features]
            if predictor is not None:
                bonus = predictor.pred_bonus(last_state, state, action)
                
                # bonus
                global fixed_level, minClip, flyingMaxClip

                if(info['level'] != fixed_level):
                    lastDistance = 0

                diff = (nowDistance - lastDistance)
                diff = (2 / float(2 * normalizationParameter)) * (diff + normalizationParameter) - 1
                
                if(diff <= minClip):
                    diff = minClip

                bonus = bonus + diff
                
                if(terminal):
                    # arrive the goal
                    if(nowDistance > 2500):
                        bonus = 0.00000001
                        nowMaxDistance = 800
                        nowMaxDistanceCounter = 0
                    else:
                        if(EPS_threshold <= 0.2 and nowMaxDistanceCounter < 2):
                            if(nowDistance / 100 <= nowMaxDistance / 100 - 2 and nowMaxDistance > 800):
                                nowMaxDistance = nowMaxDistance - 100
                                nowMaxDistanceCounter = nowMaxDistanceCounter + 1
                else:
                    # if the agent break the max distance record, then give it huge reward!
                    if(nowDistance / 100 > nowMaxDistance / 100):
                        nowMaxDistance = nowDistance
                        bonus = bonus + 1
                        nowMaxDistanceCounter = 0
                        rollout.bonuses = [b + 1 for b in rollout.bonuses]

                # if the agent is flying, then clip the reward to maxima at 0.1    
                if(isflying):
                    bonus = bonus if bonus <= flyingMaxClip else flyingMaxClip

                curr_tuple += [bonus, state]
                life_bonus += bonus
                ep_bonus += bonus

            # collect the experience
            rollout.add(*curr_tuple)
            """
            print(startFlyingIdx)
            print(rollout.bonuses)
            raw_input("")
            """
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            if terminal or length >= timestep_limit:
                # prints summary of each life if envWrap==True else each game
                if predictor is not None:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (rewards, length, life_bonus))
                    life_bonus = 0
                else:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])

                print("normalizationParameter: {0}".format(normalizationParameter))
                print("nowMaxDistance: {0}".format(nowMaxDistance))
                print("realMaxDistance: {0}".format(realMaxDistance))
                print("EPS_threshold: {0}".format(EPS_threshold))
                print("EPS_step: {0}".format(EPS_step))
                print("")

                length = 0
                rewards = 0
                terminal_end = True
                last_features = policy.get_initial_features()  # reset lstm memory
                # TODO: don't reset when gym timestep_limit increases, bootstrap -- doesn't matter for atari?
                # reset only if it hasn't already reseted
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()

            if info:
                # summarize full game including all lives (even if envWrap=True)
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                if terminal:
                    summary.value.add(tag='global/episode_value', simple_value=float(values))
                    values = 0
                    if predictor is not None:
                        summary.value.add(tag='global/episode_bonus', simple_value=float(ep_bonus))
                        ep_bonus = 0
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            if terminal_end:
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(self, env, task, visualise, unsupType, summary_writer, envWrap=False, designHead='universe', noReward=False):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.task = task
        self.unsup = unsupType is not None # unsupType default is "action"
        self.envWrap = envWrap
        self.env = env
        self.distance = 0

        predictor = None
        numaction = env.action_space.n
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                if self.unsup:
                    with tf.variable_scope("predictor"):
                        if 'state' in unsupType:
                            self.ap_network = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.ap_network = StateActionPredictor(env.observation_space.shape, numaction, designHead)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                pi.global_step = self.global_step
                
                if self.unsup:
                    with tf.variable_scope("predictor"):
                        if 'state' in unsupType:
                            self.local_ap_network = predictor = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.local_ap_network = predictor = StateActionPredictor(env.observation_space.shape, numaction, designHead)

            # Computing a3c loss: https://arxiv.org/abs/1506.02438
            # print('a3c.loss()') 
            self.tfNowDistance = tf.placeholder(tf.float32, [], name="tfNowDistance")
            self.tfLastDistance = tf.placeholder(tf.float32, [], name="tfLastDistance")
            self.tfGradientDistance = tf.placeholder(tf.float32, [], name="tfGradientDistance")

            self.ac = tf.placeholder(tf.float32, [None, numaction], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)
            # 1) the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)  # Eq (19)
            # 2) loss of value function: l2_loss = (x-y)^2/2
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))  # Eq (28)
            # 3) entropy to ensure randomness
            entropy = - tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))
            # final a3c loss: lr of critic is half of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy * constants['ENTROPY_BETA']
            print(pi_loss)
            print(self.loss)
            
            # compute gradients
            grads = tf.gradients(self.loss * 20.0, pi.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.

            # computing predictor loss
            if self.unsup:
                if 'state' in unsupType:
                    self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
                else:
                    self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                    predictor.forwardloss * constants['FORWARD_LOSS_WT'])
                predgrads = tf.gradients(self.predloss * 20.0, predictor.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.

                # do not backprop to policy
                if constants['POLICY_NO_BACKPROP_STEPS'] > 0:
                    grads = [tf.scalar_mul(tf.to_float(tf.greater(self.global_step, constants['POLICY_NO_BACKPROP_STEPS'])), grads_i)
                                    for grads_i in grads]


            self.runner = RunnerThread(env, pi, constants['ROLLOUT_MAXLEN'], visualise,
                                        predictor, envWrap, noReward, task)

            # storing summaries
            bs = tf.to_float(tf.shape(pi.x)[0])
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss)
                tf.summary.scalar("model/value_loss", vf_loss)
                tf.summary.scalar("model/entropy", entropy)
                tf.summary.image("model/state", pi.x)  # max_outputs=10
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                
                tf.summary.scalar("distance/last_distance", self.tfLastDistance)
                tf.summary.scalar("distance/now_distance", self.tfNowDistance)
                tf.summary.scalar("distance/gradient_distance", self.tfGradientDistance)

                if self.unsup:
                    tf.summary.scalar("model/predloss", self.predloss)
                    if 'action' in unsupType:
                        tf.summary.scalar("model/inv_loss", predictor.invloss)
                        tf.summary.scalar("model/forward_loss", predictor.forwardloss)
                    tf.summary.scalar("model/predgrad_global_norm", tf.global_norm(predgrads))
                    tf.summary.scalar("model/predvar_global_norm", tf.global_norm(predictor.var_list))
                self.summary_op = tf.summary.merge_all()
            else:
                tf.scalar_summary("model/policy_loss", pi_loss)
                tf.scalar_summary("model/value_loss", vf_loss)
                tf.scalar_summary("model/entropy", entropy)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                if self.unsup:
                    tf.scalar_summary("model/predloss", self.predloss)
                    if 'action' in unsupType:
                        tf.scalar_summary("model/inv_loss", predictor.invloss)
                        tf.scalar_summary("model/forward_loss", predictor.forwardloss)
                    tf.scalar_summary("model/predgrad_global_norm", tf.global_norm(predgrads))
                    tf.scalar_summary("model/predvar_global_norm", tf.global_norm(predictor.var_list))
                self.summary_op = tf.merge_all_summaries()
            #self.summary_writer = summary_writer
            #self.summary_writer.add_summary(tf.Summary.FromString(self.summary_op), self.global_step)

            # clip gradients
            grads, _ = tf.clip_by_global_norm(grads, constants['GRAD_NORM_CLIP'])
            grads_and_vars = list(zip(grads, self.network.var_list))
            if self.unsup:
                predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
                pred_grads_and_vars = list(zip(predgrads, self.ap_network.var_list))

                '''
                # testing loss 
                distance_loss = tf.divide(tf.subtract(self.tfNowDistance, self.tfLastDistance), 100)
                dis_grads = tf.gradients(distance_loss, pi.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.
                dis_grads, _ = tf.clip_by_global_norm(dis_grads, constants['GRAD_NORM_CLIP'])
                dis_grads_and_vars = list(zip(dis_grads, self.ap_network.var_list))
                '''

                grads_and_vars = grads_and_vars + pred_grads_and_vars

            # update global step by batch size
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            # TODO: make optimizer global shared, if needed
            print("Optimizer: ADAM with lr: %f" % (constants['LEARNING_RATE']))
            print("Input observation shape: ",env.observation_space.shape)
            opt = tf.train.AdamOptimizer(constants['LEARNING_RATE'])
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            # copy weights from the parameter server to the local model
            sync_var_list = [v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]
            if self.unsup:
                sync_var_list += [v1.assign(v2) for v1, v2 in zip(predictor.var_list, self.ap_network.var_list)]
            self.sync = tf.group(*sync_var_list)

            # initialize extras
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        # get top rollout from queue (FIFO)
        rollout = self.runner.queue.get(timeout=1000.0)
        while not rollout.terminal:
            try:
                # Now, get remaining *available* rollouts from queue and append them into
                # the same one above. If queue.Queue(5): len=5 and everything is
                # superfast (not usually the case), then all 5 will be returned and
                # exception is raised. In such a case, effective batch_size would become
                # constants['ROLLOUT_MAXLEN'] * queue_maxlen(5). But it is almost never the
                # case, i.e., collecting  a rollout of length=ROLLOUT_MAXLEN takes more time
                # than get(). So, there are no more available rollouts in queue usually and
                # exception gets always raised. Hence, one should keep queue_maxlen = 1 ideally.
                # Also note that the next rollout generation gets invoked automatically because
                # its a thread which is always running using 'yield' at end of generation process.
                # To conclude, effective batch_size = constants['ROLLOUT_MAXLEN']
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        Process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        #print('a3c.process()')
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=constants['GAMMA'], lambda_=constants['LAMBDA'], clip=self.envWrap)

        #should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        should_compute_summary = self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        global nowDistance
        global lastDistance
        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
            self.tfNowDistance: nowDistance,
            self.tfLastDistance: lastDistance,
            self.tfGradientDistance: nowDistance - lastDistance
        }
        if self.unsup:
            feed_dict[self.local_network.x] = batch.si[:-1]
            feed_dict[self.local_ap_network.s1] = batch.si[:-1]
            feed_dict[self.local_ap_network.s2] = batch.si[1:]
            feed_dict[self.local_ap_network.asample] = batch.a

        # training
        fetched = sess.run(fetches, feed_dict=feed_dict)

        # testing
        #fetched = sess.run([self.global_step],feed_dict=feed_dict )

        if batch.terminal:
            print("Global Step Counter: %d"%fetched[-1])

            global logDirCounter
            if fetched[-1] >= logDirCounter and self.task == 0:
                # copy subdirectory example
                fromDirectory = "./tmp/ac4_tiles_1_3"
                toDirectory = "./model/1-3/fine_tuned/tile/ac4_y_info/40/" + str(self.task) + "_" + str(logDirCounter) + ".bk/"
                logDirCounter = logDirCounter + 500000

                copy_tree(fromDirectory, toDirectory)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

