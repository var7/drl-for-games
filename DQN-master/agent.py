from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym
import gym_rle
from replay import Experience

import pickle

class QAgent(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir

        self.env = gym.make(config['game'])
        self.env.seed(101010)
        self.ale_lives = None

        self.replay_memory = Experience(
            memory_size=config['state_memory'],
            state_shape=config['state_shape'],
            dtype=config['state_dtype']
        )

        self.net = config['q'](
            batch_size=config['batch_size'],
            state_shape=config['state_shape']+[config['state_time']],
            num_actions=config['actions'],
            summaries=True,
            **config['q_params']
        )

        # Disable the target network if needed
        if not self.config['double_q']:
            self.net.t_batch = self.net.q_batch

        with tf.variable_scope('RL_summary'):
            self.episode = tf.Variable(0.,name='episode')
            self.training_reward = tf.Variable(0.,name='training_reward')
            self.validation_reward = tf.Variable(0.,name='validation_reward')
            self.epsilon = tf.Variable(1.,name="epsilon") ################################################
            self.steps = tf.Variable(0, name="steps") ####################################################
            
            tf.summary.scalar(name='training_reward',tensor=self.training_reward)
            tf.summary.scalar(name='validation_reward',tensor=self.validation_reward)
            tf.summary.scalar(name='epsilon',tensor=self.epsilon) ########################################
            tf.summary.scalar(name='steps',tensor=self.steps) ############################################

        # Create tensorflow variables
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.update_target_network()
        self.summaries = tf.summary.merge_all()
        if log_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

        self.eps=self.session.run(self.epsilon) #########################################################
        self.steps_taken=self.session.run(self.steps) ###################################################

    def update_target_network(self):
        """
        Update the parameters of the target network

        :return:
        """
        if self.config['double_q']:
            self.session.run(self.net.assign_op)

    def _update_training_reward(self,reward):
        """
        set the value of the training reward.
        This ensures it is stored and visualised on the tensorboard
        """
        self.session.run(self.training_reward.assign(reward))

    def _update_validation_reward(self,reward):
        """
        set the value of the validation reward.
        This ensures it is stored and visualised on the tensorboard
        """
        self.session.run(self.validation_reward.assign(reward))

    def get_training_state(self):
        """
        Get the last state
        :return:
        """
        return self.replay_memory.get_last_state(self.config['state_time'])

    def sample_action(self,state,epsilon):
        """
        Sample an action for the state according to the epsilon greedy strategy

        :param state:
        :param epsilon:
        :return:
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(0,self.config['actions'])
        else:
            return self.session.run(self.net.q, feed_dict={self.net.x: state[np.newaxis].astype(np.float32)})[0].argmax()

    def update_state(self, old_state, new_frame): #COMBINE OLD STATE AND NEW FRAME
        """

        :param old_state:
        :param new_frame:
        :return:
        """
        return np.concatenate([
            old_state[:, :, 1:],
            new_frame
        ], axis=2)

    def reset_to_zero_state(self):
        """
        Reset the state history to zeros and reset the environment to fill the first state
        :return:
        """
        return np.concatenate([
                    np.zeros(self.config['state_shape']+[self.config['state_time']-1]),
                    self.config['frame'](self.env.reset())
                ],axis=2)

    def set_agent(self): ###############################################################################
        self.eps=self.session.run(self.epsilon)
        self.steps_taken=self.session.run(self.steps)

    def update_epsilon_and_steps(self): #################################################################
        if self.steps_taken > self.config['step_startrl']:
            self.eps = max(self.config['eps_minval'],self.eps*self.config['step_eps_mul']-self.config['step_eps_min'])
        self.steps_taken += 1

    def get_steps(self): ###############################################################################
        return self.steps_taken

    def get_epsilon(self): #############################################################################
        return self.eps

    def _update_steps_and_epsilon(self): ###############################################################
        self.session.run(self.epsilon.assign(self.eps))
        self.session.run(self.steps.assign(self.steps_taken))
    
    def save_replay(self, log_dir):
        pickle.dump(self.replay_memory,open(log_dir+"/replay_memory.mem", "wb"))

    def load_replay(self, log_dir):
        self.replay_memory = pickle.load( open(log_dir+"/replay_memory.mem", "rb" ))

    def train_episode(self): #TRAIN FOR A COMPLETE EPISODE
        # Load the last state and add a reset
        state = self.update_state(
            self.get_training_state(), #GET LAST STATE FROM REPLAY MEMORY
            self.config['frame'](self.env.reset()) #ADDS A RESET STATE
        )

        #print(state) #PRINT THE STATE

        # Store the starting state in memory ONLY START STATE
        self.replay_memory.add(
            state = state[:,:,-1],
            action = np.random.randint(self.config['actions']),
            reward = 0.,
            done = False
        )

        # Use flags to signal the end of the episode and for pressing the start button
        done = False # FLAG FOR IS THE GAME DONE
        press_fire = True # FLAG FOR START GAME
        total_reward = 0 # RETURNS
        while not done: #WHILE NOT END OF EPISODE
            if press_fire: # START THE EPISODE
                press_fire = False # SO THAT IT DOES NOT RESET AGAIN AND AGAIN
                new_frame,reward,done, info = self.act(state,-1,True) # TAKE AN ACTION
                if 'ale.lives' in info:
                    self.ale_lives = info['ale.lives']
            else: #END IT?
                self.update_epsilon_and_steps()
                new_frame,reward,done, info = self.act(state,self.eps,True)
            state = self.update_state(state, new_frame) #CONCATENATE THE CURRENT AND THE NEXT STATE
            total_reward += reward #ADD THE REWARD

            if self.steps_taken > self.config['step_startrl']: #CURRENT STEP TO NEXT
                summaries,_ = self.train_batch()
                if self.steps_taken % self.config['tensorboard_interval'] == 0:
                    self.train_writer.add_summary(summaries, global_step=self.steps_taken)
                if self.steps_taken % self.config['double_q_freq'] == 0.:
                    print("double q swap") 
                    self.update_target_network() #DDQN
        return total_reward


    def validate_episode(self,epsilon,visualise=False):
        state = self.reset_to_zero_state()
        done = False
        press_fire = True
        total_reward = 0.
        while not done:
            if press_fire:
                press_fire = False
                new_frame, reward, done,_ = self.act(state=state, epsilon=-1, store=False)
            else:
                new_frame, reward, done,_ = self.act(state=state, epsilon=epsilon, store=False)
            state = self.update_state(old_state=state, new_frame=new_frame)
            total_reward += reward
            if visualise:
                self.env.render()
        return total_reward

    def act(self, state, epsilon, store=False): #PERFORM AN ACTION AND ADD TO REPLAY IF NECESSARY
        """
        Perform an action in the environment.

        If it is an atari game and there are lives, it will end the episode in the replay memory if a life is lost.
        This is important in games lik

        :param epsilon: the epsilon for the epsilon-greedy strategy. If epsilon is -1, the no-op will be used in the atari games.
        :param state: the state for which to compute the action
        :param store: if true, the state is added to the replay memory
        :return: the observed state (processed), the reward and whether the state is final
        """
        if epsilon == -1:
            action = 10 #NO OPERATION
        else:
            action = self.sample_action(state=state,epsilon=epsilon)
        raw_frame, reward, done, info = self.env.step(action) #TAKE THE ACTION IN THE ENVIRONMENT AND COLLECT THE FRAME, REWARD, IF THE GAME IS DONE and INFO
        #print raw_frame, reward, done, info #PRINT THE ABOVE THINGS
        
        # Clip rewards to -1,0,1
        reward = np.sign(reward) #WHY?

        # Preprocess the output state
        new_frame = self.config['frame'](raw_frame) #CROP THE NEW FRAME

        # If needed, store the last frame
        if store:
            # End episodes if lives are involved in the atari game.
            # This is important in e.g. breakout
            # If this is not included, dropping the ball is not penalised.
            # By marking the end of the reward propagation, the maximum reward is limited
            # This makes learning faster.
            store_done = done
            if self.ale_lives is not None and ('ale.lives' in info) and info['ale.lives']<self.ale_lives:
                store_done = True
                self.ale_lives = info['ale.lives']
            self.replay_memory.add(state[:,:,-1],action,reward,store_done)
        return new_frame, reward, done, info


    def train_batch(self): #WHERE THE TRAINING TAKES PLACE
        """
        Sample a batch of training samples from the replay memory.
        Compute the target Q values
        Perform one SGD update step

        :return: summaries, step
        summaries: the tensorflow summaries that can be put into a log.
        step, the global step from tensorflow. This represents the number of updates
        """

        # Sample experience
        xp_states, xp_actions, xp_rewards, xp_done, xp_next = self.replay_memory.sample_experience(
            self.config['batch_size'],
            self.config['state_time']
        )

        # Create a mask on which output to update
        q_mask = np.zeros((self.config['batch_size'], self.config['actions']))
        for idx, a in enumerate(xp_actions):
            q_mask[idx, a] = 1

        # Use the target network to value the next states
        next_actions, next_values = self.session.run(
            [self.net.q_batch,self.net.t_batch],
            feed_dict={self.net.x_batch: xp_next.astype(np.float32)}
        )

        # Combine the reward and the value of the next state into the q-targets
        q_next = np.array([
            next_values[idx,next_actions[idx].argmax()]
            for idx in range(self.config['batch_size'])
        ])
        q_next *= (1.-xp_done)
        q_targets = (xp_rewards + self.config['gamma']*q_next)

        # Perform the update
        feed = {
            self.net.x_batch: xp_states.astype(np.float32),
            self.net.q_targets: q_targets[:,np.newaxis]*q_mask,
            self.net.q_mask: q_mask
        }
        _, summaries, step = self.session.run([self.net._train_op, self.summaries, self.net.global_step], feed_dict=feed)


        return summaries, step