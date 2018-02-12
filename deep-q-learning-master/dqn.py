# -*- coding: utf-8 -*-
import random
import gym
import gym_rle
import numpy as np
from collections import deque
import keras

EPISODES = 1000

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

def transform_reward(reward):
    return np.sign(reward)

class DQNAgent:
    def __init__(self, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        MARIO_SHAPE = (128, 112, 1)

        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(MARIO_SHAPE, name='frames')
        actions_input = keras.layers.Input((self.action_size,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        print(normalized.shape)
        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        conv_1 = keras.layers.convolutional.Convolution2D(
            16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = keras.layers.convolutional.Convolution2D(
            32, 4, 4, subsample=(2, 2), activation='relu'
        )(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(self.action_size)(hidden)
        # Finally, we multiply the output by the mask!
        filtered_output = keras.layers.merge([output, actions_input], mode='mul')

        model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, np.ones(self.action_size))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, np.ones(self.action_size))[0]))
            target_f = self.model.predict(state, np.ones(self.action_size))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('SuperMarioAllStars-v0')
    env.render()
    state_size = env.observation_space.shape
    print(env.observation_space.shape)
    action_size = env.action_space.n
    print(state_size, action_size)
    agent = DQNAgent(action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        print(state.shape)
        # state = np.reshape(state, [1, state_size])
        state = downsample(to_grayscale(state))
        print(state.shape)
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            # next_state = np.reshape(next_state, [1, state_size])
            next_state = downsample(to_grayscale(next_state))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
