from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()

import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import mario_config

load_episode=21530
config=mario_config


agent = QAgent(config=config, log_dir=None)

tf.train.Saver().restore(agent.session,'log/episode_%d.ckpt'%(load_episode))
# tf.train.Saver().restore(agent.session,'log/DQN/2018-03-19_14-42-32_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(load_episode))
# tf.train.Saver().restore(agent.session,'log/2018-03-05_12-09-42_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(load_episode))
agent.set_agent()
eps=agent.get_epsilon()
print(agent.session.run(agent.training_reward))
print(eps)
print(agent.get_steps())
scores = [agent.validate_episode(epsilon=eps, visualise=True) for i in range(config['episodes_validate_runs'])] # WITH VALIDATION
print(scores)