from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()

import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import mario_config

load_episode=2610
config=mario_config


agent = QAgent(config=config, log_dir=None)

tf.train.Saver().restore(agent.session,'log/2018-02-19_02-33-07_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(load_episode))
agent.set_agent()
eps=agent.get_epsilon()
# print(eps)
scores = [agent.validate_episode(epsilon=0.3, visualise=True) for i in range(config['episodes_validate_runs'])] # WITH VALIDATION
print(scores)