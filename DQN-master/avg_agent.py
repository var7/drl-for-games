from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()

from plot_util import init_figure, update_figure

import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import mario_config



if __name__ == '__main__':
    config = mario_config
    config['state_memory']=1 # prevent allocating of a huge chunk of memory
    load_episode = 2610
    # epsilon = 0.1 # The epsilon for the strategy

    # Build the graph on CPU to keep gpu for training....
    with tf.device('/cpu:0'):
        agent = QAgent(config=config, log_dir=None)
    eps=[0.1,0.25,0.5]
    # Restore the values....
    tf.train.Saver().restore(agent.session,'log/2018-02-19_02-33-07_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(load_episode))
    # tot_r=[[],[],[],[],[]]
    tot_r=[]
    res=[]
    print(load_episode)
    for epsilon in eps:
        print("\n")
        c=0
        total_reward = 0.
        while c<100:
            
            # Initialise the episode
            state = agent.reset_to_zero_state()
            done = False
            
            steps = -1
            # Prepare the visualisation
            # plots = init_figure(config['actions'])
            r=0
            while not done:
                steps += 1
                q = agent.session.run(agent.net.q,feed_dict={agent.net.x:state[np.newaxis].astype(np.float32)})
                new_frame, reward, done,_ = agent.act(state=state, epsilon=epsilon, store=False)
                state = agent.update_state(old_state=state, new_frame=new_frame)
                total_reward += reward
                # r+=reward
                # if reward != 0:
                    # print(reward)
            c+=1
        print(total_reward)
        tot_r.append(total_reward/100)
        print(tot_r)