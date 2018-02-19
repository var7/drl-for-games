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
    load_episode = 1640
    epsilon = 0.5 # The epsilon for the strategy

    # Build the graph on CPU to keep gpu for training....
    with tf.device('/cpu:0'):
        agent = QAgent(config=config, log_dir=None)

    # Restore the values....
    tf.train.Saver().restore(agent.session,'log/2018-02-18_13-33-27_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(load_episode))
    tot_r=[[],[],[],[],[]]
    q_res=[[],[],[],[],[]]
    c=0
    while c<5:
        print("\n\n\n")
        # Initialise the episode
        state = agent.reset_to_zero_state()
        done = False
        total_reward = 0.
        steps = -1
        # Prepare the visualisation
        # plots = init_figure(config['actions'])
        r=0
        while not done:
            
            steps += 1
            q = agent.session.run(agent.net.q,feed_dict={agent.net.x:state[np.newaxis].astype(np.float32)})
            # q_res[c].append(np.mean(q))
            # print(type(q))
            # print()
            # print(q)
            new_frame, reward, done,_ = agent.act(state=state, epsilon=epsilon, store=False)
            state = agent.update_state(old_state=state, new_frame=new_frame)
            total_reward += reward
            r+=reward
            if reward != 0:
                print(reward)
            if reward==-1:
                tot_r[c].append(r+1)
                r=0
            # elif reward
            # update_figure(plots, steps, q, reward, agent.env.render(mode='rgb_array'))
            # plt.draw()
            # plt.pause(0.001)
        c+=1
    # print(tot_r)
    # print()
    res=[]
    # print(tot_r.shape)
    max=0
    for i in range(5):
        # plt.plot(tot_r[i])
        if(len(tot_r[i])>max):
            max=len(tot_r[i])
        # plt.plot(q_res[i])
    for i in range(5):
        for j in range(max-len(tot_r[i])):
            tot_r.append(0)

    final_result=[]
    for j in range(max):
        sum=0
        for i in range(5):
            sum+=tot_r[i][j]
        final_result.append(sum/float(max))
        
    print(final_result)
    plt.plot(final_result)
    plt.show()
    input()