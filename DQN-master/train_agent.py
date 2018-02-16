from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from agent import QAgent
from configs import mario_config
from util import get_log_dir

if __name__ == '__main__':
    config = mario_config
    log_dir = get_log_dir('log', config['game']+'_'+str(config['double_q']))#CREATE A LOG DIRECTORY FOR THE GIVEN GAME
    agent = QAgent(config=config, log_dir=log_dir) #CREATE THE AGENT WITH THE GIVEN CONFIGURATION AND SPECIFY THE DIRECTORY FOR SAVING THE LOGS
    saver = tf.train.Saver() #HELPS TO SAVE AND RETRIEVE VARIABLES TO AND FROM CHECKPOINTS

    ###CHANGE WHEN RESTARTING FROM A CHECKPOINT###
    load_episode=0
    # saver.restore(agent.session,'log/2018-02-16_18-44-13_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(load_episode))
    # agent.set_agent()

    ###END CHANGE###

    for episode in range(load_episode,config['episodes']): #FOR EACH EPISODE
        steps=agent.get_steps()
        epsilon=agent.get_epsilon()
        print('\n\nepisode: %d, step: %d, eps: %.4f\n\n---------------------' % (episode, steps, epsilon))
        
        # Store the rewards...
        agent._update_training_reward(agent.train_episode())

        if episode % config['episodes_validate']==0:
            print('Validate....\n==============')
            scores = [agent.validate_episode(epsilon=0.05) for i in range(config['episodes_validate_runs'])] # WITHOUT VISUALISATION
            # scores = [agent.validate_episode(epsilon=0.05, visualise=True) for i in range(config['episodes_validate_runs'])] # WITH VALIDATION
            agent._update_validation_reward(np.mean(scores))
            print(scores)
        # Store every validation interval
        if episode % config['episodes_save_interval']==0:
            agent._update_steps_and_epsilon()
            saver.save(agent.session,'%s/episode_%d.ckpt'%(log_dir,episode))
