import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import mario_config

import pickle

if __name__ == '__main__':
    config = mario_config
    config['state_memory']=1 # prevent allocating of a huge chunk of memory
    load_episode = range(15960, 18011,10)
    # load_episode = range(110,201,10)

    agent = QAgent(config=config, log_dir=None)

    # avg_valid=[]
    # valid=[]

    
    
    avg_valid = pickle.load( open( "Average_Valid.p", "rb" ) )
    print(len(avg_valid))
    valid = pickle.load( open( "Valid.p", "rb" ) )
    print(len(valid))

    for episode in load_episode:
        print(episode)
  
        try:
            tf.train.Saver().restore(agent.session,'log/DQN/2018-03-17_21-53-34_SuperMarioAllStarsDeterministic-v4_False/episode_%d.ckpt'%(episode))
        except:
            print(episode," not available")
            pass    
            
  
        c=0
        total_reward = 0.
        res = []
        while c<10:
            print(c)
            scores = agent.validate_episode(epsilon=0.05, visualise=False)
            res.append(scores)
            c+=1
            total_reward+=scores

        avg_valid.append(total_reward/10)
        valid.append(res)

    pickle.dump( avg_valid, open( "Average_Valid.p", "wb" ) )
    pickle.dump( valid, open( "Valid.p", "wb" ) )