import pickle
import os


import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import mario_config

j = os.listdir("log/DQN/")

for directory in j:
	episodes = []
	files = os.listdir("log/DQN/"+directory)
	for file in files:
		if(file.endswith(".meta")):
			episodes.append(file)
	s_episodes = sorted(episodes)
	
	start = int(s_episodes[0].split("_")[1].split(".")[0])
	end = int(s_episodes[-1].split("_")[1].split(".")[0])+1
	config = mario_config
	config['state_memory']=1
	# print(start,end)
	load_episode = range(start, end,10)

	agent = QAgent(config=config, log_dir=None)
	scores=[]
	for episode in load_episode:
		l = []
		l.append(episode)
		try:
		    tf.train.Saver().restore(agent.session,'log/DQN/'+directory+'/episode_%d.ckpt'%(episode))
		except:
		    print(episode," not available")
		    pass    
		l.append(agent.session.run(agent.training_reward))
		scores.append(l)
	pickle.dump( scores, open( 'log/DQN/'+directory+"/training_reward.p", "wb" ))
		

	# avg_valid = pickle.load( open( 'log/DQN/'+directory+"/training_reward.p", "rb" ) )
	# print(len(avg_valid[1]))