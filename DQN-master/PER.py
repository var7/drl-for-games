#! -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
#import random
import Sum_Tree
from configs import mario_config_PER

class BetaSchedule(object):
    def __init__(self,step):
        self.config = mario_config_PER
        self.batch_size = self.config['batch_size']

        self.beta_zero = self.config['beta_zero']
        self.learn_start = self.config['step_startrl']
        
        self.total_steps = 100#self.config['total_steps']
        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

    def get_beta(self, global_step):
        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        return beta, self.batch_size



class Experience(object):
    """ The class represents prioritized experience replay buffer.
    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.
    see https://arxiv.org/pdf/1511.05952.pdf .
    """
    
    def __init__(self, epsilon, step):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.config = mario_config_PER
        self.beta_sched = BetaSchedule(step)
        self._max_priority = 1.0

        self.index = 0
        self.count = 0
        self.isFull = False
        self.prioritized_replay_eps = epsilon

        self.memory_size = self.config['state_memory']
        
        self.tree = Sum_Tree.SumTree(self.memory_size)
        self.alpha = self.config['alpha']
        self.state_shape = self.config['state_shape']

    def get_last_state(self,state_length):
        state=np.zeros((4,84,84))
        sta = self.tree.data[self.index-state_length:self.index]
        for c,i in enumerate(sta):
            state[c][:][:]=i[1]
        return state.transpose(1,2,0)

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.count <= self.memory_size:
            self.count += 1
        if self.index % self.memory_size == 0:
            self.index = 1
            return self.index
        else:
            self.index += 1
            return self.index

    def add(self, state, action, reward, done , priority=None):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.count = min(self.memory_size,self.count+1)
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done

        data=[self.count, self.state, self.action, self.reward, self.done]

        if priority is None:
            priority = self._max_priority
        
        self.tree.add(data, priority**self.alpha)

        self.fix_index()
        

    def sample(self,  batch_size, state_length, global_step):
        beta, batch_size = self.beta_sched.get_beta(global_step)
        return self.select(beta, batch_size=batch_size, state_length=state_length)

    def select(self, beta, batch_size, state_length):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """
        
        if self.tree.filled_size() < batch_size:
            return None, None, None
        startstatedata=np.zeros((batch_size,state_length,self.state_shape[0],self.state_shape[1]))
        nextstatedata=np.zeros((batch_size,state_length,self.state_shape[0],self.state_shape[1]))
        actions=np.zeros(batch_size)
        rewards=np.zeros(batch_size)
        done = np.zeros(batch_size)


        weights = []
        priorities = []
        rand_vals = np.random.rand(batch_size)
        for s,r in enumerate(rand_vals):  #range(batch_size):
            data, priority, index = self.tree.find(r)
            if(index>self.memory_size-self.state_length):
                while(index<=self.memory_size-self.state_length):
                    data, priority, index = self.tree.find(np.random.rand())
            for i in range(state_length):
                startstatedata[s][i][:][:]=self.tree.get_tuple(index+i)
                if(i==state_length-1):
                    actions[s] = data[2]
                    rewards[s] = data[3]
                    done[s] = data[4]
                priorities.append(priority)
                weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
                indices.append(index)
                data = self.tree.find(r+1+i)[0]
                nextstatedata[s][:][:] = data[1]
                # out.append(data)

        startstatedata=startstatedata.transpose(0,2,3,1)
        nextstatedata=nextstatedata.transpose(0,2,3,1)

        self.update_priority(indices, priorities) # Revert priorities
        # weights /= max(weights) # Normalize for stability
        w = np.array(weights)
        w = np.divide(w,max(w))

        return startstatedata, actions, rewards, done, nextstatedata

    def update_priority(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, (p+self.prioritized_replay_eps)**self.alpha)
    
    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.update_priority(range(self.tree.filled_size()), priorities)

    def rebalance(self):
        pass
            
        
if __name__ == '__main__':
    e = Experience(0.1,500)
    e.add(0,np.ones((84,84))*0, 0, 0, False)
    e.add(1,np.ones((84,84))*1, 1, 1, False)
    e.add(2,np.ones((84,84))*2, 2, 2, False)
    e.add(3,np.ones((84,84))*3, 3, 3, False)
    e.add(4,np.ones((84,84))*4, 4, 4, False)
