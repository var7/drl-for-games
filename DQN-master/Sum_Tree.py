from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import math



class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = int(math.ceil(math.log(max_size+1, 2)))+1
        self.tree_size = 2**self.tree_level-1
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        # print(contents)
        # input()
        index = self.cursor
        self.cursor = (self.cursor+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2**(self.tree_level-1)-1+index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2**(self.tree_level-1)-1+index
        diff = value-self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)/2)
            self.reconstruct(tindex, diff)
    
    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):

        if 2**(self.tree_level-1)-1 <= index:
            return self.data[index-(2**(self.tree_level-1)-1)], self.tree[index], index-(2**(self.tree_level-1)-1)

        left = self.tree[2*index+1]
        # print("value : ", value)
        # print("left : ", left)
        # input()
        if value <= left:
            return self._find(value,2*index+1)
        else:
            return self._find(value-left,2*(index+1))
        
    def print_tree(self):
        for k in range(1, self.tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size

    def get_tuple(self,index):
        return self.data[index]

if __name__ == '__main__':
    import numpy as np
    index=0
    s = SumTree(40)
    s.print_tree()
    s.add("One",1)
    index+=1
    s.add("Two",1)
    index+=1
    s.add("Three",1)
    index+=1
    s.add("Four",1)
    index+=1
    s.add("Five",1)
    index+=1
    s.add("Six",1)
    index+=1
    s.add("Seven",1)
    index+=1
    s.add("Eight",1)
    index+=1
    s.add("Nine",1)
    index+=1
    s.add("Ten",1)
    index+=1
    s.add("Eleven",1)
    index+=1
    s.add("Twelve",1)
    index+=1
    s.add("Thirteen",1)
    index+=1
    s.add("Fourteen",1)
    index+=1
    s.add("Fifteen",1)
    index+=1
    s.add("Sixteen",1)
    index+=1
    s.add("Seventeen",1)
    index+=1
    s.add("Eighteen",1)
    index+=1
    s.add("Nineteen",1)
    index+=1
    s.add("Twenty",2)
    index+=1
    s.print_tree()
    
    state_length=4
    rand_vals = np.random.rand(2)
    print(rand_vals)
    for t,r in enumerate(rand_vals):  #range(batch_size):
            print("--------------------------------------")
            data, priority, idx = s.find(r)
            input()
            idx=17
            print("Index : ", index)
            print("idx : ",idx)
            if(idx > (index-state_length)):
                print("HIT")
                while(idx>(index-state_length)):
                    data, priority, idx = s.find(np.random.rand())
            print("F : ", idx)
            print(s.get_tuple(idx))
    
    