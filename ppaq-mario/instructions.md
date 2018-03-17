## Instructions to install and run ppaquette\_gym\_super_mario
This is for macOS 10.13.3

1. Create a new conda environment
```
conda create --name mario python=3.6
```  
2. Install openAI gym version 0.8.0
```
pip install gym==0.8.0
```  
3. Now clone gym-pull
```
git clone https://github.com/ppaquette/gym-super-mario
```
4. Head to folder ```gym-super-mario``` by
```
cd gym-super-mario
```
5. now run 
```
python setup.py install
```
6. check that everything is working by running in the python interpreter  


```
import gym
import ppaquette_gym_super_mario
```  
You might get an error saying ```fceux``` is not installed, you can install it by runnning ` brew install fceux`  

7. Once this works you can run `env = gym.make('ppaquette/SuperMarioBros-1-1-v0')`. ~~This will not work and throw up the error~~ it works! make sure to include the `ppaquette/` before the exact level that you want. 



## To run on DICE/GPU Cluster

