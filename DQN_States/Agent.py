
EPISODES = 15000
starting_episode = 1
updateTargetNetwork = 1000
currentIteration = 0
stop_flow = False


import random, pygame, signal, time
import os,sys
import datetime
sys.path.append('..')
#sys.path.append(os.path.abspath(os.path.join('..', 'Environments')))

import random, pygame, signal, time
from Environments.ple import PLE
from Environments.ple.games.citycopter import citycopter
from Environments.ple.games.catcher import Catcher

from pygame.constants import K_w, K_s
from .model import DQNAgent
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.layers import random_uniform_initializer





def sigint_handler(signum, frame):
    global stop_flow
    print('Going to stop the flow after the current episode terminates...')
    stop_flow = True
    
# To capture Ctrl-C events and stop gracefully
# Source: https://pythonadventures.wordpress.com/2012/11/21/handle-ctrlc-in-your-script/
signal.signal(signal.SIGINT, sigint_handler)

# These dumps can be read by plot*.py and display the rewards/loss curve
#f = open('saved/data_plots.txt', 'w')



class DQN_States:
    def __init__(self, env_name,train_flag=True,fps=20, force_fps=True, display_screen=True):
        #tf.keras.backend.clear_session()
        self.env_name=env_name
        self.game =self.make_env(env_name)
        self.env= PLE(self.game, fps=fps, force_fps=force_fps, display_screen=display_screen)
        self.action_list=self.env.getActionSet()
        self.action_size=len(self.action_list)
        self.state_size=self.env.getStateSize()
        self.agent=DQNAgent(self.state_size,self.action_size,train_flag)
        self.today=str(datetime.date.today())
        self.path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data/")
        filename =self.path+self.env_name+'/'+self.today+'/data_plots/data_plots.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.plotData_file=open(filename, 'w')
        self.env.init()
    def make_env(self,env_name):
        if(env_name=='citycopter'):
            return citycopter(512, 512)
        if(env_name=='catcher'):
            return Catcher(480,480)


    def resetEnv(self):
        self.env.reset_game()
        return self.getCurrentState()

    def getCurrentState(self):
        state_dict = self.env.getGameState()
        state = [state_dict[i] for i in state_dict]
        return np.reshape(state, [1, len(state)])

    def actInEnv(self, action_num):
        reward = self.env.act(self.action_list[action_num])
        state = self.getCurrentState()
        done = self.env.game_over()
        action = self.action_list[action_num]
        return state, reward, done, action
    def train(self):
        
        done = False
        last_loss = 0
        currentIteration = 0
        for e in range(starting_episode, EPISODES):
            state = self.resetEnv()
            total_reward = 0.0
            done = False
            while True:
                currentIteration += 1
                action = self.agent.act(state)
                next_state, reward, done, _ = self.actInEnv(action)
                reward = reward if not done else -10
                total_reward += reward
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.3}"
                        .format(e, EPISODES, total_reward, self.agent.epsilon))
                    self.plotData_file.write("{},{},{}\n".format(e, total_reward, last_loss))
                    break
                if len(self.agent.memory) > self.agent.batch_size*4:
                    last_loss = self.agent.replay_batch_gpu_optimized()
                    if currentIteration % updateTargetNetwork == 0:
                        self.agent.target_train()
                
            if stop_flow:
                break
            if e % 500 == 0:
                self.agent.save(self.path+self.env_name+'/'+self.today+"/dqn-states-{}.h5".format(e))
                print("Saved checkpoint!")
                # Decrease LR
                K.set_value(self.agent.model.optimizer.lr, self.agent.learning_rate/pow(1.1, e/500))
            
        self.plotData_file.close()
        self.agent.save(self.path+self.env_name+'/'+self.today+"/dqn-states-final.h5")
    
    def test(self):
        #self.agent.load('saved_data/'+self.env_name+'/'+self.today+"/heli-dqn-9000.h5")
        self.agent.load(self.path+self.env_name+'/'+self.today+"/heli-dqn-final.h5")
        done = False

        for e in range(starting_episode, EPISODES):
            state = self.resetEnv()
            total_reward = 0.0
            done = False
            while True:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.actInEnv(action)
                reward = reward if not done else -10
                total_reward += reward
                state = next_state
                if done:
                    print("episode: {} - score: {}"
                        .format(e, total_reward))
                    break
                
            if stop_flow:
                break




if __name__ == "__main__":
    x=DQN_States('citycopter',False)
    x.test()











