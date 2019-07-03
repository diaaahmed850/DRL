
EPISODES = 15000
starting_episode = 1
updateTargetNetwork = 1000
currentIteration = 0
stop_flow = False
import os,sys
from os.path import dirname, abspath
Environment_dir = (dirname(dirname(abspath(__file__))))+'/Environments'
sys.path.insert(0,Environment_dir)
sys.path.insert(0,(dirname(dirname(abspath(__file__)))))
from arguments import get_args
args=get_args()
print(Environment_dir)
from ple_xteam import PLE
import gym
if args.train_type=='states':
    from wrappers.wrapper_states.xteam_wrapper_states import PLEEnv

import random, pygame, signal, time
import datetime
sys.path.append('..')
#sys.path.append(os.path.abspath(os.path.join('..', 'Environments')))

import random, pygame, signal, time
from Environments.ple_xteam import PLE
from Environments.ple_xteam.games.citycopter import citycopter
from Environments.ple_xteam.games.catcher import Catcher
#from Environments.ple_xteam.games.colorswitch import colorswitch

from pygame.constants import K_w, K_s
from .model import DQNAgent
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.layers import random_uniform_initializer


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from shutil import copyfile
import imageio
from skimage.transform import resize


def sigint_handler(signum, frame):
    global stop_flow
    print('Going to stop the flow after the current episode terminates...')
    stop_flow = True
    
# To capture Ctrl-C events and stop gracefully
# Source: https://pythonadventures.wordpress.com/2012/11/21/handle-ctrlc-in-your-script/


# These dumps can be read by plot*.py and display the rewards/loss curve
#f = open('saved/data_plots.txt', 'w')



class DQN_States:
    def __init__(self, env_name,train_flag=True,fps=20, force_fps=True, display_screen=True,folder=str(datetime.date.today())):
        #tf.keras.backend.clear_session()
        print(env_name)
        self.env_name=env_name
        self.env =gym.make(self.env_name)
        #self.env= PLE(self.game, fps=fps, force_fps=force_fps, display_screen=display_screen)
        #self.action_list=self.env.getActionSet()
        self.action_size=self.env.action_space.n#len(self.action_list)
        self.state_size=self.env.observation_space.shape[0]#self.env.getStateSize()
        self.agent=DQNAgent(self.state_size,self.action_size,train_flag)
        self.today=folder
        self.path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data/")
        filename =self.path+self.env_name+'/'+self.today+'/data_plots/data_plots.txt'
        signal.signal(signal.SIGINT, sigint_handler)
        if train_flag == True:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.plotData_file=open(filename, 'w')
        else:
            self.plotData_file=open(filename, 'r')
        #self.env.init()
    def make_env(self,env_name):
        """
        if(env_name=='citycopter-v0'):
            return citycopter(512, 512)
        if(env_name=='Catcher-v0'):
            return Catcher(480,480)
        if(env_name=='colorswitch'):
            return colorswitch(500,700)
        """
        env=gym.make(env_name)
        print(env.observation_space.shape[0])
        print("asfasfasfsafasassfa")
        return env



    def resetEnv(self):
        """
        self.env.reset_game()
        return self.getCurrentState()
        """
        state=self.env.reset()
        return np.reshape(state, [1, len(state)])

    def getCurrentState(self):
        state_dict = self.env.getGameState()
        state = [state_dict[i] for i in state_dict]
        return np.reshape(state, [1, len(state)])

    def actInEnv(self, action_num):
        """
        reward = self.env.act(self.action_list[action_num])
        state = self.getCurrentState()
        done = self.env.game_over()
        action = self.action_list[action_num]
        """
        state, reward, done,info=self.env.step(action_num)
        state=np.reshape(state, [1, len(state)])
        return state, reward, done,info
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
                #reward = reward if not done else -10

                total_reward += reward
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                self.env.render()
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
        self.agent.load(self.path+self.env_name+'/'+self.today+"/dqn-states-final.h5")
        done = False
        
        for e in range(starting_episode, EPISODES):
            state = self.resetEnv()
            total_reward = 0.0
            done = False
            frames=[]
            if args.save_fig:
                frames.append(self.env.env._get_image())
            while True:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.actInEnv(action)
                if args.save_fig:
                    frames.append(self.env.env._get_image())
                reward = reward if not done else -10
                total_reward += reward
                state = next_state
                self.env.render()
                if done:
                    print("episode: {} - score: {}"
                        .format(e, total_reward))
                    if args.save_fig:
                        self.generate_gif(e,frames,total_reward)
                    break
                
            if stop_flow:
                break
    
    def generate_gif(self,num, frames_for_gif,score):
        """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
        """
        for idx, frame_idx in enumerate(frames_for_gif): 
            frames_for_gif[idx] = resize(frame_idx, (480, 480, 3), 
                                        preserve_range=True, order=0).astype(np.uint8)
            try:
                os.mkdir(self.path+self.env_name+'/'+self.today+'/gifs')
            except:
                pass
        imageio.mimsave(self.path+self.env_name+'/'+self.today+"/gifs/"+"0"+str(num)+"-"+args.env_name+"-"+args.train_type+"-"+str(score)+".mp4", 
                        frames_for_gif)#, duration=1/30)
    
    def plotLoss(self):
        FILENAME = self.path+self.env_name+'/'+self.today+"/data_plots/data_plots.txt"
        fig_path = self.path+self.env_name+'/'+self.today+"/data_plots/"
        style.use('fivethirtyeight')

        fig = plt.figure()
        ax2 = fig.add_subplot(1,1,1)
        graph_data = open(FILENAME,'r').read()
        lines = graph_data.split('\n')
        xs = []
        zs = []
        for line in lines:
            if len(line) > 1:
                x, y, z = line.split(',')
                xs.append(float(x))
                zs.append(float(z))
        ax2.clear()
        ax2.plot(xs, zs)
        plt.savefig(fig_path+'loss.png')


    def plotRewards(self):
        FILENAME = self.path+self.env_name+'/'+self.today+"/data_plots/data_plots.txt"
        fig_path = self.path+self.env_name+'/'+self.today+"/data_plots/"
        style.use('fivethirtyeight')

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        avg_over = 10
        graph_data = open(FILENAME,'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        ys_sum = 0.0
        j = 0
        for line in lines:
            if len(line) > 1:
                try:
                    x, y, z = line.split(',')
                    ys_sum += float(y)
                    j += 1
                    if j % avg_over == 0:
                        xs.append(float(x))
                        ys.append(ys_sum/float(avg_over))
                        ys_sum = 0.0

                except:
                    continue
        ax1.clear()
        ax1.plot(xs, ys)
        plt.savefig(fig_path+args.env_name+"-"+args.train_type+'-'+'rewards_plot.png')





if __name__ == "__main__":
    x=DQN_States('citycopter',False)
    x.test()











