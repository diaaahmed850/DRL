
EPISODES = 15000
starting_episode = 1
updateTargetNetwork = 1000
currentIteration = 0
stop_flow = False


import random, pygame, signal, time
import os,sys
sys.path.append('..')
#sys.path.append(os.path.abspath(os.path.join('..', 'Environments')))

import random, pygame, signal, time
from Environments.ple import PLE
from Environments.ple.games.citycopter import citycopter

from pygame.constants import K_w, K_s
from model import DQNAgent
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras.layers import random_uniform_initializer


def resetEnv(ple_env):
    ple_env.reset_game()
    return getCurrentState(ple_env)

def getCurrentState(ple_env):
    state_dict = ple_env.getGameState()
    state = [state_dict[i] for i in state_dict]
    return np.reshape(state, [1, len(state)])

action_map = [K_w, K_s]
def actInEnv(ple_env, action_num):
    reward = ple_env.act(action_map[action_num])
    state = getCurrentState(ple_env)
    done = ple_env.game_over()
    action = action_map[action_num]
    return state, reward, done, action

def make_env(env_name):
    if(env_name=='citycopter'):
        return citycopter(512, 512)
    



def sigint_handler(signum, frame):
    global stop_flow
    print('Going to stop the flow after the current episode terminates...')
    stop_flow = True
    
# To capture Ctrl-C events and stop gracefully
# Source: https://pythonadventures.wordpress.com/2012/11/21/handle-ctrlc-in-your-script/
signal.signal(signal.SIGINT, sigint_handler)

# These dumps can be read by plot*.py and display the rewards/loss curve
f = open('saved/data_plots.txt', 'w')

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    game =make_env('citycopter') #citycopter(512, 512)
    p = PLE(game, fps=20, force_fps=True, display_screen=True)
    p.init()
    state_size = 7 # TODO: Don't hardcode
    action_size = 2 # Up and No-op
    agent = DQNAgent(state_size, action_size)
    #agent.load("/tmp/heli-dqn-6000.h5")
    done = False
    last_loss = 0

    for e in range(starting_episode, EPISODES):
        state = resetEnv(p)
        total_reward = 0.0
        done = False
        while True:
            currentIteration += 1
            action = agent.act(state)
            next_state, reward, done, _ = actInEnv(p, action)
            reward = reward if not done else -10
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.3}"
                      .format(e, EPISODES, total_reward, agent.epsilon))
                f.write("{},{},{}\n".format(e, total_reward, last_loss))
                break
            if len(agent.memory) > agent.batch_size*4:
                last_loss = agent.replay_batch_gpu_optimized()
                if currentIteration % updateTargetNetwork == 0:
                    agent.target_train()
            
        if stop_flow:
            break
        if e % 500 == 0:
            agent.save("saved/heli-dqn-{}.h5".format(e))
            print("Saved checkpoint!")
            # Decrease LR
            K.set_value(agent.model.optimizer.lr, agent.learning_rate/pow(1.1, e/500))
          
f.close()
agent.save("saved/heli-dqn-final.h5")