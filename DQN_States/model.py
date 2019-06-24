from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
#from tensorflow.python.keras.backend import manual_variable_initialization
#manual_variable_initialization(True)
from collections import deque
import numpy as np
import os,random
class DQNAgent:
    def __init__(self, state_size, action_size,train_flag=True):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64
        self.memory = deque(maxlen=20000)
        self.gamma = 0.97    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.0004
        self.tau = .125
        self.train_on_TPU = False #Won't work if True
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.train=train_flag
        if self.train_on_TPU:
            self.cpu_model = self.target_model.sync_to_cpu()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size)) # default is linear activation
        if self.train_on_TPU:
            model = tf.contrib.tpu.keras_to_tpu_model(model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
            ))
            model.compile(loss='mean_squared_error',
                      optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate))
        else:
            model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        if(self.train):
            if np.random.rand() <= self.epsilon:
                return np.random.choice([0, 1], p=[0.25, 0.75]) #random.randrange(self.action_size)
    #         if self.train_on_TPU: # Sorry for this hack, Google
    #             state = state.repeat(self.batch_size, axis=0)
            act_values = self.model.predict(state) if not self.train_on_TPU else self.cpu_model.predict(state) 
            return np.argmax(act_values[0])  # returns action
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action


    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        
        history = self.model.fit(np.array(states), np.array(targets_f), batch_size=self.batch_size, epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def replay_batch_gpu_optimized(self):
        # Check the above reference implementation for correspondance
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        state = np.array(minibatch[:, 0].tolist()).squeeze()
        action = minibatch[:, 1]
        target = minibatch[:, 2]
        next_state = np.array(minibatch[:, 3].tolist()).squeeze()
        done = np.array(minibatch[:, 4], dtype=bool)
        Q_next_max = self.gamma * np.amax(self.target_model.predict(next_state, batch_size=self.batch_size), axis=1)
        target = target + (Q_next_max * np.invert(done))
        target_f = self.model.predict(state, batch_size=self.batch_size)
        target_f[range(self.batch_size), action.tolist()] = target
        
        history = self.model.fit(state, target_f, batch_size=self.batch_size, epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss
      
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        # Any better way than below? ;(
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        if self.train_on_TPU: # I hope this is not a costly operation
            self.cpu_model = self.target_model.sync_to_cpu()
    
    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)
        if self.train_on_TPU:
            self.cpu_model.load_weights(name)            

    def save(self, name):
        self.model.save_weights(name)