import tensorflow as tf
from tensorflow import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random


class Agent():
    def __init__(self,epsilon=0.3, epsilon_min=0.01, gamma=0.99, batch_size = 20):
        self.M_size = 11
        self.goal = 10
        self.M = self._create_M()
        self.state = 0
        self.action = None


        self.state_size = self.get_state_size()
        self.action_size = self.get_action_size()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = 5
        self.target_count_update = 0

        self.memory = deque(maxlen=500)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())


    def reset(self):
        self.state = 0
        self.action = None

    def get_state_size(self):
        return self.M_size

    def get_state(self):
        return self.state

    def get_action_size(self):
        return self.M_size
    def _create_M(self):
        edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
                 (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
                 (8, 9), (7, 8), (1, 7), (3, 9)]
        M = np.ones(shape=(self.M_size, self.M_size))
        M *= -1
        for point in edges:
            if point[1] == self.goal:
                M[point] = 100
            else:
                M[point] = 0
            if point[0] == self.goal:
                M[point[::-1]] = 100
            else:
                M[point[::-1]] = 0
        M[self.goal, self.goal] = 100
        return M
    def available_actions(self, state):
        current_state_row = self.M[state,]
        available_action = np.where(current_state_row >= 0)[0]
        return available_action

    def unavailable_actions(self, state):
        current_state_row = self.M[state,]
        available_action = np.where(current_state_row < 0)[0]
        return available_action
    def next_random_action(self , available_actions):
        next_action = int(np.random.choice(available_actions, 1))
        return next_action

    def _get_reward(self, action):
         return self.M[self.state][action]

    def step(self, action):
        reward = self._get_reward(action)
        self._take_action( action)
        done = self.state == self.goal
        return  reward, self.state, done

    def _take_action(self, action):
        self.state = action
    def _build_model(self):
        model = Sequential()
        model.add(Dense(3, input_dim = 1, activation = keras.activations.relu))
        model.add(Dense(5, activation = keras.activations.relu))
        model.add(Dense(self.action_size, activation = keras.activations.relu))
        model.compile(loss = 'mse',
                        optimizer = Adam(), metrics  =[])
        return model

    def act(self, state):
        if (np.random.random() <= self.epsilon):
            avai_act = self.available_actions(state)
            return self.next_random_action(avai_act)
        return self._get_max_ind_Q( self.model.predict([[state]])[0], state)

    def _save_model(self):
        self.target_model.save('nn.h5')

    def _load_model(self):
        self.target_model = tf.keras.models.load_model('nn.h5')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _get_max_ind_Q(self, arr_y, state):
        un_act = self.unavailable_actions(state)
        setval = np.zeros(shape=(1, len(un_act)))
        np.put(arr_y, un_act, setval)
        return np.argmax(arr_y)

    def _get_max_Q(self, arr_y, state):
        un_act = self.unavailable_actions(state)
        setval = np.zeros(shape=(1, len(un_act)))
        np.put(arr_y, un_act, setval)
        return np.max(arr_y)

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory,  batch_size)
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(np.array([[state]]))[0]
            if done :
                y_target[action] = reward
            else :
                q_max = self._get_max_Q(self.target_model.predict(np.array([[next_state]]))[0], next_state)
                y_target[action] = reward + self.gamma *  q_max
            x_batch.append([state])
            y_batch.append(y_target)
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size = 10, verbose = 0)

        if( self.target_count_update < self.target_update) :
            self.target_count_update += 1
        else :
            self.target_count_update = 0
            self.target_model.set_weights(self.model.get_weights())

    def train(self, episode):
        self.reset()
        for i in range(100) :
            s = self.get_state()
            a = self.act(s)
            reward, next_state, done = self.step(a)
            self.remember(s, a, reward, next_state, done)
        for i in range (episode + 1):
            print("epoch :", i+1)
            done = False
            self.reset()
            while( not done  ) :
                s = self.get_state()
                a = self.act(s)
                reward, next_state, done = self.step(a)
                self.remember(s, a, reward,next_state, done)
                self.replay(100)
        self._save_model()



    def test(self):
        done = False
        self._load_model()
        self.reset()
        s = self.get_state()
        print(s)
        while ( not done) :
            a = self._get_max_ind_Q( self.target_model.predict([[s]])[0], s)
            reward, next_state, done = self.step(a)
            s = next_state
            print(s)

agent = Agent()
agent.train(10)
agent.test()
