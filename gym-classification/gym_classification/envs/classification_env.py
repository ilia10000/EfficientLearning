# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:40:29 2018

@author: Ilia
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from keras.datasets import mnist
from gym.spaces import Discrete, Tuple, Box
from matplotlib import pyplot as plt
 
class ClassificationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, partitions=4):
        self.partitions = partitions
        self.data = mnist.load_data()
        self.observation_space = Box(low=0, high=255, shape=(28,28,1))
        self.action_space = Discrete(10)
        self.correct=[]
        self.viewed=[]
    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size
    
        If arr is a 2D array, the returned array looks like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape[:2]
        return (arr.reshape(h//nrows, nrows, -1, ncols,1)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols, 1))
    def unblockshaped(self, arr, shape):
        """
        Return an array of shape (h, w) where
        h * w = arr.size
    
        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        h, w = shape[:2]
        n, nrows, ncols = arr.shape[:-1]
        return (arr.reshape(h//nrows, -1, nrows, ncols,1)
                   .swapaxes(1,2)
                   .reshape(h, w, 1))
    def getState(self):
        blocks = self.blockshaped(self.target_image, 28//int(self.partitions**0.5), 28//int(self.partitions**0.5))
        for i in range(self.partitions):
            if i not in self.revealed:
                blocks[i] = np.zeros(np.shape(blocks[i]))
        return self.unblockshaped(blocks, np.shape(self.target_image))
    def step(self, action):
        episode_over = False
        print ("action: {0} true: {1}".format(action,self.target_class))
        if action == 10:
            print ("requesting...")
            self.counter+=0
            if len(self.revealed)<self.partitions:
                reward=-0.1*len(self.revealed)
                #self.guesses=0
                self.revealed.append(np.random.choice(list(set(range(self.partitions))-set(self.revealed))))
            if len(self.revealed)>=self.partitions:
                self.action_space.n=10
            #else:
                #continue
                #reward=-2
                #self.correct.append(0)
                #self.viewed.append(len(self.revealed))
                #episode_over=True
        elif action == self.target_class:
            self.correct.append(1)
            reward = 1#self.partitions#*(3-self.guesses)
            #reward= (self.partitions-len(self.revealed)+1)*(3-self.guesses)
            self.counter+=1
            self.guesses=0
            #self.target_image, self.target_class = self.random_image()
            #self.revealed = [np.random.choice(range(self.partitions))]
            if self.counter>0:
                self.viewed.append(len(self.revealed))
                episode_over = True
        else:
            self.guesses+=1
            self.correct.append(-1)
            reward = -1#-self.partitions
            if self.guesses>0:
                self.counter+=1
                self.guesses=0
                #self.target_image, self.target_class = self.random_image()
                #self.revealed = [np.random.choice(range(self.partitions))]
                if self.counter>0:
                    self.viewed.append(len(self.revealed))
                    episode_over = True
        ob = self.getState()
        return ob, reward, episode_over, {}
       
    def random_image(self):
        n = np.random.randint(len(self.data[0][0]))
        im = self.data[0][0][n]
        lab = self.data[0][1][n]
        return im, lab
    def reset(self):
        if len(self.correct)>1:
            #print(self.correct[-1])
            print("Correct: {0}, Wrong {1}, Skipped: {2}, Viewed: {3}".format(self.correct[-500:].count(1), self.correct[-500:].count(-1), self.correct[-500:].count(0), sum(self.viewed[-500:])))
        self.counter=0
        self.target_image, self.target_class = self.random_image()
        self.revealed = list(np.random.choice(range(self.partitions),2,replace=False))
        self.guesses=0
        self.action_space.n=11
        return self.getState()
    def render(self, mode='human', close=False):
        #print(np.shape(self.getState()))
        plt.imshow(np.reshape(self.getState(),(28,28)))
    
if __name__ == '__main__':
    from gym import wrappers, logger
    import os
    os.chdir("D:/Workspace/EfficientLearning/gym-classification/")
    import gym_classification

    from six.moves import cPickle as pickle
    import json, sys, os
    from os import path
    MAX_EPISODES = 30000
    CONSECUTIVE_EPISODES = 500   # Number of trials' rewards to average for solving
    IS_RECORDING = False 

    # Fine-tuning the EPSILON_DECAY parameters will lead to better results for 
    # some environments and worse for others. As this code is a go at a 
    # general player, it is neater to treat it as a global constant 
    env = gym.make("classification-v0")
   
     #%%
    EPSILON_DECAY = 0.999
    # Train the agent, and snapshot each stage
    if IS_RECORDING:
        env.monitor.start('results-' + "classification-v0", force=True)
    
    # Create a gym instance and initialize the agent
    agent = Agent(env.action_space.n,
                  env.observation_space.shape)
    reward, done, = 0.0, False
        
    
    # Start the game loop
    for episode in range(1, MAX_EPISODES + 1):
        obs, done = env.reset(), False
        reward=0.0
        action = agent.act(obs, reward, done, episode)
        
        while not done:
            
            # Un-comment to show the game on screen 
            #env.render()
            
            # Decide next action and feed the decision to the environment         
            obs, reward, done, _ = env.step(action) 
            #agent._N_ACTIONS_temp=11-env.action_space.n
            #print(agent._N_ACTIONS_temp)
            action = agent.act(obs, reward, done, episode)        
            
    # Save info and shut activities
    correct=env.correct
    viewed=env.viewed
    env.close()
    agent.close()
    if IS_RECORDING:
        env.monitor.close()
#%%
class Agent():

    def __init__(self, n_actions, obs_space):

        # Initialization of useful variables and constants
        self._N_ACTIONS = n_actions
        self._N_ACTIONS_temp = 0
        self._nn = FeedForwardNeuralNetwork(n_actions, obs_space)
        self._STATE_FRAMES = self._nn._STATE_FRAMES
        self._close = lambda : self._nn.close()
        
        # Hyperparameters of the training
        self._DISCOUNT_FACTOR = 0.99    # discount of future rewards
        self._TRAINING_PER_STAGE = 1
        self._MINIBATCH_SIZE = 1     
        self._REPLAY_MEMORY = 60000   

        # Exploration/exploitations parameters
        self._epsilon = 1.
        self._EPSILON_DECAY = EPSILON_DECAY
        self._EPISODES_PURE_EXPLORATION = 500
        self._MIN_EPSILON = 0.02

        # Define useful variables
        self._total_reward, self._list_rewards = 0.0, []
        self._last_action = np.zeros(self._N_ACTIONS)
        self._previous_observations, self._last_state = [], None
        self._LAST_STATE_IDX = 0
        self._ACTION_IDX = 1 
        self._REWARD_IDX = 2
        self._CURR_STATE_IDX = 3
        self._TERMINAL_IDX = 4
        
        
    def act(self, obs, reward, done, episode):
        
        self._total_reward += reward
        
        if done:
            self._list_rewards.append(self._total_reward)
            average = np.mean(self._list_rewards[-CONSECUTIVE_EPISODES:])
            print ('Episode', episode, 'Reward', self._total_reward, 
                   'Average Reward', round(average, 2))           
            self._total_reward = 0.0
            self._epsilon = max(self._epsilon * self._EPSILON_DECAY,
                                self._MIN_EPSILON) 
                        
        # Compress the input image into a pre-set format
        compressedImage = self._nn._compressImage(obs)                                    

        # If the CNN has no last state, fill it by using the current state,
        # choose a random action, and return the action to the game
        if self._last_state is None:
            self._last_state = compressedImage.copy()
            for _ in range(self._STATE_FRAMES - 1):
                self._last_state = np.append(self._last_state, 
                                             compressedImage,
                                             axis=3)
            value_per_action = self._nn.predict(self._last_state)
            chosen_action_index = np.argmax(value_per_action)  
            self._last_action = np.zeros(self._N_ACTIONS)
            self._last_action[chosen_action_index] = 1
            return (chosen_action_index)
                
        # Update the current state 
        # current_state is made by (STATE_FRAMES) reduced images
        current_state = np.append(compressedImage, 
                                  self._last_state[:,:,:,:-3], 
                                  axis=3)
         
        # Store the last transition
        new_observation = [0 for _ in range(5)]
        new_observation[self._LAST_STATE_IDX] = self._last_state.copy()
        new_observation[self._ACTION_IDX] = self._last_action.copy()
        new_observation[self._REWARD_IDX] = reward
        new_observation[self._CURR_STATE_IDX] = current_state.copy()
        new_observation[self._TERMINAL_IDX] = done
        self._previous_observations.append(new_observation)
        self._last_state = current_state.copy()
            
        # If the memory is full, pop the oldest stored transition
        while len(self._previous_observations) >= self._REPLAY_MEMORY:
            self._previous_observations.pop(0)
        
        # Only train and decide after enough episodes of random play
        if episode > self._EPISODES_PURE_EXPLORATION:
  
            for _ in range(self._TRAINING_PER_STAGE):
                self._train()       
                
            # Chose the next action with an epsilon-greedy approach
            if np.random.random() > self._epsilon:
                value_per_action = self._nn.predict(self._last_state)
                chosen_action_index = np.argmax(value_per_action[:,:self._N_ACTIONS-self._N_ACTIONS_temp])  
            else:
                chosen_action_index = np.random.randint(0, self._N_ACTIONS-self._N_ACTIONS_temp)
        
        else:
            chosen_action_index = np.random.randint(0, self._N_ACTIONS-self._N_ACTIONS_temp)
    
        next_action_vector = np.zeros([self._N_ACTIONS])
        next_action_vector[chosen_action_index] = 1
        self._last_action = next_action_vector
          
        return (chosen_action_index)
        

    def _train(self):
        
        # Sample a mini_batch to train on
        permutations = np.random.permutation(
            len(self._previous_observations))[:self._MINIBATCH_SIZE] 
        previous_states = np.concatenate(
            [self._previous_observations[i][self._LAST_STATE_IDX]
            for i in permutations], 
            axis=0)
        actions = np.concatenate(
            [[self._previous_observations[i][self._ACTION_IDX]] 
            for i in permutations], 
            axis=0)
        rewards = np.array(
            [self._previous_observations[i][self._REWARD_IDX] 
            for i in permutations]).astype('float')
        current_states = np.concatenate(
            [self._previous_observations[i][self._CURR_STATE_IDX] 
            for i in permutations], 
            axis=0)
        done = np.array(
            [self._previous_observations[i][self._TERMINAL_IDX] 
            for i in permutations]).astype('bool')

        # Calculates the value of the current_states (per action)
        valueCurrentstates = self._nn.predict(current_states)
        
        # Calculate the empirical target value for the previous_states
        valuePreviousstates = rewards.copy()
        valuePreviousstates += ((1. - done) * 
                                self._DISCOUNT_FACTOR * 
                                valueCurrentstates.max(axis=1))

        # Run a training step
        self._nn.fit(previous_states,
                          actions, 
                          valuePreviousstates)


"""
Plain Feed Forward Neural Network
The chosen activation function is the Leaky ReLU function
"""
class FeedForwardNeuralNetwork:
    
    def __init__(self, n_actions, obs_space):

        # NN variables
        self._generateNetwork(n_actions, obs_space)
        self._previous_observations = []
        self._PARAMETERS_FILE_PATH = 'Parameters_CNN.ckpt'


    def _generateNetwork(self, n_actions, obs_space):
        """
        The network is implemented in TensorFlow
        Change this method if you wish to use a different library
        """
        
        import tensorflow as tf   
        self._ALPHA = 1e-3              # learning rate    
        RESIZED_SCREEN = 84
        self._STATE_FRAMES = 1         # states/images used for taking a decision
        
        # Graph for compressing the input image 
        x, y, z = obs_space
        self._image_input_layer = tf.placeholder("float", 
            [None, x, y, z])
        image_step_size_x = int(np.ceil(float(x / RESIZED_SCREEN)))
        image_step_size_y = int(np.ceil(float(y / RESIZED_SCREEN)))
        extra_pad_x = RESIZED_SCREEN - int(x / image_step_size_x)
        extra_pad_y = RESIZED_SCREEN - int(y / image_step_size_y)
        self._image_output_layer = tf.nn.max_pool(
                self._image_input_layer, 
                ksize=[1, image_step_size_x, image_step_size_y, 1],
                strides=[1, image_step_size_x, image_step_size_y, 1], 
                padding="VALID")                                         
        
        # Function for compressing (and reshaping) the image
        self._compressImage = lambda obs : np.pad(
            self._session.run(
                self._image_output_layer, 
                feed_dict={self._image_input_layer: np.array([obs])})/255.0,    
            ((0,0), (0,extra_pad_x), (0,extra_pad_y), (0,0)),
            mode='constant')   

        CONVOLUTION_FILTER_VECTOR = [6, 6, 4]
        CONVOLUTION_STRIDE_VECTOR = [3, 3, 2]
        CONVOLUTION_KERNEL_VECTOR = [16, 16, 36]
        CONVOLUTION_INPUT_VECTOR = ([z * self._STATE_FRAMES] + 
                                    CONVOLUTION_KERNEL_VECTOR[:-1])
        FEED_FWD_VECTOR = [(3**2) * CONVOLUTION_KERNEL_VECTOR[-1], 64, 
                           n_actions]      
        
        # The chosen activation function is the Leaky ReLU function
        self._activation = lambda x : tf.maximum(0.01*x, x)

            
        # Initialization parameters
        INITIALIZATION_STDDEV = 0.1
        INITIALIZATION_MEAN = 0.00
        INITIALIZATION_BIAS = -0.001

        # Convolutional layers
        self._input_layer = tf.placeholder("float", 
                                           [None, 
                                            RESIZED_SCREEN, 
                                            RESIZED_SCREEN, 
                                            z * self._STATE_FRAMES])
        self._convolutional_weights = []
        self._convolutional_bias = []
        self._hidden_convolutional_layer = [self._input_layer]

        for i in range(len(CONVOLUTION_FILTER_VECTOR)):
            self._convolutional_weights.append(tf.Variable(tf.truncated_normal(
                [CONVOLUTION_FILTER_VECTOR[i], 
                 CONVOLUTION_FILTER_VECTOR[i], 
                 CONVOLUTION_INPUT_VECTOR[i], 
                 CONVOLUTION_KERNEL_VECTOR[i]], 
                mean=INITIALIZATION_MEAN, 
                stddev=INITIALIZATION_STDDEV)))
            self._convolutional_bias.append(tf.Variable(tf.constant(
                INITIALIZATION_BIAS, 
                shape=[CONVOLUTION_KERNEL_VECTOR[i]])))
            self._hidden_convolutional_layer.append(
                self._activation(tf.nn.conv2d(
                                    self._hidden_convolutional_layer[i], 
                                    self._convolutional_weights[i], 
                                    strides=[1, 
                                             CONVOLUTION_STRIDE_VECTOR[i],
                                             CONVOLUTION_STRIDE_VECTOR[i], 
                                             1], 
                                    padding="VALID") 
                                + self._convolutional_bias[i]))
                                
        # Feed forward layers
        self._hidden_activation_layer = [tf.reshape(
            self._hidden_convolutional_layer[-1], 
            [-1, FEED_FWD_VECTOR[0]])]
        self._feed_forward_weights = []
        self._feed_forward_bias = []

        for i in range(len(FEED_FWD_VECTOR) - 2):
            self._feed_forward_weights.append(tf.Variable(tf.truncated_normal(
                [FEED_FWD_VECTOR[i], 
                 FEED_FWD_VECTOR[i+1]], 
                mean=INITIALIZATION_MEAN, 
                stddev=INITIALIZATION_STDDEV)))
            self._feed_forward_bias.append(tf.Variable(tf.constant(
                INITIALIZATION_BIAS, shape=[FEED_FWD_VECTOR[i+1]])))
            self._hidden_activation_layer.append(self._activation(
                    tf.matmul(self._hidden_activation_layer[i], 
                              self._feed_forward_weights[i]) 
                    + self._feed_forward_bias[i])
                    )
                    
        # The calculation of the state-action value function does not 
        # require the neurons' activation function
        self._feed_forward_weights.append(tf.Variable(tf.truncated_normal(
            [FEED_FWD_VECTOR[-2], 
             FEED_FWD_VECTOR[-1]], 
            mean=INITIALIZATION_MEAN, 
            stddev=INITIALIZATION_STDDEV)))
        self._feed_forward_bias.append(tf.Variable(tf.constant(
            INITIALIZATION_BIAS, 
            shape=[FEED_FWD_VECTOR[-1]])))
        self._state_value_layer = (tf.matmul(self._hidden_activation_layer[-1], 
                                             self._feed_forward_weights[-1]) 
                                    + self._feed_forward_bias[-1])

        # Define the logic of the optimization
        self._action = tf.placeholder("float", [None, n_actions])
        self._target = tf.placeholder("float", [None])
        self._action_value_vector = tf.reduce_sum(tf.multiply(
            self._state_value_layer, self._action), reduction_indices=1)
        self._cost = tf.reduce_sum(tf.square(
            self._target - self._action_value_vector))
        self._alpha = tf.placeholder('float')
        self._train_operation = tf.train.AdamOptimizer(
            self._alpha).minimize(self._cost)
        self._session = tf.Session()

        operation_intizializer = tf.initialize_all_variables()
        self._saver = tf.train.Saver()

        try:
            self._saver.restore(self._session, self._PARAMETERS_FILE_PATH)
            print ('Calibrated parameters SUCCESSFULLY LOADED.',
                   flush=True)
        except:
            self._session.run(operation_intizializer)
            print ('It was not possible to load calibrated parameters.',
                   flush=True)
   
        # Definition of feed_forward and optimization functions
        self._feedFwd = lambda state : self._session.run(
                            self._state_value_layer, 
                            feed_dict={self._input_layer: state})
                            
        self._backProp = lambda valueStates, actions, valueTarget : (
            self._session.run(self._train_operation, 
            feed_dict={self._input_layer: valueStates,
                       self._action: actions,
                       self._target: valueTarget,
                       self._alpha : self._ALPHA}))
                                         
    def close(self):

        # If training, save the RAM memory to file
        if self._is_training:
            self._saver.save(self._session, self._PARAMETERS_FILE_PATH)
            
        # Close the session and clear TensorFlow's graphs             
        from tensorflow.python.framework import ops
        ops.reset_default_graph() 
        self._session.close()
                                         
    def predict(self, state):    
        return(self._feedFwd(state))
       
    def fit(self, valueStates, actions, valueTarget):                      
        self._backProp(valueStates, actions, valueTarget)