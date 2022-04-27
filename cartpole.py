from collections import namedtuple
import os
import sys
from sqlalchemy import false
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
from collections import deque
import random
import torch.optim as optim
import math
import time


# Check if GPU is avalible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

# Defining classes and functions ---------------------------------------------------------------

class NeuralNetwork(nn.Module):
    # Creating Neural network with two hidden layers
    # Input: 4
    # Output 2
    def __init__(self,state_size,action_size):
        super(NeuralNetwork, self).__init__()
        self.lin1 = nn.Linear(state_size,32)
        self.lin2 = nn.Linear(32,32)
        self.lin3 = nn.Linear(32,32)
        self.lin4 = nn.Linear(32,action_size)

    # Defining forward function
    def forward(self,input):
        x = input.to(device)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))        
        x = self.lin4(x)
        return x


# datatype for storing multiple things
Experience = namedtuple('Experience',
                        ('state','action','reward','next_state'))

#Create a memory class
class Memory:
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)     #deque is like a list but fast

    def __len__(self):            
        return len(self.memory)

    def store(self, experience:Experience):
        self.memory.append(experience)

    def sample(self,batch_size:int):
        return random.sample(self.memory,batch_size)

#Take action function 
def take_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    #takes a random action if under eps_threshold, uses the model otherwise
    if random.uniform(0,1) < eps_threshold:
        action = random.randint(0, 1)
    else:
        model_output = model(state)
        # action = torch.argmax(model_output, device=device)
        action = torch.argmax(model_output).item()
    return action


def optimize_model():
    if len(memory) < batch_size:#1000: #batch_size:
        return

    #Samples random batch from memory
    transitions = memory.sample(batch_size)

    #separates all transitions and storing as a tensor
    batch  = Experience(*zip(*transitions)) 
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)                
    
    #Finds indexes for states that are not terminal
    non_final_index = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool) 
    #Stores all next states that are not terminal
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    #Calculate the Q_values of a taken action using model
    state_action_values = model(state_batch).gather(1, action_batch)
    
    #Calculate the Q_values for the best action for the next state using the target model
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_index] = torch.amax(target_model(non_final_next_states),1).detach()

    #Calculate the expected state action values
    expected_state_action_values = reward_batch + (gamma * next_state_values) 

    # Setup the loss function using the mean square error
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))
    
    #Reset gradient, calculate new gradient and take a step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def plot_data(scores, max_q_mean):
    plt.figure(1)
    plt.plot(scores,'.')
    plt.xlabel('episode number')
    plt.ylabel("timesteps")
    plt.title("standuptime")
    plt.savefig("standuptime.png")

    plt.figure(2)
    plt.plot(max_q_mean,'b--')
    plt.xlabel('episode number')
    plt.ylabel("average max value of Q")
    plt.title("Max_q_mean")
    plt.savefig("max_q_mean.png")

#Initializing starts here ----------------------------------

env = gym.make('CartPole-v0')
#env = gym.make('Breakout-ram-v0')

#Constants  
batch_size = 32
TARGET_UPDATE = 2
gamma = 0.99
EPISODES = 1000                          
test_states_no = 10000
steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
learning_rate = 5e-4
state_size = env.observation_space.shape[0]
action_size = env.action_space.n



#Create replay memory
memory = Memory(500000)

#Creates model and target model
model = NeuralNetwork(state_size,action_size).to(device)
target_model = NeuralNetwork(state_size,action_size).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()


#Using the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




if __name__ == "__main__":

    #create test states for plotting q-values
    test_states = np.zeros((test_states_no, state_size))
    max_q = np.zeros((EPISODES, test_states_no))
    max_q_mean = np.zeros((EPISODES,1))

    done = True
    for i in range(test_states_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action = random.randrange(2)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state
    
    scores, episodes = [], []

    #The main for loop where we go through the whole range of Episodes
    for i_episode in range(EPISODES):

        #calculate mean q-value
        done = False
        score = 0
        
        test_states = np.float32(test_states)
        tmp = model(torch.from_numpy(test_states))
        max_q[i_episode][:] = np.max(tmp.cpu().detach().numpy(), axis=1)
        max_q_mean[i_episode] = np.mean(max_q[i_episode][:])
        #breakpoint()
        #Reset env
        next_state = env.reset()
        next_state = next_state.astype('float32')
        next_state = torch.from_numpy(next_state).unsqueeze(0)

        while not done:
            env.render(mode ="human") #Can remove when training
            time.sleep(5) # sleeps for 5 seconds
            state = next_state
            
            action = take_action(state) 
            
            #get feedback from env
            next_state, reward, done, info = env.step(action)

            #If the next_state terminates set next_state to none otherwise convert it to tensor
            if done:
                next_state = None
            else:
                next_state = next_state.astype('float32')
                next_state = torch.from_numpy(next_state).unsqueeze(0)
            
            #Convert reward to tensor
            reward = torch.tensor([reward], device=device)
            action = torch.tensor([action], device=device)
              
            score += reward.item()

            memory.store(Experience(state,action,reward,next_state))
            
            #Train the model
            optimize_model()

            #If the episode is finished
            if done:
                scores.append(score)
                print("episode:", i_episode, "  score:", score," q_value:", max_q_mean[i_episode],
                "  memory length:", len(memory), " average score: ",np.mean(scores[-100:]))

                #Check if cartpole is solved
                if np.mean(scores[-100:]) >= 195:
                    print('solved after: ', i_episode-100,' episodes')
                    plot_data(scores,max_q_mean[:i_episode])
                    sys.exit()
                break

        #Updates the target network
        if i_episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())
        



    

