import numpy as np
import math

class epsilon_greedy_agent(object):
    """
    Class to represent epsilon greedy agent

    Uses epsilon greedy algorithm to select best arm
    """
    def __init__(self,num_arms, epProb):
        self.num_arms = num_arms    #number of arms in testbed
        self.epProb = epProb    #epsilon value

        self.timeStep = 0   #step count
        self.lastAction = None      #Previous Action

        self.kAction = np.zeros(num_arms)
        self.rSum = np.zeros(num_arms)
        self.valEstimates = np.zeros(num_arms)


    def agent_name(self):
        return "epsilon_greedy_agent"

    def __str__(self):
        return "ep=" + str(self.epProb)

    def action(self):

        randProb = np.random.random()   # Pick random probability between 0-1
        if randProb < self.epProb:
            a = np.random.choice(len(self.valEstimates))    # Select random action
        else:
            maxAction = np.argmax(self.valEstimates)     # Find max value estimate

            action = np.where(self.valEstimates == np.argmax(self.valEstimates))[0]

            if len(action) == 0:
                a = maxAction
            else:
                a = np.random.choice(action)

        self.lastAction = a
        return a


    def learn(self, reward):
        At = self.lastAction

        self.kAction[At] += 1       # Add 1 to action selection
        self.rSum[At] += reward     # Add reward to sum array

        self.valEstimates[At] = self.rSum[At]/self.kAction[At]

        self.timeStep += 1


    def reset(self):
        self.timeStep = 0                    # Time Step t
        self.lastAction = None               # Store last action

        self.kAction[:] = 0                  # count of actions taken at time t
        self.rSum[:] = 0
        self.valEstimates[:] = 0   # action value estimates Qt ~= Q*(a)



class softmax_agent(object):

    def __init__(self, num_arms, temp):
        self.num_arms = num_arms      # Number of arms
        self.temp = temp              # temp

        self.timeStep = 0                    # Time Step t
        self.lastAction = None               # Store last action

        self.kAction = np.zeros(num_arms)          # count of actions taken at time t
        self.rSum = np.zeros(num_arms)             # Sums number of rewards
        self.valEstimates = np.zeros(num_arms)     # action value estimates sum(rewards)/Amount


    def agent_name(self):
        return "softmax_agent"

    def __str__(self):
        return "temp=" + str(self.temp)


    def action(self):


        randProb = np.random.random()
        softmax_probs = np.exp(self.valEstimates/self.temp) / np.sum(np.exp(self.valEstimates/self.temp), axis=0)
        a = np.random.choice(len(self.valEstimates), p=softmax_probs)
        
        self.lastAction = a
        return a


    def learn(self, reward):
        At = self.lastAction

        self.kAction[At] += 1
        self.rSum[At] += reward

        self.valEstimates[At] = self.rSum[At]/self.kAction[At]

        self.timeStep += 1


    def reset(self):
        self.timeStep = 0
        self.lastAction = None

        self.kAction[:] = 0
        self.rSum[:] = 0
        self.valEstimates[:] = 0

class UCB1(object):

    def __init__(self, num_arms, c=1):
        self.num_arms = num_arms
        self.c = c

        self.timeStep = 0
        self.lastAction = None

        self.kAction = np.zeros(num_arms)
        self.rSum = np.zeros(num_arms)
        self.valEstimates = np.zeros(num_arms)

    def agent_name(self):
        return "UCB1"

    def __str__(self):
        return "c=" + str(self.c)

    def action(self):

        for i, a in enumerate(self.valEstimates):
            if a == 0:
                self.lastAction = i
                return i

        a = np.argmax(self.valEstimates + (self.c*np.sqrt(np.log(self.timeStep)/self.kAction)))

        self.lastAction = a
        return a


    def learn(self, reward):
        At = self.lastAction

        self.kAction[At] += 1
        self.rSum[At] += reward

        self.valEstimates[At] = self.rSum[At]/self.kAction[At]

        self.timeStep += 1


    def reset(self):
        self.timeStep = 0
        self.lastAction = None

        self.kAction[:] = 0
        self.rSum[:] = 0
        self.valEstimates[:] = 0





class MEA(object):

    def __init__(self, num_arms, ep=0.5, delta=0.01):
        self.num_arms = num_arms

        self.ep = ep
        self.delta = delta

        self.ep_l = self.ep*0.25
        self.delta_l = self.delta*0.5

        self.timeStep = 0
        self.lastAction = None

        self.kAction = np.zeros(num_arms)
        self.rSum = np.zeros(num_arms)
        self.valEstimates = np.zeros(num_arms)

        self.arms = np.arange(num_arms)
        self.arm_count = np.zeros(num_arms)

    def agent_name(self):
        return "MEA"

    def __str__(self):
        return "ep=" + str(self.ep) + " delta=" + str(self.delta)

    def update_epdel(self):

        self.ep_l *= 0.75
        self.delta_l *= 0.5

    def update_count(self):
        return math.ceil((4/(self.ep_l**2))*(math.log(3/self.delta_l)))

    def action(self):

        if len(self.arms)>1:
            for i in self.arms:
                if self.arm_count[i]<self.update_count():
                    self.lastAction = i
           
                    return i
        else:
            return self.arms[0]

    def learn(self, reward):

        At = self.lastAction
        self.kAction[At] += 1
        self.arm_count[At] += 1
        self.rSum[At] += reward

        self.valEstimates[At] = self.rSum[At]/self.kAction[At]

        self.timeStep += 1

        if len(self.arms)>1:
            if np.sum(self.arm_count)==self.update_count()*len(self.arms):
                
                valEstimates = self.valEstimates[self.arms]
                #arms where valesitmates>median(valestimates)
                self.arms = self.arms[np.nonzero(valEstimates>np.median(valEstimates))[0]]
                np.random.shuffle(self.arms)
                self.arm_count[:] = 0
                #update epsilon and delta
                self.update_epdel()

    def reset(self):

        self.timeStep = 0
        self.lastAction = None

        self.kAction[:] = 0
        self.rSum[:] = 0
        self.valEstimates[:] = 0

        self.arms = np.arange(self.num_arms)
        self.arm_count[:] = 0

        self.ep_l = self.ep*0.25
        self.delta_l = self.delta*0.5



