import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Environment(object):

    # Constructor
    def __init__(self, testbed, agents, plays, iterations, name):
        self.testbed = testbed
        self.agents = agents

        self.name = name

        self.plays = plays
        self.iterations = iterations

        self.scoreAvg = 0
        self.optimlAvg = 0


    # Run Test
    def play(self):

        # Array to store the scores, number of plays X number of agents
        scoreArr = np.zeros((self.plays, len(self.agents)))
        # Array to maintain optimal count, Graph 2
        optimlArr = np.zeros((self.plays, len(self.agents)))

        # loop for number of iterations
        for iIter in tqdm(range(self.iterations)):

            #Reset testbed and all agents
            self.testbed.reset()
            for agent in self.agents:
                agent.reset()


            # Loop for number of plays
            for jPlays in range(self.plays):
                agtCnt = 0

                for kAgent in self.agents:
                    actionT =  kAgent.action()
                    # Reward - normal dist (Q*(at), variance = 1)
                    rewardT = np.random.normal(self.testbed.A[actionT], scale=1)

                    # Agent checks state
                    kAgent.learn(reward=rewardT)
                    # Add score in arrary, graph 1
                    scoreArr[jPlays,agtCnt] += rewardT

                    # check the optimal action, add optimal to array, graph 2
                    # print(actionT, self.testbed.optim)
                    if actionT == self.testbed.optim:
                        optimlArr[jPlays,agtCnt] += 1

                    agtCnt += 1

        #return averages
        scoreAvg = scoreArr/self.iterations
        optimlAvg = optimlArr/self.iterations

        self.scoreAvg, self.optimlAvg = scoreAvg, optimlAvg

        return scoreAvg, optimlAvg

    def plot(self):

        plt.title("1000-Armed TestBed - "+self.name+" Average Rewards")
        plt.plot(self.scoreAvg, linewidth=.5)
        plt.ylabel('Average Reward')
        plt.xlabel('Plays')
        plt.legend(self.agents, loc=4)
        plt.show(block=True)
        plt.savefig("./plots/Average_Rewards_"+self.name+".jpg")
        plt.clf()

        plt.title("1000-Armed TestBed - "+self.name+" % Optimal Action")
        plt.plot(self.optimlAvg * 100, linewidth=.5)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Plays')
        plt.legend(self.agents, loc=4)
        plt.show(block=True)
        plt.savefig("./plots/Optimal_Action_"+self.name+".jpg")
        plt.clf()

