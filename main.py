import numpy as np
from agents import epsilon_greedy_agent, softmax_agent, UCB1, MEA
from env import Environment
from testbed import Testbed

num_arms = 10
iterations = 100
plays = 3000
testbed = Testbed(num_arms=num_arms, mean=0, var=1)



print("\nRunning epsilon_greedy_agent")
agents = [epsilon_greedy_agent(num_arms=num_arms, epProb=0), 
            epsilon_greedy_agent(num_arms=num_arms, epProb=0.01), 
            epsilon_greedy_agent(num_arms=num_arms, epProb=0.1),
            epsilon_greedy_agent(num_arms=num_arms, epProb=1)]
environment = Environment(testbed=testbed, agents=agents, plays=plays, iterations=iterations, name="Epsilon_Greedy")

avg_rewards, optimal_percent = environment.play()

environment.plot()




print("\nRunning softmax_agent")
agents = [softmax_agent(num_arms=num_arms, temp=1),
            softmax_agent(num_arms=num_arms, temp=0.1),
            softmax_agent(num_arms=num_arms, temp=0.01)]
environment = Environment(testbed=testbed, agents=agents, plays=plays, iterations=iterations, name="Softmax")

avg_rewards, optimal_percent = environment.play()

environment.plot()



print("\nRunning UCB1")
agents = [UCB1(num_arms=num_arms, c=0.1),
			UCB1(num_arms=num_arms, c=1),
			UCB1(num_arms=num_arms, c=2)]
environment = Environment(testbed=testbed, agents=agents, plays=plays, iterations=iterations, name="UCB1")

avg_rewards, optimal_percent = environment.play()

environment.plot()


print("\nRunning MEA algorithm")

agents = [MEA(num_arms=num_arms, ep=2, delta=1),
			MEA(num_arms=num_arms, ep=1, delta=1),
			MEA(num_arms=num_arms, ep=0.1, delta=1)]
environment = Environment(testbed=testbed, agents=agents, plays=plays, iterations=iterations, name="MEA")

avg_rewards, optimal_percent = environment.play()

environment.plot()

num_arms = 1000
iterations = 100
plays = 30000
testbed = Testbed(num_arms=num_arms, mean=0, var=1)

print("\nRunning comaprison")

agents = [MEA(num_arms=num_arms, ep=5, delta=1),
			MEA(num_arms=num_arms, ep=5, delta=.1),
			MEA(num_arms=num_arms, ep=9, delta=1),
			MEA(num_arms=num_arms, ep=9, delta=.1)]
environment = Environment(testbed=testbed, agents=agents, plays=plays, iterations=iterations, name="Comparison")

avg_rewards, optimal_percent = environment.play()

environment.plot()
