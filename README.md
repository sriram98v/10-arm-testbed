# Multi Arm Bandit Implementations in Python

This repository contains the python implementation of popular multi arm bandits as described in the book:

Richard S. Sutton and Andrew G. Barto, Reinforcement Learning:  AnIntroduction, Second edition

It includes Epsilon-greedy, softmax action selection, UCB1 and Median Elimination algorithm

## Usage

To test the algorithms you can run the main.py script as is an it will produce all the graphs as depicted in the report and save them in the plots directory.

To run the agents on custom multi arm bandits problems, you can change the values of variables:

```bash
num_arms = 10
plays = 2000
iterations = 1000
```
## References

Markup :  1. Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, Second
		 	edition
         2. Github Link: Shangtong Zhang, Python implementation of Reinforcement Learning: An 
         	Introduction.
         3. Even-Dar, Eyal \& Mannor, Shie \& Mansour, Yishay. (2003). Action Elimination and 
         	Stopping Conditions for Reinforcement Learning.. 162-169. 
         4. https://github.com/jettdlee/10_armed_bandit.git
