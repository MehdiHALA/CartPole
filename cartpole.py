#!/usr/bin/env python

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd


N = 10

pos = 2.4
angle = 0.418
inf = 5

# Discretize the intervals
cart_position_bins = pd.cut([-pos, pos], bins=N)
cart_velocity_bins = pd.cut([-inf, inf], bins=2*N)
pole_angle_bins = pd.cut([-angle, angle], bins=N)
pole_velocity_bins = pd.cut([-inf, inf], bins=2*N)


def get_interval(x,L,n=N):
    '''
    Returns the index of the interval that contains x
    Args:
        x: Number to which we're trying to find the interval
        L: Intervals list
        n: number of intervals
    Return:
        Index of the corresponding interval
    '''
    for i in range(0,n):
        if x in L[i]:
            return i

#############################################################################################################################
########## Start of class
#############################################################################################################################

class AgentMonteCarlo(object):
    def __init__(self, env, epsilon=1.0, gamma=1):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q_table = dict()
        self.n = {((a,b,c,d), action):0 for a in range(N) for b in range(2*N) for c in range(N) for d in range(2*N) for action in range(3)}
        self.ep = [[]]
        random.seed()


    def create_state(self, obs):
        '''
        Create state variable from observation.

        Args:
            obs: Observation list with format [horizontal position, velocity,
                 angle of pole, angular velocity].
        Returns:
            state: State tuple
        '''
        state = (get_interval(obs[0], cart_position_bins.categories),
                get_interval(obs[1], cart_velocity_bins.categories, 2*N),
                get_interval(obs[2], pole_angle_bins.categories),
                get_interval(obs[3], pole_velocity_bins.categories, 2*N))
        return state


    def choose_action(self, state):
        '''
        Given a state, choose an action.

        Args:
            state: State of the agent.
        Returns:
            action: Action that agent will take.
        '''
        # Chooses a random action to do with a probability of epsilo
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # Find max Q value
            max_Q = self.get_maxQ(state)
            actions = []
            for action, Q in self.Q_table[state].items():
                if Q == max_Q:
                    actions.append(action)
            if len(actions) != 0:
                action = random.choice(actions)
        return action


    def create_Q(self, state, valid_actions):
        '''
        Update the Q table given a new state/action pair.

        Args:
            state: List of state booleans.
            valid_actions: List of valid actions for environment.
        '''
        if state not in self.Q_table:
            self.Q_table[state] = dict()
            for action in valid_actions:
                self.Q_table[state][action] = 0.0
        return


    def get_maxQ(self, state):
        '''
        Find the maximum Q value in a given Q table.

        Args:
            Q_table: Q table dictionary.
            state: List of state booleans.
        Returns:
            maxQ: Maximum Q value for a given state.
        '''
        maxQ = max(self.Q_table[state].values())
        return maxQ


    def train(self, episode):
        '''
        Update the Q-values

        Args:
            episode : the number of the current episode
        '''
        for step in self.ep[episode]:
            Rt = len(self.ep[episode])
            self.Q_table[step[0]][step[1]] += 1/float(self.n[(step[0], step[1])]) * (Rt - self.Q_table[step[0]][step[1]])

#############################################################################################################################
########## End of class
#############################################################################################################################


def avg100(l, index):
    '''
    Function that returns the average of the last 100 rewards

    Args:
        l: list of Rewards
        old_avg: last average calculated
        index: index of where the new average should be calculated
    Return:
    '''
    return np.mean(l[index-99:-1])


def q_learning(env, agent):
    '''
    Implement Q-learning policy.

    Args:
        env: Gym enviroment object.
        agent: Learning agent.
    Returns:
        Rewards for training/testing.
    '''
    # Start out with Q-table set to zero.
    # Whenever a new state is found, agent adds a new column/row to Q-table
    valid_actions = [0, 1]
    rewards = []
    avgs = []
    episode = 0
    avg = 0
    while(True):
        episode_rewards = 0
        obs = env.reset()
        agent.epsilon = agent.epsilon * 0.99                        # 99% of epsilon value
        for step in range(500):                                     # 500 steps max
            state = agent.create_state(obs)                         # Get state
            agent.create_Q(state, valid_actions)                    # Create state in Q_table
            action = agent.choose_action(state)                     # Choose action
            agent.n[(state, action)] += 1                           # Update agent.n
            env.render()                                            # Render
            obs, reward, done, info = env.step(action)              # Do action
            agent.ep[episode].append((state, action, reward))       # Update agent.ep
            episode_rewards += reward                               # Receive reward
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                break

        # Plot Block ############
        rewards.append(episode_rewards)
        if (episode <= 99):
            avg = np.mean(episode_rewards)
        elif (episode > 99):
            avg = avg100(rewards, episode)
        avgs.append(avg)
        #########################
        agent.train(episode)                                        # Train ( Update Q-Values )
        agent.ep.append([])                                         # Add a new episode
        # Terminal Prints
        print()
        print("Episode : " + str(episode))
        print("Current reward : " + str(episode_rewards))
        print("Avg of last 100 ( if appliable ) : " + str(avg))

        # Breaks if solved
        if avg >= 195.0 and episode > 100:
            print("Congrats !!!")
            return rewards, avgs
        episode += 1
    return rewards, avgs


def run():
    ''' Execute main program. '''
    # Create a cartpole environment
    # Observation: [horizontal pos, velocity, angle of pole, angular velocity]
    # Rewards: +1 at every step. i.e. goal is to get to an average reward ( on 100 consecutive episodes ) greater or equal to 195 the fastest possible
    env = gym.make('CartPole-v1')
    # Set environment seed
    print("Setting Environment")
    env.seed(21)
    agent = AgentMonteCarlo(env, epsilon=0.6)
    rewards, avgs = q_learning(env, agent)
    plt.plot(avgs)
    plt.xlabel("episode")
    plt.ylabel("Average of the last 100 episodes if appliable")
    plt.show()
    # No argument passed, agent defaults to Basic

if __name__ == '__main__':
    ''' Run main program. '''
    run()
