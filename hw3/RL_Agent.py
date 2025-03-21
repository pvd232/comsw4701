"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A Q-learning agent for a stochastic task environment
"""

import random
import math
import sys


class RL_Agent(object):

    def __init__(self, states, valid_actions, parameters):
        self.alpha = parameters["alpha"]
        self.epsilon = parameters["epsilon"]
        self.gamma = parameters["gamma"]
        self.Q0 = parameters["Q0"]

        self.states = states
        self.Qvalues = {}
        for state in states:
            for action in valid_actions(state):
                self.Qvalues[(state, action)] = parameters["Q0"]
        print(self.epsilon)

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def choose_action(self, state, valid_actions):
        """Choose an action using epsilon-greedy selection.

        Args:
            state (tuple): Current robot state.
            valid_actions (list): A list of possible actions.
        Returns:
            action (string): Action chosen from valid_actions.
        """

        r = random.randint(0, 10)
        if r / 10 <= self.epsilon:
            q_max = float("-inf")
            a_max = None
            for a in valid_actions:
                q = self.Qvalues[(state, a)]
                if q > q_max:
                    q_max = q
                    a_max = a
            return a_max
        else:
            a_l = len(valid_actions) - 1
            r2 = random.randint(0, a_l)
            a = valid_actions[r2]
            return a

    def update(self, state, action, reward, successor, valid_actions):
        """ Update self.Qvalues for (state, action) given reward and successor.

        Args:
            state (tuple): Current robot state.
            action (string): Action taken at state.
            reward (float): Reward given for transition.
            successor (tuple): Successor state.
            valid_actions (list): A list of possible actions at successor state.
        """
        if successor == None:
            self.Qvalues[(state, action)] = 0
        else:
            q1 = self.Qvalues[(state, action)]
            a2 = self.choose_action(successor, valid_actions)
            learn = self.alpha * (
                reward + (self.gamma * self.Qvalues[(successor, a2)]) - q1
            )
            q2 = q1 + learn
            self.Qvalues[(state, action)] = q2
