"""
COMS W4701 Artificial Intelligence - Programming Homework 3

A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None

    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass

    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass

    def policy_evaluation(self, transition):
        """Computes all values for current policy by iteration and stores them in self.values.
        Args:
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        done = False
        while not done:
            done = True
            for i in range(len(self.states)):
                s1 = self.states[i]
                v1 = self.values[s1]
                s2, r = transition(s1, self.policy[s1])
                v2 = r + (self.gamma * self.values[s2])
                if abs(v2 - v1) > 1e-6:
                    done = False
                self.values[s1] = v2
    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.
        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        for s1 in self.states:
            acts = valid_actions(s1)
            max_v = float("-inf")
            max_a = None
            for a in acts:
                s2, r = transition(s1, a)
                v = 0
                if s2 != None:
                    v = r + self.gamma * self.values[s2]
                if v > max_v:
                    max_v = v
                    max_a = a
            self.policy[s1] = max_a

    def policy_iteration(self, valid_actions, transition):
        """ Runs policy iteration to learn an optimal policy. Calls policy_evaluation() and policy_extraction().
        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        done = False
        while not done:
            done = True
            old_policy = self.policy.copy()
            self.policy_evaluation(transition)
            self.policy_extraction(valid_actions, transition)
            for s in self.states:
                p1 = old_policy[s]
                p2 = self.policy[s]
                if p1 != p2:
                    done = False
