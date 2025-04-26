#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility
from math import sqrt, log1p

class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state:np.ndarray = state
        self.player = player
        self.parent = parent
        self.children = children
        self.value = v
        self.N = N

    def get_child(self, state):
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def select(root:Node, alpha:float):
    """ Starting from given node, find a terminal node or node with unexpanded children.
    If all children of a node are in tree, move to the one with the highest UCT value.

    Args:
        root (Node): MCTS tree root node
        alpha (float): Weight of exploration term in UCT

    Returns:
        node (Node): Node at bottom of MCTS tree
    """
    curr = root
    moves = get_possible_moves(curr)
    if not moves:
        return root
    while moves:
        n_s = curr.state.copy()
        max_uct = float("-inf")
        best = None        
        for m in moves:
            s = play_move(n_s, curr.player,m[0], m[1])
            c: Node = curr.get_child(s)
            if not c:
                return curr
            else:
                uct = (c.value / c.N) + alpha(sqrt(log1p(curr.N/c.N)))
                if uct > max_uct:
                    best = c
                    max_uct = uct
        if not best:
            return curr
        curr = c
        moves = get_possible_moves(curr)                    
    return curr


def expand(node):
    """ Add a child node of state into the tree if it's not terminal.

    Args:
        node (Node): Node to expand

    Returns:
        leaf (Node): Newly created node (or given Node if already leaf)
    """
    curr = node
    moves = get_possible_moves(curr)
    if not moves:
        return node
    n_s = curr.state.copy()
    for m in moves:
        s = play_move(n_s, curr.player, m[0], m[1])
        c: Node = curr.get_child(s)
        if not c:
            c = Node(s, 3 - curr.player, curr,[],0,0)
            curr.children.append(c)
            return c            
    return node


def simulate(node:Node):
    """ Run one game rollout using from state to a terminal state.
    Use random playout policy.

    Args:
        node (Node): Leaf node from which to start rollout.

    Returns:
        utility (int): Utility of final state
    """
    curr = node
    player = curr.player
    moves = get_possible_moves(curr)
    s = node.state.copy()
    
    while moves:
        m = random.choice(moves)
        s = play_move(s, curr.player, m[0],m[1]) # update board
        player = 3 - player # update player to get new moves
        moves = get_possible_moves(s, player) # get new moves
    return compute_utility(s)


def backprop(node:Node, utility):
    """ Backpropagate result from state up to the root.
    Every node has N, number of plays, incremented
    If node's parent is dark (1), then node's value increases
    Otherwise, node's value decreases.

    Args:
        node (Node): Leaf node from which rollout started.
        utility (int): Utility of simulated rollout.
    """
    curr = node
    while curr:
        curr.N += 1
        if curr.player == 2:
            curr.value += utility
        else:
            curr.value -= utility
        curr = curr.parent
    return


def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for i in range(rollouts):
        leaf = select(root, alpha)
        new = expand(leaf)
        utility = simulate(new)
        backprop(new, utility)

    move = None
    plays = 0
    for m in get_possible_moves(state, player):
        s = play_move(state, player, m[0], m[1])
        if root.get_child(s).N > plays:
            plays = root.get_child(s).N
            move = m

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
