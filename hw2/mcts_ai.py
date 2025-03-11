#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import annotations

"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input  # type: ignore
from othello_shared import get_possible_moves, play_move, compute_utility
from typing import Optional
from math import sqrt, log
import numpy.typing as npt


class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state: npt.NDArray[np.int_] = state
        self.player: int = player
        self.parent: Node = parent
        self.children: list[Node] = children
        self.value: int = v
        self.N: int = N

    def get_child(self, state) -> Optional[Node]:
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def select(root: Node, alpha: float) -> Node:
    """Starting from the given node, find a terminal node or a node with unexpanded children.
    If all children of a node are in the tree, move to the one with the highest UCT value.

    Args:
        root (Node): MCTS tree root node.
        alpha (float): Weight of exploration term in UCT.

    Returns:
        Node: Node at the bottom of the MCTS tree.
    """
    curr = root
    moves = get_possible_moves(curr.state, curr.player)

    # Loop while there are possible moves from the current node.
    while moves:
        best_uct = float("-inf")
        best_child = None

        for m in moves:
            child_state = play_move(curr.state, curr.player, m[0], m[1])
            c = curr.get_child(child_state)
            if not c:
                # Unexpanded move found; return current node.
                return curr
            else:
                uct = (c.value / c.N) + alpha * sqrt(log(c.parent.N) / c.N)
                if uct > best_uct:
                    best_uct = uct
                    best_child = c

        if not best_child:
            return curr

        curr = best_child
        moves = get_possible_moves(curr.state, curr.player)

    return curr


def expand(node: Node) -> Node:
    """Add a child node of state into the tree if it's not terminal.

    Args:
        node (Node): Node to expand

    Returns:
        leaf (Node): Newly created node (or given Node if already leaf)
    """
    moves = get_possible_moves(node.state, node.player)
    if not moves:
        return node
    # Loop while there are possible moves from the current node.
    for m in moves:
        child_state = play_move(node.state, node.player, m[0], m[1])
        c = node.get_child(child_state)
        if not c:
            leaf = Node(child_state, 3 - node.player, node, [], 0, 0)
            node.children.append(leaf)
            return leaf
    return node


def simulate(node: Node) -> int:
    """Run one game rollout using from state to a terminal state.
    Use random playout policy.

    Args:
        node (Node): Leaf node from which to start rollout.

    Returns:
        utility (int): Utility of final state
    """
    curr_state = node.state.copy()
    player = node.player
    moves = get_possible_moves(curr_state, player)

    while moves:
        m = random.choice(moves)  # Random rollout policy
        new_state = play_move(curr_state, player, m[0], m[1])
        curr_state = new_state
        player = 3 - player
        moves = get_possible_moves(curr_state, player)

    return compute_utility(curr_state)


def backprop(node: Node, utility: int):
    """Backpropagate result from state up to the root.
    Every node has N, number of plays, incremented
    If node's parent is dark (1), then node's value increases
    Otherwise, node's value decreases.

    Args:
        node (Node): Leaf node from which rollout started.
        utility (int): Utility of simulated rollout.
    """
    curr = node
    while curr:
        if curr.player == 2:
            curr.value += utility
        else:
            curr.value -= utility
        curr.N += 1
        curr = curr.parent
    return


def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for _ in range(rollouts):
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
    print("MCTS AI")  # First line is the name of this AI
    color = int(input())  # 1 for dark (first), 2 for light (second)

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
