#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Alpha-beta minimax AI player for Othello.
"""

import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


def max_value(state, player, alpha, beta):
    """
    Args:
        state: Board state
        player: Dark (1) or light (2)
        alpha, beta values

    Returns:
        value (int): Minimax value of state
        move (tuple): Best move to make
    """
    v_max = -float("inf")
    a_max = None  
    m = get_possible_moves()
    if not m:
        return compute_utility(state), None
    for a in m:
        v2, _ = min_value(state, 3- player, alpha, beta)
        if v2 > v_max:
            v_max = v2, a_max = a
            alpha = max(alpha, v_max)
        if v2 > beta:
            return v_max, a_max
    return v_max, a_max


def min_value(state, player, alpha, beta):
    """
    Args:
        state: Board state
        player: Dark (1) or light (2)
        alpha, beta values

    Returns:
        value (int): Minimax value of state
        move (tuple): Best move to make
    """
    v_min = -float("inf")
    a_min = None
    m = get_possible_moves()
    if not m:
        return compute_utility(state), None
    for a in m:
        v2, _ = max_value(state, 3 - player,alpha,beta)
        if v2 < v_min:
            v_min = v2, a_min = a
            beta = min(beta, v_min)
        if v_min < alpha:
            return v_min, a_min
    return v_min, a_min


def minimax(state, player):
    """
    Minimax main loop
    Call max_value if player is 1 (dark), min_value if player is 2 (light)
    Then return the resultant move
    """
    if player == 1:
        _, move = max_value(state, player, -float('inf'), float('inf'))
    else:
        _, move = min_value(state, player, -float('inf'), float('inf'))
    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Minimax AI")     # First line is the name of this AI
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
            movei, movej = minimax(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
