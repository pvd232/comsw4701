import random
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
Sudoku board initializer
Credit: https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
"""


def generate(N, num_clues):
    # Generate a sudoku puzzle of order n with "num_clues" cells assigned
    # A random set of num_clues cells contain the solution values, empty cells contain 0
    # Return a numpy array representing the sudoku puzzle and list of clue indices
    n = int(np.sqrt(N))
    ran = range(n)

    rows = [g * n + r for g in sample(ran, n) for r in sample(ran, n)]
    cols = [g * n + c for g in sample(ran, n) for c in sample(ran, n)]
    nums = sample(range(1, N + 1), N)

    S = np.array([[nums[(n * (r % n) + r // n + c) % N] for c in cols] for r in rows])
    indices = sample(range(N**2), num_clues)
    values = S.flatten()[indices]

    empty_board = np.zeros(N**2, dtype=int)
    empty_board[indices] = values
    board = np.reshape(empty_board, (N, N))

    clues = []
    for i in indices:
        clues.append((i // N, i % N))
    return board, clues


def initialize(board):
    # Given a sudoku puzzle, return a board with all empty spaces filled in
    # Ensure that every major subgrid contains n unique integers from 1 to n
    N = board.shape[0]
    n = (int)(N**0.5)
    for i in range(N):
        subgrid = board[(i // n) * n : (i // n) * n + n, (i % n) * n : (i % n) * n + n]
        for j in range(1, N + 1):
            if j not in subgrid:
                idx = np.argwhere(subgrid == 0)[0]
                subgrid[idx[0], idx[1]] = j
        board[(i // n) * n : (i // n) * n + n, (i % n) * n : (i % n) * n + n] = subgrid
    return board


def successors(board, clues):
    # Return a list of all successor states, each generated
    # by swapping two non-clue entries within a major subgrid
    N = board.shape[0]
    n = (int)(N**0.5)
    successors = []

    for i in range(N):
        subgrid = []
        for j in range(n):
            for k in range(n):
                subgrid.append(((i // n) * n + j, (i % n) * n + k))

        for j in range(N):
            if subgrid[j] not in clues:
                for k in range(j + 1, N):
                    if subgrid[k] not in clues:
                        succ = np.copy(board)
                        tmp = succ[subgrid[j]]
                        succ[subgrid[j]] = succ[subgrid[k]]
                        succ[subgrid[k]] = tmp
                        successors.append(succ)
    return successors


def num_errors(board):
    """Compute and return the total number of errors on the sudoku board.
    Total number of missing values from every row and every column.
    """
    N = board.shape[0]
    digits = range(1, N + 1)
    errors = 0
    for i in range(N):
        errors += N - np.sum(np.isin(digits, board[i]))
        errors += N - np.sum(np.isin(digits, board[:, i]))
    return errors


def simulated_annealing(board, clues, startT, decay, tol=1e-4, errors=[]):
    """Run simulated annealing to solve sudoku.

    Args:
        board (numpy): NumPy array representing the initial sudoku board.
        clues (list): A list of clue cell tuples.
        startT (float): The starting value of temperature T.
        decay (float): The decay rate of temperature T.
        tol (float): The minimum threshold of temperature T.

    Returns:
        board (numpy): NumPy array representing the final sudoku board.
    """
    current_error = np.inf
    T = startT / decay

    while current_error > 0 and T >= tol:
        T *= decay
        all_succ = successors(board, clues)
        if not all_succ:
            break

        succ = random.choice(all_succ)
        succ_error = num_errors(succ)
        if succ_error < current_error or np.random.random() < np.exp(
            (current_error - succ_error) / T
        ):
            board = succ
            errors.append(current_error)
            current_error = succ_error
    if current_error == 0:
        errors.append(current_error)

    return board


def plot_batch_error(errors, num_runs):
    perf = 0
    data = []
    i = 0
    for e in errors:
        if not e:
            perf += 1
        data.append(e)
        i += 1
    print("Perf runs " + str(round((perf / num_runs) * 100)) + "%")
    # Create the histogram
    plt.hist(data, bins=30, edgecolor="black")

    # Add labels and title
    plt.xlabel("Number of Trials")
    plt.ylabel("Number of Errors")

    # Show the plot
    plt.show()


def plot_one_err(errors):
    plt.plot(errors, marker="o", linestyle="-", color="b", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Number of Errors")
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="COMSW4701 HW2",
        description="Sudoku",
    )
    parser.add_argument(
        "-n", required=True, type=int, help="Value of n specifying grid size (nxn)"
    )
    parser.add_argument("-c", required=True, type=int, help="Number of clues")
    parser.add_argument(
        "-s",
        default=100,
        type=float,
        help="Starting value of temperature T (default 100)",
    )
    parser.add_argument(
        "-d", default=0.5, type=float, help="Decay rate of temperature T (default 0.5)"
    )
    parser.add_argument("-b", type=int, help="Number of searches to run in a batch")
    args = parser.parse_args()

    if args.b is None:
        errors = []
        board, clues = generate(args.n, args.c)
        print("Sudoku puzzle:\n", board, "\n")
        sol = simulated_annealing(
            initialize(board), clues, args.s, args.d, 1e-4, errors
        )
        print(sol)
        print("Number of errors: ", num_errors(sol))
        plot_one_err(errors)
    else:
        final_errors = []
        for _ in range(args.b):
            board, clues = generate(args.n, args.c)
            sol = simulated_annealing(initialize(board), clues, args.s, args.d)
            final_errors.append(num_errors(sol))
        plot_batch_error(final_errors, args.b)


if __name__ == "__main__":
    main()
