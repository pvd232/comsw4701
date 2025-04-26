from queue import PriorityQueue
import heapq
from utils.utils import (
    PathPlanMode,
    Heuristic,
    cost,
    expand,
    visualize_expanded,
    visualize_path,
    Environment,
)
import numpy as np
import numpy.typing as npt
from typing import TypedDict, Optional
import math


def compute_heuristic(
    node: tuple[int, int], goal: tuple[int, int], heuristic: Heuristic
) -> float:
    """Computes an admissible heuristic value of node relative to goal.

    Args:
        node (tuple): The cell whose heuristic value we want to compute.
        goal (tuple): The goal cell.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.

    Returns:
        h (float): The heuristic value.

    """
    if heuristic == Heuristic.MANHATTAN:
        # Scale value by 1/2 to account for diagonal distance being 1, not 2
        # Scale value by 1/2 to account for the lowest cost tile being .5
        return math.fabs((goal[0] - node[0]) + (goal[1] - node[1])) / 4
    else:
        # Scale value by 1/2 to account for diagonal distance being 1, not sqrt(2)
        # Scale value by 1/2 to account for the lowest cost tile being .5
        return (
            math.sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2)
            * math.sqrt(2)
            / 4
        )


def uninformed_search(
    grid: npt.NDArray[np.int_],
    start: tuple[int, int],
    goal: tuple[int, int],
    mode: PathPlanMode,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """Find a path from start to goal in the gridworld using
    BFS or DFS.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.DFS or PathPlanMode.BFS.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """
    frontier: list[tuple[int, int]] = [start]
    frontier_sizes: list[int] = []
    expanded: list[tuple[int, int]] = []
    reached: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    path: list[tuple[int, int]] = []

    # Check to make sure frontier has nodes to expand
    while len(frontier):
        frontier_sizes.append(len(frontier))
        curr: tuple[int, int] = frontier.pop()
        if curr == goal:
            # Traverse back up the tree to recreate the path
            curr_node: Optional[tuple[int, int]] = curr
            while curr_node is not None:
                path.insert(0, curr_node)
                curr_node = reached[curr_node]
            break
        else:
            expanded.append(curr)
            for child in expand(grid, curr):
                if child not in reached and grid[child] != Environment.MOUNTAIN:
                    reached[child] = curr  # Key is child, value is parent
                    if mode == PathPlanMode.DFS:
                        frontier.append(child)
                    else:
                        frontier.insert(0, child)
    return path, expanded, frontier_sizes


# Define custom types for clarity
class NodeRecord(TypedDict):
    cost: float
    parent: Optional[tuple[int, int]]


# Example function signature using the new type
def a_star(
    grid: npt.NDArray[np.int_],
    start: tuple[int, int],
    goal: tuple[int, int],
    mode: PathPlanMode,
    heuristic: Heuristic,
    width: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """
    Performs A* search or beam search to find the shortest path from start to goal in the gridworld.

    Args:
        grid (numpy.ndarray): NxN numpy array representing the world with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must specify either PathPlanMode.A_STAR or PathPlanMode.BEAM_SEARCH.
        heuristic (Heuristic): The heuristic to use. Must specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        width (int): The width of the beam search. Only used if mode is PathPlanMode.BEAM_SEARCH.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing the size of the frontier at each iteration.
    """
    max_size = width if mode == PathPlanMode.BEAM_SEARCH else 0

    frontier: PriorityQueue[tuple[float, tuple[int, int]]] = PriorityQueue()
    frontier.put((0, start))
    frontier_sizes: list[int] = []
    expanded: list[tuple[int, int]] = []

    # Now using NodeRecord for the reached dictionary
    reached: dict[tuple[int, int], NodeRecord] = {
        start: {"cost": cost(grid, start), "parent": None}
    }
    curr = start
    path: list[tuple[int, int]] = []

    while not frontier.empty():
        frontier_sizes.append(frontier.qsize())
        curr = frontier.get()[1]
        if curr == goal:
            # Reconstruct path from goal back to start
            while curr:
                path.insert(0, curr)
                curr = reached[curr]["parent"]
            break
        else:
            expanded.append(curr)
            for child in expand(grid, curr):
                c = cost(grid, child)
                h = compute_heuristic(child, goal, heuristic)
                g = reached[curr]["cost"] + c
                f = g + h
                if child not in reached or f < reached[child]["cost"]:
                    if max_size == 0 or frontier.qsize() < max_size or child in reached:
                        reached[child] = {"cost": f, "parent": curr}
                        frontier.put((f, child))
                    else:
                        # If beam search is full, update if a cheaper path is found.
                        for i in range(len(frontier.queue)):
                            e = frontier.queue[i]
                            if f < e[0]:
                                reached[child] = {"cost": f, "parent": curr}
                                frontier.queue[i] = (f, child)
                                heapq.heapify(frontier.queue)  # Update the heap
                                break

    return path, expanded, frontier_sizes


def local_search(
    grid: npt.NDArray[np.int_],
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic: Heuristic,
):
    """Find a path from start to goal in the gridworld using
    local search.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.

    Returns:
        path (list): A list of cells from start to goal.
    """

    path = [start]

    # TODO:

    return path


def test_world(world_id, start, goal, h, width, animate, world_dir):  # type: ignore
    print(f"Testing world {world_id}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")

    if h == 0:
        modes = [PathPlanMode.DFS, PathPlanMode.BFS]
        print("Modes: 1. DFS, 2. BFS")
    elif h == 1 or h == 2:
        modes = [PathPlanMode.A_STAR, PathPlanMode.BEAM_SEARCH]
        if h == 1:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Manhattan heuristic")
        else:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Euclidean heuristic")
    elif h == 3 or h == 4:
        h -= 2
        modes = [PathPlanMode.LOCAL_SEARCH]
        if h == 1:
            print("Mode: LOCAL_SEARCH")
            print("Using Manhattan heuristic")
        else:
            print("Mode: LOCAL_SEARCH")
            print("Using Euclidean heuristic")

    for mode in modes:  # type: ignore

        search_type, path, expanded, frontier_size = None, [], [], []
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)  # type: ignore
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)  # type: ignore
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)  # type: ignore
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)  # type: ignore
            search_type = "BEAM_A_STAR"
        elif mode == PathPlanMode.LOCAL_SEARCH:
            path = local_search(grid, start, goal, h)  # type: ignore
            search_type = "LOCAL_SEARCH"

        if search_type:
            print(f"Mode: {search_type}")
            path_cost = 0
            for c in path:
                path_cost += cost(grid, c)
            print(f"Path length: {len(path)}")
            print(f"Path cost: {path_cost}")
            if frontier_size:
                print(f"Number of expanded states: {len(frontier_size)}")
                print(f"Max frontier size: {max(frontier_size)}\n")
            if animate == 0 or animate == 1:
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)  # type: ignore
            else:
                visualize_path(grid, start, goal, path)  # type: ignore
