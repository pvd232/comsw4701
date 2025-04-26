import  heapq 
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
from math import sqrt
from collections import deque


def compute_heuristic(node, goal, heuristic: Heuristic):
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
        return (abs(goal[0] - node[0]) + abs(goal[1] - node[1])) * .5

    else:
        return sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2) * (sqrt(2) / 4)


def uninformed_search(grid, start, goal, mode: PathPlanMode):
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
    curr = start
    frontier = deque([curr])
    frontier_sizes = []
    expanded = []
    reached = {curr: None}
    while frontier and curr != goal:
        frontier_sizes.append(len(frontier))
        curr = frontier.pop() if mode == PathPlanMode.DFS else frontier.popleft()
        expanded.append(curr)
        for child in expand(grid, curr):
            if child not in reached and grid[child] != Environment.MOUNTAIN:
                reached[child] = curr
                if mode == PathPlanMode.DFS:
                    frontier.append(child)
                else:
                    frontier.append(child)
    path = []
    if curr == goal:
        while curr:
            path.append(curr)
            curr = reached[curr]
        path.reverse()
    return path, expanded, frontier_sizes


def a_star(grid, start, goal, mode: PathPlanMode, heuristic: Heuristic, width):
    """Performs A* search or beam search to find the
    shortest path from start to goal in the gridworld.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.A_STAR or
        PathPlanMode.BEAM_SEARCH.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        width (int): The width of the beam search. This should
        only be used if mode is PathPlanMode.BEAM_SEARCH.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    reached = {start: {"cost": 0, "parent": None}}
    frontier_sizes, expanded = [], []

    while frontier:
        frontier_sizes.append(len(frontier))
        f_curr, curr = heapq.heappop(frontier)
        expanded.append(curr)

        if curr == goal:
            break

        for child in expand(grid, curr):
            if grid[child] == Environment.MOUNTAIN:
                continue

            g = reached[curr]["cost"] + cost(grid, child)
            h = compute_heuristic(child, goal, heuristic)
            f = g + h

            if child in reached and g >= reached[child]["cost"]:
                continue

            reached[child] = {"cost": g, "parent": curr}
            heapq.heappush(frontier, (f, child))

        if mode == PathPlanMode.BEAM_SEARCH and width > 0 and len(frontier) > width:
            frontier = heapq.nsmallest(width, frontier)
            heapq.heapify(frontier)

    path = []
    if curr == goal:
        while curr:
            path.append(curr)
            curr = reached[curr]["parent"]
        path.reverse()
    return path, expanded, frontier_sizes


def local_search(grid, start, goal, heuristic: Heuristic):
    path = [start]
    curr = start

    while curr != goal:
        h_curr = compute_heuristic(curr, goal, heuristic)

        best_child = None
        best_h = h_curr

        for child in expand(grid, curr):
            if grid[child] != Environment.MOUNTAIN:
                h_child = compute_heuristic(child, goal, heuristic)
                if h_child < best_h: 
                    best_child, best_h = child, h_child

        if best_child is None:  
            return []

        curr = best_child
        path.append(curr)

    return path


def test_world(world_id, start, goal, h, width, animate, world_dir):
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

    for mode in modes:

        search_type, path, expanded, frontier_size = None, [], [], []
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)
            search_type = "BEAM_A_STAR"
        elif mode == PathPlanMode.LOCAL_SEARCH:
            path = local_search(grid, start, goal, h)
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
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)
            else:
                visualize_path(grid, start, goal, path)
