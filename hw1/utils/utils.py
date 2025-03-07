import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from enum import IntEnum
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from typing import Optional, List, Tuple
from pathlib import Path


class Environment(IntEnum):
    PRAIRIE = 0
    POND = 1
    DESERT = 2
    MOUNTAIN = 3
    EXPANDED = 4


class PathPlanMode(IntEnum):
    DFS = 1
    BFS = 2
    A_STAR = 3
    BEAM_SEARCH = 4
    LOCAL_SEARCH = 5


class Heuristic(IntEnum):
    MANHATTAN = 1
    EUCLIDEAN = 2


def cost(
    grid: npt.NDArray[np.int_],
    point: Tuple[int, int],
) -> float:
    if grid[point] == Environment.PRAIRIE:
        return 1.0
    elif grid[point] == Environment.POND:
        return 2.0
    elif grid[point] == Environment.DESERT:
        return 0.5
    else:
        return np.inf


def expand(
    grid: npt.NDArray[np.int_],
    point: Tuple[int, int],
) -> List[Tuple[int, int]]:
    children: list[tuple[int, int]] = []
    neighbors = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]

    x, y = point
    for i, j in neighbors:
        if (
            x + i >= 0
            and x + i < grid.shape[0]
            and y + j >= 0
            and y + j < grid.shape[1]
            and grid[x + i, y + j] != Environment.MOUNTAIN
        ):
            children.append((x + i, y + j))
    return children


def create_pond(
    grid: npt.NDArray[np.int_],
    center_x: int,
    center_y: int,
    axis_x: int,
    axis_y: int,
) -> npt.NDArray[np.int_]:
    c_x, c_y, a, b = center_x, center_y, axis_x, axis_y
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if ((x - c_x) / a) ** 2 + ((y - c_y) / b) ** 2 < 1:
                grid[int(x), int(y)] = Environment.POND
    return grid


def create_valley(
    grid: npt.NDArray[np.int_],
    center_x: int,
    center_y: int,
    radius: int,
) -> npt.NDArray[np.int_]:
    c_x, c_y, r_2 = center_x, center_y, radius
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2) < r_2:
                grid[int(x), int(y)] = Environment.DESERT
    return grid


def create_mountain(
    grid: npt.NDArray[np.int_],
    lower_x: int,
    upper_x: int,
    lower_y: int,
    upper_y: int,
) -> npt.NDArray[np.int_]:
    grid[lower_x:upper_x, lower_y:upper_y] = Environment.MOUNTAIN
    return grid


def highlight_start_and_end(
    grid: npt.NDArray[np.int_],
    cell: Tuple[int, int],
    val: int,
) -> npt.NDArray[np.int_]:
    c_x, c_y = cell
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2) < 5:
                grid[int(x), int(y)] = val

    return grid


def sample_world_1(
    width: int = 100,
    height: int = 100,
) -> tuple[npt.NDArray[np.int_], tuple[int, int], tuple[int, int]]:
    start = (10, 10)
    goal = (87, 87)
    grid_world = np.zeros((width, height), dtype=np.int_)

    grid_world = create_pond(grid_world, 87, 40, 12, 12)

    grid_world = create_valley(grid_world, 40, 87, 12)

    grid_world = create_mountain(grid_world, 0, 35, 45, 50)
    grid_world = create_mountain(grid_world, 45, 50, 0, 35)
    grid_world = create_mountain(grid_world, 15, 75, 70, 75)
    grid_world = create_mountain(grid_world, 70, 75, 15, 75)

    return grid_world, start, goal


def sample_world_2(
    width: int = 100,
    height: int = 100,
) -> tuple[npt.NDArray[np.int_], tuple[int, int], tuple[int, int]]:
    start = (10, 10)
    goal = (90, 90)
    grid_world = np.zeros((width, height), dtype=np.int_)

    grid_world = create_pond(grid_world, 37, 10, 20, 7)
    grid_world = create_pond(grid_world, 49, 22, 7, 7)
    grid_world = create_pond(grid_world, 50, 22, 7, 7)
    grid_world = create_pond(grid_world, 49, 78, 7, 7)
    grid_world = create_pond(grid_world, 50, 78, 7, 7)
    grid_world = create_pond(grid_world, 63, 90, 20, 7)
    grid_world = create_pond(grid_world, 20, 50, 10, 10)
    grid_world = create_pond(grid_world, 80, 50, 10, 10)

    grid_world = create_valley(grid_world, 50, 50, 20)

    grid_world = create_mountain(grid_world, 20, 40, 20, 25)
    grid_world = create_mountain(grid_world, 60, 100, 20, 25)
    grid_world = create_mountain(grid_world, 0, 40, 75, 80)
    grid_world = create_mountain(grid_world, 60, 80, 75, 80)
    return grid_world, start, goal


def sample_world_3(
    width: int = 100,
    height: int = 100,
) -> tuple[npt.NDArray[np.int_], tuple[int, int], tuple[int, int]]:
    start = (10, 10)
    goal = (90, 90)
    grid_world = np.zeros((width, height), dtype=np.int_)

    grid_world = create_pond(grid_world, 25, 10, 7, 7)
    grid_world = create_pond(grid_world, 30, 10, 7, 7)
    grid_world = create_pond(grid_world, 49, 10, 30, 7)
    grid_world = create_pond(grid_world, 66, 10, 30, 7)
    grid_world = create_pond(grid_world, 85, 10, 7, 7)
    grid_world = create_pond(grid_world, 90, 10, 7, 7)
    grid_world = create_pond(grid_world, 90, 15, 7, 7)
    grid_world = create_pond(grid_world, 90, 34, 7, 30)
    grid_world = create_pond(grid_world, 90, 50, 7, 7)
    grid_world = create_pond(grid_world, 90, 55, 7, 7)
    grid_world = create_pond(grid_world, 90, 60, 7, 7)
    grid_world = create_pond(grid_world, 90, 65, 7, 7)

    grid_world = create_valley(grid_world, 15, 40, 7)
    grid_world = create_valley(grid_world, 15, 60, 7)
    grid_world = create_valley(grid_world, 15, 80, 7)
    grid_world = create_valley(grid_world, 30, 50, 7)
    grid_world = create_valley(grid_world, 30, 70, 7)
    grid_world = create_valley(grid_world, 30, 90, 7)
    grid_world = create_valley(grid_world, 45, 40, 7)
    grid_world = create_valley(grid_world, 45, 60, 7)
    grid_world = create_valley(grid_world, 45, 80, 7)
    grid_world = create_valley(grid_world, 60, 50, 7)
    grid_world = create_valley(grid_world, 60, 70, 7)
    grid_world = create_valley(grid_world, 60, 90, 7)

    grid_world = create_mountain(grid_world, 10, 80, 20, 25)
    grid_world = create_mountain(grid_world, 75, 80, 20, 80)
    grid_world = create_mountain(grid_world, 80, 100, 75, 80)

    return grid_world, start, goal


def sample_world_4(
    width: int = 50,
    height: int = 50,
) -> tuple[npt.NDArray[np.int_], tuple[int, int], tuple[int, int]]:
    start = (24, 24)
    goal = (43, 42)
    grid_world = np.zeros((width, height), dtype=np.int_)

    grid_world = create_pond(grid_world, 7, 42, 7, 7)
    grid_world = create_pond(grid_world, 40, 24, 8, 5)
    grid_world = create_pond(grid_world, 24, 7, 6, 6)
    grid_world = create_pond(grid_world, 7, 24, 7, 5)

    grid_world = create_valley(grid_world, 24, 46, 4)
    grid_world = create_valley(grid_world, 32, 38, 4)
    grid_world = create_valley(grid_world, 7, 7, 6)
    grid_world = create_valley(grid_world, 42, 7, 6)

    grid_world = create_mountain(grid_world, 12, 20, 14, 19)
    grid_world = create_mountain(grid_world, 29, 38, 14, 19)
    grid_world = create_mountain(grid_world, 0, 15, 30, 35)
    grid_world = create_mountain(grid_world, 20, 50, 30, 35)

    return grid_world, start, goal


def visualize_grid_world(grid: npt.NDArray[np.int_]) -> None:

    _, ax = plt.subplots()  # type: ignore
    grid_world = np.copy(grid)

    cmap = ListedColormap(
        [
            "#006600",  # PRAIRIE
            "#4d94ff",  # Pond
            "#FFA500",  # DESERT
            "#333333",  # Mountain
        ]
    )

    ax.imshow(grid_world, cmap=cmap)  # type: ignore
    legend_elements = [
        Patch(facecolor="#006600", label="Prairie"),
        Patch(facecolor="#4d94ff", label="Pond"),
        Patch(facecolor="#FFA500", label="Desert"),
        Patch(facecolor="#333333", label="Mountain"),
    ]

    ax.set_title("Grid World Visualization")  # type: ignore
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4)  # type: ignore
    ax.set_xticklabels([])  # type: ignore
    ax.set_yticklabels([])  # type: ignore
    plt.show()  # type: ignore


def visualize_path(
    grid: npt.NDArray[np.int_],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[Tuple[int, int]],
    blit: bool = False,
) -> None:

    fig, ax = plt.subplots()  # type: ignore
    grid_world = np.copy(grid)

    cmap = ListedColormap(
        [
            "#006600",  # PRAIRIE
            "#4d94ff",  # Pond
            "#FFA500",  # DESERT
            "#333333",  # Mountain
            "#00AA00",  # Start & Goal
        ]
    )

    grid_world = highlight_start_and_end(grid_world, start, cmap.N - 1)
    grid_world = highlight_start_and_end(grid_world, goal, cmap.N - 1)

    ax.imshow(grid_world, cmap=cmap)  # type: ignore

    legend_elements = [
        Patch(facecolor="#006600", label="Prairie"),
        Patch(facecolor="#4d94ff", label="Pond"),
        Patch(facecolor="#FFA500", label="Desert"),
        Patch(facecolor="#333333", label="Mountain"),
    ]

    (path_line,) = ax.plot([], [], color="#FF0000", label="Path")  # type: ignore

    def update_path(frame: int):
        if frame < len(path):
            x, y = zip(*path[: frame + 1])
            path_line.set_data(y, x)
        return (path_line,)

    _ = FuncAnimation(
        fig, update_path, frames=len(path), repeat=False, interval=1, blit=blit
    )
    legend_elements.append(Patch(facecolor="#FF0000", label="Path"))

    ax.set_title(f"Grid World Path Planning Result")  # type: ignore
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=5)  # type: ignore
    ax.set_xticklabels([])  # type: ignore
    ax.set_yticklabels([])  # type: ignore
    plt.show()  # type: ignore


def visualize_expanded(
    grid: npt.NDArray[np.int_],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    expanded: List[Tuple[int, int]],
    path: Optional[Path],
    animation: bool = True,
) -> None:

    fig, ax = plt.subplots()  # type: ignore
    grid_world = np.copy(grid)

    cmap = ListedColormap(
        [
            "#006600",  # PRAIRIE
            "#4d94ff",  # Pond
            "#FFA500",  # DESERT
            "#333333",  # Mountain
            "#86592d",  # Expanded
            "#00AA00",  # Start & Goal
        ]
    )
    color_index = len(cmap.colors) - 1  # type: ignore
    grid_world = highlight_start_and_end(grid_world, start, color_index)
    grid_world = highlight_start_and_end(grid_world, goal, color_index)

    legend_elements = []

    if path:
        path_x, path_y = zip(*path)
        (gw,) = ax.plot(path_y, path_x, color="#FF0000", label="Path")  # type: ignore
        legend_elements.append(Patch(facecolor="#FF0000", label="Path"))  # type: ignore

    # dumb bug fix
    fix_bug = grid_world[0, -1]
    grid_world[0, 3] = 4
    gw = ax.imshow(grid_world, cmap=cmap)  # type: ignore
    grid_world[0, 3] = fix_bug

    legend_elements.extend(  # type: ignore
        [
            Patch(facecolor="#006600", label="Prairie"),
            Patch(facecolor="#4d94ff", label="Pond"),
            Patch(facecolor="#FFA500", label="Desert"),
            Patch(facecolor="#333333", label="Mountain"),
            Patch(facecolor="#86592d", label="Expanded"),
        ]
    )  # type: ignore

    expanded = [s for s in expanded if len(s) > 0]
    all_x, all_y = [], []

    if animation:

        def update_expanded(frame: int):
            if frame < len(expanded):
                expanded_grid_world = np.copy(grid_world)
                x, y = expanded[frame]
                all_x.append(x)  # type: ignore
                all_y.append(y)  # type: ignore
                expanded_grid_world[all_x, all_y] = Environment.EXPANDED
                gw.set_array(expanded_grid_world)
            return [gw]

        _ = FuncAnimation(
            fig,
            update_expanded,
            frames=len(expanded),
            repeat=False,
            interval=1,
        )
    else:
        for s in expanded:
            x, y = s
            all_x.append(x)  # type: ignore
            all_y.append(y)  # type: ignore
        grid_world[all_x, all_y] = Environment.EXPANDED
        gw.set_array(grid_world)

    ax.set_title(f"Grid World Expanded Cells Result")  # type: ignore
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)  # type: ignore

    ax.set_xticklabels([])  # type: ignore
    ax.set_yticklabels([])  # type: ignore
    plt.show()  # type: ignore
