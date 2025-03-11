def calc_ttt_score(x_coords, o_coords, tile_scores):
    """
    Calculates the final score difference as (O's score - X's score), with an additional
    penalty if a player has three in a row (a winning line).

    Coordinates and tile_scores are 0-indexed.

    :param x_coords: list of (row, col) tuples where X is placed
    :param o_coords: list of (row, col) tuples where O is placed
    :param tile_scores: list of (((row, col), score)) pairs for all 9 squares
    :return: (int) the final score difference, after penalties.
    """
    # Build a lookup dictionary for tile scores.
    tile_dict = {position: score for (position, score) in tile_scores}

    # Calculate the base scores.
    x_score = sum(tile_dict[pos] for pos in x_coords)
    print("x_score", x_score)
    o_score = sum(tile_dict[pos] for pos in o_coords)
    print("o_score", o_score)

    winning_lines = [
        [(0, 0), (0, 1), (0, 2)],  # Row 0
        [(1, 0), (1, 1), (1, 2)],  # Row 1
        [(2, 0), (2, 1), (2, 2)],  # Row 2
        [(0, 0), (1, 0), (2, 0)],  # Col 0
        [(0, 1), (1, 1), (2, 1)],  # Col 1
        [(0, 2), (1, 2), (2, 2)],  # Col 2
        [(0, 0), (1, 1), (2, 2)],  # Diagonal \
        [(0, 2), (1, 1), (2, 0)],  # Diagonal /
    ]

    # Check whether X or O fully occupies any winning line.
    x_set = set(x_coords)
    o_set = set(o_coords)

    x_win = any(set(line).issubset(x_set) for line in winning_lines)
    o_win = any(set(line).issubset(o_set) for line in winning_lines)

    # Apply the penalty:
    if x_win:
        x_score += 3
    if o_win:
        o_score += 3

    return o_score - x_score


# --------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    x_positions = [
        [(0, 0), (1, 0), (2, 2), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (2, 2), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 2), (1, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 2), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 2), (1, 1)],
    ]
    o_positions = [
        [(2, 0), (2, 1), (1, 2), (0, 2)],
        [(2, 0), (2, 1), (1, 2), (1, 1)],
        [(2, 0), (2, 1), (1, 2), (0, 1)],
        [(2, 0), (2, 1), (1, 2), (1, 1)],
        [(2, 0), (2, 1), (1, 2)],
    ]
    scores = [
        ((0, 0), 2),
        ((0, 1), 5),
        ((0, 2), 2),
        ((1, 0), 5),
        ((1, 1), 1),
        ((1, 2), 5),
        ((2, 0), 2),
        ((2, 1), 5),
        ((2, 2), 2),
    ]

    for i in range(len(x_positions)):
        result = calc_ttt_score(x_positions[i], o_positions[i], scores)
        print(f"Game {i}, Final score = {result}")
