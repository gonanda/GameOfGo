from __future__ import absolute_import
from collections import namedtuple
import copy
from dlgo.gotypes import Player, Point

class Territory:
    def __init__(self, territory_map):
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == 'territory_b':
                self.num_black_territory += 1
            elif status == 'territory_w':
                self.num_white_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)


class GameResult(namedtuple('GameResult', 'b w')):
    @property
    def winner(self):
        if self.b > self.w:
            return Player.black
        elif self.w > self.b:
            return Player.white
        return None

    @property
    def winning_margin(self):
        return abs(self.b - self.w)

    def __str__(self):
        if self.b > self.w:
            return 'B+%d' % (self.b - self.w)
        elif self.w > self.b:
            return 'W+%d' % (self.w - self.b)
        return 'DRAW'



def evaluate_territory(board):
    new_board = copy.deepcopy(board)
    new_board.find_safe_strings_and_vital_regions()
    for r in range(1, new_board.num_rows + 1):
        for c in range(1, new_board.num_cols + 1):
            p = Point(row=r, col=c)
            if new_board.get(p) is not None:
                point_color = new_board.get(p)
                point_other_region = new_board.read_region_by_point(point_color.other,p)
                if point_other_region in new_board._safe_strings_by_region[point_color.other]:
                    new_board._remove_string(new_board.get_go_string(p))

    status = {}
    for r in range(1, new_board.num_rows + 1):
        for c in range(1, new_board.num_cols + 1):
            p = Point(row=r, col=c)
            if p in status:
                continue
            stone = new_board.get(p)
            if stone is not None:
                status[p] = new_board.get(p)
            else:
                group, neighbors = _collect_region(p, new_board)
                if len(neighbors) == 1:
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = 'territory_' + stone_str
                else:
                    fill_with = 'dame'
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def _collect_region(start_pos, board, visited=None):

    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]
    all_borders = set()
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r, col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, borders = _collect_region(next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(neighbor)
    return all_points, all_borders


def compute_game_result(game_state):
    territory = evaluate_territory(game_state.board)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones)
