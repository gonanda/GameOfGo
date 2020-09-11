from .gotypes import Player, Point
from .goboard import Move
import numpy as np

COLS = 'ABCDEFGHJKLMNOPQRSTabcdefghjklmnopqrst'
STONE_TO_CHAR = {
    None: ' . ',
    Player.white: ' o ',
    Player.black: ' x ',
}


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)

    print('%s %s' % (player, move_str))

def move_string(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)

    return ('%s %s' % (player, move_str))

def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))

def point_from_coords(coords):
    col = COLS.index(coords[0])%19 + 1
    row = int(coords[1:])
    return Point(row=row, col=col)

def move_from_string(string, num_rows, num_cols):
    if string=='p' or string=='pass':
        return Move.pass_turn()
    elif string=='r' or string=='resign':
        return Move.resign()
    elif len(string)<2 or len(string)>3:
        return None

    if string[0] in COLS:
        col = COLS.index(string[0])%19 + 1
    else:
        return None

    try:
        row = int(string[1:])
    except ValueError:
        return None

    if 0<row<num_rows+1 and 0<col<num_cols+1:
        return Move.play(Point(col=col, row=row))
    else:
        return None


    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1
        
