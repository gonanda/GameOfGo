import enum
from collections import namedtuple

COLS = 'ABCDEFGHJKLMNOPQRST'

class Player(enum.Enum):

    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

    def __str__(self):
        if self == Player.black:
            player_string = 'black'
        else:
            player_string = 'white'
        return '%s' % player_string

    __repr__ = __str__


class Point(namedtuple('Point', 'row col')):

    def neighbors(self):
        return [
            Point(self.row - 1,self.col),
            Point(self.row + 1,self.col),
            Point(self.row,self.col - 1),
            Point(self.row,self.col + 1),
        ]

    def __str__(self):
        return '%s%d' % (COLS[self.col - 1], self.row)

    __repr__ = __str__
