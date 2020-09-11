import numpy as np

from .goboard import Move
from .gotypes import Player, Point


class Encoder:
    def __init__(self, board_size, params=(4,1,1,1,1)):
        self.board_size = board_size
        self.params = params
        self.num_lib = params[0]
        self.num_flags = 0
        if self.params[1]>0:
            self.pss_flag = True
            self.num_flags += 1
        else:
            self.pss_flag = False
        if self.params[2]>0:
            self.ss_flag = True
            self.num_flags += 1
        else:
            self.ss_flag = False
        if self.params[3]>0:
            self.hp_flag = True
            self.num_flags += 1
        else:
            self.hp_flag = False
        if self.params[4]>0:
            self.vp_flag = True
            self.num_flags += 1
        else:
            self.vp_flag = False
        self.num_planes = 3 + 2*self.num_lib + 2*self.num_flags

# new layout
# 0. move would be illegal due to ko
# 1. if we get komi
# 2. - n + 1. our stones with 1, 2, ... n+ liberties
# n + 2. our pss
# n + 3. our ss
# n + 4. our hp
# n + 5. our vp
# starting from n + 2. to n + 6. (depending on chosen planes) same for other player
# required parameters: number of distinct liberties, flags for pss, ps, hp, vp
# resulting number of planes 3 + 2*num_lib + 2*num_flags

    def pred_ss(self): #positive flag predecessors of safe stones plane
        if self.pss_flag:
            return 1
        else:
            return 0

    def pred_hp(self): #positive flag predecessors of healthy points plane
        if self.ss_flag:
            return self.pred_ss()+1
        else:
            return self.pred_ss()

    def pred_vp(self): #positive flag predecessors of vital points plane
        if self.hp_flag:
            return self.pred_hp()+1
        else:
            return self.pred_hp()

    def our_komi(self):
        return 1

    def our_lib(self, n):
        return 1 + n

    def our_pss(self):
        if self.pss_flag:
            return self.num_lib + 2
        else:
            return None

    def our_ss(self):
        if self.ss_flag:
            return self.num_lib + 2 + self.pred_ss()
        else:
            return None

    def our_hp(self):
        if self.hp_flag:
            return self.num_lib + 2 + self.pred_hp()
        else:
            return None

    def our_vp(self):
        if self.hp_flag:
            return self.num_lib + 2 + self.pred_vp()
        else:
            return None

    def their_komi(self):
        return self.num_lib + self.num_flags + 2

    def their_lib(self, n):
        return self.num_lib + self.num_flags + 2 + n

    def their_pss(self):
        if self.pss_flag:
            return 2*self.num_lib + self.num_flags + 3
        else:
            return None

    def their_ss(self):
        if self.ss_flag:
            return 2*self.num_lib + self.num_flags + 3 + self.pred_ss()
        else:
            return None

    def their_hp(self):
        if self.hp_flag:
            return 2*self.num_lib + self.num_flags + 3 + self.pred_hp()
        else:
            return None

    def their_vp(self):
        if self.vp_flag:
            return 2*self.num_lib + self.num_flags + 3 + self.pred_vp()
        else:
            return None

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player

        if self.pss_flag:
            ps_str = {Player.black: set().union(*game_state.board._potentially_safe_strings_by_region[Player.black].values()),
                    Player.white: set().union(*game_state.board._potentially_safe_strings_by_region[Player.white].values())}
        if self.ss_flag:
            s_str = {Player.black: set().union(*game_state.board._safe_strings_by_region[Player.black].values()),
                    Player.white: set().union(*game_state.board._safe_strings_by_region[Player.white].values())}
        if self.hp_flag:
            h_reg = {Player.black: set().union(*game_state.board._healthy_regions_by_string[Player.black].values()),
                    Player.white: set().union(*game_state.board._healthy_regions_by_string[Player.white].values())}
        if self.vp_flag:
            v_reg = {Player.black: set().union(*game_state.board._vital_regions_by_string[Player.black].values()),
                    Player.white: set().union(*game_state.board._vital_regions_by_string[Player.white].values())}


        if game_state.next_player == Player.white:
            board_tensor[self.our_komi()] = 1
        else:
            board_tensor[self.their_komi()] = 1
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(next_player,Move.play(p)):
                        board_tensor[0][r][c] = 1
                    if self.hp_flag:
                        if game_state.board._region_by_point_black.get(p) in  h_reg[Player.black]:
                            if next_player==Player.black:
                                board_tensor[self.our_hp()][r][c] = 1
                            else:
                                board_tensor[self.their_hp()][r][c] = 1
                        if game_state.board._region_by_point_white.get(p) in  h_reg[Player.white]:
                            if next_player==Player.white:
                                board_tensor[self.our_hp()][r][c] = 1
                            else:
                                board_tensor[self.their_hp()][r][c] = 1
                    if self.vp_flag:
                        if game_state.board._region_by_point_black.get(p) in  v_reg[Player.black]:
                            if next_player==Player.black:
                                board_tensor[self.our_vp()][r][c] = 1
                            else:
                                board_tensor[self.their_vp()][r][c] = 1
                        if game_state.board._region_by_point_white.get(p) in  v_reg[Player.white]:
                            if next_player==Player.white:
                                board_tensor[self.our_vp()][r][c] = 1
                            else:
                                board_tensor[self.their_vp()][r][c] = 1

                else:
                    liberty_plane = min(self.num_lib, game_state.board.num_liberties(go_string)) + 1
                    if go_string.color != next_player:
                        liberty_plane += self.num_lib + self.num_flags + 1
                    board_tensor[liberty_plane][r][c] = 1
                    if go_string.color != next_player:
                        if go_string.color==Player.white:
                            if self.hp_flag:
                                if game_state.board._region_by_point_black.get(p) in  h_reg[Player.black]:
                                    board_tensor[self.our_hp()][r][c] = 1
                            if self.vp_flag:
                                if game_state.board._region_by_point_black.get(p) in  v_reg[Player.black]:
                                    board_tensor[self.our_vp()][r][c] = 1
                        else:
                            if self.hp_flag:
                                if game_state.board._region_by_point_white.get(p) in  h_reg[Player.white]:
                                    board_tensor[self.our_hp()][r][c] = 1
                            if self.vp_flag:
                                if game_state.board._region_by_point_white.get(p) in  v_reg[Player.white]:
                                    board_tensor[self.our_vp()][r][c] = 1
                        if self.pss_flag:
                            if go_string in ps_str[go_string.color]:
                                board_tensor[self.their_pss()][r][c] = 1
                        if self.ss_flag:
                            if go_string in s_str[go_string.color]:
                                board_tensor[self.their_ss()][r][c] = 1
                    else:
                        if go_string.color==Player.white:
                            if self.hp_flag:
                                if game_state.board._region_by_point_black.get(p) in  h_reg[Player.black]:
                                    board_tensor[self.their_hp()][r][c] = 1
                            if self.vp_flag:
                                if game_state.board._region_by_point_black.get(p) in  v_reg[Player.black]:
                                    board_tensor[self.their_vp()][r][c] = 1
                        else:
                            if self.hp_flag:
                                if game_state.board._region_by_point_white.get(p) in  h_reg[Player.white]:
                                    board_tensor[self.their_hp()][r][c] = 1
                            if self.vp_flag:
                                if game_state.board._region_by_point_white.get(p) in  v_reg[Player.white]:
                                    board_tensor[self.their_vp()][r][c] = 1
                        if self.pss_flag:
                            if go_string in ps_str[go_string.color]:
                                board_tensor[self.our_pss()][r][c] = 1
                        if self.ss_flag:
                            if go_string in s_str[go_string.color]:
                                board_tensor[self.our_ss()][r][c] = 1
        return board_tensor

    def encode_move(self, move):
        if move.is_play:
            return (self.board_size * (move.point.row - 1) + (move.point.col - 1))
        elif move.is_pass:
            return self.board_size * self.board_size
        raise ValueError('Cannot encode resign move')

    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row + 1, col=col + 1))

    def num_moves(self):
        return self.board_size * self.board_size + 1

    def shape(self):
        return self.num_planes, self.board_size, self.board_size


def create(board_size, params):
    return Encoder(board_size, params)

