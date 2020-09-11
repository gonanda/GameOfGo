import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import sys
import h5py
import dlgo.goboard as goboard
from dlgo.agent import Agent, decode_agent
from dlgo.gotypes import Player, Point
from dlgo.utils import print_board, print_move, move_string, point_from_coords, move_from_string
from six.moves import input

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--black', default='human')
    parser.add_argument('--black-num-rounds', type=int, default=0)
    parser.add_argument('--black-search-time', type=int, default=0)
    parser.add_argument('--white', default='human')
    parser.add_argument('--white-num-rounds', type=int, default=0)
    parser.add_argument('--white-search-time', type=int, default=0)

    args = parser.parse_args()

    if args.black != 'human' and args.black_num_rounds <= 0 and args.black_search_time <= 0:
        sys.exit('Error: EITHER --black-num-rounds OR --black-search-time must be set to positive integer')

    if args.black != 'human' and args.black_num_rounds > 0 and args.black_search_time > 0:
        sys.exit('Error: EITHER --black-num-rounds OR --black-search-time may be set')

    if args.white != 'human' and args.white_num_rounds <= 0 and args.white_search_time <= 0:
        sys.exit('Error: EITHER --white-num-rounds OR --white-search-time must be set to positive integer')

    if args.black != 'human' and args.white_num_rounds > 0 and args.white_search_time > 0:
        sys.exit('Error: EITHER --white-num-rounds OR --white-search-time may be set')

    def load_agent(filename):
        with h5py.File(filename, 'r') as h5file:
            return decode_agent(h5file)

    blackAgent = None
    whiteAgent = None

    board_size = args.board_size

    if args.black != 'human':
        blackAgent = load_agent(args.black)
        blackAgent.set_concent_param(0)
        if blackAgent.encoder.board_size != board_size:
            sys.exit('Error: Agent provided for the black player has wrong board size (%d instead of %d)' % (blackAgent.encoder.board_size,board_size))
        if args.black_num_rounds > 0:
            blackAgent.set_num_rounds(args.black_num_rounds)
        else:
            blackAgent.set_search_time(args.black_search_time)

    if args.white != 'human':
        whiteAgent = load_agent(args.white)
        whiteAgent.set_concent_param(0)
        if whiteAgent.encoder.board_size != board_size:
            sys.exit('Error: Agent provided for the white player has wrong board size (%d instead of %d)' % (whiteAgent.encoder.board_size,board_size))
        if args.white_num_rounds > 0:
            whiteAgent.set_num_rounds(args.white_num_rounds)
        else:
            whiteAgent.set_search_time(args.white_search_time)

    agents = {Player.black: blackAgent, Player.white: whiteAgent}

    game = goboard.GameState.new_game(board_size)

    turn = 0
    move = None


    move_message = 'new game'

    while not (game.is_over()):

        print(chr(27) + "[2J")
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print('turn %s:' % turn)        
        print_board(game.board)

        print(move_message)

        if agents[game.next_player] is None:
            human_move = input('%s player enter move: ' % game.next_player)
            move = move_from_string(string=human_move.strip(), num_rows=game.board.num_rows, num_cols=game.board.num_cols)
        else:
            possMoves = len(game.sensible_legal_moves())
            move = agents[game.next_player].select_move(game)

        if move is None:
            move_message = 'invalid input'
        elif game.is_valid_move(move):
            game = game.apply_move(move)
            move_message = move_string(game.next_player.other, move)
            turn += 1
        else:
            move_message = 'illegal move'

    print(chr(27) + "[2J")
    print('turn %s:' % turn)        
    print_board(game.board)
    print(move_message)
    print(goboard.compute_game_result(game))

if __name__ == '__main__':
    main()
