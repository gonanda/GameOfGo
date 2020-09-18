import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


import argparse
import datetime
import multiprocessing
import random
import shutil
import time
from collections import namedtuple

import h5py
import numpy as np

from dlgo import kerasutil
from dlgo import scoring
from dlgo.agent import Agent, decode_agent
from dlgo.encoder import Encoder
from dlgo.goboard import GameState, Player, Point
from dlgo.utils import print_board


def load_agent(agent_num, load_args):
    if agent_num == 1:
        with h5py.File(load_args.agent1, 'r') as h5file:
            agent = decode_agent(h5file)
            agent.set_num_rounds(load_args.num_rounds1)
            agent.set_c(load_args.c1)
            agent.set_concent_param(load_args.concent_param1)
            agent.set_dirichlet_weight(load_args.dirichlet_weight1)
    else:
        with h5py.File(load_args.agent2, 'r') as h5file:
            agent = decode_agent(h5file)
            agent.set_num_rounds(load_args.num_rounds2)
            agent.set_c(load_args.c2)
            agent.set_concent_param(load_args.concent_param2)
            agent.set_dirichlet_weight(load_args.dirichlet_weight2)
    return agent

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

def play_games(args):
    num_games, board_size, gpu_frac, load_args = args

    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(1, load_args)
    agent2 = load_agent(2, load_args)

    wins1, wins1b, wins1w, wins2, wins2b, wins2w = 0, 0, 0, 0, 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins1 += 1
            if color1 == Player.black:
                wins1b += 1
            else:
                wins1w += 1
        else:
            print('Agent 2 wins')
            wins2 += 1
            if color1 == Player.black:
                wins2w += 1
            else:
                wins2b += 1
        print('Agent 1 record: %d/%d' % (wins1, wins1 + wins2))
        print('Agent 2 record: %d/%d' % (wins2, wins1 + wins2))
        color1 = color1.other
    return wins1, wins1b, wins1w, wins2, wins2b, wins2w


def evaluate(num_games, num_workers, board_size, load_args):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            games_per_worker, board_size, gpu_frac, load_args
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    t_wins1, t_wins1b, t_wins1w, t_wins2, t_wins2b, t_wins2w = 0, 0, 0, 0, 0, 0
    for wins1, wins1b, wins1w, wins2, wins2b, wins2w in game_results:
        t_wins1 += wins1
        t_wins1b += wins1b
        t_wins1w += wins1w
        t_wins2 += wins2
        t_wins2b += wins2b
        t_wins2w += wins2w
    print('FINAL RESULTS:')
    print('Agent1 total: %d/%d (%.3f)' % (t_wins1,2*load_args.games_per_color,float(t_wins1)/float(2*load_args.games_per_color)))
    print('Agent1 as black: %d/%d (%.3f)' % (t_wins1b,load_args.games_per_color,float(t_wins1b)/float(load_args.games_per_color)))
    print('Agent1 as white: %d/%d (%.3f)' % (t_wins1w,load_args.games_per_color,float(t_wins1w)/float(load_args.games_per_color)))
    print('Agent2 total: %d/%d (%.3f)' % (t_wins2,2*load_args.games_per_color,float(t_wins2)/float(2*load_args.games_per_color)))
    print('Agent2 as black: %d/%d (%.3f)' % (t_wins2b,load_args.games_per_color,float(t_wins2b)/float(load_args.games_per_color)))
    print('Agent2 as white: %d/%d (%.3f)' % (t_wins2w,load_args.games_per_color,float(t_wins2w)/float(load_args.games_per_color)))
    print('Black total: %d/%d (%.3f)' % (t_wins1b+t_wins2b,2*load_args.games_per_color,float(t_wins1b+t_wins2b)/float(2*load_args.games_per_color)))
    print('White total: %d/%d (%.3f)' % (t_wins1w+t_wins2w,2*load_args.games_per_color,float(t_wins1w+t_wins2w)/float(2*load_args.games_per_color)))
    pool.close()
    pool.join()
    return  t_wins1, t_wins1b, t_wins1w, t_wins2, t_wins2b, t_wins2w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num-rounds1', type=int, default=300)
    parser.add_argument('--num-rounds2', type=int, default=300)
    parser.add_argument('--c1', type=float, default=2.0)
    parser.add_argument('--c2', type=float, default=2.0)
    parser.add_argument('--concent-param1', type=float, default=0.03)
    parser.add_argument('--concent-param2', type=float, default=0.03)
    parser.add_argument('--dirichlet-weight1', type=float, default=0.5)
    parser.add_argument('--dirichlet-weight2', type=float, default=0.5)
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--games-per-color', type=int, default=64)
    parser.add_argument('--log-file', required=True)

    args = parser.parse_args() 

    agent1 = args.agent1
    agent2 = args.agent2

    t_wins1, t_wins1b, t_wins1w, t_wins2, t_wins2b, t_wins2w = evaluate(
        num_games=2*args.games_per_color,
        num_workers=args.num_workers,
        board_size=args.board_size,
        load_args=args)

    with open(args.log_file, 'a') as logf:
        logf.seek(0,2)
        if logf.tell()==0:
            logf.write('agent1,agent2,nr1,nr2,c1,c2,cp1,cp2,dw1,dw2,a1b,a1w,a2b,a2w\n')
        logf.write('%s,%s,%d,%d,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d\n' % (args.agent1,args.agent2,args.num_rounds1,args.num_rounds2,args.c1,args.c2,args.concent_param1,args.concent_param2,args.dirichlet_weight1,args.dirichlet_weight2,t_wins1b,t_wins1w,t_wins2b,t_wins2w))
   

if __name__ == '__main__':
    main()
