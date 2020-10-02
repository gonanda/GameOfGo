import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


import argparse
import json
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


def load_agent(ag_num, num, work_dir, load_args):
    file_path = os.path.join(work_dir,'agent_%08d.hdf5' % num)
    with h5py.File(file_path, 'r') as h5file:
        agent = decode_agent(h5file)
        if ag_num == 1:
            agent.set_num_rounds(load_args.nr1)
            agent.set_c(load_args.c1)
            agent.set_concent_param(load_args.cp1)
            agent.set_dirichlet_weight(load_args.dw1)
        else:
            agent.set_num_rounds(load_args.nr2)
            agent.set_c(load_args.c2)
            agent.set_concent_param(load_args.cp2)
            agent.set_dirichlet_weight(load_args.dw2)
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

    agent1 = load_agent(1, load_args.agent1_num, load_args.agent1_dir, load_args)
    agent2 = load_agent(2, load_args.agent2_num, load_args.agent2_dir, load_args)

    ag1b = []
    ag2b = []

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins.')
        elif game_record.winner == color1.other:
            print('Agent 2 wins.')
        else:
            print('Agents play a draw.')
        if game_record.winner==Player.black:
            black_score = game_record.margin
        elif game_record.winner==Player.white:
            black_score = -game_record.margin
        else:
            black_score = 0
        if color1 == Player.black:
            ag1b.append(black_score)
        else:
            ag2b.append(black_score)
        color1 = color1.other

    return (ag1b, ag2b)


def evaluate(num_games, num_workers, board_size, load_args):
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            games_per_worker,
            board_size,
            gpu_frac,
            load_args
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    ag1b = []
    ag2b = []
    for ag1b_work, ag2b_work in game_results:
        ag1b += ag1b_work
        ag2b += ag2b_work
    stats_data = []
    split_work_dir1 = os.path.split(os.path.abspath(load_args.agent1_dir))
    split_work_dir2 = os.path.split(os.path.abspath(load_args.agent2_dir))
    stats_path = os.path.join(split_work_dir1[0],'stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as infile:
            stats_data = json.load(infile)

    stats_data.append({
        'ag_b_dir': split_work_dir1[1],
        'ag_b_num': load_args.agent1_num,
        'ag_w_dir': split_work_dir2[1],
        'ag_w_num': load_args.agent2_num,
        'nr_b': load_args.nr1,
        'nr_w': load_args.nr2,
        'c_b': load_args.c1,
        'c_w': load_args.c2,
        'cp_b': load_args.cp1,
        'cp_w': load_args.cp2,
        'dw_b': load_args.dw1,
        'dw_w': load_args.dw2,
        'scores': ag1b})

    stats_data.append({
        'ag_b_dir': split_work_dir2[1],
        'ag_b_num': load_args.agent2_num,
        'ag_w_dir': split_work_dir1[1],
        'ag_w_num': load_args.agent1_num,
        'nr_b': load_args.nr2,
        'nr_w': load_args.nr1,
        'c_b': load_args.c2,
        'c_w': load_args.c1,
        'cp_b': load_args.cp2,
        'cp_w': load_args.cp1,
        'dw_b': load_args.dw2,
        'dw_w': load_args.dw1,
        'scores': ag2b})

    with open(stats_path,'w') as outfile:
        json.dump(stats_data,outfile,indent=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-per-worker-and-color', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--agent1-dir', required=True)
    parser.add_argument('--agent2-dir', required=True)
    parser.add_argument('--agent1-num', type=int, required=True)
    parser.add_argument('--agent2-num', type=int, required=True)
    parser.add_argument('--nr1', type=int, default=100)
    parser.add_argument('--nr2', type=int, default=100)
    parser.add_argument('--c1', type=float, default=2.0)
    parser.add_argument('--c2', type=float, default=2.0)
    parser.add_argument('--cp1', type=float, default=0.03)
    parser.add_argument('--cp2', type=float, default=0.03)
    parser.add_argument('--dw1', type=float, default=0.3)
    parser.add_argument('--dw2', type=float, default=0.3)

    args = parser.parse_args() 


    with h5py.File(os.path.join(args.agent1_dir,'agent_00000000.hdf5'), 'r') as h5file:
        board_size = int(h5file['encoder'].attrs['board_size'])
    print('read the following board size: %d' % board_size)


    evaluate(
        num_games=2*args.games_per_worker_and_color*args.num_workers,
        num_workers=args.num_workers,
        board_size=board_size,
        load_args=args)


if __name__ == '__main__':
    main()
