import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import json
import argparse
import multiprocessing
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from dlgo import kerasutil
from dlgo import scoring
from dlgo.experience import ExperienceCollector, combine_experience, load_experience
from dlgo.agent import Agent, decode_agent
from dlgo.encoder import Encoder
from dlgo.goboard import GameState, Player, Point
from dlgo.utils import print_board


def load_agent(num, work_dir, load_args):
    file_path = os.path.join(work_dir,'agent_%08d.hdf5' % num)
    with h5py.File(file_path, 'r') as h5file:
        agent = decode_agent(h5file)
        agent.set_num_rounds(load_args.nr)
        agent.set_c(load_args.c)
        agent.set_concent_param(load_args.cp)
        agent.set_dirichlet_weight(load_args.dw)
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
        margin=game_result.winning_margin
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(args):
    work_dir, board_size, agent1_num, agent2_num, num_games, experience_filename, gpu_frac, load_args = args
    kerasutil.set_gpu_memory_target(gpu_frac)
    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())
    agent1 = load_agent(agent1_num, work_dir, load_args)
    agent2 = load_agent(agent2_num, work_dir, load_args)
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    ag1b = []
    ag2b = []

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=game_record.margin)
            collector2.complete_episode(reward=-game_record.margin)
        elif game_record.winner == color1.other:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-game_record.margin)
            collector2.complete_episode(reward=game_record.margin)
        else:
            print('Agents play a draw.')
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)
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

    experience = combine_experience([collector1,collector2])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)
    return (ag1b, ag2b)


def generate_experience(work_dir, lear_agent, ref_agent, experience_file, num_games, board_size, num_workers, load_args):
    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        print("filename for worker %d:    %s" % (i,filename))
        experience_files.append(filename)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            work_dir,
            board_size,
            lear_agent,
            ref_agent,
            games_per_worker,
            experience_files[_],
            gpu_frac,
            load_args,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(do_self_play, worker_args)

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    with h5py.File(first_filename, 'r') as expf:
        combined_buffer = load_experience(expf)
    for filename in other_filenames:
        with h5py.File(filename, 'r') as expf:
            next_buffer = load_experience(expf)
        combined_buffer = combine_experience([combined_buffer, next_buffer])

    ag1b = []
    ag2b = []
    for ag1b_work, ag2b_work in game_results:
        ag1b += ag1b_work
        ag2b += ag2b_work
    stats_data = []
    split_work_dir = os.path.split(os.path.abspath(work_dir))
    stats_path = os.path.join(split_work_dir[0],'stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as infile:
            stats_data = json.load(infile)

    stats_data.append({
        'ag_b_dir': split_work_dir[1],
        'ag_b_num': lear_agent,
        'ag_w_dir': split_work_dir[1],
        'ag_w_num': ref_agent,
        'nr_b': load_args.nr,
        'nr_w': load_args.nr,
        'c_b': load_args.c,
        'c_w': load_args.c,
        'cp_b': load_args.cp,
        'cp_w': load_args.cp,
        'dw_b': load_args.dw,
        'dw_w': load_args.dw,
        'scores': ag1b})

    stats_data.append({
        'ag_b_dir': split_work_dir[1],
        'ag_b_num': ref_agent,
        'ag_w_dir': split_work_dir[1],
        'ag_w_num': lear_agent,
        'nr_b': load_args.nr,
        'nr_w': load_args.nr,
        'c_b': load_args.c,
        'c_w': load_args.c,
        'cp_b': load_args.cp,
        'cp_w': load_args.cp,
        'dw_b': load_args.dw,
        'dw_w': load_args.dw,
        'scores': ag2b})

    with open(stats_path,'w') as outfile:
        json.dump(stats_data,outfile,indent=0)

    print('Saving into %s...' % experience_file)
    with h5py.File(experience_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    temp_komi = sum(ag1b+ag2b)/len(ag1b+ag2b)
    ag1b_komi = [x - temp_komi for x in ag1b]
    ag2b_komi = [x - temp_komi for x in ag2b]

    ag1_wins = 0
    for res in ag1b_komi:
        if res>0:
            ag1_wins += 1
    for res in ag2b_komi:
        if res<0:
            ag1_wins += 1

    ag1_win_rate = ag1_wins/len(ag1b+ag2b)

    print('win rate of learning agent %d against reference agent %d: %f' % (lear_agent,ref_agent,ag1_win_rate))
    print('required win rate: %f' % load_args.evfrac)

    if ag1_win_rate>load_args.evfrac:
        new_ref = lear_agent
        print('agent %d is the new reference' % lear_agent)
    else:
        new_ref = ref_agent
        print('agent %d remains the reference' % ref_agent)

    ref_path = os.path.join(work_dir,'references.json')
    ref_data = []
    if os.path.exists(ref_path):
        with open(ref_path) as infile:
            ref_data = json.load(infile)
    ref_data.append({
        'agent': new_ref,
        'new': ag1_win_rate>load_args.evfrac,
        'winrate': ag1_win_rate,
        'threshold': load_args.evfrac,
        'meanscore': temp_komi
        })
    with open(ref_path,'w') as outfile:
        json.dump(ref_data,outfile,indent=0)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)

    pool.close()
    pool.join()

def train_worker(work_dir,lear_agent,next_agent,experience_file,lr,mo,batch_size,policy_loss_weight,epochs,load_args):
    learning_agent = load_agent(lear_agent, work_dir, load_args)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = load_experience(expf)
    learning_agent.train(exp_buffer, learning_rate=lr, momentum=mo, batch_size=batch_size, policy_loss_weight=policy_loss_weight, epochs=epochs)
    with h5py.File(os.path.join(work_dir,'agent_%08d.hdf5' % next_agent), 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)

    ag_path = os.path.join(work_dir,'agents.json')
    ag_data = []
    if os.path.exists(ag_path):
        with open(ag_path) as infile:
            ag_data = json.load(infile)
    ag_data.append({
        'agent': next_agent,
        'lr': lr,
        'mo': mo,
        'bs': batch_size,
        'plw': policy_loss_weight,
        'ep': epochs
        })
    with open(ag_path,'w') as outfile:
        json.dump(ag_data,outfile,indent=0)


def train_on_experience(work_dir,lear_agent,next_agent,experience_file,lr,mo,batch_size,policy_loss_weight,epochs,load_args):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            work_dir,
            lear_agent,
            next_agent,
            experience_file,
            lr,
            mo,
            batch_size,
            policy_loss_weight,
            epochs,
            load_args
        )
    )
    worker.start()
    worker.join()

def parse_cmds(cmdfile, parser):
    global args
    with open(cmdfile) as f:
        cmdline=f.readline()
    args = parser.parse_args(cmdline.split())

def main():
    cmdparser = argparse.ArgumentParser()
    cmdparser.add_argument('--work-dir')

    cmdargs = cmdparser.parse_args()
    cmdfile = os.path.join(cmdargs.work_dir,'options')

    parser = argparse.ArgumentParser()

    parser.add_argument('--games-per-worker-and-color', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--nr', type=int, default=300)
    parser.add_argument('--c', type=float, default=2.0)
    parser.add_argument('--cp', type=float, default=0.03)
    parser.add_argument('--dw', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--mo', type=float, default=0.9)
    parser.add_argument('--bs', type=int, default=2048)
    parser.add_argument('--plw', type=float, default=0.2)
    parser.add_argument('--ep', type=int, default=20)
    parser.add_argument('--evfrac', type=float, default=0.55)
    parser.add_argument('--stop', type=int, default=0)

    parse_cmds(cmdfile, parser)

    with h5py.File(os.path.join(cmdargs.work_dir,'agent_00000000.hdf5'), 'r') as h5file:
        board_size = int(h5file['encoder'].attrs['board_size'])
    print('read the following board size: %d' % board_size)
    experience_file = os.path.join(cmdargs.work_dir, 'exp_temp.hdf5')

    while args.stop==0:
        parse_cmds(cmdfile, parser)

        ag_path = os.path.join(cmdargs.work_dir,'agents.json')
        if os.path.exists(ag_path):
            with open(ag_path) as infile:
                ag_dat = json.load(infile)
                lear_agent = ag_dat[-1]['agent']
        else:
            lear_agent = 0
 
        ref_path = os.path.join(cmdargs.work_dir,'references.json')
        if os.path.exists(ref_path):
            with open(ref_path) as infile:
                ref_dat = json.load(infile)
                ref_agent = ref_dat[-1]['agent']
        else:
            ref_agent = 0

        generate_experience(
            work_dir=cmdargs.work_dir,
            lear_agent=lear_agent,
            ref_agent=ref_agent,
            experience_file=experience_file,
            num_games=2*args.games_per_worker_and_color*args.num_workers,
            board_size=board_size,
            num_workers=args.num_workers,
            load_args=args)

        train_on_experience(
            work_dir=cmdargs.work_dir,
            lear_agent=lear_agent,
            next_agent=lear_agent+2*args.games_per_worker_and_color*args.num_workers,
            experience_file=experience_file,
            lr=args.lr,
            mo=args.mo,
            batch_size=args.bs,
            policy_loss_weight=args.plw,
            epochs=args.ep,
            load_args=args)

        parse_cmds(cmdfile, parser)

if __name__ == '__main__':
    main()
