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


def load_agent(filename, load_args):
    with h5py.File(filename, 'r') as h5file:
        agent = decode_agent(h5file)
        agent.set_num_rounds(load_args.num_rounds)
        agent.set_c(load_args.c)
        agent.set_concent_param(load_args.concent_param)
        agent.set_dirichlet_weight(load_args.dirichlet_weight)
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


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent1_filename, agent2_filename, num_games, experience_filename, gpu_frac, load_args):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename, load_args)
    agent2 = load_agent(agent2_filename, load_args)

    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

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
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
        color1 = color1.other

    experience = combine_experience([collector1,collector2])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)


def generate_experience(learning_agent, reference_agent, experience_file, num_games, board_size, num_workers, load_args):
    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        print("filename for worker %d:    %s" % (i,filename))
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                games_per_worker,
                filename,
                gpu_frac,
                load_args
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()

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
    print('Saving into %s...' % experience_file)
    with h5py.File(experience_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file, lr, mo, batch_size, policy_loss_weight, epochs, load_args):
    learning_agent = load_agent(learning_agent, load_args)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = load_experience(expf)
    learning_agent.train(exp_buffer, learning_rate=lr, momentum=mo, batch_size=batch_size, policy_loss_weight=policy_loss_weight, epochs=epochs)
    with h5py.File(output_file, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


def train_on_experience(learning_agent, output_file, experience_file, lr, mo, batch_size, policy_loss_weight, epochs, load_args):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            learning_agent,
            output_file,
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


def play_games(args):
    agent1_fname, agent2_fname, num_games, board_size, gpu_frac, load_args = args

    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname, load_args)
    agent2 = load_agent(agent2_fname, load_args)

    wins, losses = 0, 0
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
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses


def evaluate(learning_agent, reference_agent, num_games, num_workers, board_size, load_args):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac, load_args
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins

def parse_cmds(cmdfile, parser):
    global args
    with open(cmdfile) as f:
        cmdline=f.readline()
    args = parser.parse_args(cmdline.split())

def main():
    cmdparser = argparse.ArgumentParser()
    cmdparser.add_argument('--lrnagent', required=True)
    cmdparser.add_argument('--refagent', required=True)
    cmdparser.add_argument('--options', required=True)
    cmdparser.add_argument('--work-dir', '-d')
    cmdparser.add_argument('--log-file', required=True)
    cmdparser.add_argument('--board-size', '-b', type=int, default=19)
    cmdparser.add_argument('--offset', type=int, default=0)

    cmdargs = cmdparser.parse_args()
    cmdfile = os.path.join(cmdargs.work_dir,cmdargs.options)

    parser = argparse.ArgumentParser()

    parser.add_argument('--games-per-batch', '-g', type=int, default=1000)
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--num-rounds', type=int, default=300)
    parser.add_argument('--c', type=float, default=2.0)
    parser.add_argument('--concent-param', type=float, default=0.03)
    parser.add_argument('--dirichlet-weight', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--mo', type=float, default=0.)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--policy-loss-weight', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--evnum', type=int, default=96)
    parser.add_argument('--evthres', type=int, default=60)
    parser.add_argument('--stop', type=int, default=0)

    parse_cmds(cmdfile, parser)

    logf = open(cmdargs.log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Starting with learner %s  and reference %s at %s\n' % (cmdargs.lrnagent, cmdargs.refagent, datetime.datetime.now()))

    learning_agent = os.path.join(cmdargs.work_dir,cmdargs.lrnagent)
    reference_agent = os.path.join(cmdargs.work_dir,cmdargs.refagent)
    experience_file = os.path.join(cmdargs.work_dir, 'exp_temp.hdf5')
    tmp_agent = os.path.join(cmdargs.work_dir, 'agent_temp.hdf5')
    working_agent = os.path.join(cmdargs.work_dir, 'agent_cur.hdf5')
    total_games = 0
    while args.stop==0:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n' % (total_games,))
        logf.flush()
        parse_cmds(cmdfile, parser)
        generate_experience(
            learning_agent=learning_agent,
            reference_agent=reference_agent,
            experience_file=experience_file,
            num_games=args.games_per_batch,
            board_size=cmdargs.board_size,
            num_workers=args.num_workers,
            load_args=args)
        total_games += args.games_per_batch
        parse_cmds(cmdfile, parser)
        train_on_experience(
            learning_agent=learning_agent,
            output_file=tmp_agent,
            experience_file=experience_file,
            lr=args.lr,
            mo=args.mo,
            batch_size=args.bs,
            policy_loss_weight=args.policy_loss_weight,
            epochs=args.epochs,
            load_args=args)
        parse_cmds(cmdfile, parser)
        wins = evaluate(
            learning_agent=tmp_agent,
            reference_agent=reference_agent,
            num_games=args.evnum,
            num_workers=args.num_workers,
            board_size=cmdargs.board_size,
            load_args=args)
        print('Won %d / %d games (%.3f)' % (
            wins, args.evnum, float(wins) / float(args.evnum)))
        logf.write('Won %d / %d games (%.3f)\n' % (
            wins, args.evnum, float(wins) / float(args.evnum)))
        shutil.copy(tmp_agent, working_agent)
        learning_agent = working_agent
        if wins >= args.evthres:
            next_filename = os.path.join(cmdargs.work_dir,'agent_%08d.hdf5' % (total_games+cmdargs.offset))
            shutil.move(tmp_agent, next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n' % next_filename)
        else:
            print('Keep learning\n')
        logf.flush()
        parse_cmds(cmdfile, parser)

if __name__ == '__main__':
    main()
