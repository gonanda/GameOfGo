import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import sys
import h5py
from dlgo.agent import Agent, decode_agent

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bot')
    args = parser.parse_args()

    def load_agent(filename):
        with h5py.File(filename, 'r') as h5file:
            return decode_agent(h5file)

    bot = load_agent(args.bot)

    print('\n')
    print('bot: %s' % args.bot)
    print('\n')
    print('board size: %d' % bot.encoder.board_size)
    print('encoded liberties: %d' % bot.encoder.num_lib)
    print('encoded: psave %d   save %d   healthy %d   vital %d' % (bot.encoder.params[1],bot.encoder.params[2],bot.encoder.params[3],bot.encoder.params[4]))
    print('use only sensible moves: %s' % bot.only_sensible)
    print('\n')
    print('number of rounds: %d' % bot.num_rounds)
    print('temperature c: %f' % bot.c)
    print('concentration parameter: %f' % bot.concent_param)
    print('dirichlet weight: %f' % bot.dirichlet_weight)
 
if __name__ == '__main__':
    main()
