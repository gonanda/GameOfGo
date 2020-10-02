import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import argparse

import h5py

from keras.layers import Dense, Input
from keras.models import Model
import dlgo.networks as networks
from dlgo.agent import Agent
from dlgo.experience import ExperienceCollector
from dlgo.encoder import Encoder

from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--network', default='simple_zero_network')
    parser.add_argument('--feature-planes', nargs=5, type=int, default=(4,1,1,1,1))
    parser.add_argument('--num-rounds', type=int, default=300)
    parser.add_argument('--c', type=float, default=2.0)
    parser.add_argument('--concent-param', type=float, default=0.03)
    parser.add_argument('--dirichlet-weight', type=float, default=0.3)
    parser.add_argument('--only-sensible', type=int, default=1)
    parser.add_argument('--work-dir')
    args = parser.parse_args()

    encoder = Encoder(args.board_size, args.feature_planes)

    model = getattr(networks,args.network)(encoder.shape(),encoder.num_moves())

    if args.only_sensible > 0:
        only_sensible = True
    else:
        only_sensible = False

    new_agent = Agent(
                model=model,
                encoder=encoder,
                num_rounds=args.num_rounds,
                c=args.c,
                concent_param=args.concent_param,
                dirichlet_weight=args.dirichlet_weight,
                only_sensible=only_sensible)

    os.mkdir(args.work_dir)
    with h5py.File(os.path.join(args.work_dir,'agent_00000000.hdf5'), 'w') as outfile:
        new_agent.serialize(outfile)
    opt_str = '--games-per-worker-and-color 32 --num-workers 8 --nr %d --c %f --cp %f --dw %f --lr 0.01 --mo 0.9 --bs 2048 --plw 0.2 --ep 20 --evfrac 0.55 --stop 0' % (args.num_rounds, args.c, args.concent_param, args.dirichlet_weight)
    with open(os.path.join(args.work_dir,'options'),'w') as outfile:
        outfile.write(opt_str)

if __name__ == '__main__':
    main()
