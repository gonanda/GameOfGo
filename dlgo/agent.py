import numpy as np
import threading
import time
from keras.optimizers import SGD
from dlgo import kerasutil
from .encoder import Encoder
from .goboard import Move


__all__ = [
    'Agent',
]


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


class TreeNode:

    def __init__(self, state, value, priors, parent, last_move, only_sensible):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if only_sensible:
                if state.is_sensible_move(move):
                    self.branches[move] = Branch(p)
            else:
                if state.is_valid_move(move):
                    self.branches[move] = Branch(p)
        self.children = {}

    def moves(self):
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0


class Agent():

    def __init__(self, model, encoder, num_rounds=100, search_time=0, c=2.0, concent_param=0.03, dirichlet_weight=0.5, only_sensible=True):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = num_rounds
        self.search_time = search_time
        self.c = c
        self.concent_param = concent_param
        self.dirichlet_weight = dirichlet_weight

        self.only_sensible = only_sensible

        self.stop_search = False
        self.search_counter = 0

    def search_timer(self):
        time.sleep(self.search_time)
        self.stop_search = True


    def select_move(self, game_state):
        root = self.create_node(game_state, is_root=True, only_sensible=self.only_sensible)

        self.stop_search = False

        if self.num_rounds==0:
            threading.Thread(target=self.search_timer).start()

        if self.search_time==0:
            self.search_counter=self.num_rounds

        while not self.stop_search:
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                if not node.state.is_over():
                    next_move = self.select_branch(node)
            if not node.state.is_over():
                new_state = node.state.apply_move(next_move)
                child_node = self.create_node(new_state, move=next_move, parent=node, only_sensible=self.only_sensible)
                move = next_move
                value = -1 * child_node.value
            else:
                move = node.last_move
                value = -node.value
                node = node.parent
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value
            if self.search_time==0:
                self.search_counter-=1
                if self.search_counter==0:
                    self.stop_search = True


        if self.collector is not None:
            # original position
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx)) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(root_state_tensor, visit_counts)
            # transposed position
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).transpose()) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.transpose(root_state_tensor,(0,2,1)), visit_counts)
            # flip row
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).flip_row(self.encoder.board_size)) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.flip(root_state_tensor,(1)), visit_counts)
            # flip row and transpose
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).flip_row(self.encoder.board_size).transpose()) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.transpose(np.flip(root_state_tensor,(2)),(0,2,1)), visit_counts)
            # flip col
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).flip_col(self.encoder.board_size)) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.flip(root_state_tensor,(2)), visit_counts)
            # flip col and transpose
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).flip_col(self.encoder.board_size).transpose()) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.transpose(np.flip(root_state_tensor,(1)),(0,2,1)), visit_counts)
            # flip row and col
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).flip_row_col(self.encoder.board_size)) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.flip(root_state_tensor,(1,2)), visit_counts)
            # flip row and col and transpose
            visit_counts = np.array([
                root.visit_count(self.encoder.decode_move_index(idx).flip_row_col(self.encoder.board_size).transpose()) for idx in range(self.encoder.num_moves())
            ])
            self.collector.record_decision(np.transpose(np.flip(root_state_tensor,(1,2)),(0,2,1)), visit_counts)

        return max(root.moves(), key=root.visit_count)


    def set_collector(self, collector):
        self.collector = collector

    def set_num_rounds(self, num_rounds):
        self.num_rounds = num_rounds
        self.search_time = 0

    def set_search_time(self, search_time):
        self.search_time = search_time
        self.num_rounds = 0

    def set_c(self, c):
        self.c = c

    def set_concent_param(self, concent_param):
        self.concent_param = concent_param

    def set_dirichlet_weight(self, dirichlet_weight):
        self.dirichlet_weight = dirichlet_weight

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)
        return max(node.moves(), key=score_branch)

    def create_node(self, game_state, move=None, parent=None, is_root=False, only_sensible=True):
        if game_state.is_over():
            priors = np.zeros(self.encoder.num_moves())
            if game_state.winner() == game_state.next_player:
                value=game_state.margin()/(self.encoder.board_size*self.encoder.board_size)
            elif game_state.winner() == game_state.next_player.other:
                value=-game_state.margin()/(self.encoder.board_size*self.encoder.board_size)
            else:
                value=0.
        else:
            state_tensor = self.encoder.encode(game_state)
            model_input = np.array([state_tensor])
            priors, values = self.model.predict(model_input)
            priors = priors[0]
            if self.concent_param>0 and is_root:
                priors = ((1.-self.dirichlet_weight)*priors+self.dirichlet_weight*np.random.dirichlet(np.full(priors.shape,self.concent_param)))
            value = values[0][0]
        move_priors = {self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors)}
        new_node = TreeNode(game_state, value, move_priors, parent, move, only_sensible)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(self, experience, learning_rate, momentum, batch_size, policy_loss_weight, epochs):
        num_examples = experience.states.shape[0]

        model_input = experience.states

        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_examples, 1))
        action_target = experience.visit_counts / visit_sums

        num_points = self.encoder.board_size*self.encoder.board_size
        value_target = experience.rewards / num_points

        self.model.compile(SGD(lr=learning_rate, momentum=momentum), loss=['categorical_crossentropy', 'mse'], loss_weights=[policy_loss_weight,1.-policy_loss_weight])
        history = self.model.fit(model_input, [action_target, value_target], batch_size=batch_size, epochs=epochs)
        return history.history

    def serialize(self, h5file):
        h5file.create_group('search')
        h5file['search'].attrs['num_rounds'] = self.num_rounds
        h5file['search'].attrs['c'] = self.c
        h5file['search'].attrs['concent_param'] = self.concent_param
        h5file['search'].attrs['dirichlet_weight'] = self.dirichlet_weight
        if self.only_sensible:
            h5file['search'].attrs['only_sensible'] = 1
        else:
            h5file['search'].attrs['only_sensible'] = 0
        h5file.create_group('encoder')
        h5file['encoder'].attrs['board_size'] = self.encoder.board_size
        h5file['encoder'].attrs['params'] = self.encoder.params
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

def decode_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    board_size = h5file['encoder'].attrs['board_size']
    params = h5file['encoder'].attrs['params']
    encoder = Encoder(board_size, params)
    num_rounds = h5file['search'].attrs['num_rounds']
    c = h5file['search'].attrs['c']
    concent_param = h5file['search'].attrs['concent_param']
    dirichlet_weight = h5file['search'].attrs['dirichlet_weight']
    if h5file['search'].attrs['only_sensible'] == 1:
        only_sensible = True
    else:
        only_sensible = False
    return Agent(model=model,
                encoder=encoder,
                num_rounds=num_rounds,
                c=c,
                concent_param=concent_param,
                dirichlet_weight=dirichlet_weight,
                only_sensible=only_sensible)

