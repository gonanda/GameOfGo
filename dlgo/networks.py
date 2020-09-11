from keras.layers import *
from keras.models import Model


def simple_zero_network(input_shape, num_moves):
    board_input = Input(shape=input_shape, name='board_input')
    pb = board_input
    for i in range(4):
        pb = Conv2D(64, (3, 3), padding='same', data_format='channels_first', activation='relu')(pb)

    policy_conv = Conv2D(2, (1, 1), data_format='channels_first', activation='relu')(pb)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(num_moves, activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1, 1), data_format='channels_first', activation='relu')(pb)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    return Model(inputs=[board_input],outputs=[policy_output, value_output])

def leaky_network(input_shape, num_moves):
    leaky_alpha=0.1
    board_input = Input(shape=input_shape, name='board_input')
    pb = board_input
    for i in range(4):
        pb = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(pb)
        pb = LeakyReLU(alpha=leaky_alpha)(pb)

    policy_conv = Conv2D(2, (1, 1), data_format='channels_first')(pb)
    policy_conv = LeakyReLU(alpha=leaky_alpha)(policy_conv)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(num_moves, activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1, 1), data_format='channels_first')(pb)
    value_conv = LeakyReLU(alpha=leaky_alpha)(value_conv)
    value_flat = Flatten()(value_conv)
    value_flat = LeakyReLU(alpha=leaky_alpha)(value_flat)
    value_hidden = Dense(256)(value_flat)
    value_hidden = LeakyReLU(alpha=leaky_alpha)(value_hidden)
    value_output = Dense(1, activation='tanh')(value_hidden)

    return Model(inputs=[board_input],outputs=[policy_output, value_output])

