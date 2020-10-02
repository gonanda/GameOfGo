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

leaky_alpha=0.1

def leaky_network(input_shape, num_moves):
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

def dual_residual_network(input_shape, num_moves,  blocks=3):
    inputs = Input(shape=input_shape)
    first_conv = conv_bn_lrelu_block(name="init")(inputs)
    res_tower = residual_tower(blocks=blocks)(first_conv)
    policy = policy_head(num_moves)(res_tower)
    value = value_head()(res_tower)
    return Model(inputs=inputs, outputs=[policy, value])

def conv_bn_lrelu_block(name, activation=True, filters=256, kernel_size=(3,3), strides=(1,1), padding="same", init="he_normal"):
    def f(inputs):
        conv = Conv2D(filters=filters, 
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=init,
                data_format='channels_first',
                name="{}_conv_block".format(name))(inputs)
        batch_norm = BatchNormalization(axis=1, name="{}_batch_norm".format(name))(conv)
        return LeakyReLU(alpha=leaky_alpha, name="{}_leakyrelu".format(name))(batch_norm) if activation else batch_norm
    return f

def residual_block(block_num, **args):
    def f(inputs):
        res = conv_bn_lrelu_block(name="residual_1_{}".format(block_num), activation=True, **args)(inputs)
        res = conv_bn_lrelu_block(name="residual_2_{}".format(block_num) , activation=False, **args)(res)
        res = add([inputs, res], name="add_{}".format(block_num))
        return LeakyReLU(alpha=leaky_alpha, name="{}_leakyrelu".format(block_num))(res) 
    return f

def residual_tower(blocks, **args):
    def f(inputs):
        x = inputs
        for i in range(blocks):
            x = residual_block(block_num=i)(x)
        return x
    return f

def policy_head(num_moves):
    def f(inputs):
        conv = Conv2D(filters=2, 
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                name="policy_head_conv_block")(inputs)
        batch_norm = BatchNormalization(axis=1, name="policy_head_batch_norm")(conv)
        activation = LeakyReLU(alpha=leaky_alpha, name="policy_head_leakyrelu")(batch_norm)
        flat = Flatten()(activation)
        return Dense(num_moves, activation="softmax", name="policy_head_dense")(flat)
    return f

def value_head():
    def f(inputs):
        conv = Conv2D(filters=1, 
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                name="value_head_conv_block")(inputs)
        batch_norm = BatchNormalization(axis=1, name="value_head_batch_norm")(conv)
        activation = LeakyReLU(alpha=leaky_alpha, name="value_head_leakyrelu")(batch_norm)
        flat = Flatten()(activation)
        dense =  Dense(units= 256, name="value_head_dense")(flat)
        dense = LeakyReLU(alpha=leaky_alpha, name="value_head_lrelu")(dense)
        return Dense(units= 1, name="value_head_output", activation="tanh")(dense)
    return f
