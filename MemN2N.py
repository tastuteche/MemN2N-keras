import keras.backend as K
import numpy as np
from keras.engine.topology import Layer

# https://arxiv.org/pdf/1503.08895.pdf


class MemN2N(Layer):
    def __init__(self, output_dim, num_hops, **kwargs):
        self._embedding_szie = output_dim
        self.num_hops = num_hops
        super(MemN2N, self).__init__(**kwargs)

    def build(self, input_shape):
        #[(None, 10, 36), (None, 36)]
        vocab_size = input_shape[0][2]
        initial_A_value = np.random.uniform(
            0, 1, size=[self._embedding_szie, vocab_size])
        initial_B_value = np.random.uniform(
            0, 1, size=[self._embedding_szie, vocab_size])
        # initial_H_value = np.random.uniform(0, 1, size=[self._embedding_szie, self._embedding_szie])
        initial_C_value = np.random.uniform(
            0, 1, size=[self._embedding_szie, vocab_size])
        self.memory_size = input_shape[0][1]
        self.A_dV = K.variable(initial_A_value)
        self.B_dV = K.variable(initial_B_value)
        # self.H = K.variable(initial_H_value)
        self.C_dV = K.variable(initial_C_value)
        self.trainable_weights = [self.A_dV, self.B_dV, self.C_dV]
        super(MemN2N, self).build(input_shape)

    def call(self, inputs, mask=None):
        input_a_memory = inputs[0]
        input_c_memory = inputs[0]

        input_b_query = inputs[1]

        mem_m_storage_representation = K.dot(
            input_a_memory, K.transpose(self.A_dV))
        mem_c_output_representation = K.dot(
            input_c_memory, K.transpose(self.C_dV))
        mem_u_query_representation = K.dot(
            input_b_query, K.transpose(self.B_dV))
        match_p_storage_query = K.softmax(
            K.reshape(K.dot(mem_m_storage_representation, K.transpose(mem_u_query_representation)), (1, self.memory_size)))
        mem_o = K.reshape(
            K.dot(match_p_storage_query, mem_c_output_representation), (1, self._embedding_szie))
        mem_u_query_hop_N = mem_o + mem_u_query_representation

        for _ in range(self.num_hops - 1):
            match_p_hop_N = K.softmax(K.reshape(
                K.dot(mem_m_storage_representation, K.transpose(mem_u_query_hop_N)), (1, self.memory_size)))
            mem_o_hop_N = K.reshape(
                K.dot(match_p_hop_N, mem_c_output_representation), (1, self._embedding_szie))
            mem_u_query_hop_N = mem_o_hop_N + mem_u_query_hop_N

        return mem_u_query_hop_N

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._embedding_szie
