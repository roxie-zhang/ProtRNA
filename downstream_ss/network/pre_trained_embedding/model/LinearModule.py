import numbers
from typing import Union, Sequence

import tensorflow as tf

class Linear(tf.Module):

    def output_shape(self):
        return self._output_shape

    def __init__(self,
                 num_output: Union[int, Sequence[int]],
                 num_input_dims: int = 1,
                 num_input = 1,
                 name: str = 'linear',
                 initializer='linear',
                 global_config=None,
                 dtype=tf.float32,
                 load=False):

        super(Linear, self).__init__(name=name)

        with self.name_scope:
            self.DTYPE = dtype
            self.gc = global_config
            if self.gc == None:
                self.gc = {'iter': 0}
            if isinstance(num_output, numbers.Integral):
                self.output_shape = (num_output,)
            else:
                self.output_shape = tuple(num_output)

            self.num_output = num_output
            self.num_input = num_input

            shape = [self.num_input, self.num_output]

            self.w = tf.Variable(tf.zeros(shape, dtype=self.DTYPE), name='weights', dtype=self.DTYPE)
            self.b = tf.Variable(tf.zeros(self.output_shape, dtype=self.DTYPE), name='bias', dtype=self.DTYPE)
            
    def __call__(self, inputs, **kwargs):
        output = tf.matmul(inputs, self.w) + self.b
        return output
