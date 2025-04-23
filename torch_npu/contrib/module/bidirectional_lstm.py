import warnings

import torch
import torch_npu

warnings.filterwarnings(action='once', category=FutureWarning)


class BiLSTM(torch.nn.Module):
    r"""Applies an NPU compatible bidirectional LSTM operation to an input
    sequence.

    The implementation of this BidirectionalLSTM is mainly based on the principle of bidirectional LSTM.
    Since NPU do not support the parameter bidirectional in torch.nn.lstm to be True,
    we reimplement it by joining two unidirection LSTM together to form a bidirectional LSTM

    Paper: [Bidirectional recurrent neural networks]

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`


    Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the cell state for `t = seq_len`.


    Examples::
        >>> r = BiLSTM(512, 256)
        >>> input_tensor = torch.randn(26, 2560, 512)
        >>> output = r(input_tensor)
    """
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()

        warnings.warn("torch_npu.contrib.BiLSTM is deprecated. "
                      "Please check document for replacement.", FutureWarning)
        self.fw_rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=False)
        self.bw_rnn = torch.nn.LSTM(input_size, hidden_size, bidirectional=False)

    def forward(self, inputs):
        input_fw = inputs
        recurrent_fw, _ = self.fw_rnn(input_fw)
        input_bw = torch.flip(inputs, [0])
        recurrent_bw, _ = self.bw_rnn(input_bw)
        recurrent_bw = torch.flip(recurrent_bw, [0])
        recurrent = torch.cat((recurrent_fw, recurrent_bw), 2)

        return recurrent