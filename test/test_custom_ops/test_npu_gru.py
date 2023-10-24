import torch
from torch import _VF
import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestGru(TestCase):
    def cpu_op_exec(self, input_data, hx, weight_input, weight_hidden, bias_input, bias_hidden,
                    seq_length, num_layers, has_biases=True, batch_first=False, dropout=0.0, bidirectional=False):
        """
        def gru(
            data: Tensor,
            batch_sizes: Tensor,
            hx: Tensor,
            params: Union[Tuple[Tensor, ...], List[Tensor]],
            has_biases: _bool,
            num_layers: _int,
            dropout: _float,
            train: _bool,
            bidirectional: _bool
            ) -> Tuple[Tensor, Tensor]: ...

            result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
                             self.num_layers, self.dropout, self.training, self.bidirectional)

        def gru(
            input: Tensor,
            hx: Tensor,
            params: Union[Tuple[Tensor, ...], List[Tensor]],
            has_biases: _bool,
            num_layers: _int,
            dropout: _float,
            train: _bool,
            bidirectional: _bool,
            batch_first: _bool
            ) -> Tuple[Tensor, Tensor]: ...

            result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
                             self.dropout, self.training, self.bidirectional, self.batch_first)
        """
        train = True
        weight_input = weight_input.transpose(0, 1)
        weight_hidden = weight_hidden.transpose(0, 1)
        weights = (weight_input, weight_hidden, bias_input, bias_hidden)
        output, hidden = _VF.gru(input_data, hx, weights, has_biases,
                                 num_layers, dropout, train, bidirectional, batch_first)

        return output, hidden

    def npu_to_exec(self, input_data, hx, weight_input, weight_hidden, bias_input, bias_hidden,
                    seq_length, num_layers, has_biases=True, batch_first=False, dropout=0.0, bidirectional=False):
        """
        npu_gru(
            Tensor input,
            Tensor hx,
            Tensor weight_input,
            Tensor weight_hidden,
            Tensor bias_input,
            Tensor bias_hidden,
            Tensor seq_length,
            bool has_biases,
            int num_layers,
            float dropout,
            bool train,
            bool bidirectional,
            bool batch_first
        ) -> Tensor[]
        """
        y, output_h, update, reset, new, hidden_new = torch_npu.npu_gru(
            input_data, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases=has_biases,
            num_layers=num_layers, dropout=dropout, train=True, bidirectional=bidirectional, batch_first=batch_first)
        return y, output_h

    def test_gru_no_bidirection(self):
        input_size = 10
        hidden_size = 6
        batch_size = 3
        num_layers = 1
        seq_length = 6
        has_biases = True
        input_shape = [seq_length, batch_size, input_size]
        h_0_shape = [num_layers, batch_size, hidden_size]
        seq_length_t = torch.Tensor([seq_length]).int().npu()

        _, input_data = create_common_tensor(["float16", 29, input_shape], -1, 1)
        _, h_0 = create_common_tensor(["float16", 29, h_0_shape], -1, 1)
        _, weight_ih = create_common_tensor(["float16", 4, (input_size, 3 * hidden_size)], -1, 1)
        _, weight_hh = create_common_tensor(["float16", 4, (hidden_size, 3 * hidden_size)], -1, 1)
        _, bias_ih = create_common_tensor(["float16", 2, (3 * hidden_size)], -1, 1)
        _, bias_hh = create_common_tensor(["float16", 2, (3 * hidden_size)], -1, 1)

        # cpu
        cpu_out, cpu_h = self.cpu_op_exec(input_data, h_0, weight_ih, weight_hh, bias_ih, bias_hh,
                                          seq_length_t, num_layers, has_biases=has_biases)
        # npu
        npu_out, npu_h = self.npu_to_exec(input_data, h_0, weight_ih, weight_hh, bias_ih, bias_hh,
                                          seq_length_t, num_layers, has_biases=has_biases)

        cpu_out = cpu_out.cpu().detach().numpy()
        cpu_h = cpu_h.cpu().detach().numpy()
        npu_out = npu_out.cpu().detach().numpy()
        npu_h = npu_h.cpu().detach().numpy()

        self.assertRtolEqual(cpu_out, npu_out)
        self.assertRtolEqual(cpu_h[-1], npu_h[-1])


if __name__ == "__main__":
    run_tests()
