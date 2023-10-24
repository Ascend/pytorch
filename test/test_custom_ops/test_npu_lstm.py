import torch
from torch import _VF
import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestLSTM(TestCase):
    def cpu_to_exec(self, input_data, weight, bias, seq_len, h, c, has_biases=True,
                    num_layers=1, dropout=0.0, train=True, bidirectional=False,
                    batch_first=False, flag_seq=False, direction=False):
        """
        def lstm(
            data: Tensor,
            batch_sizes: Tensor,
            hx: Union[Tuple[Tensor, ...], List[Tensor]],
            params: Union[Tuple[Tensor, ...], List[Tensor]],
            has_biases: _bool,
            num_layers: _int,
            dropout: _float,
            train: _bool,
            bidirectional: _bool
        ) -> Tuple[Tensor, Tensor, Tensor]: ...

        result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                          self.num_layers, self.dropout, self.training, self.bidirectional)

        def lstm(
            input: Tensor,
            hx: Union[Tuple[Tensor, ...], List[Tensor]],
            params: Union[Tuple[Tensor, ...], List[Tensor]],
            has_biases: _bool,
            num_layers: _int,
            dropout: _float,
            train: _bool,
            bidirectional: _bool,
            batch_first: _bool
        ) -> Tuple[Tensor, Tensor, Tensor]: ...

        result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                          self.dropout, self.training, self.bidirectional, self.batch_first)
        """
        input_size = input_data.shape[-1]
        weight_ih = weight[:input_size, :]
        weight_hh = weight[input_size:, :]
        weight_ih = weight_ih.transpose(0, 1)
        weight_hh = weight_hh.transpose(0, 1)
        bias_ih = bias
        bias_hh_shape = bias_ih.shape
        _, bias_hh = create_common_tensor(["float16", 2, bias_hh_shape], 0, 0)  # zeros
        weights = (weight_ih, weight_hh, bias_ih, bias_hh)

        ret = _VF.lstm(input_data, (h, c), weights, has_biases, num_layers,
                       dropout, train, bidirectional, batch_first)
        output, hn, hc = ret[0], ret[1], ret[2]
        return output, hn, hc

    def npu_to_exec(self, input_data, weight, bias, seq_len, h, c, has_biases=True,
                    num_layers=1, dropout=0.0, train=True, bidirectional=False,
                    batch_first=False, flag_seq=False, direction=False):
        """
        npu_lstm(
            Tensor input,
            Tensor weight,
            Tensor bias,
            Tensor seqMask,
            Tensor h,
            Tensor c,
            bool has_biases,
            int num_layers,
            float dropout,
            bool train,
            bool bidirectional,
            bool batch_first,
            bool flagSeq,
            bool direction)
        -> Tensor[] # yOutput, hOutput, cOutput, iOutput, jOutput, fOutput, oOutput, tanhc
        """
        result = torch_npu.npu_lstm(input_data, weight, bias, seq_len, h, c,
                                    has_biases, num_layers, dropout, train,
                                    bidirectional, batch_first, flag_seq, direction)
        y, h, c = result[0], result[1], result[2]
        return y, h, c

    def test_lstm_single_direction(self):
        input_size = 10
        hidden_size = 5
        num_layers = 1
        batch_szie = 3
        seq_length = 5
        bidirectional = False
        d = 2 if bidirectional else 1
        seq_length_t = torch.Tensor((seq_length)).int().npu()
        input_shape = (seq_length, batch_szie, input_size)
        h0_shape = (d * num_layers, batch_szie, hidden_size)
        c0_shape = (d * num_layers, batch_szie, hidden_size)

        _, input_data = create_common_tensor(["float16", 29, input_shape], -1, 1)
        _, h0_data = create_common_tensor(["float16", 29, h0_shape], -1, 1)
        _, c0_data = create_common_tensor(["float16", 29, c0_shape], -1, 1)
        _, weight_ih = create_common_tensor(["float16", 2, (input_size, 4 * hidden_size)], -1, 1)
        _, weight_hh = create_common_tensor(["float16", 2, (hidden_size, 4 * hidden_size)], -1, 1)
        _, bias_ih = create_common_tensor(["float16", 2, (4 * hidden_size)], 0, 1)
        weight = torch.cat((weight_ih, weight_hh), dim=0)
        bias = bias_ih

        # cpu
        cpu_out, cpu_hn, cpu_cn = self.cpu_to_exec(input_data, weight, bias, seq_length_t,
                                                   h0_data, c0_data, has_biases=True)
        # npu
        npu_out, npu_hn, npu_cn = self.npu_to_exec(input_data, weight, bias, seq_length_t,
                                                   h0_data, c0_data, has_biases=True)

        cpu_out = cpu_out.cpu().detach().numpy()
        cpu_hn = cpu_hn.cpu().detach().numpy()
        cpu_cn = cpu_cn.cpu().detach().numpy()
        npu_out = npu_out.cpu().detach().numpy()
        npu_hn = npu_hn.cpu().detach().numpy()
        npu_cn = npu_cn.cpu().detach().numpy()

        self.assertRtolEqual(cpu_out, npu_out)
        self.assertRtolEqual(cpu_hn[-1], npu_hn[-1])
        self.assertRtolEqual(cpu_cn[-1], npu_cn[-1])


if __name__ == "__main__":
    run_tests()
