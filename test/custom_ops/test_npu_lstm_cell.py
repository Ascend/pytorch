import torch
from torch import _VF
import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestLSTMCell(TestCase):
    def cpu_to_exec(self, input_data, weight_ih, weight_hh, h0_data, c0_data, bias_ih, bias_hh):
        """def lstm_cell(
            input: Tensor,
            hx: Union[Tuple[Tensor, ...], List[Tensor]],
            w_ih: Tensor,
            w_hh: Tensor,
            b_ih: Optional[Tensor]=None,
            b_hh: Optional[Tensor]=None
            ) -> Tuple[Tensor, Tensor]: ...

            ret = _VF.lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,)
        """
        weight_ih = weight_ih.transpose(0, 1)
        weight_hh = weight_hh.transpose(0, 1)
        h_out, c_out = _VF.lstm_cell(input_data, (h0_data, c0_data),
                                     weight_ih, weight_hh, bias_ih, bias_hh)
        return h_out, c_out

    def npu_to_exec(self, input_data, weight_ih, weight_hh, h0_data, c0_data, bias_ih, bias_hh):
        y_out, h_out, c_out, i_out, j_out, f_out, o_out, tanhc = torch_npu.npu_lstm_cell(
            input_data, weight_ih, weight_hh, h0_data, c0_data, bias_ih, bias_hh)
        return h_out, c_out

    def test_lstm_cell(self):
        input_size = 8
        hidden_size = 7
        time_step = 1
        batch_size = 3

        input_shape = (batch_size, input_size)
        h0_shape = (batch_size, hidden_size)
        c0_shape = (batch_size, hidden_size)

        _, input_data = create_common_tensor(["float16", 29, input_shape], -1, 1)
        _, h0_data = create_common_tensor(["float16", 29, h0_shape], -1, 1)
        _, c0_data = create_common_tensor(["float16", 29, c0_shape], -1, 1)
        _, weight_ih = create_common_tensor(["float16", 2, (input_size, 4 * hidden_size)], -1, 1)
        _, weight_hh = create_common_tensor(["float16", 2, (hidden_size, 4 * hidden_size)], -1, 1)
        _, bias_ih = create_common_tensor(["float16", 2, (4 * hidden_size)], -1, 1)
        _, bias_hh = create_common_tensor(["float16", 2, (4 * hidden_size)], -1, 1)

        # cpu
        cpu_h_out, cpu_c_out = self.cpu_to_exec(input_data, weight_ih, weight_hh, h0_data, c0_data,
                                                bias_ih, bias_hh)
        # npu
        npu_h_out, npu_c_out = self.npu_to_exec(input_data, weight_ih, weight_hh, h0_data, c0_data,
                                                bias_ih, bias_hh)

        cpu_h_out = cpu_h_out.cpu().detach().numpy()
        cpu_c_out = cpu_c_out.cpu().detach().numpy()
        npu_h_out = npu_h_out.cpu().detach().numpy()
        npu_c_out = npu_c_out.cpu().detach().numpy()

        self.assertRtolEqual(cpu_h_out, npu_h_out)
        self.assertRtolEqual(cpu_c_out, npu_c_out)


if __name__ == "__main__":
    run_tests()
