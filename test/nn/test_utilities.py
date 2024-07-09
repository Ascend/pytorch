from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestUtilities(TestCase):
    def set_prune(self):
        net = nn.Sequential(OrderedDict([
            ('first', nn.Linear(10, 4)),
            ('second', nn.Linear(4, 1)),
        ]))
        net = net.npu()

        parameters_to_prune = (
            (net.first, 'weight'),
            (net.second, 'weight'),
        )
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=10,
        )
        return net

    def test_clip_grad_norm_(self):
        input1 = torch.tensor([125, 75, 45, 15, 5]).npu()
        output = nn.utils.clip_grad_norm_(input1, max_norm=1, norm_type=2)
        self.assertEqual(output is not None, True)

    def test_clip_grad_value_(self):
        x = torch.tensor([1., 2.])

        x.grad = torch.tensor([0.3, 1.])

        torch.nn.utils.clip_grad_value_(x, clip_value=0.4)
        expected_cpu_xgrad = torch.tensor([0.3000, 0.4000])
        self.assertEqual(expected_cpu_xgrad.numpy(), x.grad.cpu().numpy())

    def test_prune_global_unstructured(self):
        out = self.set_prune()
        self.assertEqual(out is not None, True)

    def test_parameters_to_vector(self):
        out = self.set_prune()

        output = sum(torch.nn.utils.parameters_to_vector(out.buffers()) == 0)
        expected_cpu_output = torch.tensor([10])
        self.assertEqual(expected_cpu_output.numpy(), output.cpu().numpy())

    def test_prune_PruningContainer(self):
        m = nn.Conv3d(2, 2, 2).npu()
        prune.l1_unstructured(m, name='weight', amount=0.1)
        weight_mask0 = m.weight_mask  # save it for later sanity check

        # prune again
        prune.ln_structured(m, name='weight', amount=0.3, n=2, dim=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertIsInstance(
            hook,
            torch.nn.utils.prune.PruningContainer
        )

    def test_prune_Identity(self):
        model = nn.Linear(2, 3).npu()
        m = nn.utils.prune.Identity()
        m.apply(model, 'bias')
        self.assertEqual(m is not None, True)

    def test_prune_RandomUnstructured(self):
        model = nn.Linear(2, 3).npu()
        m = nn.utils.prune.RandomUnstructured(amount=1)
        m.apply(model, name='weight', amount=1)
        self.assertEqual(m is not None, True)

    def test_prune_L1Unsctructured(self):
        # if you call pruning twice, the hook becomes a PruningContainer
        m = nn.Conv3d(2, 2, 2).npu()
        prune.l1_unstructured(m, name='weight', amount=0.1)
        weight_mask0 = m.weight_mask  # save it for later sanity check

        # prune again
        prune.ln_structured(m, name='weight', amount=0.3, n=2, dim=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertIsInstance(
            hook,
            torch.nn.utils.prune.PruningContainer
        )
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        self.assertEqual(hook._tensor_name, 'weight')

        # check that the pruning container has the right length
        # equal to the number of pruning iters
        self.assertEqual(len(hook), 2)  # m.weight has been pruned twice

        # check that the entries of the pruning container are of the expected
        # type and in the expected order
        self.assertIsInstance(hook[0], torch.nn.utils.prune.L1Unstructured)

    def test_prune_RandomStructured(self):
        model = nn.Linear(2, 3).npu()
        m = nn.utils.prune.RandomStructured(amount=1)
        m.apply(model, name='weight', amount=1)
        self.assertEqual(m is not None, True)

    def test_prune_LnStructured(self):
        model = nn.Linear(2, 3).npu()
        m = nn.utils.prune.LnStructured(amount=1, n=2)
        m.apply(model, name='weight', amount=1, n=2, dim=-1)
        self.assertEqual(m is not None, True)

    def test_prune_CustomFromMask(self):
        r"""Test that the CustomFromMask is capable of receiving
        as input at instantiation time a custom mask, and combining it with
        the previous default mask to generate the correct final mask.
        """
        # new mask
        mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]]).npu()
        # old mask
        default_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]).npu()

        # some tensor (not actually used)
        t = torch.rand_like(mask.to(dtype=torch.float32)).npu()

        p = prune.CustomFromMask(mask=mask)

        computed_mask = p.compute_mask(t, default_mask)
        expected_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]]).to(
            dtype=t.dtype
        )

        self.assertEqual(computed_mask.cpu().numpy(), expected_mask.numpy())

    def test_prune_identity(self):
        model = nn.Linear(2, 3).npu()
        m = nn.utils.prune.identity(model, 'bias')
        expected_cpu_mbias_mask = torch.tensor([1., 1., 1.])
        self.assertEqual(expected_cpu_mbias_mask.numpy(), m.bias_mask.cpu().numpy())

    def test_prune_random_unstructured(self):
        m = nn.utils.prune.random_unstructured(nn.Linear(2, 3).npu(), 'weight', amount=1)
        output = torch.sum(m.weight_mask == 0)
        expected_cpu_output = torch.tensor([1])
        self.assertEqual(expected_cpu_output.numpy(), output.cpu().numpy())

    def test_prune_l1_unstructured(self):
        m = nn.utils.prune.l1_unstructured(nn.Linear(2, 3).npu(), 'weight', amount=0.2)
        output = m.state_dict().keys()
        self.assertEqual(m is not None, True)

    def test_prune_random_structured(self):
        m = nn.utils.prune.random_structured(
            nn.Linear(5, 3).npu(), 'weight', amount=3, dim=1
        )
        columns_pruned = int(sum(torch.sum(m.weight, dim=0) == 0))
        self.assertEqual(3, columns_pruned)

    def test_prune_ln_structured(self):
        m = prune.ln_structured(
            nn.Conv2d(5, 3, 2).npu(), 'weight', amount=0.3, dim=1, n=float('-inf')
        )

    def test_prune_custom_from_mask(self):
        model = nn.Linear(5, 3)
        mask = torch.Tensor([0, 1, 0])
        model = model.npu()
        mask = mask.npu()
        m = prune.custom_from_mask(
            model, name='bias', mask=mask
        )
        self.assertEqual(m.bias_mask is not None, True)

    def test_prune_remove(self):
        model = nn.Linear(5, 7)
        model = model.npu()
        m = prune.random_unstructured(model, name='weight', amount=0.2)
        m = prune.remove(m, name='weight')
        self.assertEqual(m is not None, True)

    def test_prune_is_pruned(self):
        m = nn.Linear(5, 7)
        m = m.npu()
        output = prune.is_pruned(m)
        self.assertFalse(output)

        prune.random_unstructured(m, name='weight', amount=0.2)
        output = prune.is_pruned(m)
        self.assertTrue(output)

    def test_weight_norm(self):
        input1 = nn.Linear(20, 40).npu()
        m = torch.nn.utils.weight_norm(input1, name='weight')
        weight_g_size = m.weight_g.size()
        weight_v_size = m.weight_v.size()
        self.assertExpectedInline(str(weight_g_size), '''torch.Size([40, 1])''')
        self.assertExpectedInline(str(weight_v_size), '''torch.Size([40, 20])''')

    def test_remove_weight_norm(self):
        m = torch.nn.utils.weight_norm(nn.Linear(20, 40).npu())
        output = torch.nn.utils.remove_weight_norm(m)
        self.assertExpectedInline(str(output), '''Linear(in_features=20, out_features=40, bias=True)''')

    def test_spectral_norm(self):
        m = torch.nn.utils.spectral_norm(nn.Linear(20, 40).npu())
        output = m.weight_u.size()
        self.assertExpectedInline(str(output), '''torch.Size([40])''')

    def test_remove_spectral_norm(self):
        m = torch.nn.utils.spectral_norm(nn.Linear(40, 10).npu())
        output = torch.nn.utils.remove_spectral_norm(m)
        self.assertExpectedInline(str(output), '''Linear(in_features=40, out_features=10, bias=True)''')


class TestUtilityFunction(TestCase):
    def test_rnn_pack_padded_sequence(self):
        batch_size = 3   # 这个batch有3个序列
        max_len = 6       # 最长序列的长度是6
        embedding_size = 8  # 嵌入向量大小8
        hidden_size = 16   # 隐藏向量大小16
        vocab_size = 20    # 词汇表大小20

        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]   # batch中每个seq的有效长度。
        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # LSTM的RNN循环神经网络
        lstm = torch.nn.LSTM(embedding_size, hidden_size)

        embedding = embedding.npu()
        lstm = lstm.npu()

        # 由大到小排序
        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)

        PAD_token = 0  # 填充下标是0

        def pad_seq(seq, seq_len, max_length):
            seq = seq
            seq += [PAD_token for _ in range(max_length - seq_len)]
            return seq

        pad_seqs = []  # 填充后的数据
        for i, j in zip(input_seq, lengths):
            pad_seqs.append(pad_seq(i, j, max_len))

        pad_seqs = torch.tensor(pad_seqs).npu()
        embeded = embedding(pad_seqs)

        # 压缩，设置batch_first为true
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=True)
        # 这里如果不写batch_first,你的数据必须是[s,b,e]，不然会报错lenghth错误

        # 　利用lstm循环神经网络测试结果
        state = None
        pade_outputs, _ = lstm(pack, state)
        # 设置batch_first为true;你可以不设置为true,为false时候只影响结构不影响结果
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=True)
        self.assertExpectedInline(str(others), '''tensor([6, 5, 3])''')

    def test_rnn_pad_sequence(self):
        a = torch.ones(25, 300)
        b = torch.ones(22, 300)
        c = torch.ones(15, 300)
        output = torch.nn.utils.rnn.pad_sequence([a.npu(), b.npu(), c.npu()]).size()
        self.assertExpectedInline(str(output), '''torch.Size([25, 3, 300])''')

    def test_nn_Flatten(self):
        input1 = torch.randn(32, 1, 5, 5).npu()
        m = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1),
            nn.Flatten()
        )
        m = m.npu()
        output = m(input1)

        self.assertExpectedInline(str(output.size()), '''torch.Size([32, 288])''')


if __name__ == "__main__":
    run_tests()
