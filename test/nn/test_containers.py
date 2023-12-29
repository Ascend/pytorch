from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)


class TestNN(TestCase):
    def test_parameter(self):
        cpu_output = torch.nn.Parameter(torch.tensor([[1, 2, 3, 4, 5.0]]))
        npu_output = torch.nn.Parameter(torch.tensor([[1, 2, 3, 4, 5.0]], device=device))
        self.assertEqual(cpu_output, npu_output)


class TestContainers(TestCase):
    def test_module(self):

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 2, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return x

        model = Model().npu()
        x = torch.randn(1, 1, 32, 32).npu()
        output = model(x)
        self.assertEqual(output is not None, True)

    def test_sequential(self):
        torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)

    def test_moduleList(self):

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x

        model = MyModule().npu()
        x = torch.randn(1, 1, 10, 10).npu()
        output = model(x)
        self.assertEqual(output is not None, True)

    def test_ModuleDict(self):
        class ModuleDict(nn.Module):
            def __init__(self):
                super(ModuleDict, self).__init__()
                self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(10, 10, 3),
                    'pool': nn.MaxPool2d(3)
                })

                self.activations = nn.ModuleDict({
                    'relu': nn.ReLU(),
                    'prelu': nn.PReLU()
                })

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x

        net = ModuleDict().npu()

        fake_img = torch.randn((4, 10, 32, 32)).npu()

        output = net(fake_img, 'conv', 'relu')

    def make_param(self):
        return Parameter(torch.randn(10, 10, device=device))

    def check_list(self, parameters, param_list):
        self.assertEqual(len(parameters), len(param_list))
        for p1, p2 in zip(parameters, param_list):
            self.assertIs(p1, p2)
        for p1, p2 in zip(parameters, param_list.parameters()):
            self.assertIs(p1, p2)
        for i, _ in enumerate(parameters):
            self.assertIs(parameters[i], param_list[i])

    def model(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        return s

    def test_ParameterList(self):
        parameters = [self.make_param(), self.make_param()]
        param_list = nn.ParameterList(parameters)
        self.check_list(parameters, param_list)
        parameters += [self.make_param()]
        param_list += [parameters[-1]]
        self.check_list(parameters, param_list)
        parameters.append(self.make_param())
        param_list.append(parameters[-1])
        self.check_list(parameters, param_list)
        next_params = [self.make_param(), self.make_param()]
        parameters.extend(next_params)
        param_list.extend(next_params)
        self.check_list(parameters, param_list)
        parameters[2] = self.make_param()
        param_list[2] = parameters[2]
        self.check_list(parameters, param_list)
        parameters[-1] = self.make_param()
        param_list[-1] = parameters[-1]
        self.check_list(parameters, param_list)
        idx = torch.tensor(2, dtype=torch.int32)
        parameters[2] = self.make_param()
        param_list[idx] = parameters[2]
        self.assertIs(param_list[idx], parameters[2])
        self.check_list(parameters, param_list)
        self.assertEqual(param_list[1:], nn.ParameterList(parameters[1:]))
        self.assertEqual(param_list[3:], nn.ParameterList(parameters[3:]))
        self.assertEqual(param_list[:-1], nn.ParameterList(parameters[:-1]))
        self.assertEqual(param_list[:-3], nn.ParameterList(parameters[:-3]))
        self.assertEqual(param_list[::-1], nn.ParameterList(parameters[::-1]))
        with self.assertRaises(TypeError):
            param_list += self.make_param()
        with self.assertRaises(TypeError):
            param_list.extend(self.make_param())
        s = self.model()
        parameters = list(s.parameters())
        param_list = nn.ParameterList()
        param_list.extend(s.parameters())
        self.check_list(parameters, param_list)

    def check_dict(self, parameter_dict, parameters):
        self.assertEqual(len(parameter_dict), len(parameters))
        for k1, m2 in zip(parameters, parameter_dict.parameters()):
            self.assertIs(parameters[k1], m2)
        for k1, k2 in zip(parameters, parameter_dict):
            self.assertIs(parameters[k1], parameter_dict[k2])
        for k in parameter_dict:
            self.assertIs(parameter_dict[k], parameters[k])
        for k in parameter_dict.keys():
            self.assertIs(parameter_dict[k], parameters[k])
        for k, v in parameter_dict.items():
            self.assertIs(v, parameters[k])
        for k1, m2 in zip(parameters, parameter_dict.values()):
            self.assertIs(parameters[k1], m2)
        for k in parameters.keys():
            self.assertTrue(k in parameter_dict)

    def test_ParameterDict(self):
        parameters = OrderedDict([
            ('p1', Parameter(torch.randn(10, 10, device=device))),
            ('p2', Parameter(torch.randn(10, 10, device=device))),
            ('p3', Parameter(torch.randn(10, 10, device=device))),
        ])
        parameter_dict = nn.ParameterDict(parameters)
        self.check_dict(parameter_dict, parameters)
        parameters['p4'] = Parameter(torch.randn(10, 10))
        parameter_dict['p4'] = parameters['p4']
        self.check_dict(parameter_dict, parameters)
        next_parameters = [
            ('p5', Parameter(torch.randn(10, 10, device=device))),
            ('p2', Parameter(torch.randn(10, 10, device=device))),
        ]
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        self.check_dict(parameter_dict, parameters)
        next_parameters = OrderedDict([
            ('p6', Parameter(torch.randn(10, 10, device=device))),
            ('p5', Parameter(torch.randn(10, 10, device=device))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        self.check_dict(parameter_dict, parameters)
        next_parameters = {
            'p8': Parameter(torch.randn(10, 10, device=device)),
            'p7': Parameter(torch.randn(10, 10, device=device))
        }
        parameters.update(sorted(next_parameters.items()))
        parameter_dict.update(next_parameters)
        self.check_dict(parameter_dict, parameters)
        del parameter_dict['p3']
        del parameters['p3']
        self.check_dict(parameter_dict, parameters)
        with self.assertRaises(TypeError):
            parameter_dict.update(1)
        with self.assertRaises(TypeError):
            parameter_dict.update([1])
        with self.assertRaises(ValueError):
            parameter_dict.update(Parameter(torch.randn(10, 10, device=device)))
        p_pop = parameter_dict.pop('p4')
        self.assertIs(p_pop, parameters['p4'])
        parameters.pop('p4')
        self.check_dict(parameter_dict, parameters)
        parameter_dict.clear()
        self.assertEqual(len(parameter_dict), 0)
        parameters.clear()


if __name__ == "__main__":
    run_tests()
