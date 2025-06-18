## 简介
test目录为PTA相关的测试用例。
## 安装依赖

`pip3 install -r requirements.txt`

## 补全脚本
该操作需要联网
```
cd test
bash get_synchronized_files.sh
```
通过以上操作，会自动补齐testfiles_synchronized.txt和testfolder_synchronized.txt中的文件或文件夹。
## 跳过失败用例
`export DISABLED_TESTS_FILE=./unsupported_test_cases/.pytorch-disabled-tests.json`

如果不是在test目录下运行测试用例，需要传入.pytorch-disabled-tests.json的绝对路径。
## 执行方式
### 执行单个测试脚本
运行以test开头的文件。以test_autocast.py为例：

方式一：

`python test_autocast.py`


方式二：

`python run_test.py -i test_autocast`

说明：部分以test开头的脚本不是直接运行的脚本，比如jit中的测试脚本是通过test_jit.py执行的。

### 执行具体的用例
通过-k参数传入具体的用例名。以test_autocast.py为例：

方式一：

`python test_autocast.py -v -k test_autocast_nn_fp32`

方式二：

`python run_test.py -v -i test_autocast -- -k test_autocast_nn_fp32`

### 执行全量UT的方式
```
# 进入到test目录的上一级
cd ../
```

运行非分布式全量用例：

```
python ci/access_control_test.py --all
```

运行分布式全量用例：

```
python ci/access_control_test.py --distributed
```

## FAQ
1. 报错："dictionary changed size during interation".  

    如果python 环境是3.8.1版本，报错在unitest/case.py中，可考虑是sys.modules被修改导致的。第三方包可能会有对sys.modules的修改，比如beartype。 
    此问题为python 3.8.1版本/3.9.0版本的已知bug，可按照 https://github.com/python/cpython/issues/73806 中修改方式修改，将
    `for v in sys.modules.values()` 
    改为
    `for v in list(sys.modules.values())`

    可用于复现问题的用例：
    `python test_jit.py -v -k test_annotated_empty_dict`

2. test_public_bindings.py 用例的功能

    该用例是为了校验接口的公开规范性，如果该用例报错，请确认报错的接口是否要公开，并按照报错的提示进行修改。
