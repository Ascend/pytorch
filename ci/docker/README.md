## Ascend Pytorch Dockerfile Repository

This folder hosts the `Dockerfile` to build docker images with various platforms.

### Build torch_npu from Docker container

**Clone torch-npu**

```Shell
git clone https://github.com/ascend/pytorch.git --depth 1
```

**Build docker image**

```Shell
cd pytorch/ci/docker/{arch} # {arch} for X86 or ARM
docker build -t manylinux-builder:v1 .
```
**Enter docker Container**

```Shell
docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
# {code_path} is the torch_npu source code path
```
**Compile torch_npu**

Take Python 3.8 as an example
```Shell
cd /home/pytorch
bash ci/build.sh --python=3.8
```
