FROM pytorch/manylinux-builder:cpu-2.1

# Install python
RUN cd /usr/local/bin \
    && ln -s /opt/_internal/cpython-3.7.5/bin/pip3.7 pip3.7 \
    && ln -s /opt/_internal/cpython-3.8.1/bin/pip3.8 pip3.8 \
    && ln -s /opt/_internal/cpython-3.9.0/bin/pip3.9 pip3.9 \
    && ln -s /opt/_internal/cpython-3.10.1/bin/pip3.10 pip3.10 \
    && ln -s /opt/_internal/cpython-3.11.0/bin/pip3.11 pip3.11 \
    && ln -s /opt/_internal/cpython-3.7.5/bin/python3.7 python3.7 \
    && ln -s /opt/_internal/cpython-3.8.1/bin/python3.8 python3.8 \
    && ln -s /opt/_internal/cpython-3.9.0/bin/python3.9 python3.9 \
    && ln -s /opt/_internal/cpython-3.10.1/bin/python3.10 python3.10 \
    && ln -s /opt/_internal/cpython-3.11.0/bin/python3.11 python3.11 \
    && ln -s python3.8 python3

# Set pip source
RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://mirrors.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

# Install pip package(build)
RUN pip3.9 install pyyaml  \
    && pip3.9 install torch==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3.9 install numpy==1.21.3

RUN pip3.10 install pyyaml  \
    && pip3.10 install torch==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3.10 install numpy==1.21.3

RUN pip3.11 install pyyaml  \
    && pip3.11 install torch==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3.11 install numpy==1.23.2

WORKDIR /home
