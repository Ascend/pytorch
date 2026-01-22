FROM pytorch/manylinux2_28_aarch64-builder:cpu-aarch64-66ed646875945a78e9671d054c763b71d55c742a

ENV PATH /usr/local/bin:$PATH

RUN echo "alias ll='ls -l --color=auto'" >> /root/.bashrc

# Set pip&python
RUN cd /usr/local/bin \
    && ln -sf /opt/_internal/cpython-3.10.18/bin/pip3.10 pip3.10 \
    && ln -sf /opt/_internal/cpython-3.11.13/bin/pip3.11 pip3.11 \
    && ln -sf /opt/_internal/cpython-3.12.11/bin/pip3.12 pip3.12 \
    && ln -sf /opt/_internal/cpython-3.10.18/bin/pip3.10 pip3 \
    && ln -sf /opt/_internal/cpython-3.10.18/bin/python3.10 python3.10 \
    && ln -sf /opt/_internal/cpython-3.11.13/bin/python3.11 python3.11 \
    && ln -sf /opt/_internal/cpython-3.12.11/bin/python3.12 python3.12 \
    && ln -sf /opt/_internal/cpython-3.10.18/bin/python3.10 python3

# Set pip source
RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://mirrors.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

ARG CONFIG_FOR_LCOV=0

RUN if [ "$CONFIG_FOR_LCOV" = "1" ]; then \
    mkdir -p /etc/yum.repos.d/backup && \
    mv /etc/yum.repos.d/*.repo /etc/yum.repos.d/backup/ && \
    curl -o /etc/yum.repos.d/Centos7-aliyun.repo https://mirrors.wlnmp.com/centos/Centos7-aliyun-altarch.repo -k && \
    yum clean all && \
    yum makecache; \
    fi

RUN if [ "$CONFIG_FOR_LCOV" = "1" ]; then \
    yum install -y perl-Module-Load-Conditional && \
    yum install -y perl-Digest-MD5 && \
    yum install -y "perl(DateTime)" && \
    yum install -y perl-CPAN*; \
    fi

# Install pip package(build)
RUN pip3.10 install pyyaml  \
    && pip3.10 install torch==2.10.0 \
    && pip3.10 install numpy==1.25.2

RUN pip3.11 install pyyaml  \
    && pip3.11 install torch==2.10.0 \
    && pip3.11 install numpy==1.25.2

RUN pip3.12 install pyyaml  \
    && pip3.12 install torch==2.10.0 \
    && pip3.12 install numpy==1.25.2

WORKDIR /home
