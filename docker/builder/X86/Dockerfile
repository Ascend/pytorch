FROM quay.io/pypa/manylinux_2_28_x86_64:2025.04.26-0

ENV PATH /usr/local/bin:$PATH

RUN echo "alias ll='ls -l --color=auto'" >> /root/.bashrc

# Set pip&python
RUN cd /usr/local/bin \
    && ln -sf /opt/python/cp310-cp310/bin/pip3.10 pip3.10 \
    && ln -sf /opt/python/cp311-cp311/bin/pip3.11 pip3.11 \
    && ln -sf /opt/python/cp312-cp312/bin/pip3.12 pip3.12 \
    && ln -sf /opt/python/cp310-cp310/bin/pip3.10 pip3 \
    && ln -sf /opt/python/cp310-cp310/bin/python3.10 python3.10 \
    && ln -sf /opt/python/cp311-cp311/bin/python3.11 python3.11 \
    && ln -sf /opt/python/cp311-cp311/bin/python3.12 python3.12 \
    && ln -sf /opt/python/cp310-cp310/bin/python3.10 python3 \
    && ln -sf /opt/python/cp310-cp310/bin/python3.10 python

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

RUN yum install -y vim-common\
    && yum install -y ninja-build \
    && yum install -y dos2unix \
    && yum install -y gcc-toolset-14 \
    && echo "source /opt/rh/gcc-toolset-14/enable" >> /root/.bashrc

# Install pip package(build)
RUN pip3.10 install pyyaml  \
    && pip3.10 install setuptools  \
    && pip3.10 install torch==2.11.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3.10 install numpy==1.23.2

RUN pip3.11 install pyyaml  \
    && pip3.11 install setuptools  \
    && pip3.11 install torch==2.11.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3.11 install numpy==1.26.2

RUN pip3.12 install pyyaml  \
    && pip3.12 install setuptools  \
    && pip3.12 install torch==2.11.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3.12 install numpy==1.26.2

WORKDIR /home
