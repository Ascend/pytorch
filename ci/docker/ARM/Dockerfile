FROM quay.io/pypa/manylinux2014_aarch64:2023-10-07-c1e05d1

# Set pip
RUN cd /usr/local/bin \
    && ln -s /opt/_internal/cpython-3.9.18/bin/pip3.9 pip3.9 \
    && ln -s /opt/_internal/cpython-3.10.13/bin/pip3.10 pip3.10 \
    && ln -s /opt/_internal/cpython-3.11.6/bin/pip3.11 pip3.11 \
    && ln -s python3.9 python3

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
RUN pip3.9 install pyyaml  \
    && pip3.9 install torch==2.5.1 \
    && pip3.9 install numpy==1.21.3

RUN pip3.10 install pyyaml  \
    && pip3.10 install torch==2.5.1 \
    && pip3.10 install numpy==1.21.3

RUN pip3.11 install pyyaml  \
    && pip3.11 install torch==2.5.1 \
    && pip3.11 install numpy==1.23.2

WORKDIR /home
