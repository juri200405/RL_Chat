FROM nvidia/cuda:11.0-base

SHELL ["/bin/bash", "-c"]

# build時だけ使う変数の宣言
# 値はdocker-compose.ymlから渡される
ARG U_ID
ARG G_ID
ARG USER_NAME
ARG HOME_DIR

ENV DEBIAN_FRONTEND=noninteractive

# ソフトウェアインストール
RUN apt update --fix-missing && \
    apt upgrade --fix-missing -y && \
    apt install -y \
        locales \
        wget \
        ca-certificates \
        git \
        make \
        curl \
        xz-utils \
        file \
        sudo \
        net-tools \
	graphviz \
        && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean && \
    localedef -i ja_JP -c -f UTF-8 -A /usr/share/locale/locale.alias ja_JP.UTF-8

# user作成
# uid,gid,usernameはホストのものにする
RUN groupadd -g ${G_ID} ${USER_NAME} && \
    useradd -u ${U_ID} -g ${G_ID} -s /bin/bash -m ${USER_NAME}
USER ${U_ID}
WORKDIR ${HOME_DIR}

# 環境変数
ENV PATH ~/conda/bin:$PATH
ENV LANG ja_JP.utf8

# minicondaインストール
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/conda && \
    rm ~/miniconda.sh && \
    ~/conda/bin/conda clean -tipsy && \
#    ln -s ~/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ~/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find ~/conda/ -follow -type f -name '*.a' -delete && \
    find ~/conda/ -follow -type f -name '*.js.map' -delete && \
    ~/conda/bin/conda clean -afy && \
    . ~/.bashrc

RUN conda update -n base -c defaults conda

# pythonのパッケージをconda環境にインストール
COPY ./environment.yml ./
RUN conda env create -f environment.yml
RUN echo "conda activate pl" >> ~/.bashrc

# RUN conda activate pl
# RUN git clone https://github.com/NVIDIA/apex
# RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

ENV PYTHONPATH src:torchviz
ENV LRU_CACHE_CAPACITY 1
