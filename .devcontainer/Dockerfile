FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 必要なパッケージを root 権限で全てインストール
RUN apt update && apt install -y \
    python3 python3-pip \
    git curl \
 && pip3 install --no-cache-dir ipykernel nbstripout \
 && python3 -m ipykernel install --prefix=/usr/local --name=myenv --display-name "python-gpu-dev-container"

# vscodeユーザーを作成（UID/GIDはデフォルト）
RUN groupadd -g 1000 vscode \
 && useradd -m -u 1000 -g 1000 -s /bin/bash vscode