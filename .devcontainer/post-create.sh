#!/bin/bash
apt update
apt install -y python3 python3-pip
pip install --no-cache-dir ipykernel nbstripout
python3 -m ipykernel install --user --name=myenv --display-name 'python-gpu-dev-container'