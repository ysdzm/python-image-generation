#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
pip install --no-cache-dir ipykernel nbstripout
nbstripout --install
python3 -m ipykernel install --user --name=myenv --display-name 'python-devcontainer'