#!/bin/bash
pip install --no-cache-dir ipykernel nbstripout
nbstripout --install
python -m ipykernel install --user --name=myenv --display-name 'python-devcontainer'