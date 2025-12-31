#!/usr/bin/env bash
virtualenv -p python3.10 stco-env
source stco-env/bin/activate
echo $(which python3)
# python3 -m pip install -r ./requirements.txt
