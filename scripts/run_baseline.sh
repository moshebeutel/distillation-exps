#!/bin/bash
nohup mlflow server --host 127.0.0.1 --port 8080&
python3 ../doubledistill/ddistexps/baseline/main.py  --expname debug
