#!/bin/bash
jupyter nbconvert --to script Training.ipynb
# execute converted file and use the arguments after the '--'
ipython Training.py -- -c '{"training_parameter":{"loss_weights":{"classifier":0.1}}}' # here you could place even more arguments...