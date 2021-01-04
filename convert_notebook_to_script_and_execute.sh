#!/bin/bash
jupyter nbconvert --to script main.ipynb
# execute converted file and use the arguments after the '--'
ipython main.py -- # --batch_size 16 --epochs 1000 # here you could place even more arguments...