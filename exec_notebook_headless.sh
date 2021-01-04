#!/bin/bash
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --inplace main.ipynb