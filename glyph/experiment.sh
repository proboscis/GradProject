#!/bin/bash
mkdir $2
unbuffer ./experiment.py $1 $2 $3 | tee "$2/log.txt"
