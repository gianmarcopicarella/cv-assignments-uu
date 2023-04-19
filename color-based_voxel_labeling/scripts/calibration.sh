#!/bin/bash

step=20

python3 ../calibration.py -l ../data/checkerboard.xml -c ../data/cam1 -s $step
python3 ../calibration.py -l ../data/checkerboard.xml -c ../data/cam2 -s $step
python3 ../calibration.py -l ../data/checkerboard.xml -c ../data/cam3 -s $step
python3 ../calibration.py -l ../data/checkerboard.xml -c ../data/cam4 -s $step