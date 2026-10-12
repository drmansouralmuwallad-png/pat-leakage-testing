#!/bin/bash
cd /home/claude/bench
for ds in M5 M4D M4H; do python run_bench.py $ds ridge; done > logs_ridge.txt 2>&1
python pat.py > logs_pat.txt 2>&1
for ds in M5 M4D M4H; do python run_bench.py $ds lgbm; done > logs_lgbm.txt 2>&1
echo ALLDONE > alldone.flag
