#!/bin/bash
cd /home/claude/bench
for ds in M4D M4H; do python run_bench.py $ds ridge; done >> logs_ridge2.txt 2>&1
for ds in M4D M4H; do python run_bench.py $ds lgbm; done >> logs_lgbm2.txt 2>&1
python pat.py > logs_pat2.txt 2>&1
echo ALLDONE > alldone.flag
