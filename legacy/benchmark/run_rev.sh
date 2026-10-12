#!/bin/bash
cd /home/claude/bench
python recover_bench.py > logs_rev.txt 2>&1
python pat_crn.py >> logs_rev.txt 2>&1
python timing_fit.py >> logs_rev.txt 2>&1
python pat_ablation.py >> logs_rev.txt 2>&1
python pat.py > logs_pat3.txt 2>&1
python run_bench.py ELEC ridge >> logs_rev.txt 2>&1
python run_bench.py ELEC lgbm >> logs_rev.txt 2>&1
for ds in M5 M4D M4H ELEC; do NOSHIFT=1 python run_bench.py $ds xgb; done >> logs_xgb.txt 2>&1
for ds in M5 M4D M4H ELEC; do NOSHIFT=1 python run_bench.py $ds mlp; done >> logs_mlp.txt 2>&1
echo ALLDONE > alldone_rev.flag
