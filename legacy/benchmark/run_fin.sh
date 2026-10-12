#!/bin/bash
cd /home/claude/bench
python pat.py > logs_pat2.txt 2>&1
python pat_nv.py > logs_patnv2.txt 2>&1
python run_bench.py M4H lgbm >> logs_lgbm2.txt 2>&1
echo ALLDONE > alldone.flag
