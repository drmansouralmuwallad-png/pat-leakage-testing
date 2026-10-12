#!/bin/bash
cd /home/claude/bench
for ds in M5 M4D M4H ELEC; do SUBSET=15 NOSHIFT=1 python run_bench.py $ds mlp; done >> logs_mlp.txt 2>&1
echo ALLDONE > alldone_mlp.flag
