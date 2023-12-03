#!/bin/sh

conda activate cfl-lbm
pip install -e .
python source/run_cfl.py --analysis bdi --include_dem 0
python source/extended_analyses/cluster_questions.py --exp_id 0 --n_clusters 3
python source/extended_analyses/compare_aggregates.py --exp_id 0
python source/extended_analyses/compare_cca.py --total_bdi 0
python source/extended_analyses/compare_mbdi.py
python source/extended_analyses/compare_naive.py