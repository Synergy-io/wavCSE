# Fully shared task pairs

Run `bash configs/task_combinations/run_fully_shared_pairs_30ep.sh 1` from
`improvements/clustering`. It runs KS–SI, SI–ER, and KS–ER sequentially on the
chosen GPU, with seed 42, all layers, and 30 epochs.

Pair thresholds 1.0 and 1.1 make beta equal to 1 after the existing adaptive
warm-up. Both candidate matrices share every row; classifiers remain separate.
This is a fixed full-sharing comparison with the earlier distance-gated runs.

Results are in `fully_shared/results/<task_type>/<timestamp>/` relative to
`logs/epoch30/task_combinations/`. Checkpoints are in
`checkpoints_ncmtl_task_combinations_fully_shared_30ep/<task_type>/<timestamp>/`
relative to `improvements/clustering/`.
Console logs use `*_fully_shared_seed42_30ep_*.log` in `clustering/logs/`;
application logs also use the default `clustering/logs/` directory. Logs can be
moved here manually after training.
