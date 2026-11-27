# Pairwise distance gate: 0.30 / 0.50

Runs KS–SI, SI–ER, and KS–ER sequentially for 30 epochs, seed 42, using all
layers. After warm-up, beta is 1 at normalized distances <= 0.30, 0 at >= 0.50,
and interpolated linearly between. Training settings match the earlier runs.

Results are written to `gate_0p30_0p50/results/<task_type>/<timestamp>/` under
`logs/epoch30/task_combinations/`. Checkpoints use the separate
`checkpoints_ncmtl_task_combinations_gate_0p30_0p50_30ep` root.
Console logs contain `gate_0p30_0p50` in their filenames under `clustering/logs/`;
application logs also stay in `clustering/logs/` for manual organization.

From `improvements/clustering`, launch with
`bash configs/task_combinations/run_gate_0p30_0p50_pairs_30ep.sh 1`.
