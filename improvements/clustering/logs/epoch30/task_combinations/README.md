# Task combination runs (30 epochs)

Run `bash configs/task_combinations/run_task_combinations_30ep.sh 0` from
`improvements/clustering`, or launch it with `nohup` and `disown`.
The script runs `ks_si_er` first, then `ks_si`, `si_er`, `ks_er`, `ks`, `si`,
and `er` sequentially on the selected GPU. Console logs are named by task and
start time under `improvements/clustering/logs/`. The application logger also
writes its timestamped files directly there.
Training metrics, predictions, and cluster summaries are under
`logs/epoch30/task_combinations/results/<task_type>/<timestamp>/`; checkpoints
are under `checkpoints_ncmtl_task_combinations_30ep/<task_type>/<timestamp>/`.

The three-task soft sharing rule compares the three candidate pairs. A two-task
run has one pair; after adaptive warm-up, each row is gated by its pair distance
divided by the sum of the two row norms. A distance at or below 0.05 shares
fully, at or above 0.20 remains independent, and between them shares partially.
A single-task run has no clustering or sharing; its candidate layer remains.
These runs use seed 42 and all 25 WavLM Large embedding rows.
