# Dual GPU Training Guide

This setup allows you to run two RAMUSA training instances simultaneously on different GPUs.

## Configuration Files

- **`ramusa_config.yml`** - GPU 0 configuration
  - Device: `cuda:0`
  - Results: `results_ramusa/`
  - Checkpoints: `checkpoints_ramusa/`

- **`ramusa_config_gpu1.yml`** - GPU 1 configuration
  - Device: `cuda:1`
  - Results: `results_ramusa_gpu1/`
  - Checkpoints: `checkpoints_ramusa_gpu1/`

## Quick Start

### Option 1: Use the Automated Script (Easiest)

```bash
cd improvements/lowrank
bash run_dual_gpu.sh
```

This will:
- Start training on GPU 0 in the background
- Start training on GPU 1 in the background  
- Display process IDs and log file locations
- Return control to your terminal immediately

### Option 2: Manual Terminal 1 & 2

**Terminal 1 (GPU 0):**
```bash
cd improvements/lowrank
python run.py --task_type ks_si_er --config configs/ramusa_config.yml --device_index 0
```

**Terminal 2 (GPU 1):**
```bash
cd improvements/lowrank
python run.py --task_type ks_si_er --config configs/ramusa_config_gpu1.yml --device_index 1
```

### Option 3: Background with nohup

```bash
cd improvements/lowrank

# GPU 0
nohup python run.py --task_type ks_si_er --config configs/ramusa_config.yml --device_index 0 > /tmp/run_gpu0.log 2>&1 &

# GPU 1
nohup python run.py --task_type ks_si_er --config configs/ramusa_config_gpu1.yml --device_index 1 > /tmp/run_gpu1.log 2>&1 &
```

## Monitoring

### View Live Logs

```bash
# GPU 0
tail -f /tmp/run_gpu0_ramusa_config.log

# GPU 1
tail -f /tmp/run_gpu1_ramusa_config.log
```

### Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 'nvidia-smi'

# Or just check once
nvidia-smi
```

### List Running Processes

```bash
ps aux | grep "run.py"
```

## Output Locations

Both instances will save results to separate directories:

**GPU 0:**
```
results_ramusa/YYYY_MM_DD_HH_MM_SS/
├── train_ks_si_er.txt
├── eval_metrics_best.txt
├── eval_metrics_opt.txt
└── eval_predictions_*.csv

checkpoints_ramusa/YYYY_MM_DD_HH_MM_SS/
├── train_ks_si_er_best.pth
└── train_ks_si_er_opt.pth
```

**GPU 1:**
```
results_ramusa_gpu1/YYYY_MM_DD_HH_MM_SS/
├── train_ks_si_er.txt
├── eval_metrics_best.txt
├── eval_metrics_opt.txt
└── eval_predictions_*.csv

checkpoints_ramusa_gpu1/YYYY_MM_DD_HH_MM_SS/
├── train_ks_si_er_best.pth
└── train_ks_si_er_opt.pth
```

## Customizing Configurations

To test different hyperparameters on the two GPUs:

1. Edit `ramusa_config_gpu1.yml` to change settings (e.g., `smoothing_beta`, `min_spectral_gap`)
2. Keep `ramusa_config.yml` as baseline
3. Compare results in the separate output directories

Example changes in `ramusa_config_gpu1.yml`:

```yaml
ramusa:
  adaptive_tau:
    smoothing_beta: 0.80      # More stable
    min_spectral_gap: 0.15    # More sensitive
```

## Stopping Training

### Individual Process
```bash
# Replace PID with actual process ID from ps output
kill <PID>
```

### All RAMUSA Processes
```bash
pkill -f "run.py"
```

## Troubleshooting

### "CUDA out of memory" on one GPU
- Reduce `batch_size` in the config
- Check if another process is using GPU memory
- Use `nvidia-smi` to verify GPU availability

### Processes not starting
- Check if ports are available
- Verify GPU indices are correct: `nvidia-smi`
- Check Python environment: `which python && python --version`

### Results not saving
- Verify directory permissions: `ls -la results_ramusa*`
- Check disk space: `df -h`
- Check log files for detailed errors

## Key Points

✅ **Isolation:** Each instance has separate:
- Output directories
- Checkpoint files
- Log files
- GPU memory

✅ **Synchronization:** Training on both GPUs:
- Starts immediately after launch
- Runs fully independently
- Can complete at different times

✅ **Comparison:** Easy A/B testing with:
- Identical code, different configs
- Same dataset
- Different hyperparameters
- Separate result tracking

## Next Steps

1. Verify both GPUs are available: `nvidia-smi`
2. Start dual training: `bash run_dual_gpu.sh`
3. Monitor both processes: `watch nvidia-smi`
4. Compare results when complete
