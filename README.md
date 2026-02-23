# AI Optimizers Benchmark

Publication-quality benchmarking suite comparing **Adam**, **AdaMuon**, and **K-FAC** optimizers across 6 workloads.

## Quick Setup (GPU Server)

```bash
# 1. Clone and install
git clone <your-repo-url>
cd ai-optimizers-benchmark
uv sync

# 2. Run ALL experiments overnight (single command)
nohup bash scripts/run_all.sh results/ 8  2>&1 | tee benchmark.log &

# The "8" is the number of GPUs — adjust for your setup
```

## Monitor Progress 

```bash
# Live log output
tail -f benchmark.log

# Quick progress summary (updates in real-time)
cat results/PROGRESS.txt

# Check a specific experiment
tail -f results/logs/cifar10_resnet18_adam_seed42.log
```

## After Completion

```bash
# Generate all publication plots + LaTeX tables
uv run python scripts/plot_results.py --results-dir results/
# Outputs: plots/*.pdf, plots/*.svg, plots/summary_table.tex
```

## Run a Single Experiment

```bash
uv run python scripts/run_experiment.py \
  --config configs/cifar10_resnet18.yaml \
  --optimizer adam \
  --seed 42 \
  --gpu 0
```

## Workloads

| Config | Dataset | Model | Epochs |
|--------|---------|-------|--------|
| `fmnist_simplecnn` | Fashion-MNIST | SimpleCNN | 40 |
| `cifar10_resnet18` | CIFAR-10 | ResNet-18 | 100 |
| `cifar100_resnet34` | CIFAR-100 | ResNet-34 | 100 |
| `svhn_wrn164` | SVHN | WRN-16-4 | 80 |
| `adult_mlp` | UCI Adult | MLP (256-128-64) | 50 |
| `covertype_mlp` | Covertype | MLP (512-256-128) | 50 |

## Optimizers

- **Adam** — `torch.optim.Adam` (PyTorch built-in)
- **AdaMuon** — Single-GPU implementation from [arXiv 2507.11005](https://arxiv.org/abs/2507.11005)
- **K-FAC** — Martens & Grosse (ICML 2015), supports Conv2d + Linear

## Resume Support

If the run crashes or you need to restart, just run the same command again.
The script automatically skips already-completed experiments (checks for `metrics.json`).

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU(s)
- [UV](https://docs.astral.sh/uv/) package manager
