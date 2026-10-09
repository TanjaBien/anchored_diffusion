<div align="center"> <h1>Characterizing and Mitigating Distributional Bias in Physics-Informed Generative Models</h1> </div>
<h4 align="center">

Official implementation of the **data-anchored residual penalty** for physics-informed diffusion models. Residual
penalties such as PIDM's `‖R(x̂₀)‖²` can favor states where the same prediction error costs less (*avoidance*) and
extend this preference to physically admissible states outside the data (*leakage*). The anchored penalty weights
the prediction error by the residual Jacobian at the clean training sample and normalizes it,
`‖J_R(x₀)(x̂₀ − x₀)‖² / ‖J_R(x₀)‖²_F`, without changing the model or the sampler. This repository contains the code
for training, sampling and evaluation on a synthetic shocked circle and the advection, Burgers and Darcy equations.

### Installation
Set up your environment using Conda:

```bash
conda env create -f environment.yml
conda activate anchored
```

### Datasets

The data is expected under `data/` (or set `--data_dir`).

1. **Advection and Burgers:** generated with spectral solvers. The default settings and seeds reproduce the
   paper's datasets exactly.

```bash
python src/datasets/generation/advection.py   # data/advection/1d_advection_beta1.0.h5
python src/datasets/generation/burgers.py     # data/burgers/1d_burgers_nu0.001.h5
```

2. **Darcy (manual):** the dataset of [PIDM](https://github.com/jhbastek/PhysicsInformedDiffusionModels).
   * Download it from the [ETHZ Research Collection](https://doi.org/10.3929/ethz-b-000674074).
   * Place its `darcy` folder (with `train/` and `valid/`) in `data/`.

3. **Shocked circle:** no action required, the data is sampled on the fly.

Check the data before training. The script reports the residual of the ground truth and, for Darcy, the error of the
stored pressure against a fresh solve, and saves a few fields with their residual to `results/data_validation/`:

```bash
python -m src.datasets.validation
```

Expected mean |R| of the data: advection ~6e-6, Burgers ~2.5e-4, Darcy ~2.8e-3 (solution error ~3.4e-5),
shocked circle ~1e-7.

## Training

Each config in `config/` holds the complete training protocol of the paper; any option can be overridden on the
command line. The method is selected with `--method` and the penalty weight with `--c_residual`:

```bash
python main.py --config config/burgers.yaml                                  # anchored
python main.py --config config/burgers.yaml --method pidm --c_residual 1e-6  # PIDM
python main.py --config config/burgers.yaml --method dm                      # no penalty
```

| `--method` | penalty | time weight |
|---|---|---|
| `anchored` | `‖J_R(x₀) d‖² / ‖J_R(x₀)‖²_F`, `d = x̂₀ − x₀` | `min(SNR_t, γ)` |
| `pidm` | `‖R(x̂₀)‖²` | `1 / (2 β̃_t)` |
| `dm` | none | |


Runs are saved to `trained_models/<dataset>/`. To add a method, see `src/methods/registry.py`; to add a dataset,
`src/datasets/registry.py` and `src/residuals/`.

### Evaluation
When training ends, the evaluation samples are drawn and the metrics of the paper (mean |R|, MMD, σ ratio, std
ratio, Darcy solution error; circular W₁ and distance to the data plane on the shocked circle) are written to
`<run>/evaluation/report.json`. To reproduce a table of the paper, run the weight sweep and collect the results:

```bash
bash scripts/sweep.sh burgers                  # DM, PIDM and anchored over the paper's weights
python -m src.evaluation.tables burgers        # results/table_burgers.csv
```

To run everything on one machine, `bash scripts/run_all.sh` starts a tmux session that trains and evaluates all
four datasets and writes their tables (`tmux attach -t anchored` to watch; on a GPU with less than ~30 GB, add
`ARGS_BURGERS="--train_batch_size 4"`). The sweep runs on one GPU by default; `GPUS="0 1 2 3"` runs one job per GPU. A run can be re-evaluated with
`python -m src.evaluation.generation trained_models/burgers/<run> --force`. The metrics are defined in
`src/evaluation/paper_metrics.py`.


## Acknowledgments

This project builds upon code and data from:

* [PIDM](https://github.com/jhbastek/PhysicsInformedDiffusionModels) (Physics-Informed Diffusion Models): U-Net, Darcy data and residual
* [torchfsm](https://qiauil.github.io/torchfsm) (spectral PDE solvers) and [findiff](https://github.com/maroba/findiff)

## License

This project is released under the [MIT License](LICENSE). The U-Net (`src/models/unet.py`) is adapted from
PIDM and video-diffusion-pytorch, both MIT-licensed; see the file header.
