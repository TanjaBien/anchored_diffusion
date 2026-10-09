"""Train a diffusion model with the anchored physics term (or PIDM, or the data-only DM), then
draw its evaluation samples and compute the paper's metrics.

    python main.py --config config/burgers.yaml
    python main.py --config config/burgers.yaml --method pidm --c_residual 1e-6
    python main.py --resume_checkpoint trained_models/burgers/<run>
"""
from src.config import checkpoint_step, device, fix_seeds, latest_checkpoint, parse_args_and_config, run_directory
from src.datasets import build_problem_setup
from src.diffusion_process import DenoisingDiffusion, load_model
from src.evaluation.generation import evaluate_run
from src.models import build_model
from src.residuals import build_residual
from src.training import describe, train


def main():
    _args, config = parse_args_and_config()
    fix_seeds(config['seed'])
    setup = build_problem_setup(config)
    model = build_model(setup)
    start_iteration = 0
    if config['resume_checkpoint']:
        checkpoint = latest_checkpoint(config['resume_checkpoint'])
        load_model(checkpoint, model)
        start_iteration = (checkpoint_step(checkpoint) or 0) + 1
    residual = build_residual(setup, device)
    diffusion = DenoisingDiffusion(config['diff_steps'], device, config['snr_gamma'])

    run_dir = run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(describe(model, config), f'-> {run_dir}')
    train(config, setup, model, residual, diffusion, run_dir, start_iteration)

    if config['generate_after_training']:
        del model
        evaluate_run(run_dir, force=True)


if __name__ == '__main__':
    main()
