import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
import logging
import os
import click

# while not called directly, we need to import this so the environments are registered
import balance_robot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create directories to hold models, logs, sim videos
MODEL_DIR = "models"
LOG_DIR = "logs"
RECORDING_DIR = "movies"


@click.command(name="test", help="Test the current model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def test(ctx: dict, environment: str):
    """ Test a model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_class = ctx.obj['ALGORITHM_CLASS']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_class.__name__}", "best_model.zip")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting test simulation")
    logger.info(f"Algorithm: {algorithm_class.__name__}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    model = algorithm_class.load(model_file, env=env)

    obs = env.reset()[0]   
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break


@click.group()
@click.option(
    '-a',
    '--algorithm',
    required=True,
    type=str,
    help="Stable Baseline3 algorithm (eg; A2C, DDPG, DQN, PPO, SAC, TD3)"
)
@click.option(
    '-m',
    '--model',
    default=None,
    type=click.Path(exists=False),
    help="Path to model file"
)
@click.pass_context
def cli(ctx: dict, algorithm: str, model: str | os.PathLike):
    algorithm_class = getattr(stable_baselines3, algorithm, None)
    if algorithm_class is None:
        raise RuntimeError(f"Could not find Stable Baselines3 algorithm for: {algorithm}")

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['ALGORITHM_CLASS'] = algorithm_class
    ctx.obj['MODEL_PATH'] = model
    print(model)


cli.add_command(test)


def __make_folders():
    # create folders to store training data, models, logs, etc
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)


if __name__ == '__main__':
    __make_folders()
    cli(obj={})

