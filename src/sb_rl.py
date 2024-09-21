import click
import gymnasium as gym
import logging
import numpy as np
import os
import onnx
import onnxruntime as ort
import stable_baselines3
import torch

from pathlib import Path
from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
)
from stable_baselines3.common.monitor import Monitor

# while not called directly, we need to import this so the environments are registered
import balance_robot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create directories to hold models, logs, sim videos
MODEL_DIR = "models"
LOG_DIR = "logs"
RECORDING_DIR = "movies"


@click.command(name="convert", help="Convert a PyTorch model to ONNX format")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def convert(ctx: dict, environment: str):
    """ Converts model to ONNX format """
    env = gym.make(environment, render_mode='rgb_array')

    algorithm_class = ctx.obj['ALGORITHM_CLASS']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_class.__name__}", "best_model.zip")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Converting model to ONNX")
    logger.info(f"Algorithm: {algorithm_class.__name__}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    model = algorithm_class.load(model_file, env=env)

    # Get the underlying PyTorch model (policy network)
    pytorch_model = model.policy

    # The model must be in evaluation mode for export. If it's in training mode, behaviors
    # like dropout and batch normalization will yield non-deterministic results.
    pytorch_model.eval()

    # Create a dummy input based on the observation space
    dummy_input = torch.zeros(1, *env.observation_space.shape)

    input_path = Path(model_file)
    output_path = input_path.with_suffix('.onnx')

    logger.info(f"Output file: {str(output_path)}")
    # Export the model to ONNX format
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_path),
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )


@click.command(name="test", help="Test the current model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.pass_context
def test(ctx: dict, environment: str, show_io: bool):
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

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        action, _ = model.predict(obs)
        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(action)))
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="test-onnx", help="Test an ONNX model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.pass_context
def test_onnx(ctx: dict, environment: str, show_io: bool):
    """ Test an ONNX model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_class = ctx.obj['ALGORITHM_CLASS']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_class.__name__}", "best_model.onnx")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting ONNX test simulation")
    logger.info(f"Algorithm: {algorithm_class.__name__}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(model_file)
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        input_tensor = np.array([obs], dtype=np.float32)
        onnx_outputs = session.run([output_name], {input_name: input_tensor})[0][0]
        # WHY!?
        # is this because the gym action space is -2.0 to 2.0 and not -1.0 to 1.0?
        onnx_outputs[0] = onnx_outputs[0] * 2
        onnx_outputs[1] = onnx_outputs[1] * 2

        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(onnx_outputs)))

        obs, _, terminated, truncated, _ = env.step(onnx_outputs)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="train", help="Train a model with a given environment")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def train(ctx: dict, environment: str):
    """ Train a model by with a given MuJoCo environment """

    algorithm_class = ctx.obj['ALGORITHM_CLASS']

    env = gym.make(environment, render_mode='rgb_array')
    env = Monitor(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=RECORDING_DIR,
        video_length=0,
        episode_trigger = lambda x: x % 50 == 0,
    )

    logger.info(f"Starting training process")
    logger.info(f"Algorithm: {algorithm_class.__name__}")
    logger.info(f"Environment: {environment}")

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # no model given, so create a new one
        model = algorithm_class('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=LOG_DIR)
        logger.info(f"Model: starting with new model")
    elif os.path.isfile(model_file):
        # start with an existing model
        model = algorithm_class.load(model_file, env=env, tensorboard_log=LOG_DIR)
        logger.info(f"Model: starting with {model_file}")
    else:
        raise RuntimeError(f"Model file {model_file} does not exist")

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=6000, verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10000,
        verbose=1
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=10000,
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_train_callback,
        verbose=1,
        best_model_save_path=os.path.join(MODEL_DIR, f"{environment}_{algorithm_class.__name__}"),
    )

    model.learn(
        total_timesteps=int(1e10),
        tb_log_name=f"{environment}_{algorithm_class.__name__}",
        callback=eval_callback
    )


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


cli.add_command(convert)
cli.add_command(test)
cli.add_command(test_onnx)
cli.add_command(train)


def __make_folders():
    # create folders to store training data, models, logs, etc
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)


if __name__ == '__main__':
    __make_folders()
    cli(obj={})

