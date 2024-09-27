import click
import gymnasium as gym
import logging
import numpy as np
import os
import onnx
import onnxruntime as ort
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.base_class
import tensorflow as tf
import torch

from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

# while not called directly, we need to import this so the environments are registered
import balance_robot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create directories to hold models, logs, sim videos
MODEL_DIR = "models"
LOG_DIR = "logs"
RECORDING_DIR = "movies"


def algorithm_factory(algorithm_name: str, env: gym.Env) -> BaseAlgorithm:
    """
    Factory method to create algorithm. Will also include any extra params needed for this
    particular case (balancing robot)
    """
    if algorithm_name == "DDPG":
        policy_kwargs = dict(
            net_arch=dict(pi=[300, 200], qf=[200, 150])
        )
        action_noise = NormalActionNoise(
            mean=np.zeros(2),
            sigma=0.01 * np.ones(2)
        )
        model = stable_baselines3.DDPG(
            "MlpPolicy",
            env=env,
            verbose=1,
            device='cpu',
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
            action_noise=action_noise
        )
        return model
    else:
        algorithm_class = getattr(stable_baselines3, algorithm_name, None)
        if algorithm_class is None:
            return None
        model = algorithm_class(
            'MlpPolicy',
            env,
            verbose=1,
            device='cpu',
            tensorboard_log=LOG_DIR
        )
        return model


@click.command(name="convert", help="Convert a PyTorch model to ONNX format")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def convert(ctx: dict, environment: str):
    """ Converts model to ONNX format """
    env = gym.make(environment, render_mode='rgb_array')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}", "best_model.zip")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Converting model to ONNX")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    algorithm_class = getattr(stable_baselines3, algorithm_name, None)
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
@click.option('--show-i', is_flag=True, default=False, help="log model inputs to std out in Python array syntax")
@click.pass_context
def test(ctx: dict, environment: str, show_io: bool, show_i: bool):
    """ Test a model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}", "best_model.zip")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    algorithm_class = getattr(stable_baselines3, algorithm_name, None)
    model = algorithm_class.load(model_file, env=env)

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        action, _ = model.predict(obs)
        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(action)))
        if show_i and run_loop_count % 30 == 0:
            logger.info(str(list(obs)) + ",")

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

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}", "best_model.onnx")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting ONNX test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
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

        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(onnx_outputs)))

        obs, _, terminated, truncated, _ = env.step(onnx_outputs)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="test-tflite", help="Test a tflite model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.pass_context
def test_tflite(ctx: dict, environment: str, show_io: bool):
    """ Test a tflite model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # then assume default name
        model_file = os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}", "best_model.tflite")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting tflite test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        input_tensor = np.array([obs], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(output_data)))

        obs, _, terminated, truncated, _ = env.step(output_data)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="test-tflite-quant", help="Test a quantized tflite model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def test_tflite_quant(ctx: dict, environment: str):
    """ Test a tflite model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        raise RuntimeError(f"Must provide model file")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting tflite quantized test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        # convert observation values to quantized values using parameters from the tflite
        # quantized model
        # TODO: this needs to be changed each time the quantization is done
        obs_quant = [(obs_value / 0.026083452627062798 - 67) for obs_value in obs]

        # need to make sure the quantized values are within the range of an int8, otherwise
        # the values get wrapped and -129 becomes +127! Which is obviously bad for the
        # robots balance
        obs_quant = np.clip(obs_quant, a_min = -128, a_max = 127)

        input_tensor = np.array([obs_quant], dtype=np.int8)
        logger.info(input_tensor)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data_quant = interpreter.get_tensor(output_details[0]['index'])[0]

        logger.info(output_data_quant)

        # convert quantized model outputs back to floats as expected by the sim
        # TODO: this needs to be changed each time the quantization is done
        output_data = np.array(
            [
                output_data_quant[0] * 0.0078125,
                output_data_quant[1] * 0.0078125,
            ],
            dtype=np.float32
        )

        obs, _, terminated, truncated, _ = env.step(output_data)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="train", help="Train a model with a given environment")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
@click.pass_context
def train(ctx: dict, environment: str):
    """ Train a model by with a given MuJoCo environment """

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    env = gym.make(environment, render_mode='rgb_array')
    env = Monitor(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=RECORDING_DIR,
        video_length=0,
        episode_trigger = lambda x: x % 50 == 0,
    )

    logger.info(f"Starting training process")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # no model given, so create a new one
        # model = algorithm_class('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=LOG_DIR)
        model = algorithm_factory(algorithm_name=algorithm_name, env=env)
        logger.info(f"Model: starting with new model")
    elif os.path.isfile(model_file):
        # start with an existing model
        algorithm_class = getattr(stable_baselines3, algorithm_name, None)
        if algorithm_class is None:
            raise RuntimeError(f"Couldn't find algorithm {algorithm_name}")
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
        best_model_save_path=os.path.join(MODEL_DIR, f"{environment}_{algorithm_name}"),
    )

    model.learn(
        total_timesteps=int(1e10),
        tb_log_name=f"{environment}_{algorithm_name}",
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
    ctx.obj['ALGORITHM_NAME'] = algorithm
    ctx.obj['MODEL_PATH'] = model


cli.add_command(convert)
cli.add_command(test)
cli.add_command(test_onnx)
cli.add_command(test_tflite)
cli.add_command(test_tflite_quant)
cli.add_command(train)


def __make_folders():
    # create folders to store training data, models, logs, etc
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)


if __name__ == '__main__':
    __make_folders()
    cli(obj={})

