# Reinforcement Learning for a Self Balancing Robot
Code for creating a trained policy that can be used by a two wheeled self balancing robot. This includes the following steps;
- Training a policy using reinforcement learning (Stable Baselines3 / PyTorch) in several simulation environments (MuJoCo)
- Testing the policy within the simulation environments
- Quantization of the trained model from float32 to int8 so the policy can be run on microcontrollers using TensorFlow Lite (TFLite/LiteRT)

![MuJoCo simulation](./docs/mujoco_rl_robot.gif) ![Real robot](./docs/real_robot.gif)


# Setup

Create and activate the conda environment. This will install dependencies needed for training, testing, and converting the model.

    conda env create -f conda-environment.yaml
    conda activate robot-mujoco-rl


# General operation

The training, testing, and conversion process is for the most part is performed by a command line application implemented in [`sb_rl.py`](./src/sb_rl.py).

Command line usage instructions can be found by running the following command.

    python sb_rl.py --help

Which will display

    Usage: sb_rl.py [OPTIONS] COMMAND [ARGS]...

    Options:
    -a, --algorithm TEXT  Stable Baseline3 algorithm (eg; A2C, DDPG, DQN, PPO,
                            SAC, TD3)  [required]
    -m, --model PATH      Path to model file
    --help                Show this message and exit.

    Commands:
    convert              Convert a PyTorch model to ONNX format
    test                 Test the current model
    test-onnx            Test an ONNX model
    test-tflite          Test a tflite model
    test-tflite-arduino  Test a quantized tflite model on the arduino
    test-tflite-quant    Test a quantized tflite model
    train                Train a model with a given environment

Command help can be shown by including the command name, for example

    python sb_rl.py -a PPO train --help

Example command lines are included in the following sections.


# Training process

The following commands will train a policy in multiple environments. These commands all use the PPO algorithm, which is recommended.

Create and train a policy to balance using Env01. Use tensorboard to monitor training progress and stop when policy balances consistently.

    python sb_rl.py -a PPO train -e Env01-v2

Start with the pre-trained policy from the previous command, and perform additional training with Env03 to improve balancing robustness. As per previous step, stop when the robot maintains balance consistently.

    python sb_rl.py -a PPO -m ./models/Env01-v2_PPO/best_model.zip train -e Env03-v2

Review performance of the trained policy by testing interactively in the environment. This will launch a MuJoCo simulation similar to that shown in the gif above.

        python sb_rl.py -a PPO -m ./models/Env03-v2_PPO/best_model.zip test -e Env03-v2


# Process commands

Convert the PyTorch model to ONNX

    python sb_rl.py -a SAC -m ../backup/best_model.zip  convert -e Env01-v1

Test the ONNX model. Confirm the robot still balances as well as the PyTorch model.

    python sb_rl.py -a SAC -m ../backup/best_model.onnx test-onnx -e Env01-v1

Start the onnx2tf docker contain, will open shell

    docker run --rm -it \
        -v `pwd`:/workdir \
        -w /workdir \
        docker.io/pinto0309/onnx2tf:1.25.12

In the docker shell convert ONNX model to tflite (is this correct???)

    onnx2tf -i ./balance-robot-mujoco-rl/backup/best_model.onnx -b 1 -osd

Exit docker container

Test the tflite model. Confirm the robot still balances as well as the PyTorch model.

    python sb_rl.py -a SAC -m ../../saved_model/best_model_float32.tflite test-tflite -e Env01-v1

Copy/paste output of following command into `quantize_tflite.py`

    python sb_rl.py -a SAC -m ../backup/best_model.zip test -e Env01-v1 --show-i

Get the input name from the outputs of the following command `inputs['input'] tensor_info:` -> `input`. Update the name in the `representative_dataset` function included in `quantize_tflite.py`

    saved_model_cli show --dir saved_model/ --all

Transform the tflite model into an int8 quantized tflite model

    python balance-robot-mujoco-rl/src/quantize_tflite.py

Open int8_model.tflite in Neuron, get quantization params and update `test_tflite_quant` function in sb_rl.py

Test the quantized model

    python sb_rl.py -a SAC -m ../../saved_model/int8_model.tflite test-tflite-quant -e Env01-v1

Convert quantized model into c array

    xxd -i int8_model.tflite > model.h

The c array can then be included in the microcontroller code along with a suitable tflite library to run inference.


# Training notes

## Environment descriptions

# Env01
Env01-v1 : Simple environment, used to teach robot to balance on two wheels.
Env01-v2 : Same as v1, but adds noise to pitch and pitch dot observations. Use this for training instead of v1.

# Env02
Env02-v1 : Same as Env01, but adds randomised friction to contact between wheel and ground.

# Env03
Env03-v1 : Blocks are thrown at robot from any random direction. While fun to watch, is generally too chaotic for good training performance.
Env03-v1-fail: big "FAIL" block is dropped on robot when it falls over. Don't use for training, created for comedic value.
Env03-v2 : Blocks come from either front or back (random), but are thrown from the same side consistently over an episode. This is case that caused the most problems in v1, isolating this behaviour in v2 resulted in much better training.


## Reinforcement Learning Algorithms
Some rough notes on training this self balancing robot model.

A2C
- Didn't train well

TD3
- Didn't train well

SAC
- Trained well
- Reasonable model size
- Results in a an Exp (exponential) op being included in quantized tflite model. Unfortunately this op doesn't support int8/uint8 inputs in the tflite lib, and as a result this **model could not be run on the microcontroller**.

DDPG
- Trained ok, once
- Resulting model had weird behavior where it would balance, but delta wheel speed alternated rapidly between min and max
- Large model size
- Successfully ran on microcontroller
- Microcontroller tflite inference results on int8 quantized model do not match that of desktop tensorflow using same model. Weird model behavior prevented further validation.

PPO
- On par, maybe better than SAC in respect to training
- Small model size
- Model include an Exp op (this prevented SAC from being used), but the onnx2tf model simplification step removes this op from the model.
- Successfully ran on microcontroller
- Microcontroller tflite inference results on int8 quantized model do not match that of desktop tensorflow using same model. Tested MuJoCo simulation with real time inference results from microcontroller and the model works.
- **Recommended algorithm**
