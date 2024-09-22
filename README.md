# balance-robot-mujoco-rl
Use reinforcement learning to train self balancing robot in MuJoCo environment




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