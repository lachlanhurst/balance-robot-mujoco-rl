# Script converts a saved pytorch model to onnx format
import argparse
import gymnasium as gym
import stable_baselines3
import torch
import os
from stable_baselines3 import PPO

# while not called directly, we need to import this so the environments are registered
import balance_robot

model_dir = "models"

if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')    
    args = parser.parse_args()

    # Dynamic way to import algorithm. For example, passing in DQN is equivalent to hardcoding:
    # from stable_baselines3 import DQN
    sb3_class = getattr(stable_baselines3, args.sb3_algo)

    env = gym.make(args.gymenv, render_mode='rgb_array')
    model = sb3_class.load(os.path.join(model_dir, f"{args.gymenv}_{args.sb3_algo}", "best_model"), env=env)

    # Get the underlying PyTorch model (policy network)
    pytorch_model = model.policy

    # Create a dummy input based on the observation space
    dummy_input = torch.zeros(1, *env.observation_space.shape)

    output_filename = os.path.join(model_dir, f"{args.gymenv}_{args.sb3_algo}", "best_model.onnx")
    print(f"Output file: {output_filename}")

    # Export the model to ONNX format
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_filename,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
