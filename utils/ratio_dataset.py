import os
import gymnasium as gym
import minari
import torch
import random
import numpy as np
from tqdm import trange
from collections import defaultdict

def ratio_dataset(dataset_path, env_name, ratio):
    random.seed(1234)
    np.random.seed(1234)
    
    dataset = minari.load_dataset(env_name, download=True)
    all_episodes = list(dataset.iterate_episodes())  # <-- correct way to access episode data

    total_episodes = len(all_episodes)
    num_sample = int(total_episodes * ratio)
    selected_episodes = random.sample(all_episodes, num_sample)

    new_dataset = {
        "observations": np.concatenate([traj.observations for traj in selected_episodes], axis=0),
        "actions": np.concatenate([traj.actions for traj in selected_episodes], axis=0),
        "rewards": np.concatenate([traj.rewards for traj in selected_episodes], axis=0),
        "terminals": np.concatenate([traj.terminations for traj in selected_episodes], axis=0),
        "timeouts": np.concatenate([traj.truncations for traj in selected_episodes], axis=0),
    }

    # os.mkdir(os.path.join(dataset_path, "original"))
    save_path = os.path.join(dataset_path, "original", f"{env_name}_ratio_{ratio}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(new_dataset, save_path)

    print(f"Save downsample dataset in {dataset_path}")
    print(f"=============================================")
    print(f"Env: {env_name}")
    print(f"Trajectory number: {total_episodes} -> {num_sample}")
    total_transitions = sum(len(ep.actions) for ep in all_episodes)
    selected_transitions = sum(len(ep.actions) for ep in selected_episodes)
    print(f"Transition number: {total_transitions} -> {selected_transitions}")
    print(f"=============================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--dataset_path", type=str, default="your_path_of_dataset")
    args = parser.parse_args()
    ratio_dataset(args.dataset_path, args.env_name, args.ratio)







