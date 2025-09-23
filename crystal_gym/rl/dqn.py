"""
Deep Q-Network (DQN) implementation for crystal design optimization.
"""

import os
import random
import signal
import time
from copy import deepcopy
from functools import partial
from typing import Dict, Any, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import ReplayBuffer, ListStorage

import hydra
from crystal_gym.agents import MEGNetRL
from crystal_gym.env import CrystalGymEnv
from crystal_gym.utils import collate_function

# Global signal handler for graceful shutdown
caught_signal = False

def catch_signal(sig: int, frame: Any) -> None:
    """Signal handler for graceful shutdown."""
    global caught_signal
    caught_signal = True


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, kwargs: Dict[str, Any]) -> callable:
    """
    Create an environment factory function.
    
    Args:
        env_id: Environment identifier
        idx: Environment index
        capture_video: Whether to record video
        run_name: Name for the run
        kwargs: Environment configuration parameters
        
    Returns:
        Environment factory function
    """
    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, kwargs=kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

class QNetwork(nn.Module):
    """
    Q-Network for Deep Q-Learning.
    
    Supports both MLP and MEGNetRL architectures for crystal design optimization.
    """
    
    def __init__(self, env: gym.Env, network_type: str = "mlp") -> None:
        """
        Initialize the Q-Network.
        
        Args:
            env: Gymnasium environment
            network_type: Type of network architecture ("mlp" or "MEGNetRL")
        """
        super().__init__()
        self.network_type = network_type

        if network_type == "mlp":
            # Simple MLP architecture
            obs_dim = np.array(env.single_observation_space.shape).prod()
            self.network = nn.Sequential(
                nn.Linear(obs_dim, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, env.single_action_space.n),
            )
        elif network_type == "MEGNetRL":
            self.qnet = MEGNetRL(num_actions = env.single_action_space.n,
                                 ntypes_state =  20)
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (observation)
            
        Returns:
            Q-values for each action
        """
        if self.network_type == "mlp":
            return self.network(x)
        elif self.network_type == "MEGNetRL":
            q_vals = self.qnet(
                x, 
                x.edata['e_feat'], 
                x.ndata['atomic_number'], 
                x.lengths_angles_focus
            )
            return q_vals


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """
    Linear schedule for epsilon-greedy exploration.
    
    Args:
        start_e: Starting epsilon value
        end_e: Ending epsilon value
        duration: Duration of the schedule
        t: Current timestep
        
    Returns:
        Current epsilon value
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@hydra.main(version_base=None, config_path="../config", config_name="dqn")
def main(args: DictConfig) -> None:
    """
    Main training function for DQN agent.
    
    Args:
        args: Hydra configuration containing all hyperparameters
    """
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGTERM, catch_signal)
    run_name = f"{args.algo.env_id}__{args.exp.exp_name}__{args.exp.seed}"
    
    # Set random seeds for reproducibility
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    torch.manual_seed(args.exp.seed)
    torch.backends.cudnn.deterministic = args.exp.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.exp.cuda else "cpu")

    # Environment setup
    kwargs = {'env': dict(args.env), 'qe': dict(args.qe)}
    kwargs['env']['run_name'] = run_name
    envs = make_env(args.algo.env_id, 0, args.exp.capture_video, run_name, kwargs)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize networks
    q_network = QNetwork(envs, network_type="MEGNetRL").to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.algo.learning_rate)
    target_network = QNetwork(envs, network_type="MEGNetRL").to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize replay buffer
    rb = ReplayBuffer(
        storage=ListStorage(max_size=args.algo.buffer_size),
        batch_size=args.algo.batch_size,
        collate_fn=partial(collate_function, p_hat=args.env.p_hat, agent = args.algo.agent),
        pin_memory=True,
        prefetch=16,
    )

    # Checkpoint and resume setup
    save_path = os.path.join(os.getcwd(), "models", run_name)
    start_iteration = 0
    global_step = 0
    run_id = None
    
    try:
        os.makedirs(save_path)
    except OSError:
        # Directory exists, try to resume from checkpoint
        files = os.listdir(save_path)
        if len(files) > 0:
            indices = sorted([int(file.split('_')[-1].split('.')[0]) for file in files if 'ckpt' in file])
            ind = indices[-1]
            try:
                run_state = torch.load(os.path.join(save_path, f"ckpt_{ind}.pt"))
            except:
                if len(indices) > 1:
                    ind = indices[-2]
                    run_state = torch.load(os.path.join(save_path, f"ckpt_{ind}.pt"))
                else:
                    run_state = None
            
            if run_state is not None:
                # Load model states
                q_network.load_state_dict(run_state["states"]["q_network"])
                target_network.load_state_dict(run_state["states"]["target_network"])
                optimizer.load_state_dict(run_state["states"]["optimizer"])

                # Load replay buffer
                rb_state = run_state["states"]["rb"]
                rb.extend(rb_state["_storage"]["_storage"])

                run_id = run_state["run_id"]
                start_iteration = global_step = run_state["global_step"]
                print(f"Resuming from iteration {start_iteration}")
        
    # Initialize logging
    if args.exp.track:
        try:
            print(f"Resuming from previous run {run_id}")
        except:
            run_id = None
        wandb.init(
            project=args.wandb.wandb_project_name,
            group=args.wandb.wandb_group,
            sync_tensorboard=True,
            config=OmegaConf.to_container(args, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            mode=args.wandb.mode,
            id=run_id,
            resume="allow"
        )
        run_id = wandb.run.id
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    start_time = time.time()

    # Start training
    obs, _ = envs.reset(seed=args.exp.seed)
    for global_step in range(start_iteration, args.algo.total_timesteps):
        # Action selection with epsilon-greedy exploration
        epsilon = linear_schedule(
            args.algo.start_e, 
            args.algo.end_e, 
            args.algo.exploration_fraction * args.algo.total_timesteps, 
            global_step
        )
        
        if random.random() < epsilon:
            # Random action for exploration
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]).item()
        else:
            # Greedy action selection
            obs = obs.to(device)
            obs.lengths_angles_focus = obs.lengths_angles_focus.to(device)
            q_values = q_network(obs)
            actions = torch.argmax(q_values).detach().cpu().numpy().item()

        # Execute action in environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episode statistics
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_error", info["episode"]["error_flag"], global_step)
                    property_key = args.env.property
                    if property_key in info['episode']:
                        writer.add_scalar(f"charts/episodic_{property_key}", info["episode"][property_key], global_step)
                        if 'sim_time' in info['episode']:
                            writer.add_scalar("charts/episodic_sim_time", info["episode"]["sim_time"], global_step)

        # Store experience in replay buffer
        real_next_obs = deepcopy(next_obs)
        obs_dict = envs.graph_to_dict(obs)
        next_obs_dict = envs.graph_to_dict(real_next_obs)
        rb.add((obs_dict, next_obs_dict, actions, rewards, terminations, infos))
        
        # Update observation for next step
        if not terminations:
            obs = next_obs
        else:
            obs, _ = envs.reset()
            obs = obs.to(device)

        # Training step
        if global_step > args.algo.learning_starts:
            if global_step % args.algo.train_frequency == 0:
                # Sample batch from replay buffer
                data = rb.sample()
                (
                    observations_sampled,
                    next_observations_sampled,
                    actions_sampled,
                    rewards_sampled,
                    dones_sampled,
                ) = data
                rewards_sampled = rewards_sampled.to(dtype=torch.float32)
                
                # Compute target Q-values
                with torch.no_grad():
                    target_max, _ = target_network(next_observations_sampled).max(dim=1)
                    td_target = (rewards_sampled.flatten() + 
                               args.algo.gamma * target_max * 
                               (1.0 - dones_sampled.flatten().to(torch.float32)))
                
                # Compute current Q-values
                old_val = q_network(observations_sampled).gather(1, actions_sampled.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # Log training metrics
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    print(f"SPS: {sps}")
                    writer.add_scalar("charts/SPS", sps, global_step)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network with soft update
            if global_step % args.algo.target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.algo.tau * q_param.data + (1.0 - args.algo.tau) * target_param.data
                    )
        # Save checkpoint
        if (global_step % args.exp.save_freq == 0 or caught_signal) and global_step > 0:
            states = {
                "q_network": q_network.state_dict(),
                "target_network": target_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rb": rb.state_dict()
            }

            run_state = {
                "run_name": run_name,
                "run_id": run_id,
                "global_step": global_step,
                "states": states
            }
            torch.save(run_state, os.path.join(save_path, f"ckpt_{global_step}.pt"))
            
            # Clean up old checkpoints (keep only last 10)
            files = os.listdir(save_path)
            if len(files) > 10:
                files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[:-10]
                [os.remove(os.path.join(save_path, file)) for file in files]

    # Cleanup
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
