"""
Rainbow DQN implementation for crystal design optimization.

This module implements a Rainbow DQN agent that combines multiple DQN improvements:
- Double DQN
- Dueling Network Architecture
- Multi-step learning

Supports both MLP and MEGNetRL architectures for crystal design optimization.
"""

import os
import random
import signal
import time
from collections import deque
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
from torch.nn import Linear, ReLU, Sequential
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import ListStorage, PrioritizedReplayBuffer, ReplayBuffer

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

def multi_step_reward(rewards: list, gamma: float) -> float:
    """
    Compute multi-step reward for n-step learning.
    
    Args:
        rewards: List of rewards from n consecutive steps
        gamma: Discount factor
        
    Returns:
        Discounted sum of rewards
    """
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret

def beta_schedule(start_beta: float, end_beta: float, duration: int, t: int) -> float:
    """
    Linear schedule for prioritized experience replay beta parameter.
    
    Args:
        start_beta: Starting beta value
        end_beta: Ending beta value
        duration: Duration of the schedule
        t: Current timestep
        
    Returns:
        Current beta value
    """
    slope = (end_beta - start_beta) / duration
    return min(slope * t + start_beta, end_beta)

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

class RainbowAgent(nn.Module):
    """
    Rainbow DQN Agent with dueling network architecture.
    
    Combines MEGNetRL feature extraction with dueling network heads for
    improved value function estimation in crystal design optimization.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 network_type: str, 
                 dueling: bool) -> None:
        """
        Initialize the Rainbow DQN agent.
        
        Args:
            env: Gymnasium environment
            network_type: Type of network architecture ("mlp" or "MEGNetRL")
            dueling: Whether to use dueling network architecture
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_type = network_type
        self.dueling = dueling
        
        if network_type == 'MEGNetRL':
            # MEGNetRL feature extractor
            self.qnet = MEGNetRL(
                num_actions=env.single_action_space.n,
                ntypes_state=env.single_action_space.n, 
                critic=False
            )
            
            if dueling:
                # Dueling network heads
                self.value = Sequential(
                    Linear(env.single_action_space.n, 64),
                    ReLU(),
                    Linear(64, 64),
                    ReLU(),
                    Linear(64, 1),
                )
                self.advantage = Sequential(
                    Linear(env.single_action_space.n, 64),
                    ReLU(),
                    Linear(64, 64),
                    ReLU(),
                    Linear(64, env.single_action_space.n),
                )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Rainbow network.
        
        Args:
            x: Input observation (graph)
            
        Returns:
            Q-values for each action
        """
        features = self.qnet(x, x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
        
        if self.network_type == 'MEGNetRL' and self.dueling:
            # Dueling network: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value(features)
            advantage = self.advantage(features)
            q_values = value + advantage - advantage.mean()
        else:
            q_values = features
            
        return q_values


@hydra.main(version_base=None, config_path="../config", config_name="rainbow")
def main(args: DictConfig) -> None:
    """
    Main training function for Rainbow DQN agent.
    
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
    kwargs['env']['agent'] = args.algo.agent

    envs = make_env(args.algo.env_id, 0, args.exp.capture_video, run_name, kwargs)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize networks
    q_network = RainbowAgent(envs, network_type="MEGNetRL", dueling=args.algo.dueling).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.algo.learning_rate)
    target_network = RainbowAgent(envs, network_type="MEGNetRL", dueling=args.algo.dueling).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Initialize replay buffer
    if args.algo.replay_type == "uniform":
        rb = ReplayBuffer(
            storage=ListStorage(max_size=args.algo.buffer_size),
            batch_size=args.algo.batch_size,
            collate_fn=partial(collate_function, p_hat=args.env.p_hat, agent=args.algo.agent),
            pin_memory=True,
            prefetch=16,
        )
    elif args.algo.replay_type == "prioritized":
        rb = PrioritizedReplayBuffer(
            alpha=0.6,
            beta=0.4,
            storage=ListStorage(max_size=args.algo.buffer_size),
            batch_size=args.algo.batch_size,
            collate_fn=partial(collate_function, p_hat=args.env.p_hat, agent=args.algo.agent),
            pin_memory=True,
            prefetch=16,
        )
    else:
        raise ValueError(f"Unsupported replay type: {args.algo.replay_type}")

    # Multi-step learning queues
    state_deque = deque(maxlen=args.algo.multi_step)
    reward_deque = deque(maxlen=args.algo.multi_step)
    action_deque = deque(maxlen=args.algo.multi_step)

    # Checkpoint and resume setup
    if args.env.p_hat == 5.0:
        run_name += "_p5"
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
                if args.algo.replay_type == "prioritized":
                    rb.sampler.load_state_dict(rb_state["_sampler"])
                
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

        # Multi-step learning: store experience in queues
        real_next_obs = deepcopy(next_obs)
        obs_dict = envs.graph_to_dict(obs)
        next_obs_dict = envs.graph_to_dict(real_next_obs)

        state_deque.append(obs_dict)
        reward_deque.append(rewards)
        action_deque.append(actions)
        
        # Add experience to replay buffer when we have enough steps or episode ends
        if len(state_deque) == args.algo.multi_step or terminations:
            n_reward = multi_step_reward(reward_deque, args.algo.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            rb.add((n_state, next_obs_dict, n_action, n_reward, terminations, infos))
        
        # Update observation for next step
        if not terminations:
            obs = next_obs
        else:
            # Episode ended, reset environment and process remaining experiences
            obs, _ = envs.reset()
            obs = obs.to(device)
            
            # Process remaining experiences in queues
            state_deque.popleft()
            reward_deque.popleft()
            action_deque.popleft()
            
            for i in range(len(state_deque)):
                n_reward = multi_step_reward(reward_deque, args.algo.gamma)
                n_state = state_deque[i]
                n_action = action_deque[i]
                if i + 1 <= len(state_deque) - 1:
                    next_obs_dict_old = state_deque[i+1]
                else:
                    next_obs_dict_old = deepcopy(next_obs_dict)
                rb.add((n_state, next_obs_dict_old, n_action, n_reward, terminations, infos))
                reward_deque.popleft()

            # Clear queues for next episode
            state_deque.clear()
            reward_deque.clear()
            action_deque.clear()

        # Training step
        if global_step > args.algo.learning_starts:
            if global_step % args.algo.train_frequency == 0:
                # Sample batch from replay buffer
                if args.algo.replay_type == "uniform":
                    data = rb.sample()
                elif args.algo.replay_type == "prioritized":
                    data, info = rb.sample(return_info=True)
                else:
                    raise ValueError(f"Unsupported replay type: {args.algo.replay_type}")
                
                (
                    observations_sampled,
                    next_observations_sampled,
                    actions_sampled,
                    rewards_sampled,
                    dones_sampled,
                ) = data
                rewards_sampled = rewards_sampled.to(dtype=torch.float32)
                
                # Compute target Q-values with double DQN
                with torch.no_grad():
                    target_qvals = target_network(next_observations_sampled)
                    if args.algo.double:
                        # Double DQN: use main network to select action, target network to evaluate
                        next_actions = torch.argmax(q_network(next_observations_sampled), dim=1)
                        target_max = target_qvals.gather(1, next_actions.unsqueeze(1)).squeeze()
                    else:
                        # Standard DQN: use target network for both selection and evaluation
                        target_max, _ = target_qvals.max(dim=1)
                    
                    # Multi-step target with n-step return
                    td_target = (rewards_sampled.flatten() + 
                               (args.algo.gamma ** args.algo.multi_step) * target_max * 
                               (1.0 - dones_sampled.flatten().to(torch.float32)))
                
                # Compute current Q-values
                old_val = q_network(observations_sampled).gather(1, actions_sampled.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # Update priorities for prioritized experience replay
                if args.algo.replay_type == "prioritized":
                    rb.update_priority(info["index"], loss.cpu().detach().numpy())
                    beta = beta_schedule(0.4, 1.0, args.algo.total_timesteps, global_step - args.algo.learning_starts)
                    rb.sampler._beta = beta

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
                "states": states,
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
