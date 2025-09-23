"""
Proximal Policy Optimization (PPO) implementation for crystal design optimization.

"""

import os
import random
import signal
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import dgl
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import hydra
from crystal_gym.agents import MEGNetRL
from crystal_gym.env import CrystalGymEnv

# Constants
MAX_ATOMS = 20

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


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initialize layer weights using orthogonal initialization.
    
    Args:
        layer: PyTorch layer to initialize
        std: Standard deviation for weight initialization
        bias_const: Constant value for bias initialization
        
    Returns:
        Initialized layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    PPO Agent with separate actor and critic networks.
    
    Supports both MLP and MEGNetRL architectures for crystal design optimization.
    """
    
    def __init__(self, envs: gym.Env, network_type: str = "mlp") -> None:
        """
        Initialize the PPO agent.
        
        Args:
            envs: Gymnasium environment
            network_type: Type of network architecture ("mlp" or "MEGNetRL")
        """
        super().__init__()
        self.network_type = network_type
        
        if network_type == "mlp":
            # MLP architecture for both actor and critic
            obs_dim = np.array(envs.single_observation_space.shape).prod()
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )
        elif network_type == 'MEGNetRL':
            # MEGNetRL architecture for crystal structures
            self.critic = MEGNetRL(
                num_actions=envs.single_action_space.n, 
                ntypes_state=MAX_ATOMS,
                critic=True
            )
            self.actor = MEGNetRL(
                num_actions=envs.single_action_space.n,
                ntypes_state=MAX_ATOMS
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get state value from critic network.
        
        Args:
            x: Input observation
            
        Returns:
            State value estimate
        """
        if self.network_type == "mlp":
            return self.critic(x)
        elif self.network_type == "MEGNetRL":
            return self.critic(x, x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value from the agent.
        
        Args:
            x: Input observation
            action: Optional action to evaluate (if None, sample from policy)
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        if self.network_type == "mlp":
            logits = self.actor(x)
            values = self.critic(x)
        elif self.network_type == "MEGNetRL":
            logits = self.actor(x, x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
            values = self.critic(x, x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), values



@hydra.main(version_base=None, config_path="../config", config_name="ppo")
def main(args: DictConfig) -> None:
    """
    Main training function for PPO agent.
    
    Args:
        args: Hydra configuration containing all hyperparameters
    """
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGTERM, catch_signal)

    # Calculate batch sizes
    batch_size = int(args.algo.num_envs * args.algo.num_steps)
    minibatch_size = int(batch_size // args.algo.num_minibatches)
    num_iterations = args.algo.total_timesteps // batch_size
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

    # Initialize agent and optimizer
    agent = Agent(envs, network_type="MEGNetRL").to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.algo.learning_rate, eps=1e-5)

    # Storage setup for PPO rollout data
    obs = []  # Store observations as list for graph batching
    actions = torch.zeros((args.algo.num_steps, args.algo.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)
    rewards = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)
    dones = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)
    values = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)

    # Initialize environment
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.exp.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.algo.num_envs).to(device)

    # Checkpoint and resume setup
    save_path = os.path.join(os.getcwd(), "models", run_name)
    start_iteration = 1
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
                start_iteration = run_state["iteration"]
                agent.load_state_dict(run_state["states"]["agent"])
                optimizer.load_state_dict(run_state["states"]["optimizer"])
                run_id = run_state["run_id"]
                global_step = run_state["global_step"]

                # Load rollout data
                actions = run_state["variables"]["actions"]
                values = run_state["variables"]["values"]
                logprobs = run_state["variables"]["logprobs"]
                rewards = run_state["variables"]["rewards"]
                dones = run_state["variables"]["dones"]

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


    # Main training loop
    for iteration in range(start_iteration, num_iterations + 1):
        # Learning rate annealing
        if args.algo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * args.algo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout collection
        for step in range(0, args.algo.num_steps):
            global_step += args.algo.num_envs
            obs.append(next_obs)
            dones[step] = next_done

            # Action selection
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Environment step
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations).astype(np.float32)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = next_obs.to(device), torch.Tensor([next_done]).to(device)
            
            # Reset environment if episode is done
            if next_done:
                next_obs, _ = envs.reset()
                next_obs = next_obs.to(device)

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

        # Compute advantages using GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.algo.num_steps)):
                if t == args.algo.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.algo.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.algo.gamma * args.algo.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Prepare batch data for training
        lattice_features = torch.stack([obs[i].lengths_angles_focus for i in range(len(obs))]).squeeze()
        focus_features = torch.stack([obs[i].focus for i in range(len(obs))])
        focus_list_features = torch.stack([obs[i].focus_list for i in range(len(obs))])

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO training epochs
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.algo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                
                # Create batched observation
                b_obs = dgl.batch(obs[start:end])
                b_obs.lengths_angles_focus = lattice_features[start:end].to(device=device)
                b_obs.focus = focus_features[start:end].to(device=device).squeeze()
                b_obs.focus_list = focus_list_features[start:end].to(device=device)

                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Compute KL divergence and clipping statistics
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.algo.clip_coef).float().mean().item()]

                # Normalize advantages if specified
                mb_advantages = b_advantages[mb_inds]
                if args.algo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss with PPO clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.algo.clip_coef, 1 + args.algo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with optional clipping
                newvalue = newvalue.view(-1)
                if args.algo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.algo.clip_coef,
                        args.algo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.algo.ent_coef * entropy_loss + v_loss * args.algo.vf_coef

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.algo.max_grad_norm)
                optimizer.step()

            # Early stopping if KL divergence is too high
            if args.algo.target_kl is not None and approx_kl > args.algo.target_kl:
                break

        # Compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Save checkpoint
        if iteration % args.exp.save_freq == 0 or caught_signal:
            variables = {
                "actions": actions, 
                "values": values, 
                "logprobs": logprobs, 
                "rewards": rewards, 
                "dones": dones
            }
            states = {
                "agent": agent.state_dict(), 
                "optimizer": optimizer.state_dict()
            }
            run_state = {
                "iteration": iteration,
                "run_name": run_name,
                "run_id": run_id,
                "global_step": global_step,
                "variables": variables, 
                "states": states
            }
            torch.save(run_state, os.path.join(save_path, f"ckpt_{iteration}.pt"))
            
            # Clean up old checkpoints (keep only last 10)
            files = os.listdir(save_path)
            if len(files) > 10:
                files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[:-10]
                [os.remove(os.path.join(save_path, file)) for file in files]

        # Log training metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # Cleanup
    envs.close()
    writer.close()


if __name__ == '__main__':
    main()
