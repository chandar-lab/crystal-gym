# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import wandb
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from crystal_gym.env import CrystalGymEnv
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from crystal_design.agents import MEGNetRL

# def args(cfg: DictConfig) -> None: 
#     print(OmegaConf.to_yaml(cfg))
#     # exp_name: str = os.path.basename(__file__)[: -len(".py")]
#     # """the name of this expaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaeriment"""
#     # seed: int = 1
#     # """seed of the experiment"""
#     # torch_deterministic: bool = True
#     # """if toggled, `torch.backends.cudnn.deterministic=False`"""
#     # cuda: bool = True
#     # """if toggled, cuda will be enabled by default"""
#     # track: bool = True
#     # """if toggled, this experiment will be tracked with Weights and Biases"""
#     # wandb_project_name: str = "PPO"
#     # """the wandb's project name"""
#     # wandb_entity: str = None
#     # """the entity (team) of wandb's project"""
#     # capture_video: bool = False
#     # """whether to capture videos of the agent performances (check out `videos` folder)"""

#     # # Algorithm specific arguments
#     # env_id: str = "CartPole-v1"
#     # """the id of the environment"""
#     # total_timesteps: int = 500000
#     # """total timesteps of the experiments"""
#     # learning_rate: float = 2.5e-4
#     # """the learning rate of the optimizer"""
#     # num_envs: int = 4
#     # """the number of parallel game environments"""
#     # num_steps: int = 128
#     # """the number of steps to run in each environment per policy rollout"""
#     # anneal_lr: bool = True
#     # """Toggle learning rate annealing for policy and value networks"""
#     # gamma: float = 0.99
#     # """the discount factor gamma"""
#     # gae_lambda: float = 0.95
#     # """the lambda for the general advantage estimation"""
#     # num_minibatches: int = 4
#     # """the number of mini-batches"""
#     # update_epochs: int = 4
#     # """the K epochs to update the policy"""
#     # norm_adv: bool = True
#     # """Toggles advantages normalization"""
#     # clip_coef: float = 0.2
#     # """the surrogate clipping coefficient"""
#     # clip_vloss: bool = True
#     # """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
#     # ent_coef: float = 0.01
#     # """coefficient of the entropy"""
#     # vf_coef: float = 0.5
#     # """coefficient of the value function"""
#     # max_grad_norm: float = 0.5
#     # """the maximum norm for the gradient clipping"""
#     # target_kl: float = None
#     # """the target KL divergence threshold"""

#     # # to be filled in runtime
#     # batch_size: int = 0
#     # """the batch size (computed in runtime)"""
#     # minibatch_size: int = 0
#     # """the mini-batch size (computed in runtime)"""
#     # num_iterations: int = 0
#     # """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, kwargs):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, kwargs = kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, type = "mlp"):
        super().__init__()
        if type == "mlp":
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )
        elif type == 'MEGNetRL':
            # Check if special initialization is required for MEGNetRL
            self.critic = MEGNetRL(num_actions = envs.single_action_space.n, 
                                   ntypes_state =  envs.single_action_space.n,
                                   critic = True)
            self.actor = MEGNetRL(num_actions = envs.single_action_space.n,
                                  ntypes_state =  envs.single_action_space.n)
        self.type = type

    def get_value(self, x):
        return self.critic(x,x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)

    def get_action_and_value(self, x, action=None):
        if self.type == "mlp":
            logits = self.actor(x)
            values = self.critic(x)
        elif self.type == "MEGNetRL":
            logits = self.actor(x,x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
            values = self.critic(x,x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), values



@hydra.main(version_base = None, config_path="../config", config_name="ppo")
def main(args: DictConfig) -> None:
    # args = tyro.cli(Args)
    batch_size = int(args.algo.num_envs * args.algo.num_steps)
    minibatch_size = int(batch_size // args.algo.num_minibatches)
    num_iterations = args.algo.total_timesteps // batch_size
    run_name = f"{args.algo.env_id}__{args.exp.exp_name}__{args.exp.seed}__{int(time.time())}"
    if args.exp.track:
        wandb.init(
            project=args.wandb.wandb_project_name,
            group=args.wandb.wandb_group,
            sync_tensorboard=True,
            config=dict(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            mode=args.wandb.mode,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    torch.manual_seed(args.exp.seed)
    torch.backends.cudnn.deterministic = args.exp.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.exp.cuda else "cpu")

    
    # env setup  
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )

    ## Output of SyncVectorEnv is a list of outputs from each environment --  check how to parallellize this. ; but check AsyncVectorEnv
    kwargs = {'env':dict(args.env), 'qe': dict(args.qe)}
    kwargs['env']['run_name'] = run_name
    envs = make_env(args.algo.env_id, 0, args.exp.capture_video, run_name, kwargs)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, type = "MEGNetRL").to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.algo.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs = []
    actions = torch.zeros((args.algo.num_steps, args.algo.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)
    rewards = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)
    dones = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)
    values = torch.zeros((args.algo.num_steps, args.algo.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.exp.seed)
    next_obs = next_obs.to(device)
    next_done = torch.zeros(args.algo.num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.algo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * args.algo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.algo.num_steps):
            global_step += args.algo.num_envs
            # obs[step] = next_obs
            obs.append(next_obs)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations).astype(np.float32)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = next_obs.to(device), torch.Tensor([next_done]).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        if 'bg' in info['episode']:
                            writer.add_scalar("charts/episodic_bg", info["episode"]["bg"], global_step)
                        # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
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

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # Create a batch of graphs
        lattice_features = torch.stack([obs[i].lengths_angles for i in range(len(obs))])
        lattice_features = torch.cat((lattice_features, torch.tensor([1.12]*lattice_features.shape[0])[:,None]), dim = 1)
        focus_features = torch.stack([obs[i].focus for i in range(len(obs))])
        focus_list_features = torch.stack([obs[i].focus_list for i in range(len(obs))])


        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.algo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                
                end = start + minibatch_size
                b_obs = dgl.batch(obs[start:end])
                b_obs.lengths_angles_focus = lattice_features[start:end].to(device = device)
                b_obs.focus = focus_features[start:end].to(device = device).squeeze()
                b_obs.focus_list = focus_list_features[start:end].to(device = device)

                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.algo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.algo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.algo.clip_coef, 1 + args.algo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.algo.ent_coef * entropy_loss + v_loss * args.algo.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.algo.max_grad_norm)
                optimizer.step()

            if args.algo.target_kl is not None and approx_kl > args.algo.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
if __name__ == '__main__':
    main()
