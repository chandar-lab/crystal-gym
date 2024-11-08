# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from crystal_gym.agents import MEGNetRL
import hydra
from omegaconf import DictConfig, OmegaConf
from crystal_gym.env import CrystalGymEnv
from copy import deepcopy
from torchrl.data import ReplayBuffer, ListStorage
from functools import partial
from crystal_gym.utils import collate_function
import signal
import wandb

caught_signal = False

def catch_signal(sig, frame):
    global caught_signal
    caught_signal = True


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


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs, type = "mlp"):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        if type == "mlp":
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.Flatten(),
            )

            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))
        elif type == "MEGNetRL":
            self.qnet = MEGNetRL(num_actions = envs.single_action_space.n,
                                 ntypes_state =  envs.single_action_space.n)
        self.type = type

    def forward(self, x):
        if self.type == "mlp":
            x = F.relu(self.conv(x / 255.0))
            x = F.relu(self.fc1(x))
            q_vals = self.fc_q(x)
        elif self.type == "MEGNetRL":
            q_vals = self.qnet(x,x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs, type = "mlp"):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        if type == "mlp":
            self.conv = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.Flatten(),
            )

            with torch.inference_mode():
                output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

            self.fc1 = layer_init(nn.Linear(output_dim, 512))
            self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

        if type == "MEGNetRL":
            self.actor = MEGNetRL(num_actions = envs.single_action_space.n,
                                  ntypes_state =  envs.single_action_space.n)
        self.type = type

    def forward(self, x):
        if self.type == "mlp":
            x = F.relu(self.conv(x))
            x = F.relu(self.fc1(x))
            logits = self.fc_logits(x)
        elif self.type == "MEGNetRL":
            logits = self.actor(x,x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)

        return logits

    def get_action(self, x):
        if self.type == "mlp":
            logits = self(x / 255.0)
        elif self.type == "MEGNetRL":
            logits = self(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)

        return action, log_prob, action_probs

@hydra.main(version_base = None, config_path="../config", config_name="sac")
def main(args: DictConfig) -> None:


    signal.signal(signal.SIGTERM, catch_signal)

    run_name = f"{args.algo.env_id}__{args.exp.exp_name}__{args.exp.seed}"


    # TRY NOT TO MODIFY: seeding
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    torch.manual_seed(args.exp.seed)
    torch.backends.cudnn.deterministic = args.exp.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.exp.cuda else "cpu")

    # # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    kwargs = {'env':dict(args.env), 'qe': dict(args.qe)}
    kwargs['env']['run_name'] = run_name
    envs = make_env(args.algo.env_id, 0, args.exp.capture_video, run_name, kwargs)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs, type = "MEGNetRL").to(device)
    qf1 = SoftQNetwork(envs, type = "MEGNetRL").to(device)
    qf2 = SoftQNetwork(envs, type = "MEGNetRL").to(device)
    qf1_target = SoftQNetwork(envs, type = "MEGNetRL").to(device)
    qf2_target = SoftQNetwork(envs, type = "MEGNetRL").to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.algo.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.algo.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.algo.autotune:
        target_entropy = -args.algo.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.algo.q_lr, eps=1e-4)
    else:
        alpha = args.algo.alpha

    rb = ReplayBuffer(
            storage=ListStorage(max_size=args.algo.buffer_size),
            batch_size = args.algo.batch_size,
            collate_fn = partial(collate_function, p_hat = args.env.p_hat),
            pin_memory = True,
            prefetch = 16,
        )

    # rb = ReplayBuffer(
    #     args.algo.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     handle_timeout_termination=False,
    # )
    start_time = time.time()

    # SAVE AND LOAD
    save_path = os.path.join(os.getcwd(), "models", run_name)
    try:
        start_iteration = 0
        global_step = 0
        os.makedirs(save_path)
    except OSError:
        files = os.listdir(save_path)
        if len(files) > 0:
            indices = sorted([int(file.split('_')[-1].split('.')[0]) for file in files if 'ckpt' in file])
            ind = indices[-1]
            try:
                run_state = torch.load(os.path.join(save_path, f"ckpt_{ind}.pt"))
            except:
                ind = indices[-2]
                run_state = torch.load(os.path.join(save_path, f"ckpt_{ind}.pt"))

            # start_iteration = run_state["iteration"]
            actor.load_state_dict(run_state["states"]["actor"])
            qf1.load_state_dict(run_state["states"]["qf1"])
            qf2.load_state_dict(run_state["states"]["qf2"])
            qf1_target.load_state_dict(run_state["states"]["qf1_target"])
            qf2_target.load_state_dict(run_state["states"]["qf2_target"])
            actor_optimizer.load_state_dict(run_state["states"]["actor_optimizer"])
            q_optimizer.load_state_dict(run_state["states"]["q_optimizer"])

            if args.algo.autotune:
                a_optimizer.load_state_dict(run_state["states"]["a_optimizer"])
                log_alpha.data[0] = run_state["states"]["log_alpha"][0]

            rb_state = run_state["states"]["rb"]
            rb.extend(rb_state["_storage"]["_storage"])

            run_id = run_state["run_id"]

            start_iteration = global_step = run_state["global_step"]

            print("Resuming from iteration ", start_iteration)

    if args.exp.track:
        try:
            print("Resuming from previous run ", run_id)
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
            id = run_id,
            resume="allow"
        )
        run_id = wandb.run.id
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.exp.seed)
    obs = obs.to(device)
    for global_step in range(start_iteration, args.algo.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.algo.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs = obs.to(device)
            obs.lengths_angles_focus = obs.lengths_angles_focus.to(device)
            actions, _, _ = actor.get_action(obs)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_error", info["episode"]["error_flag"], global_step)
                    if 'bg' in info['episode']:
                        writer.add_scalar("charts/episodic_bg", info["episode"]["bg"], global_step)
                        writer.add_scalar("charts/episodic_sim_time", info["episode"]["sim_time"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                # break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = deepcopy(next_obs)
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        obs_dict = envs.graph_to_dict(obs)
        next_obs_dict = envs.graph_to_dict(real_next_obs)
        rb.add((obs_dict, next_obs_dict, actions, rewards, terminations, infos))

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        if not terminations:
            obs = next_obs
        else:
            obs, _ = envs.reset()
            obs = obs.to(device)

        # ALGO LOGIC: training.
        if global_step > args.algo.learning_starts:
            if global_step % args.algo.update_frequency == 0:
                data = rb.sample()
                (
                    observations_sampled,
                    next_observations_sampled,
                    actions_sampled,
                    rewards_sampled,
                    dones_sampled,
                ) = data
                rewards_sampled = rewards_sampled.to(dtype=torch.float32)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(next_observations_sampled)
                    qf1_next_target = qf1_target(next_observations_sampled)
                    qf2_next_target = qf2_target(next_observations_sampled)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = rewards_sampled.flatten() + (1.0 - dones_sampled.flatten().to(torch.float32)) * args.algo.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(observations_sampled)
                qf2_values = qf2(observations_sampled)
                qf1_a_values = qf1_values.gather(1, actions_sampled.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, actions_sampled.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(observations_sampled)
                with torch.no_grad():
                    qf1_values = qf1(observations_sampled)
                    qf2_values = qf2(observations_sampled)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.algo.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.algo.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.algo.tau * param.data + (1 - args.algo.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.algo.tau * param.data + (1 - args.algo.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.algo.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if (global_step % args.exp.save_freq == 0 or caught_signal) and global_step > 0:
            states = {"actor": actor.state_dict(), "qf1": qf1.state_dict(), "qf2": qf2.state_dict(), "qf1_target": qf1_target.state_dict(), "qf2_target": qf2_target.state_dict(), "actor_optimizer": actor_optimizer.state_dict(), "q_optimizer": q_optimizer.state_dict(), "rb": rb.state_dict()}
            if args.algo.autotune:
                states["a_optimizer"] = a_optimizer.state_dict()
                states["log_alpha"]  = log_alpha.item()

            run_state = {
                            "run_name": run_name,
                            "run_id": run_id,
                            "global_step": global_step,
                            "states": states
                        }
            torch.save(run_state, os.path.join(save_path, f"ckpt_{global_step}.pt"))
            files = os.listdir(save_path)
            if len(files) > 10:
                files = sorted(files, key = lambda x: int(x.split('_')[-1].split('.')[0]))[:-10]
                [os.remove(os.path.join(save_path, file)) for file in files]

    envs.close()
    writer.close()
if __name__ == "__main__":
    main()
