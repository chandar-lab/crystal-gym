import os
import torch
import random
import numpy as np
import hydra
import signal
import time
from functools import partial
import wandb
import torch.optim as optim
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear, Sequential, ReLU
from torchrl.data import ReplayBuffer, ListStorage
from crystal_gym.agents import MEGNetRL
from crystal_gym.utils import collate_function
from copy import deepcopy
from crystal_gym.env import CrystalGymEnv
import torch.nn.functional as F
import gymnasium as gym
from collections import deque


caught_signal = False

def catch_signal(sig, frame):
    global caught_signal
    caught_signal = True

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

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

class RainbowAgent(nn.Module):
    def __init__(self, 
                 env,
                 type, 
                 dueling) -> None:
        # ...
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type = type
        if self.type == 'MEGNetRL':
            if dueling:
                self.qnet = MEGNetRL(num_actions = env.single_action_space.n,
                                    ntypes_state =  env.single_action_space.n, critic=False)

                self.value = Sequential(
                                        Linear(env.single_action_space.n, 120),
                                        ReLU(),
                                        Linear(120, 84),
                                        ReLU(),
                                        Linear(84, 1),
                                    )
                self.advantage = Sequential(
                                        Linear(env.single_action_space.n, 120),
                                        ReLU(),
                                        Linear(120, 84),
                                        ReLU(),
                                        Linear(84, env.single_action_space.n),
                                    )
            else:
                                    
                self.qnet = MEGNetRL(num_actions = env.single_action_space.n,
                                    ntypes_state =  env.single_action_space.n, critic=False)
        
        
        # ...

    def forward(self, x):
        # ...
        features = self.qnet(x,x.edata['e_feat'], x.ndata['atomic_number'], x.lengths_angles_focus)
        if self.type == 'MEGNetRL':
            value = self.value(features)
            advantage = self.advantage(features)
            x = value + advantage - advantage.mean()
        else:
            x = features
        return x


@hydra.main(version_base = None, config_path="../config", config_name="rainbow")
def main(args: DictConfig) -> None:
    
    
    signal.signal(signal.SIGTERM, catch_signal)
    run_name = f"{args.algo.env_id}__{args.exp.exp_name}__{args.exp.seed}"
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    torch.manual_seed(args.exp.seed)
    torch.backends.cudnn.deterministic = args.exp.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.exp.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv(
     #   [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #)

    kwargs = {'env':dict(args.env), 'qe': dict(args.qe)}
    kwargs['env']['run_name'] = run_name
    kwargs['env']['agent'] = args.algo.agent

    envs = make_env(args.algo.env_id, 0, args.exp.capture_video, run_name, kwargs)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = RainbowAgent(envs, type = "MEGNetRL", dueling = args.algo.dueling).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.algo.learning_rate)
    target_network = RainbowAgent(envs, type = "MEGNetRL", dueling = args.algo.dueling).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        storage=ListStorage(max_size=args.algo.buffer_size),
        batch_size = args.algo.batch_size,
        collate_fn = partial(collate_function, p_hat = args.env.p_hat, agent = args.algo.agent),
        pin_memory = True,
        prefetch = 16,
    )

    state_deque = deque(maxlen=args.algo.multi_step)
    reward_deque = deque(maxlen=args.algo.multi_step)
    action_deque = deque(maxlen=args.algo.multi_step)

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

            #  = run_state["iteration"]
            q_network.load_state_dict(run_state["states"]["q_network"])
            target_network.load_state_dict(run_state["states"]["target_network"])
            optimizer.load_state_dict(run_state["states"]["optimizer"])

            rb_state = run_state["states"]["rb"]
            rb.extend(rb_state["_storage"]["_storage"])

            run_id = run_state["run_id"]
    
            start_iteration = global_step = run_state["global_step"]

            # state_deque = run_state["queues"]["state"]
            # reward_deque = run_state["queues"]["reward"]
            # action_deque = run_state["queues"]["action"]
        
    
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
    
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.exp.seed)
    for global_step in range(start_iteration, args.algo.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.algo.start_e, args.algo.end_e, args.algo.exploration_fraction * args.algo.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]).item()
        else:
            obs = obs.to(device)
            obs.lengths_angles_focus = obs.lengths_angles_focus.to(device)
            q_values = q_network(obs)
            actions = torch.argmax(q_values).detach().cpu().numpy().item()

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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = deepcopy(next_obs)
        # for idx, trunc in enumerate(truncations):
         #   if trunc:
         #       real_next_obs[idx] = infos["final_observation"][idx]
        obs_dict = envs.graph_to_dict(obs)
        next_obs_dict = envs.graph_to_dict(real_next_obs)

        state_deque.append(obs_dict)
        reward_deque.append(rewards)
        action_deque.append(actions)
        if len(state_deque) == args.algo.multi_step or terminations:
            n_reward = multi_step_reward(reward_deque, args.algo.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            rb.add((n_state, next_obs_dict, n_action, n_reward, terminations, infos))
        
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        if not terminations:
            obs = next_obs
        else:
            obs, _ = envs.reset()
            obs = obs.to(device)
            state_deque.popleft()
            reward_deque.popleft()
            action_deque.popleft()
            
            for i in range(len(state_deque)):
                n_reward = multi_step_reward(reward_deque, args.algo.gamma)
                n_state = state_deque[i]
                n_action = action_deque[i]
                next_obs_dict_old = state_deque[i+1]
                rb.add((n_state, next_obs_dict_old, n_action, n_reward, terminations, infos))
                reward_deque.popleft()

            state_deque.clear()
            reward_deque.clear()
            action_deque.clear()

        # ALGO LOGIC: training.
        if global_step > args.algo.learning_starts:
            if global_step % args.algo.train_frequency == 0:
                data = rb.sample()
                (
                    observations_sampled,
                    next_observations_sampled,
                    actions_sampled,
                    rewards_sampled,
                    dones_sampled,
                ) = data
                rewards_sampled = rewards_sampled.to(dtype=torch.float32)
                # data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_qvals = target_network(next_observations_sampled)
                    if args.algo.double:
                        next_actions = torch.argmax(q_network(next_observations_sampled), dim=1)
                        target_max = target_qvals.gather(1, next_actions.unsqueeze(1)).squeeze()
                    else:
                        target_max, _ = target_qvals.max(dim=1)
                    td_target = rewards_sampled.flatten() + (args.algo.gamma ** args.algo.multi_step) * target_max * (1.0 - dones_sampled.flatten().to(torch.float32))
                old_val = q_network(observations_sampled).gather(1, actions_sampled.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.algo.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.algo.tau * q_network_param.data + (1.0 - args.algo.tau) * target_network_param.data
                    )
        if (global_step % args.exp.save_freq == 0 or caught_signal) and global_step > 0:
            states = {"q_network": q_network.state_dict(), "target_network": target_network.state_dict(), "optimizer": optimizer.state_dict(), "rb": rb.state_dict()}
            # queues = {"state": state_deque, "reward": reward_deque, "action": action_deque}

            run_state = {
                            "run_name": run_name,
                            "run_id": run_id,
                            "global_step": global_step,
                            "states": states,
                            # "queues": queues,
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

