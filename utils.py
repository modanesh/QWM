from collections import deque

import glob
import json
import numpy as np
import os
import random
import statistics
import time
import torch
import wandb
from pathlib import Path
from torch import distributions as torch_dist
from torch import nn
from torch.nn import functional as F
from tqdm import trange
from feature_extractors.manual_extract_embeddings import PhysicalMorphologyExtractor


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step, log_enabled, envs, group):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step
        self.log_enabled = log_enabled
        self.num_envs = envs
        if self.log_enabled:
            wandb.init(
                project="QWM",
                name=str(logdir).split("/")[-1],
                group=group
            )

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def _group_key(self, name):
        lname = name.lower()

        if lname.startswith("loss/") or lname.startswith("return/"):
            return name

        if "loss" in lname:
            return f"loss/{name}"

        if ("return" in lname) or ("ret" in lname):
            return f"return/{name}"

        return name

    def write(self, fps=False, step=False):
        if self.log_enabled:
            if not step:
                step = self.step
            step //= self.num_envs
            scalars = list(self._scalars.items())
            if fps:
                scalars.append(("fps", self._compute_fps(step)))
            with (self._logdir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")

            wandb_log = {"step": step}

            # Log scalars
            for name, value in scalars:
                wandb_log[self._group_key(name)] = value

            wandb.log(wandb_log, step=step)

            self._scalars = {}
            self._images = {}
            self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        wandb.log({name: value}, step=step)

    def close(self):
        if self.log_enabled:
            wandb.finish()

class ReplayBuffer:
    """
    An efficient, vectorized online replay buffer designed for parallel environments
    and sequence-based models like Dreamer.

    It stores data in a layout of (num_envs, capacity_per_env, ...), ensuring that
    trajectories from each environment are stored contiguously in memory.
    This makes sampling valid temporal sequences extremely fast and simple.
    """
    def __init__(self, dataset_size, num_envs, obs_space, act_space, device, num_robot_types):
        """
        Args:
            dataset_size (int): The total number of transitions to store across all envs.
            num_envs (int): The number of parallel environments.
            obs_space (gym.spaces.Dict): The single-instance observation space.
            act_space (gym.spaces.Box): The single-instance action space.
        """
        assert dataset_size % num_envs == 0, "dataset_size must be divisible by num_envs."
        self.device = device
        self.num_envs = num_envs
        self.capacity_per_env = dataset_size // num_envs
        self.num_robot_types = num_robot_types

        # Data is stored with shape: (num_envs, capacity_per_env, feature_dim)
        self.data = {
            k: torch.zeros(
                (self.num_envs, self.capacity_per_env, *v.shape),
                dtype=torch.from_numpy(np.zeros(1, dtype=v.dtype)).dtype,
                device=self.device,
            )
            for k, v in obs_space.spaces.items()
        }
        self.data['action'] = torch.zeros((self.num_envs, self.capacity_per_env, *act_space.shape),
                                          dtype=torch.from_numpy(np.zeros(1, dtype=act_space.dtype)).dtype, device=self.device)
        self.data['reward'] = torch.zeros((self.num_envs, self.capacity_per_env, 1), dtype=torch.float32, device=self.device)
        self.data['done'] = torch.zeros((self.num_envs, self.capacity_per_env, 1), dtype=torch.bool, device=self.device)
        self.data['discount'] = torch.ones((self.num_envs, self.capacity_per_env, 1), dtype=torch.float32, device=self.device)
        self.data['quadruped_type'] = torch.full((self.num_envs, self.capacity_per_env, 1), -1, dtype=torch.float32, device=self.device)  # -1 is invalid robot type. have it for debugging and sanity checks

        self.env_groups = None

        self._ptr = 0
        self._full = False

    def __len__(self):
        """Returns the total number of valid transitions stored across all envs."""
        size_per_env = self.capacity_per_env if self._full else self.ptr
        return size_per_env * self.num_envs

    def index_envs_by_robot_type(self, env_robot_types):
        """
        Call this once after creating the environment to map env_ids to robot types.
        Args:
             env_robot_types (Tensor): Shape [num_envs], containing the robot type ID for each env.
        """
        self.env_groups = []
        unique_types = torch.unique(env_robot_types)
        print(f"[INFO - WM] Indexing replay buffer for {len(unique_types)} robot types.")

        for r_type in unique_types:
            # Find all env indices that belong to this robot type
            indices = torch.where(env_robot_types == r_type)[0]
            self.env_groups.append(indices.to(self.device))

        # Verify we aren't losing envs
        total_indexed = sum([len(g) for g in self.env_groups])
        assert total_indexed == self.num_envs, f"Indexed {total_indexed} envs, but num_envs is {self.num_envs}"

    def add(self, obs, action, reward, done, info):
        """
        Adds a batch of transitions directly to each environment's storage slot.
        This is highly efficient as it's a single slice assignment.
        """
        # Input shapes are (num_envs, ...). We store them at the current pointer
        # location across all environment buffers simultaneously.
        for key, value in obs.items():
            self.data[key][:, self._ptr] = value

        self.data['action'][:, self._ptr] = action
        self.data['reward'][:, self._ptr] = reward[:, None]
        self.data['done'][:, self._ptr] = done

        self.data['discount'][:, self._ptr] = info["discount"]
        if 'is_terminal' not in self.data:
            self.data['is_terminal'] = torch.zeros((self.num_envs, self.capacity_per_env, 1), dtype=torch.bool, device=self.device)

        self.data['is_terminal'][:, self._ptr] = False
        term_mask = (info["discount"].squeeze(-1) == 0)
        if term_mask.any():
            self.data['is_terminal'][term_mask, self._ptr] = True

        self._ptr = (self._ptr + 1) % self.capacity_per_env
        if self._ptr == 0 and not self._full:
            self._full = True

    def sample_sequences(self, batch_size: int, batch_length: int) -> dict:
        current_len = self.capacity_per_env if self._full else self._ptr
        if current_len < batch_length:
            raise ValueError(f"Not enough history per env ({current_len}) to sample sequences of length {batch_length}.")

        if self.env_groups is not None:
            num_groups = len(self.env_groups)
            samples_per_type = batch_size // num_groups
            remainder = batch_size % num_groups

            # Randomize which robots get the extra samples to be fair over time
            extra_slots = torch.randperm(num_groups, device=self.device) < remainder

            all_env_indices = []

            for i, group_indices in enumerate(self.env_groups):
                # How many to take for this robot?
                count = samples_per_type + (1 if extra_slots[i] else 0)

                # Pick random envs JUST from this robot's group
                rand_idx = torch.randint(0, len(group_indices), (count,), device=self.device)
                selected_envs = group_indices[rand_idx]
                all_env_indices.append(selected_envs)

            env_indices = torch.cat(all_env_indices)

            # Shuffle so the batch isn't ordered by robot type (important for BatchNorm/LayerNorm statistics)
            env_indices = env_indices[torch.randperm(batch_size, device=self.device)]

        else:
            # Fallback to random if index_envs_by_robot_type wasn't called
            env_indices = torch.randint(0, self.num_envs, (batch_size,), device=self.device)

        # Handle circular buffer: ensure sequences don't cross the write pointer
        if self._full:
            # Valid sampling: avoid crossing _ptr
            # We can sample from [_ptr, _ptr + (capacity - batch_length)]
            valid_starts = self.capacity_per_env - batch_length
            time_indices = (self._ptr + torch.randint(0, valid_starts + 1, (batch_size,), device=self.device)) % self.capacity_per_env
        else:
            # Buffer not full: simple case
            time_indices = torch.randint(0, current_len - batch_length + 1, (batch_size,), device=self.device)

        # Gather sequences with wraparound handling
        batch = {}
        for key, value in self.data.items():
            sequences = []
            for i in range(batch_size):
                env_idx = env_indices[i]
                time_start = time_indices[i].item()

                # Handle wraparound
                if self._full and time_start + batch_length > self.capacity_per_env:
                    # Split into two parts
                    first_part_len = self.capacity_per_env - time_start
                    second_part_len = batch_length - first_part_len

                    first_part = value[env_idx, time_start:self.capacity_per_env]
                    second_part = value[env_idx, 0:second_part_len]
                    sequence = torch.cat([first_part, second_part], dim=0)
                else:
                    # No wraparound
                    sequence = value[env_idx, time_start:time_start + batch_length]

                sequences.append(sequence)
            batch[key] = torch.stack(sequences, dim=0)

        return batch

    @property
    def ptr(self):
        return self._ptr

    @property
    def full(self):
        return self._full

    def save(self, filepath):
        """Saves the entire replay buffer data to a single file."""
        print(f"\n[INFO - WM] Saving replay buffer dataset to {filepath}...")
        # Move data to CPU before saving to avoid GPU metadata issues
        data_cpu = {k: v.cpu() for k, v in self.data.items()}
        state = {
            'data': data_cpu,
            'ptr': self._ptr,
            'full': self._full
        }
        torch.save(state, filepath)
        print("[INFO - WM] Dataset saved successfully.")

    def load(self, filepath):
        """Loads a replay buffer dataset from a file."""
        print(f"[INFO - WM] Loading replay buffer dataset from {filepath}...")
        state = torch.load(filepath, map_location=self.device, weights_only=False)

        loaded_data = state['data']
        num_envs, capacity, _ = next(iter(loaded_data.values())).shape
        if num_envs != self.num_envs or capacity != self.capacity_per_env:
            print(f"[INFO - WM] Warning: Buffer dimensions mismatch. Loaded: (envs={num_envs}, cap={capacity}), "
                  f"Current: (envs={self.num_envs}, cap={self.capacity_per_env}).")
            self.num_envs = num_envs
            self.capacity_per_env = capacity

        self.data = loaded_data
        self._ptr = state['ptr']
        self._full = state['full']

        print(f"[INFO - WM] Dataset loaded. Buffer is {'full' if self._full else 'not full'} with pointer at {self._ptr}.")
        print(f"[INFO - WM] Total valid transitions: {self.__len__()}")

    @classmethod
    def from_files(cls, filepaths, obs_space, act_space, device):
        """
        Creates a new replay buffer by loading and merging data from multiple files.

        Args:
            filepaths (list[str]): A list of file paths to the saved datasets.
            obs_space (gym.spaces.Dict): The observation space, assumed to be consistent across datasets.
            act_space (gym.spaces.Box): The action space, assumed to be consistent.
            device: The torch device to load the data onto.

        Returns:
            An instance of ReplayBuffer populated with the combined data.
        """
        if not filepaths:
            raise ValueError("Filepaths list cannot be empty.")

        print(f"[INFO - WM] Combining {len(filepaths)} datasets...")

        all_data_chunks = {}
        total_envs = 0

        # Load the first file to determine keys and capacity
        first_state = torch.load(filepaths[0], map_location='cpu', weights_only=False)
        keys = first_state['data'].keys()
        capacity_per_env = next(iter(first_state['data'].values())).shape[1]

        for key in keys:
            all_data_chunks[key] = []

        # Loop through all files and collect data chunks
        for i, path in enumerate(filepaths):
            print(f"[INFO - WM] Loading {path}...")
            state = torch.load(path, map_location='cpu', weights_only=False)
            data = state['data']

            # Sanity check: ensure capacity and keys match
            current_cap = next(iter(data.values())).shape[1]
            if current_cap != capacity_per_env:
                raise ValueError(f"Capacity mismatch in {path}. Expected {capacity_per_env}, found {current_cap}.")
            if data.keys() != keys:
                raise ValueError(f"Data keys mismatch in {path}.")

            num_envs_in_file = next(iter(data.values())).shape[0]
            total_envs += num_envs_in_file

            for key in keys:
                all_data_chunks[key].append(data[key])

        # Concatenate all chunks along the 'num_envs' dimension
        combined_data = {}
        for key in keys:
            combined_data[key] = torch.cat(all_data_chunks[key], dim=0).to(device)

        # Create the new buffer instance
        dataset_size = total_envs * capacity_per_env
        buffer = cls(dataset_size, total_envs, obs_space, act_space, device)

        # Populate the new buffer
        buffer.data = combined_data
        buffer._full = True  # The combined buffer is considered static and full
        buffer._ptr = 0

        print("[INFO - WM] Datasets successfully merged.")
        print(f"[INFO - WM] New buffer stats: {total_envs} meta-environments, {capacity_per_env} steps each.")
        print(f"[INFO - WM] Total transitions: {buffer.__len__():,}")

        return buffer


def test_agent(agent, test_envs, test_episodes, policy_action_scale):
    """
    Run evaluation in parallel Isaac envs until each environment has completed
    the requested number of episodes.

    Args:
        agent: The policy to evaluate (no training).
        test_envs: Vectorized Isaac Gym environments.
        test_episodes: Number of test episodes *per environment*.

    Returns:
        avg_reward: Mean reward across all completed episodes.
        avg_length: Mean episode length across all completed episodes.
        num_episodes: Total number of completed episodes.
    """
    num_envs = test_envs.num_envs
    target_episodes = test_episodes * num_envs

    obs = test_envs.reset()
    agent_state = None

    # Track per-env rewards and lengths
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float32, device=test_envs.device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.int32, device=test_envs.device)

    completed_rewards = []
    completed_lengths = []

    while len(completed_rewards) < target_episodes:
        # Agent acts without training
        action_dist, agent_state = agent(obs, reset=obs["is_first"], state=agent_state, training=False)
        unscaled_policy_action = action_dist["action"]
        scaled_policy_action = unscaled_policy_action * policy_action_scale

        # Step vectorized envs
        obs, reward, done, info = test_envs.step(scaled_policy_action)
        cur_reward_sum += reward
        cur_episode_length += 1

        if done.any():
            done_indices = torch.where(done)[0]
            completed_rewards.extend(cur_reward_sum[done_indices].cpu().numpy())
            completed_lengths.extend(cur_episode_length[done_indices].cpu().numpy())
            cur_reward_sum[done_indices] = 0
            cur_episode_length[done_indices] = 0

    avg_reward = float(np.mean(completed_rewards))
    avg_length = float(np.mean(completed_lengths))

    return avg_reward, avg_length, len(completed_rewards)


def train_agent(
        agent,
        env,
        online_buffer,
        logger,
        train_dataset=None,
        prefill_steps=0,
        steps=0,
        state=None,
        message=None,
        update_logger_step=True,
        policy_action_scale=1.0,
):
    """
    Args:
        agent: The policy to collect data with.
        env (IsaacWrapper): The vectorized environment wrapper.
        online_buffer (ReplayBuffer): The online buffer to store transitions in.
        logger (Logger): The logger for metrics.
        train_dataset (iterator, optional): Iterator for the training batch.
        prefill_steps (int, optional): Steps before training starts.
        steps (int): The number of agent steps to perform.
        state (tuple, optional): The state to resume from.
    """

    if state is None:
        step_counter = 0
        obs = env.reset()
        agent_state = None
        best_train_return = float("-inf")
        cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        episode_returns = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        episodes_logged_count = 0
        dataset_save_counter = 0
        save_interval = None
        if hasattr(env, 'reward_scale_per_env'):
            per_robot_returns = {int(i.item()): deque(maxlen=100) for i in env.quadrupeds_types.unique()}
            per_robot_episode_counts = {int(i.item()): 0 for i in env.quadrupeds_types.unique()}
        else:
            per_robot_returns = None
            per_robot_episode_counts = None
    else:
        save_interval = steps // 10
        step_counter, obs, agent_state, best_train_return, cur_reward_sum, cur_episode_length, episode_returns, episode_lengths, episodes_logged_count, dataset_save_counter, per_robot_returns, per_robot_episode_counts = state

    # Determine if we're doing training (agent is not just random policy)
    is_training_mode = train_dataset is not None and isinstance(agent, torch.nn.Module)

    for _ in trange(int(steps), desc=message, ncols=100, dynamic_ncols=False, leave=True):
        # Training step (only after prefill and with real agent)
        if is_training_mode and logger.step >= prefill_steps:
            try:
                data = next(train_dataset)
                if data is not None:
                    agent._train(data)
            except StopIteration:
                # Should not happen with proper iterator, but handle gracefully
                raise RuntimeError("Training dataset iterator exhausted unexpectedly.")

        # Data collection step
        action_dist, agent_state = agent(obs, obs['is_first'], agent_state, training=is_training_mode)
        unscaled_policy_action = action_dist['action']
        scaled_policy_action = unscaled_policy_action * policy_action_scale

        next_obs, reward, done, info = env.step(scaled_policy_action)
        online_buffer.add(obs, scaled_policy_action, reward, done, info)

        step_counter += 1
        if update_logger_step:
            logger.step += env.num_envs

        cur_reward_sum += reward
        cur_episode_length += 1

        # Handle finished episodes
        if done.any():
            done_indices = torch.where(done)[0]

            # Collect completed episode data
            episode_returns.extend(cur_reward_sum[done_indices].cpu().numpy())
            episode_lengths.extend(cur_episode_length[done_indices].cpu().numpy())

            if per_robot_returns is not None:
                quadruped_types_done = env.quadrupeds_types[done_indices].squeeze(-1).cpu().numpy()
                rewards_done = cur_reward_sum[done_indices].cpu().numpy()
                for robot_type, reward_val in zip(quadruped_types_done, rewards_done):
                    robot_type = int(robot_type)
                    per_robot_returns[robot_type].append(reward_val)
                    per_robot_episode_counts[robot_type] += 1

            # Reset trackers for completed episodes
            cur_reward_sum[done_indices] = 0
            cur_episode_length[done_indices] = 0

            # Log every 100 completed episodes
            if len(episode_returns) == 100:  # Deque is full
                episodes_logged_count += 100

                # Calculate metrics from the deque
                mean_return = statistics.mean(episode_returns)
                std_return = statistics.stdev(episode_returns)
                mean_length = statistics.mean(episode_lengths)
                std_length = statistics.stdev(episode_lengths)

                # Log metrics
                logger.scalar("mean_return", mean_return)
                logger.scalar("std_return", std_return)
                logger.scalar("mean_episode_length", mean_length)
                logger.scalar("std_episode_length", std_length)
                logger.scalar("episodes_logged_total", episodes_logged_count)

                # Log replay buffer status
                logger.scalar("online_buffer_filled (%)", online_buffer.ptr / online_buffer.capacity_per_env if not online_buffer.full else 1.0)

                # Log action and observation statistics
                action_stats = tensorstats(scaled_policy_action, prefix="action_stats", section=True)
                for k, v in action_stats.items():
                    logger.scalar(k, v)

                obs_stats = tensorstats(obs['proprio'], prefix="obs_stats", section=True)
                for k, v in obs_stats.items():
                    logger.scalar(k, v)

                if per_robot_returns is not None and hasattr(env._env.unwrapped, 'robots'):
                    robot_names = list(env._env.unwrapped.robots.keys())
                    for i, (robot_idx, robot_values) in enumerate(per_robot_returns.items()):
                        if len(robot_values) > 0:
                            mean_return_robot = statistics.mean(per_robot_returns[robot_idx])
                            logger.scalar(f"return/{robot_names[i]}_mean_return", mean_return_robot)
                            logger.scalar(f"return/{robot_names[i]}_episode_count", per_robot_episode_counts[robot_idx])

                logger.write()

                # Save best model if this is a new best
                if mean_return > best_train_return:
                    best_train_return = mean_return
                    if isinstance(agent, torch.nn.Module):
                        checkpoint = {
                            "agent_state_dict": agent.state_dict(),
                            "optims_state_dict": recursively_collect_optim_state_dict(agent),
                            "step": logger.step,
                            "step_counter": step_counter,
                            "best_train_return": best_train_return,
                        }
                        torch.save(checkpoint, logger._logdir / f"best_model.pt")
                        print(f"\n[INFO - WM] New best model saved! Mean return: {mean_return:.2f} at step {logger.step // env.num_envs}\n")

                episode_returns.clear()
                episode_lengths.clear()

        obs = next_obs

        if save_interval is not None and step_counter % save_interval == 0 and isinstance(agent, torch.nn.Module):
            checkpoint = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": recursively_collect_optim_state_dict(agent),
                "step": logger.step,
                "step_counter": step_counter,
                "best_train_return": best_train_return,
            }
            torch.save(checkpoint, logger._logdir / f"step_{step_counter}_model.pt")
            print(f"\n[INFO - WM] New best model saved! Mean return: {mean_return:.2f} at step {logger.step // env.num_envs}\n")


    # Return the state needed to resume the simulation
    return (step_counter, obs, agent_state, best_train_return, cur_reward_sum, cur_episode_length, episode_returns, episode_lengths, episodes_logged_count, dataset_save_counter, per_robot_returns, per_robot_episode_counts)


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torch_dist.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
            self,
            logits,
            low=-20.0,
            high=20.0,
            transfwd=symlog,
            transbwd=symexp,
            device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
                F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
                + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torch_dist.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
                torch.sqrt((event - self.mean) ** 2 + self._threshold ** 2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torch_dist.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torch_dist.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
            self,
            name,
            parameters,
            lr,
            eps=1e-4,
            clip=None,
            wd=None,
            wd_pattern=r".*",
            opt="adam",
            use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = to_np(loss)
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None, section=False):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        if section:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
        obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


class USDFeatureManager:
    """
    Manages USD feature extraction and provides embeddings for robots.
    """

    def __init__(self, usd_dir, device='cuda'):
        """
        Args:
            usd_dir: Directory containing USD files
            device: Device to store tensors on
        """
        self.device = device
        self.usd_dir = usd_dir

        # Find all USD files
        self.usd_files = sorted(glob.glob(os.path.join(self.usd_dir, "*.usd")))
        if not self.usd_files:
            raise FileNotFoundError(f"No USD files found in {usd_dir}")

        # Extract robot names (without .usd extension)
        self.robot_names = [Path(p).stem for p in self.usd_files]
        print(f"[INFO - USD] Found {len(self.robot_names)} robots: {self.robot_names}")

        # Extract and normalize features
        # self.extractor = USDFeatureExtractor()
        # self._extract_and_cache_features()
        self._load_and_cache_features()

    def _extract_and_cache_features(self):
        """Extract features from all USDs and normalize them."""
        print("[INFO - USD] Extracting features from USD files...")

        raw_features = []
        for usd_path in self.usd_files:
            features = self.extractor.extract_all_features(usd_path)
            raw_features.append(features)

        # Normalize features across all robots
        features_array = np.array(raw_features)
        normalized_features = self.extractor.scaler.fit_transform(features_array)

        # Convert to torch tensor and cache
        self.features_tensor = torch.tensor(
            normalized_features,
            dtype=torch.float32,
            device=self.device
        )

        print(f"[INFO - USD] Extracted features shape: {self.features_tensor.shape}")
        print(f"[INFO - USD] Feature dimension: {self.feature_dim}")

    def _load_and_cache_features(self):
        feature_file = os.path.join(self.usd_dir, "usd_physical_features_minmax_1.npz")
        if not os.path.exists(feature_file):
            extractor = PhysicalMorphologyExtractor()
            try:
                features, filenames = extractor.process_directory(self.usd_dir)

                robots = [os.path.splitext(f)[0] for f in filenames]
                np.savez_compressed(
                    feature_file,
                    features=features,
                    feature_names=extractor.feature_names,
                    robots=robots
                )
                print(f"[INFO - USD] Saved features to {feature_file}")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Failed to extract features: {e}")

        print(f"[INFO - USD] Loading pre-extracted features from {feature_file}...")
        with np.load(feature_file) as data:
            features_array = data['features']
            self.features_tensor = torch.tensor(features_array, dtype=torch.float32, device=self.device)

        print(f"[INFO - USD] Loaded features shape: {self.features_tensor.shape}")

    @property
    def feature_dim(self):
        """Returns the dimensionality of USD features."""
        return self.features_tensor.shape[1]

    @property
    def num_robots(self):
        """Returns the number of robots."""
        return len(self.robot_names)

    def get_features_by_name(self, robot_name):
        """
        Get USD features for a robot by name.

        Args:
            robot_name: Name of the robot (without .usd extension)
        Returns:
            features: Tensor of shape [feature_dim]
        """
        try:
            idx = self.robot_names.index(robot_name)
            return self.features_tensor[idx]
        except ValueError:
            raise ValueError(f"Robot '{robot_name}' not found. Available: {self.robot_names}")

    def get_features_by_index(self, indices):
        """
        Get USD features for robots by their indices.

        Args:
            indices: Tensor of robot indices [...,]
        Returns:
            features: Tensor of shape [..., feature_dim]
        """
        return self.features_tensor[indices]

    def get_features_by_type(self, robot_types, type_to_name_map):
        """
        Get USD features for robots by their type IDs.

        Args:
            robot_types: Tensor of robot type IDs [..., 1] or [...,]
            type_to_name_map: Dict mapping type ID to robot name
        Returns:
            features: Tensor of shape [..., feature_dim]
        """
        # Squeeze if necessary
        if robot_types.dim() > 1:
            robot_types = robot_types.squeeze(-1)

        # Convert types to indices
        indices = torch.zeros_like(robot_types, dtype=torch.long)
        for type_id, robot_name in type_to_name_map.items():
            try:
                robot_idx = self.robot_names.index(robot_name)
                mask = (robot_types == type_id)
                indices[mask] = robot_idx
            except ValueError:
                raise ValueError(f"[WARNING - USD] Robot '{robot_name}' not found for type {type_id}")

        return self.get_features_by_index(indices)

    def create_type_to_name_map(self, quadruped_types_config):
        """
        Create a mapping from quadrupeds_types to robot names.

        Args:
            quadruped_types_config: Dict from config, e.g., {'anymal_d': 0, 'unitree_a1': 1}
        Returns:
            type_to_name_map: Dict mapping type ID to robot name
        """
        return {v: k for k, v in quadruped_types_config.items()}


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def sample_sequences(online_buffer, batch_size, batch_length):
    if not online_buffer.ptr > batch_length and not online_buffer.full:
        return None

    online_count = batch_size
    online_batch = online_buffer.sample_sequences(online_count, batch_length)
    return online_batch