from isaaclab.app import AppLauncher
import gymnasium as gym
import numpy as np
import torch
import sys
import atexit


class HeteroIsaacWrapper:
    def __init__(
            self,
            task,
            num_envs=64,
            headless=True,
            seed=10,
            video=False,
            video_dir=None,
            video_name_prefix="run",
            quadrupeds_types=None,
            quadrupeds=['anymal_d'],
            domain_randomization=True,
    ):
        self.app_launcher = AppLauncher(headless=headless)
        self.simulation_app = self.app_launcher.app

        # Register cleanup on exit
        atexit.register(self._emergency_cleanup)

        import isaaclab_tasks
        from isaaclab_tasks.utils import load_cfg_from_registry

        self.isaac_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        self.isaac_cfg.scene.num_envs = num_envs
        self.isaac_cfg.seed = seed

        self.isaac_cfg.domain_randomization = domain_randomization
        self.isaac_cfg.observation_noise.enabled = domain_randomization

        self.num_envs = num_envs
        self.video = video
        self._closed = False

        self._cleanup_sys_argv()
        render_mode = "rgb_array" if video else None
        self._env = gym.make(task, cfg=self.isaac_cfg, render_mode=render_mode, quadrupeds=quadrupeds)
        if video:
            video_kwargs = {"video_folder": f"{video_dir}/videos", "video_length": 1000, "name_prefix":video_name_prefix}
            self._env = gym.wrappers.RecordVideo(self._env, **video_kwargs)

        self.device = self._env.unwrapped.device
        self.quadrupeds_types = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.long)

        max_type_id = max(quadrupeds_types.values()) if quadrupeds_types else len(quadrupeds) - 1
        reward_scales_lookup = torch.ones(max_type_id + 1, device=self.device)

        idx = 0
        for robot, robot_articulation in self._env.unwrapped.robots.items():
            self.quadrupeds_types[idx:idx + robot_articulation.num_instances] = quadrupeds_types[robot]
            idx += robot_articulation.num_instances
        quadruped_type_flat = self.quadrupeds_types.squeeze(-1).long()  # (num_envs,)
        self.reward_scale_per_env = reward_scales_lookup[quadruped_type_flat]

    def _cleanup_sys_argv(self):
        """Removes arguments that might conflict with Isaac Lab's parser."""
        args_to_remove = ["--log_enabled", "--configs", "--run_name"]
        for arg in args_to_remove:
            if arg in sys.argv:
                try:
                    index = sys.argv.index(arg)
                    sys.argv.pop(index)  # remove the argument
                    sys.argv.pop(index)  # remove its value
                except IndexError:
                    pass  # Arg was last, no value followed

    @property
    def observation_space(self):
        """Defines the observation space for the agent in the expected Dict format."""
        proprio_space = self._env.unwrapped.single_observation_space["policy"]
        spaces = {
            "proprio": gym.spaces.Box(-np.inf, np.inf, proprio_space.shape, dtype=np.float32),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "quadruped_type": gym.spaces.Box(0, 10, (1,), dtype=np.float32),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """The action space is directly taken from the underlying Isaac environment."""
        return self._env.unwrapped.single_action_space

    def reset(self):
        """
        Resets all parallel environments. This is typically only called once at the beginning.
        """
        # Reset the Isaac environment.
        observations_tensor, infos = self._env.reset()

        # For the reset observation, all environments are "first"
        is_first_reset = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)

        obs = {
            "proprio": observations_tensor["policy"],
            # At reset, no state is terminal.
            "is_terminal": torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device),
            # All observations from reset are "first"
            "is_first": is_first_reset,
            "quadruped_type": self.quadrupeds_types,
        }
        return obs

    def step(self, action):
        """
        Performs a step in all parallel environments, correctly handling auto-resets.
        """
        observations, rewards, terminated, truncated, infos = self._env.step(action)
        done = torch.logical_or(terminated, truncated).unsqueeze(1)
        rewards *= self.reward_scale_per_env

        infos["discount"] = (~terminated).float().unsqueeze(1)

        obs = {
            "proprio": observations["policy"],
            # It is dealt with in the ReplayBuffer class, so we set all to False here.
            "is_terminal": torch.full_like(done, False),
            # an observation is the 'first' in an episode if the PREVIOUS step resulted in a 'done'.
            "is_first": done,
            "quadruped_type": self.quadrupeds_types,
        }

        return obs, rewards, done, infos

    def close(self):
        """Properly close the environment and simulation app."""
        if self._closed:
            return

        try:
            # Close the gymnasium environment first
            if hasattr(self, '_env') and self._env is not None:
                self._env.close()

            # Give some time for cleanup
            import time
            time.sleep(0.1)

            # Close the simulation app
            if hasattr(self, 'simulation_app') and self.simulation_app is not None:
                self.simulation_app.close()

        except Exception as e:
            print(f"[WARNING - WM] {e}")
        finally:
            self._closed = True
            # Unregister the emergency cleanup since we've done it properly
            try:
                atexit.unregister(self._emergency_cleanup)
            except ValueError:
                pass  # Already unregistered

    def _emergency_cleanup(self):
        """Emergency cleanup if close() wasn't called explicitly."""
        if not self._closed:
            self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()

    def __getattr__(self, name):
        if hasattr(self._env, name):
            return getattr(self._env, name)
        if hasattr(self._env.env, name):
            return getattr(self._env.env, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
