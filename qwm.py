import argparse
import os
import pathlib
import sys
from envs.hq_isaac import HeteroIsaacWrapper
import models
import utils
import uuid
import torch
from torch import nn
from torch import distributions as torch_dist
import numpy as np
from ruamel.yaml import YAML

os.environ["MUJOCO_GL"] = "osmesa"

yaml = YAML(typ='safe', pure=True)

sys.path.append(str(pathlib.Path(__file__).parent))


class QWM(nn.Module):
    def __init__(self, obs_space, act_space, config, logger=None, dataset=None, test=False, usd_manager=None):
        super(QWM, self).__init__()
        if not test:
            batch_steps = config.batch_size * config.batch_length
            self._should_train = utils.Every(batch_steps / config.train_ratio)
            self._step = logger.step // config.action_repeat
        else:
            self._step = 0
        self._config = config
        self._logger = logger
        self._should_log = utils.Every(config.log_every)
        self._should_pretrain = utils.Once()
        self._should_reset = utils.Every(config.reset_every)
        self._should_expl = utils.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config, usd_manager=usd_manager)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (config.compile and os.name != "nt"):
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        self._expl_behavior = self._task_behavior.to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            if self._should_pretrain():
                steps = self._config.pretrain
            else:
                steps = self._should_train(step)
            for _ in range(steps):
                data = next(self._dataset)
                if isinstance(data, tuple):
                    data = data[0]
                self._train(data)
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += 1
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = policy_scaled_action = None
        else:
            latent, policy_scaled_action = state
        obs = self._wm.preprocess(obs)

        usd_features = None
        if self._wm.usd_manager:
            robot_types = obs["quadruped_type"].squeeze(-1)  # [num_envs]
            usd_features = self._wm.usd_manager.get_features_by_type(robot_types, self._wm.type_to_name_map)  # Shape: [num_envs, feature_dim]

        embed = self._wm.encoder(obs, usd_features)
        latent, _ = self._wm.dynamics.obs_step(latent, policy_scaled_action, embed, obs["is_first"], obs["quadruped_type"], usd_features=usd_features)
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            policy_unscaled_action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            policy_unscaled_action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            policy_unscaled_action = actor.sample()
        logprob = actor.log_prob(policy_unscaled_action)
        policy_scaled_action = None if policy_unscaled_action is None else policy_unscaled_action * self._config.policy_action_scale
        latent = {k: v.detach() for k, v in latent.items()}
        policy_output = {"action": policy_unscaled_action.detach(), "logprob": logprob}
        state = (latent, policy_scaled_action.detach() if policy_scaled_action is not None else None)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        usd_features = context.get('usd_features', None)
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        # Pass quadruped_type from data to _train
        metrics.update(self._task_behavior._train(start, reward, data['quadruped_type'], usd_features=usd_features)[-1])

        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def make_env(config, video_dir):
    env = HeteroIsaacWrapper(
        config.task,
        num_envs=config.envs,
        headless=config.headless,
        seed=config.seed,
        video=config.record_video,
        video_dir=video_dir,
        quadrupeds_types=config.quadrupeds_types,
        quadrupeds=list(config.quadrupeds),
        domain_randomization=config.domain_randomization,
    )
    return env


def main(config):
    utils.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        utils.enable_deterministic_run()

    if not config.test:
        config.steps //= config.action_repeat
        config.log_every //= config.action_repeat
        config.time_limit //= config.action_repeat

        logdir = pathlib.Path(f"./logdir/{config.run_name}_{str(uuid.uuid4())[:8]}").expanduser()
        print("[INFO - WM] Logdir", logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        logger = utils.Logger(logdir, 0, config.log_enabled, config.envs, config.run_group)

        print("[INFO - WM] Create envs.")
        train_envs = make_env(config, logdir)
        acts = train_envs.action_space

        online_buffer = utils.ReplayBuffer(
            dataset_size=config.dataset_size,
            num_envs=config.envs,
            obs_space=train_envs.observation_space,
            act_space=train_envs.action_space,
            device=config.device,
            num_robot_types=len(config.quadrupeds),
        )
        # Get the robot types from the environment wrapper
        # The wrapper has 'quadrupeds_types' which is a tensor of shape [num_envs, 1]
        # We need to pass this to the buffer for stratification
        if hasattr(train_envs, "quadrupeds_types"):
             # Ensure it's on the same device and flattened
             r_types = train_envs.quadrupeds_types.to(config.device).view(-1)
             online_buffer.index_envs_by_robot_type(r_types)
        else:
             print("[WARNING] Could not find 'quadrupeds_types' in env. Stratified sampling disabled.")

        def dataset_generator():
            while True:
                data = utils.sample_sequences(
                    online_buffer,
                    config.batch_size,
                    config.batch_length,
                )
                yield data
        train_dataset = dataset_generator()

        print("[INFO - WM] Action Space", acts)
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[-1]

        # Initialize USD feature manager
        try:
            usd_manager = utils.USDFeatureManager(config.usd_dir, device=config.device)
            print(f"[INFO - WM] Loaded USD features from {config.usd_dir}")
        except Exception as e:
            raise RuntimeError(f"[ERROR - WM] Failed to load USD features: {e}")

        # Initialize Agent
        print("[INFO - WM] Initializing agent...")
        agent = QWM(
            train_envs.observation_space,
            train_envs.action_space,
            config,
            logger,
            train_dataset,
            usd_manager=usd_manager,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)

        # Check for fine-tuning checkpoint
        if hasattr(config, "finetune") and config.finetune:
            load_path = pathlib.Path(f"./logdir/{config.from_checkpoint}").expanduser()
            model_file = "best_model.pt" if (load_path / "best_model.pt").exists() else "last_model.pt"
            full_path = load_path / model_file
            print(f"[INFO - WM] Fine-tuning from checkpoint: {full_path}")

            checkpoint = torch.load(full_path, map_location=config.device, weights_only=False)
            agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
            utils.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])

            print("[INFO - WM] Weights and Optimizer states loaded successfully.")
            print(f"[INFO - WM] Previous Best Return: {checkpoint.get('best_train_return', 'N/A')}")
            print(f"[INFO - WM] Resuming training with Domain Randomization: {config.domain_randomization}")

            if getattr(config, "freeze_world_model", False):
                print("[INFO - WM] Freezing world model for fine-tuning")
                # Freeze encoder, dynamics, and heads
                for param in agent._wm.parameters():
                    param.requires_grad = False

                # Ensure policy is trainable
                for param in agent._task_behavior.parameters():
                    param.requires_grad = True

                print("[INFO - WM] Only policy (actor + critic) will be trained")

        prefill = config.prefill if logger.step == 0 else 0
        state = None

        if prefill > 0:
            print(f"[INFO - WM] Prefill dataset ({prefill} steps).")

            if hasattr(config, "finetune") and config.finetune:
                # If fine-tuning, use the loaded agent to collect initial data.
                # This helps the buffer adapt to the new randomized domain without
                # storing pure noise/random actions which might cause immediate falls.
                prefill_actor = agent
                prefill_msg = "[INFO - WM] Prefilling with PRE-TRAINED Agent (Fine-tuning)"
                print(prefill_msg)
                agent.eval()
            else:
                # Standard training from scratch: use random actions
                mean = torch.zeros(config.envs, config.num_actions, device=config.device)
                std = torch.ones(config.envs, config.num_actions, device=config.device)
                random_actor = torch_dist.independent.Independent(torch_dist.normal.Normal(mean, std), 1)

                def random_agent(o, d, s, **kwargs):
                    action = random_actor.sample()
                    logprob = random_actor.log_prob(action)
                    return {"action": action, "logprob": logprob}, None

                prefill_actor = random_agent
                prefill_msg = "[INFO - WM] Prefilling with RANDOM Agent"
                print(prefill_msg)

            prefill_agent_steps = prefill // config.envs
            state = utils.train_agent(
                prefill_actor,
                train_envs,
                online_buffer,
                logger,
                train_dataset=None,  # No training during prefill
                prefill_steps=prefill,
                steps=prefill_agent_steps,
                message=prefill_msg,
                update_logger_step=False,
                policy_action_scale=config.policy_action_scale,
            )

            if hasattr(config, "finetune") and config.finetune:
                agent.train()

        logger.step = prefill
        print(f"[INFO - WM] Logger step set to: {logger.step}")

        print("[INFO - WM] Starting training loop...")
        logger.write()
        _ = utils.train_agent(
            agent,
            train_envs,
            online_buffer,
            logger,
            train_dataset=train_dataset,
            prefill_steps=prefill,
            steps=config.steps,
            state=state,
            message="[INFO - WM] Training agent",
            policy_action_scale=config.policy_action_scale,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": utils.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "last_model.pt")
        logger.close()
        train_envs.close()
    else:
        print("[INFO - WM] Running evaluation only (no training).")
        checkpoint_name = "best_model.pt" if config.best else "last_model.pt"
        full_checkpoint_path = pathlib.Path(f"./logdir/{config.checkpoint_path}/{checkpoint_name}").expanduser()
        checkpoint = torch.load(full_checkpoint_path, map_location=config.device, weights_only=False)

        print(f"[INFO - WM] Model stats during training, loaded from {full_checkpoint_path}:\n"
              f"\ttotal steps: {checkpoint['step']:,}\n"
              f"\tsteps per env: {checkpoint['step_counter']:,}\n"
              f"\tbest return: {checkpoint['best_train_return']:.2f}")

        test_envs = make_env(config, f"./logdir/{config.checkpoint_path}")
        acts = test_envs.action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[-1]

        # Initialize USD manager for test mode
        try:
            usd_manager = utils.USDFeatureManager(config.usd_dir, device=config.device)
            print(f"[INFO - WM] Loaded USD features from {config.usd_dir}")
        except Exception as e:
            raise RuntimeError(f"[ERROR - WM] Failed to load USD features: {e}")

        agent = QWM(
            test_envs.observation_space,
            test_envs.action_space,
            config,
            test=config.test,
            usd_manager=usd_manager,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        avg_eps_reward, avg_eps_length, num_eps_completed = utils.test_agent(agent, test_envs, config.episodes, config.policy_action_scale)
        print(f"[INFO - WM] Finished {num_eps_completed} episodes "
              f"Average reward: {avg_eps_reward:.2f}, "
              f"Average length: {avg_eps_length:.2f}")
        test_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.load((pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text())

    name_list = ["defaults"]
    if args.configs.startswith("hetero_quadruped_pwm"):
        base = "hetero_quadruped_pwm_base"
    else:
        base = "hetero_quadruped_proprio_base"
    if args.configs.startswith("hetero_quadruped") and base in configs:
        name_list.append(base)
    name_list.append(args.configs)

    defaults = {}
    for name in name_list:
        utils.recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = utils.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
