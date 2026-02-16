import copy

import networks
import torch
import utils
from torch import nn

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config, usd_manager=None):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.usd_manager = usd_manager

        if hasattr(config, "quadrupeds"):
            num_robot_types = len(config.quadrupeds_types)
        else:
            num_robot_types = 1
        self.register_buffer("wm_reward_ema_vals", torch.zeros((num_robot_types, 2), device=self._config.device))
        self.reward_ema_loss_balancer = RewardEMA(device=self._config.device, alpha=self._config.per_robot_reward_ema_alpha)

        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        # Determine USD conditioning settings
        usd_feature_dim = usd_manager.feature_dim if usd_manager else None

        self.encoder = networks.MultiEncoder(shapes, **config.encoder,
            usd_feature_dim=usd_feature_dim,
            usd_tower_ratio=config.usd_tower_ratio,
        )
        self.embed_size = self.encoder.outdim

        if hasattr(config, "quadrupeds"):
            quadrupeds_action_scale = []
            quadrupeds_action_offset = []
            quadrupeds_model_type = []
            for r in config.quadrupeds:
                quadrupeds_action_scale.append(config.quadrupeds_action_scale[r])
                quadrupeds_action_offset.append(config.quadrupeds_action_offset[r])
                quadrupeds_model_type.append(config.quadrupeds_types[r])
            self.quadrupeds_action_scale = torch.tensor(quadrupeds_action_scale, device=config.device).unsqueeze(-1)  # (num_quadrupeds, 1)
            self.quadrupeds_action_offset = torch.tensor(quadrupeds_action_offset, device=config.device)  # (num_quadrupeds, action_dim)
            self.quadrupeds_model_type = torch.tensor(quadrupeds_model_type, device=config.device)  # (num_quadrupeds,)
            if usd_manager:
                self.type_to_name_map = usd_manager.create_type_to_name_map(config.quadrupeds_types)
            else:
                self.type_to_name_map = {v: k for k, v in config.quadrupeds_types.items()}
        else:
            self.quadrupeds_action_scale = torch.tensor(config.quadrupeds_action_scale[config.target_robot], device=config.device).unsqueeze(0)  # (1, )
            self.quadrupeds_action_offset = torch.tensor(config.quadrupeds_action_offset[config.target_robot], device=config.device).unsqueeze(0)  # (action_dim, )
            self.quadrupeds_model_type = torch.tensor(config.quadrupeds_types[config.target_robot], device=config.device).unsqueeze(0)  # (1, )
            self.type_to_name_map = {config.quadrupeds_types[config.target_robot]: config.target_robot}

        print(f"[INFO - WM] Quadruped action scales shape: {self.quadrupeds_action_scale.shape}")
        print(f"[INFO - WM] Quadruped action offsets shape: {self.quadrupeds_action_offset.shape}")
        print(f"[INFO - WM] Quadruped model types shape: {self.quadrupeds_model_type.shape}")
        print(f"[INFO - WM] Type to name mapping: {self.type_to_name_map}")

        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            self.quadrupeds_action_scale,
            self.quadrupeds_action_offset,
            self.quadrupeds_model_type,
            usd_feature_dim=usd_feature_dim if usd_feature_dim else 0,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        decoder_shapes = shapes.copy()

        self.heads["decoder"] = networks.MultiDecoder(feat_size, decoder_shapes, **config.decoder)

        # Single robot or homogeneous case
        self.family_to_robot = None
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.reward_head["units"],
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = utils.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(f"[INFO - WM] Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables.")
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _get_robot_families(self, quadrupeds):
        """
        Group robots into families based on naming patterns.
        Returns: dict mapping family_name -> list of robot names
        """
        families = {}
        for robot in quadrupeds:
            if robot.startswith("anymal"):
                family = "anymal"
            elif robot.startswith("unitree"):
                family = "unitree"
            elif robot.startswith("spot"):
                family = "spot"
            else:
                family = "other"

            if family not in families:
                families[family] = []
            families[family].append(robot)

        return families

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with utils.RequiresGrad(self):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                # Get USD features if needed
                # data["quadruped_type"]: [batch_size, batch_length, 1]
                robot_types_flat = data["quadruped_type"].reshape(-1)
                usd_features = self.usd_manager.get_features_by_type(robot_types_flat, self.type_to_name_map)
                # Reshape to [batch_size, batch_length, feature_dim]
                batch_size, batch_length = data["quadruped_type"].shape[:2]
                usd_features = usd_features.reshape(batch_size, batch_length, -1)

                # Encode observations
                embed = self.encoder(data, usd_features)

                post, prior = self.dynamics.observe(
                    embed,
                    data["action"],
                    data["is_first"],
                    data['quadruped_type'],
                    usd_features=usd_features
                )

                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale)
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()

                    pred = head(feat)

                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)

                    # Apply loss weighting ONLY to the reward head
                    if name == 'reward':
                        with torch.no_grad():
                            quadruped_type_flat = data['quadruped_type'].reshape(-1)
                            unique_types = torch.unique(quadruped_type_flat)
                            weights = torch.ones_like(loss, dtype=torch.float32)
                            for type_idx in unique_types:
                                type_idx_int = int(type_idx.item())
                                mask = (quadruped_type_flat == type_idx).reshape(loss.shape)
                                rewards_masked = data[name][mask]
                                if rewards_masked.numel() > 0:
                                    _, scale = self.reward_ema_loss_balancer(rewards_masked, self.wm_reward_ema_vals[type_idx_int])
                                    weight = 1.0 / torch.clip(scale, min=1.0)
                                    weights[mask] = weight
                            weights = weights.reshape(loss.shape)
                        loss = loss * weights
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            metrics["prior_ent"] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
            metrics["post_ent"] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
                usd_features=usd_features,
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def _predict_reward_multi_head(self, reward_heads, feat, quadrupeds_types):
        """
        Route each environment to its appropriate reward head based on robot type.

        Args:
            reward_heads: ModuleDict of reward heads
            feat: Features [batch, time, feat_dim]
            quadrupeds_types: Robot types [batch, time, 1]

        Returns:
            Combined predictions routed to appropriate heads
        """
        batch_size, time_steps, feat_dim = feat.shape

        # Flatten for easier processing
        feat_flat = feat.reshape(batch_size * time_steps, feat_dim)
        types_flat = quadrupeds_types.reshape(-1)

        # Determine output shape by checking what the head produces
        first_head = next(iter(reward_heads.values()))
        sample_output = first_head(feat[:1, :1])  # [1, 1, ...] -> distribution

        if self._config.reward_head["dist"] == "symlog_disc":
            output_logits = torch.zeros(batch_size * time_steps, sample_output.logits.shape[-1], device=feat.device)

            # Route each sample to its family's head
            for robot_name, family_name in self.robot_to_family.items():
                mask = (types_flat == self._config.quadrupeds_types[robot_name])
                if mask.any():
                    feat_subset = feat_flat[mask]
                    pred_subset = reward_heads[family_name](feat_subset)
                    output_logits[mask] = pred_subset.logits
                else:
                    raise ValueError(f"No samples found for robot type: {robot_name}")

            # Reshape back to [batch, time, 255]
            output_logits = output_logits.reshape(batch_size, time_steps, sample_output.logits.shape[-1])

            return utils.DiscDist(logits=output_logits)

        else:
            # SymlogDist or other cases: we need to combine means
            output_mean = torch.zeros(batch_size * time_steps, 1, device=feat.device)

            # Route each sample to its family's head
            for robot_name, family_name in self.robot_to_family.items():
                mask = (types_flat == self._config.quadrupeds_types[robot_name])
                if mask.any():
                    feat_subset = feat_flat[mask]
                    pred_subset = reward_heads[family_name](feat_subset)
                    # Get the mode/mean as a tensor
                    output_mean[mask] = pred_subset.mode().reshape(-1, 1)

            # Reshape back to [batch, time, 1]
            output_mean = output_mean.reshape(batch_size, time_steps, 1)

            return utils.SymlogDist(output_mean)

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: v.clone().detach().to(device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1) if obs["discount"].dim() == 2 else obs["discount"]

        # Required fields - validate their presence
        assert "is_first" in obs, "Missing 'is_first' in observations"
        assert "is_terminal" in obs, "Missing 'is_terminal' in observations"
        assert "quadruped_type" in obs, "Missing 'quadruped_type' in observations - this is required for proper action scaling"

        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1) if obs["is_terminal"].dim() != 3 else (1.0 - obs["is_terminal"])
        return obs


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = utils.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(f"[INFO - WM] Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables.")
        self._value_opt = utils.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(f"[INFO - WM] Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables.")
        self.num_robot_types = len(config.quadrupeds_types)
        print(f"[INFO - WM] Initializing RewardEMA for {self.num_robot_types} robot types.")
        # register ema_vals to nn.Module for enabling torch.save and torch.load
        self.register_buffer("ema_vals", torch.zeros((self.num_robot_types, 2), device=self._config.device))
        self.reward_ema = RewardEMA(device=self._config.device)

    def _train(self, start, objective, quadruped_type, usd_features=None, train_in_imagination=False):
        self._update_slow_target()
        metrics = {}

        # Flatten quadruped_type: (batch_size, batch_length, 1) -> (batch_flat,)
        quadruped_type_flat = quadruped_type.reshape(-1)
        q_indices = (quadruped_type_flat[:, None] == self._world_model.quadrupeds_model_type[None]).float().argmax(dim=1)
        sim_action_scale = self._world_model.quadrupeds_action_scale[q_indices]  # (batch_flat, 1)
        if sim_action_scale.dim() == 1:
            sim_action_scale = sim_action_scale.unsqueeze(-1)
        sim_action_offset = self._world_model.quadrupeds_action_offset[q_indices]  # (batch_flat, action_dim)

        # start is post from wm._train, which is (batch, time, ...).
        # We need usd features matching this batch/time structure.
        # If passed from outside, flatten them.
        if usd_features is not None:
            # Flatten to (batch*time, feature_dim)
            usd_features_flat = usd_features.reshape(-1, usd_features.shape[-1])
        else:
            usd_features_flat = None

        with utils.RequiresGrad(self.actor):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                # Imagine trajectories
                imag_feat, imag_state, imag_unscaled_action = self._imagine(
                    start,
                    self.actor,
                    self._config.imag_horizon,
                    sim_action_scale,
                    sim_action_offset,
                    usd_features=usd_features_flat
                )

                # Vectorized scaling for reward computation
                imag_policy_scaled_action = imag_unscaled_action * self._config.policy_action_scale

                imag_scaled_action = imag_policy_scaled_action * sim_action_scale + sim_action_offset

                reward = self._compute_imagined_reward(objective, imag_feat, imag_state, imag_scaled_action, quadruped_type_flat)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                target, weights, base = self._compute_target(imag_feat, imag_state, reward)
                actor_loss, mets = self._compute_actor_loss(imag_feat, imag_unscaled_action, target, weights, base, quadruped_type_flat, train_in_imagination)
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with utils.RequiresGrad(self.value):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(utils.tensorstats(value.mode(), "value"))
        metrics.update(utils.tensorstats(target, "target"))
        metrics.update(utils.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                utils.tensorstats(
                    torch.argmax(imag_unscaled_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(utils.tensorstats(imag_scaled_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with utils.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_scaled_action, weights, metrics

    def _compute_imagined_reward(self, objective, imag_feat, imag_state, imag_action, quadrupeds_types):
        """
        Compute rewards during imagination, routing to appropriate reward heads.
        """
        if not isinstance(self._world_model.heads["reward"], nn.ModuleDict):
            # Single head case
            return objective(imag_feat, imag_state, imag_action)

        horizon, batch_size, feat_dim = imag_feat.shape
        feat_flat = imag_feat.reshape(horizon * batch_size, feat_dim)

        # Expand quadrupeds_types for all timesteps
        # quadrupeds_types is [batch_flat] from the starting states
        types_expanded = quadrupeds_types.unsqueeze(0).expand(horizon, -1).reshape(-1)

        rewards_flat = torch.zeros(horizon * batch_size, device=imag_feat.device)
        for robot_name, family_name in self._world_model.robot_to_family.items():
            mask = (types_expanded == self._config.quadrupeds_types[robot_name])
            if mask.any():
                feat_masked = feat_flat[mask]  # [num_masked, feat_dim]
                reward_pred = self._world_model.heads["reward"][family_name](feat_masked).mode()  # [num_masked, 1]
                rewards_flat[mask] = reward_pred.squeeze(-1)  # Squeeze to [num_masked]
        rewards = rewards_flat.reshape(horizon, batch_size, 1)
        return rewards

    def _imagine(self, start, policy, horizon, sim_action_scale, sim_action_offset, usd_features=None):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        # USD features are constant across imagination horizon
        # They should be [batch_flat, feature_dim]
        # start has shape [batch*time, ...], so batch_flat = batch*time
        if usd_features is not None:
            # Ensure correct shape
            batch_flat = start['stoch'].shape[0]
            if usd_features.shape[0] != batch_flat:
                raise ValueError(f"USD features shape mismatch: got {usd_features.shape[0]}, expected {batch_flat}")

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            unscaled_action = policy(inp).sample()

            # Scale to policy range
            policy_scaled_action = unscaled_action * self._config.policy_action_scale

            # Apply scaling for simulation
            scaled_action = policy_scaled_action * sim_action_scale + sim_action_offset

            succ = dynamics.img_step(state, scaled_action, usd_features=usd_features)
            return succ, feat, unscaled_action

        succ, feats, actions = utils.static_scan(
            step,
            [torch.arange(horizon)],
            (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = utils.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
            self,
            imag_feat,
            imag_action,
            target,
            weights,
            base,
            quadruped_type_flat,
            train_in_imagination=False
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)

        # Per-robot-type reward normalization
        normed_target = target.clone()
        normed_base = base.clone()
        unique_types = torch.unique(quadruped_type_flat)
        for type_idx in unique_types:
            type_idx = int(type_idx.item())
            mask_b = (quadruped_type_flat == type_idx)[None, :, None]
            mask_tb = mask_b.expand_as(target)
            masked_returns = target[mask_tb].detach()
            if masked_returns.numel() > 0:
                offset, scale = self.reward_ema(masked_returns, self.ema_vals[type_idx])
                normed_target_type = (target - offset) / scale
                normed_base_type = (base - offset) / scale
                normed_target = torch.where(mask_tb, normed_target_type, normed_target)
                normed_base   = torch.where(mask_tb, normed_base_type,   normed_base)
                robot_name = self._world_model.type_to_name_map.get(type_idx, f"type_{type_idx}")
                metrics[f"EMA_005_{robot_name}"] = to_np(self.ema_vals[type_idx, 0])
                metrics[f"EMA_095_{robot_name}"] = to_np(self.ema_vals[type_idx, 1])

        adv = normed_target - normed_base
        metrics.update(utils.tensorstats(normed_target, "normed_target"))

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)

        actor_loss = -weights[:-1] * actor_target
        if train_in_imagination:
            action_penalty = torch.mean(imag_action[:-1] ** 2, dim=-1, keepdim=True)
            actor_loss += self._config.action_penalty_coeff * action_penalty
            metrics["action_penalty"] = to_np(torch.mean(action_penalty))
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
