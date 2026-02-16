import numpy as np
import re
import torch
import torch.nn.functional as F
from torch import distributions as torch_dist
from torch import nn

import utils


class RSSM(nn.Module):
    def __init__(
            self,
            stoch=30,
            deter=200,
            hidden=200,
            rec_depth=1,
            discrete=False,
            act="SiLU",
            norm=True,
            mean_act="none",
            std_act="softplus",
            min_std=0.1,
            unimix_ratio=0.01,
            initial="learned",
            num_actions=None,
            embed=None,
            device=None,
            quadrupeds_action_scale=None,
            quadrupeds_action_offset=None,
            quadrupeds_type=None,
            usd_feature_dim=0,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete

        act_module = getattr(torch.nn, act)

        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        self._quadruped_action_scale = quadrupeds_action_scale
        self._quadruped_action_offset = quadrupeds_action_offset
        self._quadruped_type = quadrupeds_type
        self._usd_feature_dim = usd_feature_dim

        def build_layers(in_dim, out_dim, use_norm):
            layers = [nn.Linear(in_dim, out_dim, bias=False)]
            if use_norm:
                layers.append(nn.LayerNorm(out_dim, eps=1e-03))
            layers.append(act_module())
            return layers

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions

        inp_dim += self._usd_feature_dim

        self._img_in_layers = nn.Sequential(*build_layers(inp_dim, self._hidden, norm))
        self._img_in_layers.apply(utils.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(utils.weight_init)

        inp_dim = self._deter
        self._img_out_layers = nn.Sequential(*build_layers(inp_dim, self._hidden, norm))
        self._img_out_layers.apply(utils.weight_init)

        inp_dim = self._deter + self._embed
        self._obs_out_layers = nn.Sequential(*build_layers(inp_dim, self._hidden, norm))
        self._obs_out_layers.apply(utils.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._imgs_stat_layer.apply(utils.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(utils.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(utils.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(utils.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, quadruped_type, usd_features=None, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first, quadruped_type = swap(embed), swap(action), swap(is_first), swap(quadruped_type)

        if usd_features is not None:
            usd_features = swap(usd_features)
            scan_inputs = (action, embed, is_first, quadruped_type, usd_features)
            step_fn = lambda prev_state, prev_act, embed, is_first, quadruped_type, usd: self.obs_step(
                prev_state[0], prev_act, embed, is_first, quadruped_type, usd_features=usd
            )
        else:
            scan_inputs = (action, embed, is_first, quadruped_type)
            step_fn = lambda prev_state, prev_act, embed, is_first, quadruped_type: self.obs_step(
                prev_state[0], prev_act, embed, is_first, quadruped_type, usd_features=None
            )

        post, prior = utils.static_scan(
            step_fn,
            scan_inputs,
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torch_dist.independent.Independent(utils.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)
        else:
            mean, std = state["mean"], state["std"]
            dist = utils.ContDist(torch_dist.independent.Independent(torch_dist.normal.Normal(mean, std), 1))
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, quadruped_type, usd_features=None, sample=True):
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions), device=self._device)
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None] if is_first.dim() == 1 else is_first
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(is_first, is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)))
                prev_state[key] = (val * (1.0 - is_first_r) + init_state[key] * is_first_r)

        q_indices = (quadruped_type.squeeze(-1)[:, None] == self._quadruped_type[None]).float().argmax(dim=1)
        sim_action_scale = self._quadruped_action_scale[q_indices]  # (batch, 1)
        sim_action_offset = self._quadruped_action_offset[q_indices]  # (batch, action_dim)
        if sim_action_scale.dim() == 1:
            sim_action_scale = sim_action_scale.unsqueeze(-1)
        sim_scaled_prev_action = prev_action * sim_action_scale + sim_action_offset

        # Defensive check: Verify USD features match quadruped_type
        if usd_features is not None:
            batch_size = quadruped_type.shape[0]
            assert usd_features.shape[0] == batch_size, \
                f"USD features batch mismatch: got {usd_features.shape[0]}, expected {batch_size}"
            assert usd_features.shape[-1] == self._usd_feature_dim, \
                f"USD feature dim mismatch: got {usd_features.shape[-1]}, expected {self._usd_feature_dim}"

        prior = self.img_step(prev_state, sim_scaled_prev_action, usd_features=usd_features)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, usd_features=None, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)

        inputs = [prev_stoch, prev_action]
        if usd_features is None:
            raise ValueError(
                "[Error] 'usd_features' is None in img_step(). "
                "Check WorldModel.observe() and QWM._policy() logic."
            )
        if usd_features.shape[-1] != self._usd_feature_dim:
            raise ValueError(
                f"USD dim mismatch. Expected {self._usd_feature_dim}, got {usd_features.shape[-1]}"
            )
        inputs.append(usd_features)

        x = torch.cat(inputs, -1)

        x = self._img_in_layers(x)
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter_state = self._cell(x, [deter])
            deter = deter_state[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torch_dist.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
            self,
            shapes,
            mlp_keys,
            act,
            norm,
            mlp_layers,
            mlp_units,
            symlog_inputs,
            usd_feature_dim=None,
            usd_tower_ratio=0.5,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward", "quadruped_type")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("[INFO - WM] Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        input_size = sum([sum(v) for v in self.mlp_shapes.values()])

        # Separate processing towers
        if usd_feature_dim is None:
            raise ValueError("usd_feature_dim required for tower mode")

        usd_tower_units = int(mlp_units * usd_tower_ratio)

        print(f"[INFO - WM] Encoder mode: Separate towers (proprio_units={mlp_units}, usd_units={usd_tower_units})")

        # Proprioception tower
        self.proprio_tower = MLP(
            input_size,
            None,
            mlp_layers,
            mlp_units,
            act,
            norm,
            symlog_inputs=symlog_inputs,
            name="ProprioTower",
        )

        # USD tower (fewer layers since features are already informative)
        self.usd_tower = MLP(
            usd_feature_dim,
            None,
            mlp_layers // 2,  # Fewer layers for USD
            usd_tower_units,
            act,
            norm,
            symlog_inputs=False,  # USD features already normalized
            name="USDTower",
        )

        # Fusion layer
        fusion_layers = [nn.Linear(mlp_units + usd_tower_units, mlp_units, bias=False)]
        act_fn = getattr(torch.nn, act)
        if norm:
            fusion_layers.append(nn.LayerNorm(mlp_units, eps=1e-03))
        fusion_layers.append(act_fn())

        self.fusion = nn.Sequential(*fusion_layers)
        self.fusion.apply(utils.weight_init)

        self.outdim = mlp_units

    def forward(self, obs, usd_features=None):
        # Concatenate all observation components
        inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)

        # Mode 3: Separate towers + fusion
        if usd_features is None:
            raise ValueError("usd_features required")

        # Process proprioception
        proprio_embed = self.proprio_tower(inputs)

        # Handle broadcasting for USD features
        if usd_features.dim() == 2 and inputs.dim() == 3:
            # [batch, usd_dim] -> [batch, seq, usd_dim]
            usd_features = usd_features.unsqueeze(1).expand(-1, inputs.shape[1], -1)

        # Process USD features
        usd_embed = self.usd_tower(usd_features)

        # Fuse
        combined = torch.cat([proprio_embed, usd_embed], dim=-1)
        output = self.fusion(combined)

        return output


class MultiDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shapes,
            mlp_keys,
            act,
            norm,
            mlp_layers,
            mlp_units,
            vector_dist,
            outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "quadruped_type")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("[INFO - WM] Decoder MLP shapes:", self.mlp_shapes)

        self._mlp = MLP(
            feat_size,
            self.mlp_shapes,
            mlp_layers,
            mlp_units,
            act,
            norm,
            vector_dist,
            outscale=outscale,
            name="Decoder",
        )

    def forward(self, features):
        dists = {}
        dists.update(self._mlp(features))
        return dists


class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,
            shape,
            layers,
            units,
            act="SiLU",
            norm=True,
            dist="normal",
            std=1.0,
            min_std=0.1,
            max_std=1.0,
            absmax=None,
            temp=0.1,
            unimix_ratio=0.01,
            outscale=1.0,
            symlog_inputs=False,
            device="cuda",
            name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)

        act_module = getattr(torch.nn, act)

        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )

            if norm:
                self.layers.add_module(f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03))
            self.layers.add_module(f"{name}_act{i}", act_module())

            if i == 0:
                inp_dim = units
        self.layers.apply(utils.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(utils.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(utils.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(utils.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(utils.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = utils.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torch_dist.normal.Normal(mean, std)
            dist = torch_dist.transformed_distribution.TransformedDistribution(dist, utils.TanhBijector())
            dist = torch_dist.independent.Independent(dist, 1)
            dist = utils.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            dist = torch_dist.normal.Normal(torch.tanh(mean), std)
            dist = utils.ContDist(torch_dist.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "normal_std_fixed":
            dist = torch_dist.normal.Normal(mean, self._std)
            dist = utils.ContDist(torch_dist.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = utils.SafeTruncatedNormal(mean, std, -1, 1)
            dist = utils.ContDist(torch_dist.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "onehot":
            dist = utils.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = utils.ContDist(torch_dist.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax)
        elif dist == "huber":
            dist = utils.ContDist(torch_dist.independent.Independent(utils.UnnormalizedHuber(mean, std, 1.0), len(shape), absmax=self._absmax))
        elif dist == "binary":
            dist = utils.Bernoulli(torch_dist.independent.Independent(torch_dist.bernoulli.Bernoulli(logits=mean), len(shape)))
        elif dist == "symlog_disc":
            dist = utils.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = utils.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

