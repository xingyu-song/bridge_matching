# ============================================================
# Bridge Matching Target Definitions
# ============================================================
#
# This file defines different target constructions for Bridge Matching (BM),
# where each target specifies how to compute:
#   - x_t : intermediate sample along the probability path
#   - u*  : transport (deterministic) component
#   - d*  : diffusion / osmotic (stochastic) component
#
#
# ============================================================

from __future__ import annotations

import importlib

import torch
from torch import Tensor


class BaseBridgeMatchingTargets:
    def __init__(
        self,
        beta_value: float = 1e-2,
        sigma_floor: float = 5e-2,
        posterior_temperature: float = 1.0,
        posterior_topk: int | None = None,
        **kwargs,
    ):
        del kwargs
        self.beta_value = beta_value
        self.sigma_floor = sigma_floor
        self.posterior_temperature = posterior_temperature
        self.posterior_topk = posterior_topk

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


TARGET_REGISTRY = {}


def register_target(name: str):
    def decorator(cls):
        TARGET_REGISTRY[name] = cls
        return cls
    return decorator


def make_bridge_matching_targets(
    target_type: str,
    beta_value: float = 1e-2,
    sigma_floor: float = 5e-2,
    posterior_temperature: float = 1.0,
    posterior_topk: int | None = None,
    **kwargs,
) -> BaseBridgeMatchingTargets:
    if target_type not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target_type: {target_type}. Available: {list(TARGET_REGISTRY.keys())}")
    return TARGET_REGISTRY[target_type](
        beta_value=beta_value,
        sigma_floor=sigma_floor,
        posterior_temperature=posterior_temperature,
        posterior_topk=posterior_topk,
        **kwargs,
    )

@register_target("bm_linear_marginal")
class LinearMarginalBridgeMatchingTargets(BaseBridgeMatchingTargets):
    def _expand_time_like(self, t: Tensor, x: Tensor) -> Tensor:
        shape = [t.shape[0]] + [1] * (x.dim() - 1)
        return t.view(*shape)

    def mean_path(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        t_view = self._expand_time_like(t, x_1)
        return (1.0 - t_view) * x_0 + t_view * x_1

    def diffusion_v_star(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        return x_1 - x_0

    def batch_kde_score(self, x_t: Tensor) -> Tensor:
        batch = x_t.size(0)
        flat_dim = x_t[0].numel()
        x_flat = x_t.view(batch, flat_dim)

        h = max(float(self.sigma_floor), 1e-4)
        dist2 = torch.cdist(x_flat, x_flat) ** 2

        logits = -dist2 / (2.0 * h * h)
        logits.fill_diagonal_(float("-inf"))
        weights = torch.softmax(logits, dim=1)

        weighted_x = weights @ x_flat
        score = (weighted_x - x_flat) / (h * h)
        return score.view_as(x_t)

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_t = self.mean_path(x_0=x_0, x_1=x_1, t=t)
        v_star = self.diffusion_v_star(x_0=x_0, x_1=x_1)
        score_star = self.batch_kde_score(x_t=x_t)
        d_star = self.beta_value * score_star
        u_star = v_star - d_star
        return x_t, u_star, d_star

@register_target("bm_linear_marginal_ot")
class LinearMarginalOTBridgeMatchingTargets(LinearMarginalBridgeMatchingTargets):
    """`bm_linear_marginal` augmented with mini-batch OT coupling.

    Before computing the linear-marginal BM target, we permute the batch so
    each x_0 is paired with its Hungarian-optimal x_1 under squared L2 cost.
    The path itself stays the linear / CondOT path
    ``x_t = (1 - t) * x_0 + t * x_1`` (same as ``flow_matching.path.CondOTProbPath``),
    so this isolates the effect of the coupling vs. the path.
    """

    def _ot_pair(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise ImportError(
                "bm_linear_marginal_ot requires scipy. "
                "Install via `pip install scipy`."
            ) from e

        b = x_0.shape[0]
        x0_flat = x_0.view(b, -1)
        x1_flat = x_1.view(b, -1)
        cost = torch.cdist(x0_flat, x1_flat) ** 2
        _, col = linear_sum_assignment(cost.detach().cpu().numpy())
        col_t = torch.as_tensor(col, device=x_1.device, dtype=torch.long)
        return x_1.index_select(0, col_t)

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_1 = self._ot_pair(x_0=x_0, x_1=x_1)
        return super().compute(x_0=x_0, x_1=x_1, t=t)

@register_target("bm_linear_conditional")
class LinearConditionalBridgeMatchingTargets(BaseBridgeMatchingTargets):
    """Linear conditional BM target.

    Conditional linear interpolation:
        x_t = (1 - t) x_0 + t x_1.

    The forward conditional velocity is:
        v*_t = d x_t / dt = x_1 - x_0.

    We use the Gaussian conditional bridge score induced by the stochastic
    tube around the linear mean path:
        score*_t = - (x_t - m_t) / sigma_t^2,
        d*_t = beta * score*_t,
        u*_t = v*_t - d*_t.

    Unlike `bm_linear_marginal`, this target samples a noisy conditional
    intermediate point around the paired endpoint path, so the osmotic term is
    conditional on the specific pair (x_0, x_1) rather than estimated from the
    batch marginal by KDE.
    """

    def _expand_time_like(self, t: Tensor, x: Tensor) -> Tensor:
        shape = [t.shape[0]] + [1] * (x.dim() - 1)
        return t.view(*shape)

    def sigma(self, t: Tensor) -> Tensor:
        # Symmetric bridge tube: zero at endpoints, maximal near t = 0.5.
        # The floor avoids division by zero and keeps training numerically stable.
        return torch.sqrt((t * (1.0 - t)).clamp_min(self.sigma_floor ** 2))

    def sigma_dot(self, t: Tensor) -> Tensor:
        sigma_t = self.sigma(t).clamp_min(self.sigma_floor)
        return (1.0 - 2.0 * t) / (2.0 * sigma_t)

    def mean_path(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        t_view = self._expand_time_like(t, x_1)
        return (1.0 - t_view) * x_0 + t_view * x_1

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        sigma_t = self._expand_time_like(self.sigma(t), x_1)
        sigma_dot_t = self._expand_time_like(self.sigma_dot(t), x_1)

        eps = torch.randn_like(x_1)
        m_t = self.mean_path(x_0=x_0, x_1=x_1, t=t)
        x_t = m_t + sigma_t * eps

        mean_velocity = x_1 - x_0
        v_star = mean_velocity + (sigma_dot_t / sigma_t.clamp_min(1e-12)) * (x_t - m_t)
        d_star = -self.beta_value * (x_t - m_t) / (sigma_t ** 2)
        u_star = v_star - d_star
        return x_t, u_star, d_star

class _VPDiffusionBridgeMatchingTargets(BaseBridgeMatchingTargets):
    """Shared official VP Gaussian path used by BM and CFM diffusion targets."""

    def __init__(self, cfm_beta_min: float = 0.1, cfm_beta_max: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.cfm_beta_min = float(cfm_beta_min)
        self.cfm_beta_max = float(cfm_beta_max)

        path_module = importlib.import_module("flow_matching.path")
        scheduler_module = importlib.import_module("flow_matching.path.scheduler")
        AffineProbPath = path_module.AffineProbPath
        VPScheduler = scheduler_module.VPScheduler

        self.scheduler = VPScheduler(
            beta_min=self.cfm_beta_min,
            beta_max=self.cfm_beta_max,
        )
        self.path = AffineProbPath(scheduler=self.scheduler)

    def _expand_time_like(self, t: Tensor, x: Tensor) -> Tensor:
        shape = [t.shape[0]] + [1] * (x.dim() - 1)
        return t.view(*shape)

    def _path_time(self, t: Tensor) -> Tensor:
        return t.reshape(-1)

    def _sample_path(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        t_path = self._path_time(t).to(device=x_1.device, dtype=x_1.dtype)
        return self.path.sample(x_0=x_0, x_1=x_1, t=t_path), t_path

    def conditional_score(self, x_t: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)
        alpha_t = self._expand_time_like(scheduler_output.alpha_t, x_1)
        sigma_t = self._expand_time_like(scheduler_output.sigma_t, x_1)
        mean_t = alpha_t * x_1
        return -(x_t - mean_t) / sigma_t.clamp_min(1e-12).pow(2)


@register_target("bm_diffusion_conditional")
@register_target("cbm_diffusion")
class DiffusionConditionalBridgeMatchingTargets(_VPDiffusionBridgeMatchingTargets):
    """BM decomposition of the official VP/affine CFM diffusion target.

    The probability path and total velocity `v*` are exactly the official
    `AffineProbPath(VPScheduler)` target. BM only splits that target into
    `u* + d* = v*`, with `d* = beta * conditional_score`.
    """

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        path_sample, t_path = self._sample_path(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        v_star = path_sample.dx_t
        d_star = self.beta_value * self.conditional_score(
            x_t=x_t,
            x_1=x_1,
            t=t_path,
        )
        u_star = v_star - d_star
        return x_t, u_star, d_star

@register_target("bm_diffusion_marginal")
class DiffusionMarginalBridgeMatchingTargets(DiffusionConditionalBridgeMatchingTargets):
    """Diffusion marginal BM target.

    Uses the same official VP probability path as `bm_diffusion_conditional`:
        x_t = alpha_t x_1 + sigma_t x_0.

    The CFM-style conditional velocity is:
        v*_t = alpha'_t x_1 + (sigma'_t / sigma_t) (x_t - alpha_t x_1).

    For the osmotic component, instead of using the conditional Gaussian score
    around each individual x_1, we estimate the marginal score \nabla log pi_t(x_t)
    from the batch by KDE:
        d*_t = beta * \nabla log pi_t(x_t),
        u*_t = v*_t - d*_t.

    This makes the diffusion path marginal in the same spirit as
    `bm_linear_marginal`, while retaining the diffusion-style interpolation path.
    """

    def batch_kde_score(self, x_t: Tensor) -> Tensor:
        batch = x_t.size(0)
        flat_dim = x_t[0].numel()
        x_flat = x_t.view(batch, flat_dim)

        h = max(float(self.sigma_floor), 1e-4)
        dist2 = torch.cdist(x_flat, x_flat) ** 2

        logits = -dist2 / (2.0 * h * h)
        logits.fill_diagonal_(float("-inf"))
        weights = torch.softmax(logits, dim=1)

        weighted_x = weights @ x_flat
        score = (weighted_x - x_flat) / (h * h)
        return score.view_as(x_t)

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        path_sample, _ = self._sample_path(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        v_star = path_sample.dx_t
        score_star = self.batch_kde_score(x_t=x_t)
        d_star = self.beta_value * score_star
        u_star = v_star - d_star
        return x_t, u_star, d_star

@register_target("cfm_linear")
class CFMLinearBridgeMatchingTargets(BaseBridgeMatchingTargets):
    def _expand_time_like(self, t: Tensor, x: Tensor) -> Tensor:
        shape = [t.shape[0]] + [1] * (x.dim() - 1)
        return t.view(*shape)

    def sigma(self, t: Tensor) -> Tensor:
        # song.md Eq. (20): sigma_t = 1 - (1 - sigma_min) * t
        return 1.0 - (1.0 - self.sigma_floor) * t

    def sigma_dot(self, t: Tensor) -> Tensor:
        return torch.full_like(t, -(1.0 - self.sigma_floor))

    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t_view = self._expand_time_like(t, x_1)
        sigma_t = self._expand_time_like(self.sigma(t).clamp_min(self.sigma_floor), x_1)
        sigma_dot_t = self._expand_time_like(self.sigma_dot(t), x_1)

        m_t = t_view * x_1
        x_t = m_t + sigma_t * x_0
        # Match paper CFM linear/OT target directly (no BM decomposition term).
        v_star = (sigma_dot_t / sigma_t.clamp_min(1e-12)) * (x_t - m_t) + x_1
        d_star = torch.zeros_like(v_star)
        u_star = v_star
        return x_t, u_star, d_star


@register_target("cfm_diffusion")
class CFMDiffusionBridgeMatchingTargets(_VPDiffusionBridgeMatchingTargets):
    def compute(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        path_sample, _ = self._sample_path(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        v_star = path_sample.dx_t
        # Pure CFM target: use transport field only (u = v, d = 0).
        d_star = torch.zeros_like(v_star)
        u_star = v_star
        return x_t, u_star, d_star

def bridge_matching_targets(
    x_0: Tensor,
    x_1: Tensor,
    t: Tensor,
    beta_value: float,
    sigma_floor: float,
    target_type: str = "diffusion_marginal",
    posterior_temperature: float = 1.0,
    posterior_topk: int | None = None,
):
    bm = make_bridge_matching_targets(
        target_type=target_type,
        beta_value=beta_value,
        sigma_floor=sigma_floor,
        posterior_temperature=posterior_temperature,
        posterior_topk=posterior_topk,
    )
    return bm.compute(x_0=x_0, x_1=x_1, t=t)