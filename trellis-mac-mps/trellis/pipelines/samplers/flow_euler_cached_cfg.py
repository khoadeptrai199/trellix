from typing import *
from .flow_euler import FlowEulerSampler, FlowEulerGuidanceIntervalSampler

"""
Experimental: caches the classifier-free-guidance NEGATIVE (unconditional) branch prediction and
reuses it for `neg_cache_interval` consecutive steps instead of recomputing it every step, within
the guidance interval where GuidanceIntervalSamplerMixin already runs the model twice per step
(cond + neg_cond). Motivation: FasterCache (arXiv:2410.19355)-style CFG caching reports real,
training-free speedups (~1.67x on video diffusion) from the same observation -- the unconditional
branch isn't conditioned on the specific input, so its prediction changes more slowly across
adjacent diffusion timesteps than the conditional branch's does. Not a known-working technique
being reapplied here -- this is a new experiment for TRELLIS's specific combination of rectified
flow + guidance-interval CFG, not published anywhere as of this writing (checked: no distilled or
caching-augmented TRELLIS variant exists, see APPLE_SILICON.md's step-count section).

Subclasses FlowEulerGuidanceIntervalSampler directly (the real, production sampler class) rather
than editing it in place, per this project's own non-destructive versioning convention -- swap this
class in for `pipeline.sparse_structure_sampler`/`pipeline.slat_sampler` to test, leaving the
original class and its behavior completely untouched for anything that doesn't opt in.

`_inference_model` bypasses the GuidanceIntervalSamplerMixin/ClassifierFreeGuidanceSamplerMixin
super() chain entirely, calling FlowEulerSampler._inference_model (the real one-shot, single-model-
call implementation) directly -- avoids relying on cooperative multiple-inheritance super() dispatch
order for what's a fairly different control-flow needs (caching state across calls), at the cost of
duplicating the interval-check logic already in GuidanceIntervalSamplerMixin. Small, explicit
duplication judged safer here than fighting the MRO.

UPDATE: this naive version (freezing the raw negative-branch prediction) showed real, measured
quality degradation -- see APPLE_SILICON.md's CFG-caching section for the full result. Kept in
place, not deleted, per this port's versioning convention. See
`DiffCachedCfgFlowEulerGuidanceIntervalSampler` below for the fix (caches the prediction
DIFFERENCE, not the raw value), grounded in the actual mechanism real CFG-caching techniques
(FasterCache, AB-Cache) use.
"""


class CachedCfgFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, *args, neg_cache_interval: int = 2, **kwargs):
        """
        Args:
            neg_cache_interval: recompute the negative-branch prediction every N calls to
                _inference_model that fall inside cfg_interval; reuse the cached value on the
                N-1 calls in between. 1 = no caching (recompute every time, identical behavior to
                the base FlowEulerGuidanceIntervalSampler). Higher = more caching, more speed,
                more risk of quality drift.
        """
        super().__init__(*args, **kwargs)
        self.neg_cache_interval = neg_cache_interval

    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0,
               cfg_strength: float = 3.0, cfg_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True, **kwargs):
        # Reset cache state at the start of every sample() call -- sampler instances are reused
        # across pipeline.run() calls, and stale cache state from a previous sample must not leak
        # into the next one.
        self._neg_pred_cache = None
        self._interval_call_count = 0
        self.neg_recompute_count = 0  # exposed for the investigation script's speed accounting
        self.neg_reuse_count = 0
        return super().sample(model, noise, cond, neg_cond, steps, rescale_t, cfg_strength,
                               cfg_interval, verbose, **kwargs)

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
            if self._neg_pred_cache is None or self._interval_call_count % self.neg_cache_interval == 0:
                self._neg_pred_cache = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                self.neg_recompute_count += 1
            else:
                self.neg_reuse_count += 1
            self._interval_call_count += 1
            return (1 + cfg_strength) * pred - cfg_strength * self._neg_pred_cache
        else:
            return FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)


class DiffCachedCfgFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):
    """
    Fixes the real, measured quality degradation CachedCfgFlowEulerGuidanceIntervalSampler (above)
    showed -- see APPLE_SILICON.md's CFG-caching section for the full write-up. Root cause,
    confirmed against real sources (FasterCache arXiv:2410.19355, AB-Cache arXiv:2504.10540):
    freezing the raw negative-branch prediction is exactly the failure mode working CFG-caching
    techniques avoid. Neither caches the raw branch value verbatim; both cache and reuse
    DIFFERENCES/biases instead, and AB-Cache explicitly confirms that a larger `cfg_strength`
    amplifies the error from a stale cached value -- directly relevant here, since this pipeline's
    real config uses `cfg_strength=5.0`.

    Fix: cache `diff = pred - neg_pred` from the last full computation, and on a reuse step derive
    `neg_pred_estimated = pred_fresh - diff_cached` instead of reusing a frozen `neg_pred` directly.
    `pred` (the conditional branch) is always computed fresh every step regardless -- this costs no
    extra model call, it only changes what the reused call estimates. The claim this rests on
    (matching FasterCache/AB-Cache's own premise): the GAP between the conditional and
    unconditional predictions drifts more slowly than either branch alone, so an estimate that
    tracks the fresh conditional prediction is less stale than one that's frozen outright. Still a
    fixed reuse interval, not the fully adaptive error-gated trigger AB-Cache uses (that needs a
    cheap divergence proxy this port doesn't have yet) -- a real, smaller step, not the complete
    fix, and worth re-testing rather than assumed to fully resolve the drift.
    """
    def __init__(self, *args, neg_cache_interval: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_cache_interval = neg_cache_interval

    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0,
               cfg_strength: float = 3.0, cfg_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True, **kwargs):
        self._diff_cache = None
        self._interval_call_count = 0
        self.neg_recompute_count = 0
        self.neg_reuse_count = 0
        return super().sample(model, noise, cond, neg_cond, steps, rescale_t, cfg_strength,
                               cfg_interval, verbose, **kwargs)

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
            if self._diff_cache is None or self._interval_call_count % self.neg_cache_interval == 0:
                neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                self._diff_cache = pred - neg_pred
                self.neg_recompute_count += 1
            else:
                neg_pred = pred - self._diff_cache
                self.neg_reuse_count += 1
            self._interval_call_count += 1
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
