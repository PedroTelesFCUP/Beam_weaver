# !/usr/bin/env python3
import os
import sys
import time
import math
import cmath
import pickle
import random
import numpy as np
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cProfile, pstats, io
from math import ceil
import warnings
import collections  
from contextlib import nullcontext
from collections import deque, Counter
from collections import defaultdict
# Torch
import torch
import torch.nn as nn
from torch.nn import LayerNorm, SiLU
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Gym and stable_baselines3
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor
from stable_baselines3.sac.policies import SACPolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    Distribution, CategoricalDistribution, DiagGaussianDistribution
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor

try:                       
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

def _noop_add_to_buffer(*args, **kwargs):
    """Picklable replacement for replay_buffer.add during warm-up."""
    return None

###############################################################################
#                           GLOBAL CONSTANTS
###############################################################################
print("Torch CUDA available?", torch.cuda.is_available())


PHASE_ENDS = [5_000, 5_500, 100_500, 655_000, 2_880_000, 3_120_000, 3_470_000, 3_520_000]
PROC_NAMES = ["rayleigh", "compton", "photo", "pair"]   # 0,1,2,3
AVOGADRO = 6.02214076e23
O_ATOMIC_MASS = 16.0  # g/mol for oxygen-16
mec2 = 0.51099895069
r0 = 2.8179e-13 # cm 
HC_KEV_A = 12.3984    # Planck*c in keV·Å (for E(keV) -> wave number in 1/Å)

# Binding energies in eV for each oxygen shell:
PHOTO_SHELL_BINDINGS = {
    "H_K": 13.6,
    "O_K": 532.0,
    "O_L1": 40.0,
    "O_L2": 17.0,
    "O_L3": 17.0,
}
N_STEPS_RETURN = 25 
# ───── PHYSICS‑AWARE NORMALISERS ──────────────────────────────────────────────
_ANG_DENOM = math.pi                    # scale raw angular errors → [0,1]
_EPS_DEN   = 1e-12                      # avoid /0 throughout
def _safe_div(num, den):                # simple helper
    return num / (den + _EPS_DEN)


    
###############################################################################
#                CURRICULUM LEARNING PHASES
##############################################################################
phase_ends = PHASE_ENDS
def _unwrap_env(env):
    """Drill through any wrappers until we find your base Penelope env."""
    while hasattr(env, "env"):
        env = env.env
    return env

def _set_phase(model, phase: int) -> None:
    """
    Phase‑by‑phase summary
    ───────────────────────────────────────────────────────────
      phase 0 : train DISCRETE head only        |  MC‑forcing = ON
      phase 1 : train DISCRETE head only        |  MC‑forcing = fades OUT
      phase 2 : train CONTINUOUS‑mfp only       |  MC‑forcing = ON
      phase 3 : train CONTINUOUS‑mfp only       |  MC‑forcing = fades OUT
      phase 4+: full hybrid (all other cont.)   |  MC‑forcing = ON
                MFP output is FROZEN
    ───────────────────────────────────────────────────────────
      * The big feature‐extractor is **always** trainable.
    """

    def is_phys(name: str) -> bool:
        """Returns True for any of the physics heads/backbone."""
        return name.startswith(
            ("phys_backbone", "energy_head", "angle_head", "nsec_head", "proc_head")
        )
    # ------------------------------------------------------------------
    # Parameter freezing logic  (call *after* initialize_mu() has marked
    # the μ residual & its σ with  p._hard_frozen = True  and set
    # p.requires_grad = False for them).
    # ------------------------------------------------------------------
    n_train, n_frozen = 0, 0

    for name, p in model.policy.actor.named_parameters():

        # μ residual row + σ were tagged _hard_frozen inside initialize_mu()
        if getattr(p, "_always_frozen", False):
            p.requires_grad = False
            n_frozen += 1
            continue  # nothing else to decide for this param

        # ---------------------- curriculum switch ---------------------
        if phase < 2:  # PHASE 0–1 → train discrete only
            p.requires_grad = name.startswith(("features_extractor",
                                               "discrete_head"))
        else:          # PHASE ≥2  → freeze discrete, train the rest
            p.requires_grad = not name.startswith("discrete_head")

        # ---------------------- bookkeeping ---------------------------
        if p.requires_grad:
            n_train += 1
        else:
            n_frozen += 1

    print(f"🔧 Phase {phase}: {n_train} params trainable, {n_frozen} frozen.")

    # tiny flag on the model
    model.phase = phase
    if hasattr(model, "policy") and hasattr(model.policy, "actor"):
        model.policy.actor.phase = phase          # ➋ lets the actor know
        if phase == 0:
            print("🔄 Entering phase 0: Initializing logits and mus from physics")
            base_env = model.get_env().envs[0].unwrapped
            actor = model.policy.actor
            actor.initialize_all_logits_from_physics()
            actor.E_min = base_env.E_min        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            actor.E_max = base_env.E_max        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            actor.initialize_mu()
            actor.freeze_mu_residual()
            actor.freeze_gaussian_sigma(target_dim=-1, log_std_value=-4.5)
            actor.initialize_continuous_buffer()

        if phase == 2:
            print("🔄 Entering phase 2: Training kernel heads! Discrete head now frozen; μ remains frozen.")     
            # Sanity‐check: only the μ–residual row is frozen in continuous_head[-1]
            # (energy & angle means/log‐σ must still be trainable)
            print("👩‍⚕️🏥 Phase 2 sanity check: check if only the μ–residual is frozen; all other continuous_head params must be trainable")
            actor = model.policy.actor
            for name, param in actor.continuous_head[-1].named_parameters():
                if name.endswith(".weight"):
                    # Check the last output row (index -1)
                    row_count = param.shape[0]
                    # The last row itself was frozen via requires_grad_(False) on that slice
                    # so we check the view:
                    frozen = not param[row_count - 1].requires_grad
                    assert frozen, "❌ μ–mean row is NOT frozen!"
                    # And make sure all other rows are still trainable
                    for i in range(row_count - 1):
                        assert param[i].requires_grad, f"❌ Row {i} of weight was frozen by mistake!"
                if name.endswith(".bias"):
                    # The bias tensor is 1‐D of length = row_count
                    length = param.shape[0]
                    frozen = not param[length - 1].requires_grad
                    assert frozen, "❌ μ–bias row is NOT frozen!"
                    for i in range(length - 1):
                        assert param[i].requires_grad, f"Bias element {i} was frozen by mistake!"
            print("✅  Only μ–residual mean is frozen; all other continuous_head params are trainable.")
            # ─────────────────────────────────────────────────────────────

            
            # grab any of the wrapped envs – all share the same window
            base_env = model.get_env().envs[0].unwrapped
            actor = model.policy.actor

    # make sure every wrapped env knows the new phase
    if model.get_env() is not None:
        for venv in model.get_env().envs:
            base = _unwrap_env(venv)
            base.phase = phase

    # Special message when entering phase 4 (freezing MFP)
    if phase == 4:
        print("🔒 MFP output frozen - continuing training for angle and energy parameters")

    print(f"\n🔄  PHASE {phase} ACTIVATED")
##############################################################################
#                 PEEK AT REPLAY BUFFER 
##############################################################################
def print_last_transitions(model, last_k=10):
    rb    = model.replay_buffer
    size  = rb.buffer_size if getattr(rb, "full", False) else rb.pos
    shown = 0
    print("\n--- last live (done=False) transitions ---")
    for ofs in range(1, size+1):
        idx = (rb.pos - ofs) % rb.buffer_size
        if not rb.dones[idx, 0]:          # keep only live steps
            r = float(rb.rewards[idx, 0])
            print(f"idx={idx:5d}  r={r:+.4f}")
            shown += 1
            if shown == last_k:
                break

class BufferPeekCallback(BaseCallback):
    """
    Every `every` env steps compute stats **only on the last `tail`**
    transitions that are ALREADY in the replay buffer.  Useful when you
    care about what has just been written, not the older experience.

    • Logs to TensorBoard:
        debug/tail_r_mean, _min, _max
    • Prints to stdout if verbose > 0
    """
    def __init__(self, every: int = 1000, tail: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.every, self.tail = every, tail

    def _on_step(self) -> bool:
        if self.n_calls % self.every:        # run only every `every` calls
            return True

        rb   = self.model.replay_buffer
        full = getattr(rb, "full", False)
        size = rb.buffer_size if full else rb.pos
        if size == 0:
            return True                      # buffer still empty

        # ---- indices of the last `tail` inserts ----------------------
        k = min(self.tail, size)
        # rb.pos is the *next* write position, so the newest transition
        # is at (pos - 1) modulo buffer_size
        last_idx = (rb.pos - 1) % rb.buffer_size
        idx = (last_idx - np.arange(k)) % rb.buffer_size

        rewards = rb.rewards[idx, 0].squeeze()
        r_mean  = float(rewards.mean())
        r_min   = float(rewards.min())
        r_max   = float(rewards.max())

        # ---- TensorBoard --------------------------------------------
        self.logger.record("debug/tail_r_mean", r_mean)
        self.logger.record("debug/tail_r_min",  r_min)
        self.logger.record("debug/tail_r_max",  r_max)

        # ---- optional console print ---------------------------------
        if self.verbose:
            print(f"[tail @{self.num_timesteps:>7}] "
                  f"mean={r_mean:+.4f}  min={r_min:+.4f}  max={r_max:+.4f}")
            print("   last rewards:", rewards.tolist())
        return True
##############################################################################
# n‑STEP REPLAY BUFFER  (back‑port of SB3 ≥ 2.7 nightly)
##############################################################################
class NStepReplayBuffer(ReplayBuffer):
    """
    Simple n‑step extension of the vanilla circular replay buffer.
    • Supports 1 environment; off‑policy algorithms only.
    • Stores (obs, action, n‑step return, γⁿ, next_obs, done_n)
    • sample() returns a namedtuple with .indices and .discounts
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        n_steps: int = 3,
        gamma: float = 0.995,
        device: torch.device | str = "cpu",
        optimize_memory_usage: bool = False,
        **kwargs,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            optimize_memory_usage=optimize_memory_usage,     
            **kwargs,
        )
        self.infos: list[dict] = []
        # how many steps to bootstrap over
        self.n_steps = max(1, n_steps)
        self.gamma   = gamma
        # small ring to accumulate the last n_steps (r, next_obs, done)

        self.discounts = np.zeros(buffer_size, dtype=np.float32)
        self.n_step_transitions = collections.deque(maxlen=self.n_steps)
        self.episode_transitions = []
        # for each slot in the main buffer, store the γⁿ used when that slot was written



    def add(self, obs, next_obs, action, reward, done, infos=None):
        # Guard for info dict--->list?
        if isinstance(infos, (list, tuple)):
            # 1-env DummyVecEnv ⇒ infos == [dict]
            if len(infos) and isinstance(infos[0], dict):
                infos = infos[0]
            else:                       # something odd – fall back to empty dict
                infos = {}
        # Append the current transition to the episode buffer
        self.episode_transitions.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "infos": infos or {}
        })

        # Process n-step transitions when episode ends
        if done:
            episode_length = len(self.episode_transitions)
            for start_idx in range(episode_length):
                # Calculate the maximum possible steps from this start_idx
                max_steps = min(self.n_steps, episode_length - start_idx)
                
                total_reward = 0.0
                gamma_pow = 1.0
                last_idx = start_idx
                for step in range(max_steps):
                    trans = self.episode_transitions[start_idx + step]
                    total_reward += trans["reward"] * gamma_pow
                    gamma_pow *= self.gamma
                    last_idx = start_idx + step
                    if trans["done"]:
                        break  # Stop at terminal state

                # Add the n-step transition to the buffer
                first_trans = self.episode_transitions[start_idx]
                last_trans = self.episode_transitions[last_idx]
                info_dict = first_trans["infos"] or {}
                action_to_write = info_dict.get(
                    "override_action",          # present when we forced MC
                    first_trans["action"]       # otherwise the agent’s own
                )
                wrapped_infos = [info_dict] 
                super().add(
                    obs      = first_trans["obs"],
                    next_obs = last_trans["next_obs"],
                    action   = action_to_write,   # <── NEW (one-liner change)
                    reward   = total_reward,
                    done     = last_trans["done"],
                    infos    = wrapped_infos,
                )

                # Store discount factor for this transition
                write_idx = (self.pos - 1) % self.buffer_size
                if write_idx < len(self.infos):
                    self.infos[write_idx] = info_dict
                else:
                    self.infos.append(info_dict)
                    if len(self.infos) > self.buffer_size: 
                        self.infos.pop(0)
                self.discounts[write_idx] = self.gamma ** (last_idx - start_idx + 1)

            # Reset episode buffer after processing
            self.episode_transitions = []
#            print(f"Buffer size: {self.pos}/{self.buffer_size}")

    def sample(self, batch_size: int, env: None = None):
        """
        Returns a namedtuple with fields:
          observations, actions, next_observations, dones, rewards,
          indices, discounts
        """
        if (self.full and self.buffer_size == 0) or (not self.full and self.pos == 0):
            raise ValueError("Trying to sample from empty replay buffer")
        # pick random indices uniformly from [0 .. current_size)
        max_idx = self.buffer_size if self.full else self.pos
        batch_inds = np.random.choice(max_idx, size=batch_size, replace=False)

        infos_batch = [self.infos[idx] if idx < len(self.infos) else {}
                        for idx in batch_inds]
        # gather each array
        obs_batch       = self.observations[batch_inds, 0]
        actions_batch   = self.actions[batch_inds, 0]
        next_obs_batch  = self.next_observations[batch_inds, 0]
        dones_batch     = self.dones[batch_inds, 0]
        rewards_batch   = self.rewards[batch_inds, 0]
        discounts_batch = self.discounts[batch_inds]

        # package into a custom namedtuple

        ReplayBufferNStepSamples = namedtuple(
            "ReplayBufferNStepSamples",
            ["observations","actions","next_observations","dones","rewards","indices","infos","discounts","n_steps"],
        )

        # convert everything to torch.Tensor on the right device
        obs_t     = torch.as_tensor(obs_batch,      dtype=torch.float32, device=self.device)
        acts_t    = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        next_obs_t= torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        # dones need to be floats so that (1 - dones) works
        dones_t   = torch.as_tensor(
                        dones_batch.astype(np.float32),
                        dtype=torch.float32,
                        device=self.device
                    ).view(-1,1)
        rews_t    = torch.as_tensor(
                        rewards_batch,
                        dtype=torch.float32,
                        device=self.device
                    ).view(-1,1)
        
        # if you need discounts for any custom loss you can keep them as well
        disc_t    = torch.as_tensor(discounts_batch,
                                    dtype=torch.float32, device=self.device).view(-1, 1)

        return ReplayBufferNStepSamples(
            obs_t, acts_t, next_obs_t, dones_t, rews_t, batch_inds, infos_batch, disc_t, 
            torch.full((batch_size,1), self.n_steps, dtype=torch.long, device=self.device)
        )

###############################################################################
#                        WATER PHOTO SHELL DATA
###############################################################################

class WaterPhotoShellData:
    def __init__(self, csv_path="WaterPhotoShells.csv"):
        df = pd.read_csv(csv_path)
        self.Egrid = df["E_MeV"].values
        self.HKvals  = df["H_K_cm2g"].values
        self.OKvals  = df["O_K_cm2g"].values
        self.OL1vals = df["O_L1_cm2g"].values
        self.OL2vals = df["O_L2_cm2g"].values
        self.OL3vals = df["O_L3_cm2g"].values

        if not all(self.Egrid[i] <= self.Egrid[i+1] for i in range(len(self.Egrid)-1)):
            sort_idx = self.Egrid.argsort()
            self.Egrid  = self.Egrid[sort_idx]
            self.HKvals = self.HKvals[sort_idx]
            self.OKvals = self.OKvals[sort_idx]
            self.OL1vals = self.OL1vals[sort_idx]
            self.OL2vals = self.OL2vals[sort_idx]
            self.OL3vals = self.OL3vals[sort_idx]

    def _loglog_interp(self, E, grid, vals):
        if E <= grid[0]:
            return vals[0]
        if E >= grid[-1]:
            return vals[-1]
        left = 0
        right = len(grid) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if grid[mid] > E:
                right = mid
            else:
                left = mid
        x1 = grid[left]
        x2 = grid[right]
        y1 = vals[left]
        y2 = vals[right]
        if y1 <= 0 or y2 <= 0:
            return 0.0
        lx1 = math.log(x1)
        lx2 = math.log(x2)
        ly1 = math.log(y1)
        ly2 = math.log(y2)
        frac = (math.log(E) - lx1) / (lx2 - lx1)
        return math.exp(ly1 + frac * (ly2 - ly1))

    def pick_shell(self, E):
        HK  = self._loglog_interp(E, self.Egrid, self.HKvals)
        OK  = self._loglog_interp(E, self.Egrid, self.OKvals)
        OL1 = self._loglog_interp(E, self.Egrid, self.OL1vals)
        OL2 = self._loglog_interp(E, self.Egrid, self.OL2vals)
        OL3 = self._loglog_interp(E, self.Egrid, self.OL3vals)

        total = HK + OK + OL1 + OL2 + OL3
        if total < 1e-30:
            return (None, 0.0)

        r = random.random() * total
        if r < HK:
            return ("H_K", HK)
        r -= HK
        if r < OK:
            return ("O_K", OK)
        r -= OK
        if r < OL1:
            return ("O_L1", OL1)
        r -= OL1
        if r < OL2:
            return ("O_L2", OL2)
        return ("O_L3", OL3)
# Deprecated it only accounted for oxygen
#class OxygenPhotoShellData:
#    def __init__(self, csv_path="OxygenPhotoShells.csv"):
#        df = pd.read_csv(csv_path)
#        self.Egrid = df["E_MeV"].values
#        self.Kvals = df["K_cm2g"].values
#        self.L1vals = df["L1_cm2g"].values
#        self.L2vals = df["L2_cm2g"].values
#        self.L3vals = df["L3_cm2g"].values
#        
#        # Ensure ascending order
#        if not all(self.Egrid[i] <= self.Egrid[i+1] for i in range(len(self.Egrid)-1)):
#            sort_idx = self.Egrid.argsort()
#            self.Egrid = self.Egrid[sort_idx]
#            self.Kvals = self.Kvals[sort_idx]
#            self.L1vals = self.L1vals[sort_idx]
#            self.L2vals = self.L2vals[sort_idx]
#            self.L3vals = self.L3vals[sort_idx]
#
#    def _loglog_interp(self, E, grid, vals):
#        if E <= grid[0]:
#            return vals[0]
#        if E >= grid[-1]:
#            return vals[-1]
#        left= 0
#        right= len(grid)-1
#        while right-left>1:
#            mid= (left+right)//2
#            if grid[mid]> E:
#                right= mid
#            else:
#                left= mid
#        x1= grid[left]
#        x2= grid[right]
#        y1= vals[left]
#        y2= vals[right]
#        if y1<=0 or y2<=0:
#            return 0.0
#        logE= math.log(E)
#        lx1= math.log(x1)
#        lx2= math.log(x2)
#        ly1= math.log(y1)
#        ly2= math.log(y2)
#        frac= (logE - lx1)/(lx2 - lx1)
#        return math.exp(ly1 + frac*(ly2-ly1))
        
#    def pick_shell(self, E):
#        Kval  = self._loglog_interp(E, self.Egrid, self.Kvals)
#        L1val = self._loglog_interp(E, self.Egrid, self.L1vals)
#        L2val = self._loglog_interp(E, self.Egrid, self.L2vals)
#        L3val = self._loglog_interp(E, self.Egrid, self.L3vals)
#        total = Kval + L1val + L2val + L3val
#        if total < 1e-30:
#            return (None, 0.0)
#        r = random.random()*total
#        if r < Kval: return ("K", Kval)
#        r -= Kval
#        if r < L1val: return ("L1", L1val)
#        r -= L1val
#        if r < L2val: return ("L2", L2val)
#        return ("L3", L3val)
    


###############################################################################
#                           COMPTON SAMPLER
###############################################################################
def _kn_dcs(T):
    return (math.pi*r0**2/mec2)*(
        (mec2/(mec2 + T))**2 * (mec2/(mec2 + T) + T/(mec2 + T) - (T/(mec2 + T))**2)
    )

# Planck*c in keV·Å (for E(keV) -> wave number in Å^-1):
HC_MEV_A = 12.3984 / 1000  # MeV·Å (converted from keV·Å)

class ComptonSampler:
    def __init__(self, E_mev, sq_csv="water_sq.csv"):
        self.E = E_mev
        self.alpha = E_mev / mec2
        self.tau = 2.0 * self.alpha
        self.eps0 = 1.0 / (1.0 + self.tau)
        self.eps0_2 = self.eps0 ** 2
        self.a1 = math.log(1.0 + self.tau)  # <-- Added
        self.a2 = self.tau - self.a1        # <-- Added
        # Load S(q) table (q = sin(theta/2)/lambda * 2)
        data = np.genfromtxt(sq_csv, delimiter=',', names=True)
        self.q_grid = np.sort(data['q'])
        self.S_grid = data['S_q'][np.argsort(data['q'])]
        self.S_max = 10.0  # For water (Z=10)

    def compute_q(self, eps, cost):
        """Calculate momentum transfer q (Å⁻¹) using relativistic formula."""
        E_out = self.E * eps
        return np.sqrt(
            self.E**2 + E_out**2 - 2 * self.E * E_out * cost
        ) / HC_MEV_A  # Correct unit conversion

    def sample(self):
        """Sample energy transfer with PENELOPE's rejection function."""
        for _ in range(1000):  # Max attempts
            # 1. Generate eps using PENELOPE's composition method
            if random.random() < self.a1/(self.a1 + self.a2):
                eps = math.exp(random.random() * self.a1) * self.eps0
            else:
                eps = math.sqrt(self.eps0_2 + random.random()*(1 - self.eps0_2))
            
            # 2. Compute scattering angle variables
            cost = 1.0 - (1.0 - eps) / (eps * self.alpha)
            cost = np.clip(cost, -1.0, 1.0)
            theta = math.acos(cost)
            
            # 3. Compute momentum transfer and table x
            q = self.compute_q(eps, cost)
            x_table = q / 2  # Convert to table's x = sin(theta/2)/lambda
            
            # 4. Interpolate S(q)
            S_q = np.interp(
                x_table, 
                self.q_grid, 
                self.S_grid, 
                left=0.0, 
                right=self.S_max
            )
            
            # 5. Compute PENELOPE's rejection function T(cosθ)
            kappa = self.alpha  # E/mec²
            tau_val = 1.0 / (1.0 + kappa * (1.0 - cost))
            numerator = (
                1.0 
                + (kappa**2 - 2*kappa - 2)*tau_val 
                + (2*kappa + 1)*tau_val**2 
                + kappa**2 * tau_val**3
            )
            denominator = (1.0 + 2*kappa)**2
            T_penelope = numerator / denominator
            
            # 6. Combined acceptance probability
            accept_prob_compt = (S_q / self.S_max) * T_penelope
            if random.random() <= accept_prob_compt:
                return (self.E*(1 - eps), eps)  # (T_meV, eps)
        
        # Fallback after max attempts
        return (self.E*(1 - self.eps0), self.eps0)
        
###############################################################################
#                PENELOPE-LIKE WATER DATA
###############################################################################
class PenelopeLikeWaterData:
    def __init__(
        self,
        final_csv_path: str,
        rayleigh_csv_path: str,
        density=1.0,
        water_shell_csv="WaterPhotoShells.csv",
        coherent_ff_csv="water_fq.csv",
    ):
        # Load Rayleigh (coherent) cross-sections
        df_rayleigh = pd.read_csv(rayleigh_csv_path)
        self.E_coh = df_rayleigh["E"].values
        self.sigma_coh = df_rayleigh["coh"].values
        # Sort Rayleigh data
        sort_idx_coh = np.argsort(self.E_coh)
        self.E_coh = self.E_coh[sort_idx_coh]
        self.sigma_coh = self.sigma_coh[sort_idx_coh]
        self.log_E_coh = np.log(self.E_coh)
        self.log_sigma_coh = np.log(self.sigma_coh + 1e-12)

        # Load Final cross-sections (photoelectric, Compton, pair production)
        df_final = pd.read_csv(final_csv_path)
        self.E_final = df_final["E"].values
        self.sigma_pho = df_final["photoelectric"].values
        self.sigma_inc = df_final["compton"].values
        self.sigma_ppr = df_final["pair_triplet"].values
        # Sort Final data
        sort_idx_final = np.argsort(self.E_final)
        self.E_final = self.E_final[sort_idx_final]
        self.sigma_pho = self.sigma_pho[sort_idx_final]
        self.sigma_inc = self.sigma_inc[sort_idx_final]
        self.sigma_ppr = self.sigma_ppr[sort_idx_final]
        self.log_E_final = np.log(self.E_final)
        self.log_sigma_pho = np.log(self.sigma_pho + 1e-12)
        self.log_sigma_inc = np.log(self.sigma_inc + 1e-12)
        self.log_sigma_ppr = np.log(self.sigma_ppr + 1e-12)

        self.density = density
        self.water_shell_data = WaterPhotoShellData(water_shell_csv)
        # Load tabulated coherent form factor for water.
        # NOTE: the CSV column name is 'q', but the stored axis is actually
        # Hubbell's x = sin(theta/2)/lambda in Å^-1.
        ff_data = np.genfromtxt(coherent_ff_csv, delimiter=",", names=True)
        self.ff_x = np.asarray(ff_data["q"], dtype=float)
        self.ff_F = np.asarray(ff_data["F_q"], dtype=float)

        sort_idx_ff = np.argsort(self.ff_x)
        self.ff_x = self.ff_x[sort_idx_ff]
        self.ff_F = self.ff_F[sort_idx_ff]
        
        self.HC_KEV_A = 12.3984
        self.F0 = self.coherent_form_factor(0.0)
    
    def partial_cs(self, E):
        c = self.loglog_interp(E, self.E_coh, self.sigma_coh) * self.density
        i = self.loglog_interp(E, self.E_final, self.sigma_inc) * self.density
        p = self.loglog_interp(E, self.E_final, self.sigma_pho) * self.density
        r = self.loglog_interp(E, self.E_final, self.sigma_ppr) * self.density
        total = c + i + p + r
        return (c, i, p, r, total)

    def mu_total(self, E):
        (coh,inc,pho,ppr,tot)= self.partial_cs(E)
        return tot
        
    def partial_cs_vectorized(self, E):
        E = np.asarray(E)
        # Rayleigh (coh) interpolation
        logE_coh = np.log(np.clip(E, self.E_coh[0], self.E_coh[-1]))
        coh = np.exp(np.interp(logE_coh, self.log_E_coh, self.log_sigma_coh)) * self.density
        # Photoelectric, Compton, Pair from Final CSV
        logE_final = np.log(np.clip(E, self.E_final[0], self.E_final[-1]))
        pho = np.exp(np.interp(logE_final, self.log_E_final, self.log_sigma_pho)) * self.density
        inc = np.exp(np.interp(logE_final, self.log_E_final, self.log_sigma_inc)) * self.density
        ppr = np.exp(np.interp(logE_final, self.log_E_final, self.log_sigma_ppr)) * self.density
        return (coh, inc, pho, ppr)

    def mu_total_vectorized(self, E):
        coh, inc, pho, ppr = self.partial_cs_vectorized(E)
        return coh + inc + pho + ppr
        
    def pick_photo_shell(self, E):
        name,_ = self.water_shell_data.pick_shell(E)
        if name is None: 
            return None
        mapping = {"H_K": 0, "O_K": 1, "O_L1": 2, "O_L2": 3, "O_L3": 4}
        return mapping[name]
    def coherent_form_factor(self, q_ang_inv):
        """
        Interpolate the tabulated water coherent form factor.
    
        q_ang_inv : full momentum transfer in Å^-1 used in the Rayleigh sampler.
        The Hubbell table is tabulated in x = sin(theta/2)/lambda, so:
            x = q / 2
        """
        x_table = 0.5 * float(q_ang_inv)
        return float(
            np.interp(
                x_table,
                self.ff_x,
                self.ff_F,
                left=self.ff_F[0],
                right=self.ff_F[-1],
            )
        )
### DEPRECATED
#    def iaea_form_factor(self, q):
#        # 3-term Gaussian approximation
#        a=[0.4899,0.2626,0.2254]
#        b=[1.4752,4.1567,15.8047]
#        s=0.0
#        for ai,bi in zip(a,b):
#            s += ai * math.exp(-bi*(q/(4*math.pi))**2)
#        return s

    def loglog_interp(self, E, grid, vals):
        if E<=grid[0]: return vals[0]
        if E>=grid[-1]: return vals[-1]
        left=0; right=len(grid)-1
        while right-left>1:
            mid=(left+right)//2
            if grid[mid]>E:
                right=mid
            else:
                left=mid
        x1=grid[left]; x2=grid[right]
        y1=vals[left]; y2=vals[right]
        if y1<=0 or y2<=0:
            return 0.0
        lE=math.log(E); lx1=math.log(x1); lx2=math.log(x2)
        ly1=math.log(y1); ly2=math.log(y2)
        frac=(lE - lx1)/(lx2-lx1)
        return math.exp(ly1 + frac*(ly2-ly1))

###############################################################################
#                PHOTON INTERACTION HELPERS
###############################################################################
def rotate_direction(old_dir, theta, phi):
    """
    Rotate a direction vector by theta and phi angles.
    
    Args:
        old_dir: Original direction vector (3-element iterable)
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        
    Returns:
        New direction vector (numpy array) after rotation
    """
    (u, v, w) = old_dir
    mag = math.sqrt(u*u + v*v + w*w)
    if mag < 1e-14:
        return np.array([
            math.sin(theta)*math.cos(phi),
            math.sin(theta)*math.sin(phi),
            math.cos(theta)
        ])
    
    ux = u / mag
    uy = v / mag
    uz = w / mag
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)
    
    if abs(uz) < 0.9999:
        denom = math.sqrt(1.0 - uz*uz)
        u2 = ux*ct + ((ux*uz*cp - uy*sp) / denom) * st
        v2 = uy*ct + ((uy*uz*cp + ux*sp) / denom) * st
        w2 = uz*ct - denom*cp*st
    else:
        u2 = st*cp
        v2 = st*sp
        w2 = ct if uz > 0 else -ct
    
    return np.array([u2, v2, w2])

def sample_rayleigh(E, old_dir, data):
    shell_onehot = [0, 0, 0, 0, 0]  # No shell activation for Rayleigh
    EkeV = E * 1e3  # Convert MeV to keV
    k = EkeV / data.HC_KEV_A  # Wave number in Å⁻¹
    F0 = data.coherent_form_factor(0.0)
    
    # PENELOPE-style screening angle parameters
    a = 0.025  # Screening parameter (adjusted for water)
    max_iter = 100
    
    for _ in range(max_iter):
        # Sample from modified angular distribution
        xi1, xi2 = random.random(), random.random()
        cost = 1 - (2 * a * xi1) / (1 + a - xi1)
        cost = np.clip(cost, -1.0, 1.0)
        theta = math.acos(cost)
        
        # Calculate momentum transfer
        q = 2 * k * math.sin(theta/2)  # in Å⁻¹
        
        # Get form factor
        Fq = data.coherent_form_factor(q)
        ratio = (Fq / F0) ** 2
        
        # Rejection sampling with modified angular dependence
        if xi2 <= ratio * (1 + cost**2)/2:
            phi = 2 * math.pi * random.random()
            new_dir = rotate_direction(old_dir, theta, phi)
            return (new_dir, E, [], "rayleigh", shell_onehot)
    
    # Fallback to small-angle scattering
    theta = math.acos(0.999)
    phi = 2 * math.pi * random.random()
    new_dir = rotate_direction(old_dir, theta, phi)
    return (new_dir, E, [], "rayleigh", shell_onehot)

def sample_compton(E, old_dir, data=None):
    """
    Sample a Compton scattering event using PENELOPE methodology.
    
    Args:
        E: Photon energy in keV
        old_dir: Incident photon direction (3D vector)
        data: Additional data for physics calculations (optional)
        
    Returns:
        Tuple of (new_dir, Eout, secondaries, process_name, shell_info)
    """
    # Create sampler and sample energy transfer
    comp = ComptonSampler(E)
    T, eps = comp.sample()
    
    # CRITICAL: Use the correct Compton formula for scattering angle
    alpha = E / mec2
    cost = 1.0 - (1.0 - eps) / (eps * alpha)
    
    # Clamp cosine to valid range (-1 to 1)
    cost_clamped = max(-1.0, min(1.0, cost))
    
    # Sample azimuthal angle (uniform in 0 to 2π)
    phi = 2.0 * math.pi * random.random()
    
    # Calculate polar angle
    theta = math.acos(cost_clamped)
    
    # Sanity check for numerical issues
    if not np.isfinite(theta):
        print(f"Warning: Non-finite theta. cost={cost}, eps={eps}, alpha={alpha}")
        theta = 0.0  # Safe default
    
    # Compute new photon direction
    new_dir = rotate_direction(old_dir, theta, phi)
    
    # Outgoing photon energy
    Eout = E * eps  # or equivalently: E - T
    
    # Calculate electron recoil direction using momentum conservation
    electron_mom = np.array(old_dir) * E - new_dir * Eout
    norm = np.linalg.norm(electron_mom)
    
    if norm < 1e-14:
        edir = np.array([0, 0, 0])
    else:
        edir = electron_mom / norm
    
    # Create secondaries list with electron information
    secs = [("electron", T, edir, "compton_e")]
    
    # Default shell onehot (no photoelectric event)
    shell_onehot = [0, 0, 0, 0, 0]
    
    return (new_dir, Eout, secs, "compton", shell_onehot)


def sample_dipole_direction():
    beta=2.0
    bound=1+beta
    while True:
        cost=2*random.random()-1
        pdf=1+ beta*cost*cost
        if random.random()*bound<pdf:
            phi= 2*math.pi*random.random()
            st= math.sqrt(max(0,1-cost*cost))
            return np.array([st*math.cos(phi), st*math.sin(phi), cost])

def sample_photoelectric(E, old_dir, data):
    idx = data.pick_photo_shell(E)
    if idx is None:
        shell_onehot = [0, 0, 0, 0, 0]
        return (old_dir, E, [], "photo_none", shell_onehot)
    shellNames = ["H_K", "O_K", "O_L1", "O_L2", "O_L3"]
    shellName = shellNames[idx]
    shell_onehot = [0, 0, 0, 0, 0]
    shell_onehot[idx] = 1
    Eb_eV = PHOTO_SHELL_BINDINGS[shellName]
    Eb_MeV = Eb_eV * 1e-6
    E_e = E - Eb_MeV
    if E_e <= 0:
        return (old_dir, E, [], "photo_failed", shell_onehot)
    
    secs = []
    # Determine angular distribution based on shell
    if shellName in ("H_K", "O_K"):
        # Sauter distribution for K-shell
        T = E_e / mec2  # Kinetic energy in units of mec²
        beta = math.sqrt(T * (T + 2)) / (T + 1)
        while True:
            xi1 = random.random()
            xi2 = random.random()
            # Compute proposed cos(theta)
            numerator = xi1 * (2 + beta) - 1
            denominator = beta * (1 - xi1) + 1
            cos_theta = numerator / denominator
            # Handle potential numerical inaccuracies
            if cos_theta < -1 or cos_theta > 1:
                continue  # Reject invalid angles
            # Compute acceptance probability
            g = (1 - beta * cos_theta) / 2
            if xi2 <= g:
                break
        theta = math.acos(cos_theta)
    else:
        # Isotropic distribution for L-shells
        cos_theta = 2 * random.random() - 1
        theta = math.acos(cos_theta)
    
    phi = 2 * math.pi * random.random()
    e_dir = rotate_direction(old_dir, theta, phi)
    secs.append(("electron", E_e, e_dir, f"photo_{shellName}"))
    
    return (np.array([0.0, 0.0, 0.0]), 0.0, secs, "photo", shell_onehot)

def sample_pair(E, old_dir, data):
    if E<2*mec2:
        return (old_dir,E,[],"pair_subthresh")
    E_avail= E-2*mec2
    def sample_epsilon(Ea):
        max_val=(Ea/2)**2 + (2/3)*(mec2)**2
        while True:
            ep= random.uniform(0,Ea)
            f= ep*(Ea-ep)+(2/3)*mec2**2
            if random.random()<= (f/max_val):
                return ep
    eps_e= sample_epsilon(E_avail)
    eps_p= E_avail - eps_e
    E_e_total= eps_e+mec2
    E_p_total= eps_p+mec2
    def sample_theta(Etot):
        xi= random.random()
        if xi<1e-12: xi=1e-12
        sqrt_term= math.sqrt(-2*math.log(xi))
        return math.sqrt(mec2/Etot)* sqrt_term
    theta_e= sample_theta(E_e_total)
    theta_p= sample_theta(E_p_total)
    phi_e= 2*math.pi*random.random()
    phi_p= phi_e+math.pi
    dir_e= rotate_direction(old_dir,theta_e,phi_e)
    dir_p= rotate_direction(old_dir,theta_p,phi_p)
    secs=[("electron", eps_e, dir_e,"pair_e"),
          ("positron", eps_p, dir_p,"pair_p")]
    # Default shell onehot (no photoelectric event)
    shell_onehot = [0,0,0,0]
    return (np.array([0,0,0]), 0.0, secs, "pair", shell_onehot)

def photon_interact(E, direction, data:PenelopeLikeWaterData):
    (coh,inc,pho,ppr,tot)= data.partial_cs(E)
    if tot<1e-30:
        return (direction,E,[],"none", [0,0,0,0])
    r= random.random()*tot
    if r<coh:
        return sample_rayleigh(E,direction,data)
    r-=coh
    if r<inc:
        return sample_compton(E,direction,data)
    r-=inc
    if r<pho:
        return sample_photoelectric(E,direction,data)
    return sample_pair(E,direction,data)

###############################################################################
#            ELECTRON TRANSPORT
###############################################################################
def load_stopping_power(csv_path="ElectronStoppingPower.csv"):
    """Load both collisional and radiative stopping powers"""
    df = pd.read_csv(csv_path)
    return (
        df["E_MeV"].values,          # Energy grid (MeV)
        df["S_col_MeV_per_cm"].values,   # Collisional stopping power
        df["S_rad_MeV_per_cm"].values    # Radiative stopping power
    )

def stopping_power(E, Egrid, S_col_vals, S_rad_vals):
    """Return interpolated collisional and radiative stopping powers"""
    # Linear interpolation in log-log space (common for stopping powers)
    log_E = np.log(np.clip(E, 1e-6, None))
    log_S_col = np.interp(log_E, np.log(Egrid), np.log(S_col_vals))
    log_S_rad = np.interp(log_E, np.log(Egrid), np.log(S_rad_vals))
    return np.exp(log_S_col), np.exp(log_S_rad)
    
def transport_electron_csda(E_electron, direction, start_pos, dose_tally, env, n_steps=5, Egrid=None,  S_col_vals=None, S_rad_vals=None):
    """
    Transport electron with CSDA approximation including bremsstrahlung
    
    Returns:
        tuple: (total_deposited_energy, secondary_photons)
    """
    pos = np.array(start_pos)
    remaining_energy = E_electron
    total_deposit = 0.0
    secondaries = []
    
    while remaining_energy > 0.001:  # 1 keV cutoff
        # Get stopping powers
        S_col, S_rad = stopping_power(remaining_energy, Egrid, S_col_vals, S_rad_vals)
        S_total = S_col + S_rad
        
        if S_total <= 0:
            break
            
        # Adaptive step size (ensure at least 0.1 keV energy loss)
        step = max(0.001, 0.0001/S_total)  # 0.001 cm minimum step
        
        # Calculate energy losses
        dE_total = S_total * step
        dE_col = S_col * step
        dE_rad = S_rad * step
        
        # Update position and tally
        pos += step * direction
        k = int((pos[2] - env.zmin) / env.dz)
        if 0 <= k < env.pdd_bins:
            dose_tally[k] += dE_col  # Only collisional loss deposits locally
            
        # Handle bremsstrahlung production
        if dE_rad > 1e-6:  # 1 eV threshold
            # Sample photon energy from simplified spectrum
            photon_energy = sample_brem_energy(dE_rad)
            
            # Sample direction (forward-peaked approximation)
            theta = math.acos(1 - random.random()**0.5)  # 1-cosθ ~ uniform
            phi = 2 * math.pi * random.random()
            photon_dir = rotate_direction(direction, theta, phi)
            
            secondaries.append(("photon", photon_energy, photon_dir, "brem"))
        
        remaining_energy -= dE_total
        total_deposit += dE_col
    
    return total_deposit, secondaries

def sample_brem_energy(dE_rad):
    """Sample photon energy from normalized bremsstrahlung spectrum"""
    # Simplified Kramer spectrum approximation
    while True:
        u = random.random()
        eps = u**0.5  # dN/dE ~ 1/E
        if random.random() < eps:  # Rejection sampling
            return dE_rad * eps

    pos = np.array(start_pos)
    remaining_energy = E_electron
    total_deposit = 0.0
    
    while remaining_energy > 0.001:  # 1 keV cutoff
        S = stopping_power(remaining_energy, Egrid, S_vals)
        if S <= 0:
            break
            
        # Adaptive step size
        step = max(0.001, remaining_energy / S)  # At least 0.001 cm
        dE = S * step
        
        # Update position and tally
        pos += step * direction
        k = int((pos[2] - env.zmin) / env.dz)
        if 0 <= k < env.pdd_bins:
            dose_tally[k] += dE
            
        remaining_energy -= dE
        total_deposit += dE
    return total_deposit

def transport_electron_condensed_history(
    E_electron, 
    direction, 
    start_pos, 
    dose_tally, 
    env, 
    Egrid=None, 
    S_vals=None,
    ecut=0.001,       # MeV (1 keV) cutoff
    max_steps=2000,   # safety limit on # of substeps
    fraction_of_range=0.05,  # each sub-step is 5% of the electron's remaining range
    step_min=0.001,   # cm; we won't go below 0.001 cm
    step_max=0.1      # cm; we won't exceed 0.1 cm
):
    """
    Condensed-history electron transport with multiple scattering and dynamic step size.
    Inspired by Penelope/Geant4, but simplified.

    1) On each iteration:
       (A) Compute the electron's remaining range R(E).
       (B) Pick a sub-step = fraction_of_range * R(E), clamped by [step_min, step_max].
       (C) Compute the energy loss dE = stopping_power(E)*step.
       (D) Sample a multiple-scattering angle from the Molière/Highland formula.
       (E) Deflect the electron direction by that angle, step the electron, deposit dE in env.dose_tally.
    2) Repeat until the electron's energy E < ecut or it leaves the phantom or we exceed max_steps.

    Args:
      E_electron: float. Initial electron energy in MeV.
      direction: np.array([ux, uy, uz]) (3D unit vector).
      start_pos: tuple (x, y, z) in cm. Starting location of the electron.
      dose_tally: 1D array for dose bins in z (env.pdd_bins).
      env: environment object that has geometry (xmin,xmax,ymin,ymax,zmin,zmax, pdd_bins, dz).
      Egrid, S_vals: arrays so we can call your 'stopping_power(E, Egrid, S_vals)' function.
      ecut: MeV. If E < ecut, we stop tracking.
      max_steps: failsafe if the electron keeps going.
      fraction_of_range: fraction of the electron's *remaining* range used as sub-step each iteration.
      step_min: cm. We never take a step smaller than this, even if fraction_of_range is tiny.
      step_max: cm. We never take a step bigger than this, even if fraction_of_range is large.

    Returns:
      total_deposit: float. Total MeV deposited by the electron in env.dose_tally.
    """

    # ----------------------------------------------------------------------------
    # 1) HELPER FUNCTIONS
    # ----------------------------------------------------------------------------

    def stopping_power_e(E):
        """Use your existing interpolation for S(E)."""
        return stopping_power(E, Egrid, S_vals)

    def electron_range(E, Ecut=ecut, steps_for_range=200):
        """
        Numerically estimate the electron range (in cm) from energy E down to Ecut
        by integrating dR = dE / S(E).  We'll do a rough piecewise integration in 'steps_for_range' increments.
        """
        if E < Ecut:
            return 0.0
        # We'll do a small log-spaced or linear-spaced approach
        # so we integrate from E -> Ecut in 'steps_for_range' steps.
        dE = (E - Ecut)/steps_for_range
        E_current = E
        dist = 0.0
        for _ in range(steps_for_range):
            S_ = stopping_power_e(E_current)
            if S_ < 1e-30:
                break
            dist += dE / S_
            E_current -= dE
            if E_current <= Ecut:
                break
        return dist

    def rotate_direction(old_dir, theta, phi):
        """
        Like your existing rotate_direction, rotates old_dir by (theta, phi).
        """
        (u,v,w) = old_dir
        mag = math.sqrt(u*u + v*v + w*w)
        if mag < 1e-14:
            # pick default axis if direction is zero-ish
            return np.array([
                math.sin(theta)*math.cos(phi),
                math.sin(theta)*math.sin(phi),
                math.cos(theta)
            ])
        ux = u/mag; uy = v/mag; uz = w/mag
        st = math.sin(theta); ct = math.cos(theta)
        sp = math.sin(phi); cp = math.cos(phi)
        if abs(uz) < 0.9999:
            denom = math.sqrt(1.0 - uz*uz)
            u2 = ux*ct + ((ux*uz*cp - uy*sp)/denom)*st
            v2 = uy*ct + ((uy*uz*cp + ux*sp)/denom)*st
            w2 = uz*ct - denom*cp*st
        else:
            u2 = st*cp
            v2 = st*sp
            w2 = ct if uz > 0 else -ct
        return np.array([u2, v2, w2])

    def moliere_scattering_angle(E, step):
        """
        Molière/Highland formula for RMS multiple scattering angle:
        
          theta0 = 13.6 MeV / (beta * p) * sqrt( (step_in_radiation_lengths) ) [1 + 0.038 ln(...)]
        
        We approximate water's radiation length X0 ~ 36.08 g/cm^2, 
        and assume density=1 g/cm^3 => track length in cm => 'mass thickness' in g/cm^2 => step*g/cm^3.
        
        For electron:
         p  = sqrt(E^2 - (mec2)^2) in MeV/c
         beta = p/E
         step_in_g_per_cm2 = step * 1.0 (since density=1)
         step_in_radiation_length = step_in_g_per_cm2 / 36.08
        We'll then sample from a *Gaussian-like* distribution with sigma = theta0, ignoring edge corrections.
        """
        if E <= 0.511:
            # If near rest, handle carefully
            E = 0.5111
        # momentum:
        me = 0.511
        p = math.sqrt(E*E - me*me)  # in MeV/c
        beta = p/E  # dimensionless

        # compute fraction of rad length
        X0 = 36.08  # g/cm^2 for water
        thickness = step * 1.0  # density=1 g/cm^3 => step*g/cm^2
        tfrac = thickness / X0
        if tfrac < 1e-12:
            return 0.0

        # Highland formula:
        #   theta0 = (13.6 MeV / (beta * p)) * z * sqrt(tfrac) [1 + 0.038 ln(tfrac)]
        # for e- => z=1
        # clamp the log(...) if too small
        factor = 1.0 + 0.038*max( math.log(tfrac), -6.0 )
        theta0 = (13.6/(beta*p)) * math.sqrt(tfrac) * factor
        if theta0 < 1e-12:
            return 0.0
        
        # We'll sample from a normal distribution with sigma=theta0
        # but for larger angles, Molière is not exactly Gaussian. 
        # This is a known approximation. We'll do it anyway.
        angle = random.gauss(0.0, theta0)
        return angle

    # ----------------------------------------------------------------------------
    # 2) INITIALIZE
    # ----------------------------------------------------------------------------
    pos = np.array(start_pos, dtype=float)
    dir_ = np.array(direction, dtype=float)
    E = E_electron
    total_deposit = 0.0

    # normalize direction
    norm_dir = np.linalg.norm(dir_)
    if norm_dir < 1e-12:
        # pick a default axis if direction is near zero
        dir_ = np.array([0,0,1], dtype=float)
    else:
        dir_ /= norm_dir

    # early exit if no energy
    if E < ecut:
        return 0.0

    # ----------------------------------------------------------------------------
    # 3) MAIN LOOP
    # ----------------------------------------------------------------------------
    step_count = 0
    while E > ecut and step_count < max_steps:
        step_count += 1
        
        # 3A) compute range from E -> ecut
        R = electron_range(E, Ecut=ecut, steps_for_range=200)
        if R < 1e-12:
            # means not enough range or S(E) is too large
            # deposit what's left and stop
            deposit = E - ecut
            if deposit < 0:
                deposit = 0
            # deposit in current bin
            k = int((pos[2] - env.zmin)/env.dz)
            if 0 <= k < env.pdd_bins:
                dose_tally[k] += deposit
            total_deposit += deposit
            E = ecut
            break

        # 3B) pick sub-step distance
        sub_step = fraction_of_range * R
        # clamp to [step_min, step_max]
        if sub_step < step_min:
            sub_step = step_min
        elif sub_step > step_max:
            sub_step = step_max
        
        # 3C) stopping power => energy loss
        S_ = stopping_power_e(E)
        if S_ < 1e-30:
            # deposit leftover
            deposit = E - ecut
            if deposit < 0: 
                deposit = 0
            k = int((pos[2] - env.zmin)/env.dz)
            if 0 <= k < env.pdd_bins:
                dose_tally[k] += deposit
            total_deposit += deposit
            E = ecut
            break

        dE = S_ * sub_step
        if dE > (E - ecut):
            dE = E - ecut
        
        # 3D) Molière multiple scattering
        angle_scatter = moliere_scattering_angle(E, sub_step)
        # sample random azimuth
        phi = 2.0*math.pi*random.random()
        # rotate dir_ by angle_scatter
        new_dir = rotate_direction(dir_, abs(angle_scatter), phi)

        # 3E) move electron
        new_pos = pos + sub_step * new_dir
        
        # 3F) deposit energy
        # We'll deposit it in the bin corresponding to the END of the step
        # (some codes deposit half in start, half in end, but let's keep it simple).
        k = int((new_pos[2] - env.zmin)/env.dz)
        if 0 <= k < env.pdd_bins:
            dose_tally[k] += dE
        
        E -= dE
        pos = new_pos
        dir_ = new_dir
        total_deposit += dE
        
        # check boundary
        if (pos[0] < env.xmin or pos[0] > env.xmax or
            pos[1] < env.ymin or pos[1] > env.ymax or
            pos[2] < env.zmin or pos[2] > env.zmax):
            # electron left the phantom
            break
    
    return total_deposit
    

###############################################################################
#            ACCEPTANCE KERNELS - DENSE REWARDS FOR PHASES 2–3)
###############################################################################
# ── mirrors `sample_compton()` step -for -step ─────────────
def accept_prob_compton(E_in, E_scat, cos_theta, sampler):
    if E_scat <= 0 or E_scat >= E_in:
        return 0.0, np.zeros(180)
    eps   = E_scat / E_in
    alpha = E_in / mec2
    cost  = 1.0 - (1.0 - eps) / (eps * alpha)
    cost  = np.clip(cost, -1.0, 1.0)
    q     = sampler.compute_q(eps, cost)
    S_q   = np.interp(q/2, sampler.q_grid, sampler.S_grid,
                      left=0.0, right=sampler.S_max)
    S_fac = S_q / sampler.S_max
    kappa   = alpha
    tau_val = 1.0 / (1.0 + kappa*(1.0 - cost))
    num = (1.0 + (kappa**2 - 2*kappa - 2)*tau_val
                + (2*kappa + 1)*tau_val**2
                + kappa**2 * tau_val**3)
    den = (1.0 + 2*kappa)**2
    T_pen  = num / den
    
    # Calculate the full distribution IN DEGREES
    angles_deg = np.linspace(0, 180, 180)  # Degrees to match histogram bins
    angles_rad = np.radians(angles_deg)     # Convert to radians for physics
    dist = np.zeros(180)
    
    for i, theta_rad in enumerate(angles_rad):
        cos_i = np.cos(theta_rad)
        q_i = sampler.compute_q(eps, cos_i)
        S_q_i = np.interp(q_i/2, sampler.q_grid, sampler.S_grid,
                          left=0.0, right=sampler.S_max)
        S_fac_i = S_q_i / sampler.S_max
        tau_val_i = 1.0 / (1.0 + kappa*(1.0 - cos_i))
        num_i = (1.0 + (kappa**2 - 2*kappa - 2)*tau_val_i
                    + (2*kappa + 1)*tau_val_i**2
                    + kappa**2 * tau_val_i**3)
        T_pen_i = num_i / den
        dist[i] = float(np.clip(S_fac_i * T_pen_i * np.sin(theta_rad), 0.0, 1.0))
    
    # Normalize
    if np.sum(dist) > 0:
        dist = dist / np.sum(dist)
    
    return float(np.clip(S_fac * T_pen, 0.0, 1.0)), dist

# ── mirrors `sample_rayleigh()` step -for -step ─────────────
def accept_prob_rayleigh(E_in, cos_theta, data):
    k     = (E_in*1e3) / data.HC_KEV_A
    theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))
    q     = 2 * k * math.sin(theta_rad/2)
    F_q   = data.coherent_form_factor(q)
    ratio = (F_q / data.F0)**2
    
    # Create a full distribution based on the kernel function IN DEGREES
    angles_deg = np.linspace(0, 180, 180)  # Degrees to match histogram bins
    angles_rad = np.radians(angles_deg)     # Convert to radians for physics
    cos_values = np.cos(angles_rad)
    dist = np.zeros_like(angles_deg)
    
    for i, (angle_rad, cos_val) in enumerate(zip(angles_rad, cos_values)):
        q_i = 2 * k * math.sin(angle_rad/2)
        F_q_i = data.iaea_form_factor(q_i)
        ratio_i = (F_q_i / data.F0)**2
        dist[i] = ratio_i * (1 + cos_val**2)/2 * np.sin(angle_rad)
    
    # Normalize
    if np.sum(dist) > 0:
        dist = dist / np.sum(dist)
    
    return float(np.clip(ratio * (1 + cos_theta**2)/2, 0.0, 1.0)), dist

# ── mirrors `sample_photoelectric()` step -for -step ─────────────
def accept_prob_photo(E_in, cos_theta, shell='O_K'):
    """
    Acceptance probability for photo-electric events, consistent with the
    rejection test inside `sample_photoelectric()`.

    • K-shell  (Sauter distribution):      P_acc = (1 − β·cosθ) / 2  
    • L-shells (isotropic):                P_acc = 1.0   (no rejection)
    """
    # Create a distribution array IN DEGREES
    angles_deg = np.linspace(0, 180, 180)  # Degrees to match histogram bins
    angles_rad = np.radians(angles_deg)     # Convert to radians for physics
    dist = np.ones(180)  # Default uniform for L-shells

    # -------- L-shells: the sampler takes the angle outright -----------
    if shell in ('O_L1', 'O_L2', 'O_L3'):
        # Uniform distribution in cos theta -> sin(theta) distribution in theta
        for i, angle_rad in enumerate(angles_rad):
            dist[i] = np.sin(angle_rad)
        # Normalize
        if np.sum(dist) > 0:
            dist = dist / np.sum(dist)
        return 1.0, dist

    # -------- K-shell: rejection factor g = (1 − β cosθ)/2 -------------
    Eb_MeV = PHOTO_SHELL_BINDINGS[shell] * 1e-6
    T      = (E_in - Eb_MeV) / mec2       # kinetic energy / m_ec²
    if T <= 0.0:
        return 0.0, np.zeros(180)         # photon below binding energy

    beta = math.sqrt(T*(T+2.0)) / (T+1.0) # electron velocity / c
    prob = 0.5 * (1.0 - beta * cos_theta) # (1 − β·cosθ)/2

    # Calculate the full Sauter distribution
    for i, angle_rad in enumerate(angles_rad):
        cos_val = np.cos(angle_rad)
        dist[i] = 0.5 * (1.0 - beta * cos_val) * np.sin(angle_rad)
    
    # Normalize
    if np.sum(dist) > 0:
        dist = dist / np.sum(dist)

    # numerical safety
    return float(min(max(prob, 0.0), 1.0)), dist

# ── mirrors pair production angular distribution ──────────────────────────────
def accept_prob_pair(E_in, cos_theta):
    """
    Angular distribution for pair production electrons/positrons.
    Uses small-angle approximation: θ_characteristic ≈ mec²/E
    """
    E_avail = E_in - 2.0 * mec2
    if E_avail <= 0.0:
        return 0.0, np.zeros(180)
    
    theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))
    theta_char = mec2 / E_in  # Characteristic scattering angle in radians
    
    # Small-angle approximation with exponential fall-off
    prob = np.exp(-theta_rad / theta_char) * np.sin(theta_rad)
    
    # Create full distribution IN DEGREES
    angles_deg = np.linspace(0, 180, 180)  # Degrees to match histogram bins
    angles_rad = np.radians(angles_deg)     # Convert to radians for physics
    dist = np.zeros(180)
    
    for i, theta_rad_i in enumerate(angles_rad):
        dist[i] = np.exp(-theta_rad_i / theta_char) * np.sin(theta_rad_i)
    
    # Normalize
    if np.sum(dist) > 0:
        dist = dist / np.sum(dist)
    
    return float(prob), dist


def accept_prob(proc, *args, **kw):
    prob_func = {
        0: accept_prob_rayleigh,
        1: accept_prob_compton,
        2: accept_prob_photo,
        3: accept_prob_pair,
    }[proc]
    
    # Return both probability and distribution
    return prob_func(*args, **kw)

###############################################################################
#               HYBRID ENV WITH FLATTENED ACTION
###############################################################################
class WaterPhotonHybridEnvPenelope(gym.Env):
    global PHASE_ENDS, mec2
    """
    Single-step approach:
     - Action is a Box of shape (1 + contDim).
       The first element: discrete choice in [0..3] but stored as a float.
       The next elements: continuous parameters (mfp, Epred, secondaries, etc.)
    """
    def __init__(self, data, ecut=1e-3, max_steps=100000, NsecMax=2, *,
                train_mode=True, 
                fixed_energy=0.1, 
                energy_range=(0.001, 1.0),
                log_uniform=False,  
                n_multi: int = N_STEPS_RETURN):
        super().__init__()
        self.E_min, self.E_max = energy_range
        self.log_uniform       = log_uniform
        # ── for adaptive scaling ────────────────────────────────
        self.reward_buffer = collections.deque(maxlen=1000)
        # ── for Lagrange multiplier constraint enforcement ──────
        self.lambda_cons     = 0.1
        self.alpha_dual      = 0.1      # step‐size for λ updates
        self.beta_intrinsic   = 0.1
        self.alpha_mse = 1.0    # weight for log‐MSE reward
        self.alpha_kl  = 0.1    # weight for KL reward
        # ── for episode‐level PDD matching ──────────────────────
        # after you run an MC baseline, store its depth‐dose curve here:
        #   self.target_pdd = np.array([...])
        self.pdd_tol        = 0.05      # allowable L2 distance
        self.episode_bonus  = 10.0
        # ── for potential shaping ───────────────────────────────
        self.gamma          = 0.995
        self.prev_phi       = 0.0
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_physics_steps = 0  # NEW: Track physics steps
        self.phase = 0
        self.pdd_bins = 100
        self.target_pdd     = np.zeros(self.pdd_bins, dtype=np.float32)
        # ── for energy curriculum learning ───────────────────────
        self.original_E_min, self.original_E_max = energy_range
        # Phase 0 duration matches the first element in phase_ends array
        self.phase0_total_steps = PHASE_ENDS[0]  
        self.num_energy_regimes = 16
        self.steps_per_regime = self.phase0_total_steps // self.num_energy_regimes
        # Phase 2 duration and regime timing
        self.phase2_total_steps = PHASE_ENDS[2] - PHASE_ENDS[1]  # Phase 2 duration
        self.steps2_per_regime = self.phase2_total_steps // self.num_energy_regimes
        self.global_step_count = 0  # Track total environment steps
        self.regime_tol        = 0.001
        self.data= data
        # Energy regime boundaries (in MeV)
        self.energy_regime_boundaries = [
            0.001,  # 1 keV
            0.005,  # 5 keV
            0.01,   # 10 keV (photoelectric dominant)
            0.02,   # 20 keV
            0.03,   # 30 keV (photoelectric/Compton transition)
            0.04,   # 40 keV
            0.05,   # 50 keV (Compton becomes more important)
            0.06,   # 60 keV
            0.07,   # 70 keV
            0.08,   # 80 keV
            0.09,   # 90 keV
            0.1,    # 100 keV (Compton dominant)
            0.3,    # 300 keV (Compton fully dominant)
            0.5,    # 500 keV
            0.6,    # 600 keV
            0.8,    # 800 keV
            1.0     # 1 MeV
        ]
        self.true_prob_regimes = np.zeros((self.num_energy_regimes, 4), dtype=np.float32)
        for k in range(self.num_energy_regimes):
            emin, emax = self.energy_regime_boundaries[k], self.energy_regime_boundaries[k+1]
            E_mid = 0.5 * (emin + emax)
            coh, inc, pho, ppr, _ = self.data.partial_cs(E_mid)
            tot = coh + inc + pho + ppr + 1e-30
            self.true_prob_regimes[k] = np.array([coh, inc, pho, ppr], dtype=np.float32) / tot
        self.pred_hist_regime     = self.true_prob_regimes.copy()
        self.cum_pred_hist_regime = self.true_prob_regimes.copy()
        self.current_regime = 0
        self.E_min, self.E_max = energy_range
        self.ecut= ecut
        _all_sigma_vals = np.concatenate([
            self.data.sigma_coh,
            self.data.sigma_inc,
            self.data.sigma_pho,
            self.data.sigma_ppr,
        ])
        _log_vals = np.log10(_all_sigma_vals + 1e-30)  # avoid log(0)
        self.LOG_MIN = float(np.floor(_log_vals.min())) - 0.5   # small safety pad
        self.LOG_MAX = float(np.ceil (_log_vals.max())) + 0.5
        print(f"[obs] log-σ range set to {self.LOG_MIN:.1f} … {self.LOG_MAX:.1f}")
        self.max_steps= max_steps
        self.NsecMax= NsecMax
        self.train_mode= train_mode
        self.fixed_energy= fixed_energy
        self.n_multi = n_multi
        self.force_mc_interaction = False
        # Flattened action dimension:
        # discrete => 1
        # continuous => 2 + 3*NsecMax (like your old approach)
        base_cont   = 4 + 3*self.NsecMax
        self.cont_dim   = base_cont + 2 
        self.action_dim = 1 + self.cont_dim 
        self.action_space= spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # Physical parameter ranges
        self.param_ranges = {
            'mfp': (2.452784e-4, 1.4140e1),        # cm
            'energy': (0.001, 1.001),     # MeV
            'photon_theta': (0, math.pi),          # radians, for photon scattering
            'photon_phi': (0, 2*math.pi),          # radians, for photon scattering
            'theta': (0, math.pi),      # radians, electrons
            'phi': (0, 2*math.pi)      # radians, electrons
#            'theta_mfp': (-2.65, 8.31)              # log-rate range (from MFP)
        }
        # Build observation space (like before)
        # 1) base_obs bounds

        base_min = np.array([
            -1.0, -1.0,  0.0,   # x_norm, y_norm, z_norm
             0.0, -3.0,         # E_norm, logE
            -1.0, -1.0, -1.0,   # u_norm, v_norm, w_norm
             0.0,  0.0,         # step_frac, local_step_norm
             0.0                # mfp_norm ∈ [0,1]
        ], dtype=np.float32)
        base_max = np.array([
             1.0,  1.0,  1.0,   # x_norm, y_norm, z_norm
             1.001, 0.00043407747,       # E_norm, logE
             1.0,  1.0,  1.0,   # u_norm, v_norm, w_norm
             1.0,  1.0,         # step_frac, local_step_norm
             1.0                # mfp_norm ∈ [0,1]
        ], dtype=np.float32)

        # 2) cross sections
        cs_min = np.zeros(4, dtype=np.float32)
        cs_max = np.ones(4, dtype=np.float32)

                


        # 3) secondary features
        sec_min = np.tile(np.array([0.0, 0.0, 0.0], dtype=np.float32), self.NsecMax)
        sec_max = np.tile(np.array([2.0, 1.0, 1.0], dtype=np.float32), self.NsecMax)

        # 4) shell one-hot
        shell_min = np.zeros(4, dtype=np.float32)
        shell_max = np.ones(4, dtype=np.float32)



        # concatenate
        self.obs_min = np.concatenate([base_min, cs_min, sec_min, shell_min])
        self.obs_max = np.concatenate([base_max, cs_max,  sec_max, shell_max])

        # tell Gym
        
        self.observation_space = spaces.Box(
            low=self.obs_min,
            high=self.obs_max,
            dtype=np.float32
        )
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        # Phantom geometry
        self.xmin,self.xmax= -50.0, 50.0
        self.ymin,self.ymax= -50.0, 50.0
        self.zmin,self.zmax= 0.0, 100.0
        self.pdd_bins= 100
        self.dz = 1.0  
        E_grid = self.initialize_energy_bins()
        print(f"Initialized energy bins: N_EBINS={self.N_EBINS}")
        print(f"First few bin edges: {10**self.ebin_edges[:5]}")
        print(f"Last few  bin edges: {10**self.ebin_edges[-5:]}")
        self.E_grid     = E_grid.astype(np.float64)                      # shape (N_EBINS,)
        self.log_E_grid = np.log(self.E_grid)                            # log E midpoints
        self.hist_decay = 0.995  # decay factor for EWMA histograms
        self.steps = 0
        self.alive = True
        self.dose_tally = np.zeros(self.pdd_bins, dtype=np.float32)
        self.interaction_bank  = []
        self.interaction_stats = []
        self.use_gymnasium_api = bool(self.train_mode)
        self.reset()
        # ---------- Kernel-reward settings ----------

        self.lambda_phi  = 0.05
        # --- electron & positron φ-uniformity (new) -----------------
        self._phi_c_acc  = 0+0j    # complex sum for scattered photons
        self._phi_c_cnt  = 0

        self._phi_e_acc  = 0+0j    # complex sum for electrons
        self._phi_e_cnt  = 0

        self._phi_p_acc  = 0+0j    # complex sum for positrons
        self._phi_p_cnt  = 0
        self.shell_names = ["H_K", "O_K", "O_L1", "O_L2", "O_L3"]
        self.initialize_angle_tracking()

        # Per-bin angle tracking for phases 2+ continuous head learning
        self.angle_hist_per_bin = {}
        self.angle_target_per_bin = {}
        self.angle_kl_per_bin = {}
        
        # Initialize per-bin tracking for all energy bins
        for bin_idx in range(self.N_EBINS):
            self.angle_hist_per_bin[bin_idx] = {
                "rayleigh": deque(maxlen=200),
                "compton": deque(maxlen=200), 
                "photo": deque(maxlen=200),
                "pair": deque(maxlen=200)
            }
            self.angle_target_per_bin[bin_idx] = {
                "rayleigh": np.ones(180) / 180.0,
                "compton": np.ones(180) / 180.0,
                "photo": np.ones(180) / 180.0, 
                "pair": np.ones(180) / 180.0
            }
            self.angle_kl_per_bin[bin_idx] = {
                "rayleigh": 0.0, "compton": 0.0, "photo": 0.0, "pair": 0.0
            }        
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.steps = 0
        # Debug print to verify phase and step count
        if self.train_mode and self.global_step_count % 100 == 0:
            print(f"🔄 Reset called: phase={self.phase}, global_step={self.global_step_count}")    
            # Update energy range based on curriculum if in phase 0
            print(f"Phase 0 energy range: {self.E_min*1000:.1f} - {self.E_max*1000:.1f} keV "
                  f"(Regime {self.current_regime+1}/{self.num_energy_regimes}, "
                  f"Step {self.global_step_count})")

        # ─── Dynamic phase-0 regime advance with debug prints ─────────────────────────
        if self.phase == 0 and self.train_mode:
            # compute L1 between agent's cum-hist and true for current regime
            hist_actions = self.cum_pred_hist_regime[self.current_regime]
            if hist_actions.sum() == 0:
                # one pseudo-count per class proportional to the true probs
                hist_actions[:] = self.true_prob_regimes[self.current_regime]
            p_agent      = hist_actions / (hist_actions.sum() + 1e-15)
            p_true       = self.true_prob_regimes[self.current_regime] 
            # compute L1 distance
            d_l1 = float(np.abs(p_agent - p_true).sum())

            # DEBUG: log L1 and tolerance each reset
            if self.global_step_count % 100 == 0:
                print(f"[DEBUG] regime {self.current_regime+1}/{self.num_energy_regimes} "
                      f"at step {self.global_step_count}: L1={d_l1:.6f}, tol={self.regime_tol}")

            # advance if L1 match OR timeout
            timeout = self.steps_per_regime * (self.current_regime + 1)
            if self.global_step_count >= timeout:
                print(f"[DEBUG] → advancing from regime {self.current_regime} (L1={d_l1:.6f})")

                # Determine the next regime
                next_r = min(
                    self.current_regime + 1,
                    self.num_energy_regimes - 1
                )

                # Initialize new histograms with ground truth values
                initial_count = 100

                # Reset histograms
                self.pred_hist[:] = 0.0
                self.cum_pred_hist[:] = 0.0

                # If advancing to new regime, seed bins for that energy range
                if next_r > self.current_regime:
                    e_low = self.energy_regime_boundaries[next_r]
                    e_high = self.energy_regime_boundaries[next_r + 1]

                    # Find bins within the new energy range and seed them
                    for bin_idx in range(len(self.ebin_edges) - 1):
                        e_min = 10**self.ebin_edges[bin_idx]
                        e_max = 10**self.ebin_edges[bin_idx+1]

                        # If bin overlaps with new regime energy range
                        if (e_min <= e_high and e_max >= e_low):
                            # Seed with ground truth x initial_count
                            self.pred_hist[bin_idx] = self.true_prob[bin_idx] * initial_count
                            self.cum_pred_hist[bin_idx] = self.true_prob[bin_idx] * initial_count

                    # Reset and initialize regime-specific histograms
                    self.pred_hist_regime[:] = 0.0
                    self.cum_pred_hist_regime[:] = 0.0
                    self.pred_hist_regime[next_r] = self.true_prob_regimes[next_r] * initial_count
                    self.cum_pred_hist_regime[next_r] = self.true_prob_regimes[next_r] * initial_count

                    print(f"[DEBUG] → Re-seeded histograms for regime {next_r} with {initial_count} scaled ground truth values")

                self.current_regime = next_r

            # update energy range based on (possibly advanced) current_regime
            if self.current_regime < self.num_energy_regimes - 1:
                self.E_min = self.energy_regime_boundaries[self.current_regime]
                self.E_max = self.energy_regime_boundaries[self.current_regime + 1]
            else:
                self.E_min = self.original_E_min
                self.E_max = self.original_E_max

            # print Phase 0 regime info every 100 steps
            if self.global_step_count % 100 == 0:
                lo, hi = self.E_min * 1e3, self.E_max * 1e3
                print(f"Phase 0 energy range: {lo:.1f} – {hi:.1f} keV "
                      f"(Regime {self.current_regime+1}/{self.num_energy_regimes}, "
                      f"Step {self.global_step_count})")
        elif self.phase == 2 and self.train_mode:
            # phase 2: energy curriculum for kernel learning (same as phase 0)

    
            # Time-based regime advancement (similar to phase 0)
            if self.current_regime < self.num_energy_regimes - 1:
                # Calculate timeout based on phase 2 start
                phase2_start_step = PHASE_ENDS[1]  # When phase 2 started
                steps_in_phase2 = self.global_step_count - phase2_start_step
                timeout = self.steps2_per_regime * (self.current_regime + 1)
        
                if steps_in_phase2 >= timeout:
                    print(f"[DEBUG] → Phase 2 advancing from regime {self.current_regime} (time-based at step {self.global_step_count})")
                    next_r = min(self.current_regime + 1, self.num_energy_regimes - 1)
                    self.current_regime = next_r

                if self.global_step_count % 100 == 0:
                    print(f"Phase 2 energy range: {self.E_min*1000:.1f} - {self.E_max*1000:.1f} keV "
                          f"(Regime {self.current_regime+1}/{self.num_energy_regimes}, "
                          f"Step {self.global_step_count})")

            # Update energy range based on current regime  
            if self.current_regime < self.num_energy_regimes - 1:
                self.E_min = self.energy_regime_boundaries[self.current_regime]
                self.E_max = self.energy_regime_boundaries[self.current_regime + 1]
            else:
                self.E_min = self.original_E_min
                self.E_max = self.original_E_max
        # ───────────────────────────────────────────────────────────────────────────────

    
        # Initialize position
        half_side = 5.0
        self.x = (random.random() * 10.0) - half_side
        self.y = (random.random() * 10.0) - half_side
        self.z = 0.0
        # Store the initial physical coordinate for later tracking
        self.initial_position = (self.x, self.y, self.z)
        self.u, self.v, self.w = (0.0, 0.0, 1.0)

        # Sample energy based on current range
        if self.train_mode:
            if self.log_uniform: 
                logE  = random.uniform(math.log10(self.E_min), math.log10(self.E_max))
                self.E = 10 ** logE
            else:
                # Find bin edges that fall within our current energy range
                valid_edges = []
                for i in range(len(self.ebin_edges) - 2):
                    edge_low = 10 ** self.ebin_edges[i]
                    edge_high = 10 ** self.ebin_edges[i+1]
                    # Check if this bin overlaps with current energy range
                    if edge_high >= self.E_min and edge_low <= self.E_max:
                        valid_edges.append(i)
            
                if valid_edges:
                    # Choose from valid bins
                    k = random.choice(valid_edges)
                #    logE = 0.5 * (self.ebin_edges[k] + self.ebin_edges[k+1])
                    self.E = 10 ** self.ebin_edges[k+1]
                else:
                    # Fallback to uniform in range
                    self.E = random.uniform(self.E_min, self.E_max)
        else:
            self.E = self.fixed_energy

        self.alive = True
        self.dose_tally[:] = 0.0
        self.interaction_bank = []
        self.interaction_stats = []
        return self._get_obs(), {}


    def initialize_energy_bins(self):
        """Compute self.ebin_edges, self.true_prob and histograms from one E_grid."""
        full_E = np.asarray(self.data.E_final)
        mask   = (full_E >= self.ecut) & (full_E <= self.E_max)
        E_grid = full_E[mask]
        self.N_EBINS = E_grid.size

        # linear-space edges via midpoints
        edges_energy = np.empty(self.N_EBINS + 1, dtype=float)
        edges_energy[1:-1] = 0.5 * (E_grid[:-1] + E_grid[1:])
        edges_energy[0]     = E_grid[0]
        edges_energy[-1]    = E_grid[-1]
        self.ebin_edges = np.log10(edges_energy)

        # ground-truth probabilities per bin
        self.true_prob = np.zeros((self.N_EBINS, 4), dtype=np.float32)
        self.true_mfp_mean  = np.zeros(self.N_EBINS,       dtype=np.float32)
        self.comp_sampler_cache = []
        for i, E_mid in enumerate(E_grid):
            coh, inc, pho, ppr, _ = self.data.partial_cs(E_mid)
            tot = coh + inc + pho + ppr + 1e-30
            self.true_prob[i] = [coh/tot, inc/tot, pho/tot, ppr/tot]
            mu = self.data.mu_total(E_mid)                   
            self.true_mfp_mean[i] = 1.0 / (mu + 1e-12)     
            self.comp_sampler_cache.append(
                ComptonSampler(E_mid)  
            )
        # allocate all histograms to the same shape
        initial_count = 100
        self.pred_hist            = self.true_prob.copy() * initial_count
        self.cum_pred_hist        = self.true_prob.copy() * initial_count
        self.pred_hist_regime     = self.true_prob_regimes.copy() * initial_count
        self.cum_pred_hist_regime = self.true_prob_regimes.copy() * initial_count
        # (optional) debug print
        print(f"[bins] N_EBINS={self.N_EBINS}, shapes={self.true_prob.shape}")
        print(f"[bins] Seeded histograms with {initial_count} samples per bin based on ground truth")
        return E_grid
        
    def _small_angle(self, E_gamma):
        """
            θ_char = m_e c² / (E_gamma / 2) = 2 m_e c² / E_gamma.

        This is a standard leading-order estimate from the
        Bethe–Heitler cross section and is consistent with the
        PENELOPE angular model in the high-energy limit.  The
        estimate is used here only as a reference scale for the
        pair-production angular reward, not as a sampling kernel.
        """
        E_total_approx = max(E_gamma / 2.0, mec2) # each particle ≈ half
        return mec2_local / E_total_approx
        
    def _denormalize(self, val, param_type):
        """Map from [-1,1] to physical range"""
        min_val, max_val = self.param_ranges[param_type]
        
        # Special handling for angular parameters
        if param_type in ['photon_theta','photon_phi','theta', 'phi']:
            val = np.clip(val, -1.0, 1.0)
            return (val + 1) * (max_val - min_val)/2
        
            
        if param_type == 'energy':
            scaled = 0.5*(val + 1)*(1.001 - 0.001) + 0.001
            return np.clip(scaled, 0.001, 1.001)
        # Linear scaling for others
        
        return min_val + (val + 1) * (max_val - min_val)/2

    def reset_histogram_stats(self):
        """Reset the histogram statistics used for energy-conditioned rewards"""
        # zero out every histogram to match the current true_prob shape
        self.pred_hist            = np.zeros_like(self.true_prob)
        self.cum_pred_hist        = np.zeros_like(self.true_prob)
        self.pred_hist_regime     = np.zeros_like(self.true_prob)
        self.cum_pred_hist_regime = np.zeros_like(self.true_prob)
        # debug print with shape info
        print(f"📊 Histogram statistics reset at step {self.global_step_count}, "
              f"shape={self.pred_hist.shape}")
    
    def _print_energy_band_stats(self):
        # Print stats for six representative bins (in keV)
        reps = [
            (0.02, "20 keV"),
            (0.05, "50 keV"),
            (0.20, "200 keV"),
            (0.50, "500 keV"),
            (0.70, "700 keV"),
            (0.90, "900 keV"),
        ]
        proc_names = ["rayleigh", "compton", "photo", "pair"]

        for E_val, label in reps:
            # find the corresponding bin index
            logE = math.log10(E_val)
            bin_idx = np.searchsorted(self.ebin_edges, logE, side="right") - 1
            bin_idx = max(0, min(bin_idx, len(self.ebin_edges) - 2))

            # analytic ground-truth distribution for that bin
            p_true = self.true_prob[bin_idx]

            # agent’s empirical distribution from cumulative counts
            counts = self.cum_pred_hist[bin_idx].astype(float)
            total  = counts.sum() or 1.0
            p_agent = counts / total

            # print the block
            print(f"\n--- {label} bin ---")
            for i, name in enumerate(proc_names):
                print(f"{name:9s}: {p_true[i]*100:5.1f}%  |  agent {p_agent[i]*100:5.1f}%")
                
    ###############################################################################
    #     ANGLE TRACKING AND STATISTICS FOR ALL INTERACTIONS (AGENT AND MC)
    ###############################################################################
    def initialize_angle_tracking(self):
        """Initialize angle tracking structures for both MC and agent"""
        # Store recent angles for each interaction type, both MC and agent
        self.mc_angle_history = {
            "rayleigh": deque(maxlen=1000),
            "compton": deque(maxlen=1000),
            "photo": deque(maxlen=1000),
            "pair": deque(maxlen=1000)
        }
        self.agent_angle_history = {
            "rayleigh": deque(maxlen=1000),
            "compton": deque(maxlen=1000),
            "photo": deque(maxlen=1000),
            "pair": deque(maxlen=1000)
        }




    def update_angle_history(self, mc_interaction_type, mc_angle, agent_interaction_type, agent_angle):
        """Add angles to the history for both MC and agent interactions"""
        # Base interaction types
        mc_base_type = mc_interaction_type.split("_")[0] if mc_interaction_type else None
        agent_base_type = agent_interaction_type.split("_")[0] if agent_interaction_type else None
        
        # Add MC angle
        if mc_base_type in self.mc_angle_history:
            self.mc_angle_history[mc_base_type].append(mc_angle)
        
        # Add agent angle
        if agent_base_type in self.agent_angle_history:
            self.agent_angle_history[agent_base_type].append(agent_angle)

    def _print_regime_angle_histograms(self, width=30):
        """Print MC vs Agent angle histograms for current regime bins"""
        e_low = self.energy_regime_boundaries[self.current_regime]
        e_high = self.energy_regime_boundaries[self.current_regime + 1]
        
        # Get bins in current regime
        regime_bins = []
        for b_idx in range(len(self.ebin_edges) - 1):
            e_min = 10**self.ebin_edges[b_idx]
            e_max = 10**self.ebin_edges[b_idx+1]
            if (e_min <= e_high and e_max >= e_low):
                regime_bins.append(b_idx)
        
        if not regime_bins:
            print("No bins found for current regime")
            return
            
        # Show comparison for bins with sufficient data
        for b_idx in regime_bins[:3]:  # Show first 3 bins to avoid spam
            e_bin_min = 10**self.ebin_edges[b_idx]
            e_bin_max = 10**self.ebin_edges[b_idx+1]
            
            print(f"\nBin {b_idx}: {e_bin_min*1000:.1f} - {e_bin_max*1000:.1f} keV")
            print("-" * 60)
            
            for interaction in ["rayleigh", "compton", "photo", "pair"]:
                # Get MC angles for this bin (we'd need to track these by bin)
                # For now, use agent data vs target distribution
                agent_angles = list(self.angle_hist_per_bin[b_idx][interaction])
                target_dist = self.angle_target_per_bin[b_idx][interaction]
                
                if len(agent_angles) < 10:
                    print(f"{interaction:>10s}: Insufficient data ({len(agent_angles)} samples)")
                    continue
                    
                print(f"{interaction:>10s} ({len(agent_angles):>3d} samples):")
                
                # Create histograms
                agent_hist, bin_edges = np.histogram(agent_angles, bins=15, range=(0, 180))
                target_hist = np.interp(
                    0.5 * (bin_edges[:-1] + bin_edges[1:]),  # bin centers
                    np.linspace(0, 180, len(target_dist)),   # target angle grid
                    target_dist * len(agent_angles)          # scale to match agent count
                )
                
                # Find max for scaling
                max_count = max(agent_hist.max(), target_hist.max())
                if max_count == 0:
                    continue
                    
                # Print header
                print(f"{'Angle':<12} | {'Agent':<{width+5}} | {'Physics':<{width+5}}")
                print(f"{'-'*12} | {'-'*(width+5)} | {'-'*(width+5)}")
                
                # Print bars
                for i in range(len(agent_hist)):
                    angle_start = bin_edges[i]
                    angle_end = bin_edges[i+1]
                    
                    agent_count = agent_hist[i]
                    target_count = int(target_hist[i])
                    
                    agent_bar_len = int(agent_count / max_count * width) if max_count > 0 else 0
                    target_bar_len = int(target_count / max_count * width) if max_count > 0 else 0
                    
                    agent_bar = '█' * agent_bar_len + f" {agent_count:>3d}"
                    target_bar = '█' * target_bar_len + f" {target_count:>3d}"
                    
                    angle_range = f"{angle_start:.0f}-{angle_end:.0f}°"
                    print(f"{angle_range:<12} | {agent_bar:<{width+5}} | {target_bar:<{width+5}}")
                
                print()

    def _get_obs(self):
        # Primary state (7 values)
        x_norm = 2.0 * (self.x - self.xmin) / (self.xmax - self.xmin) - 1.0
        y_norm = 2.0 * (self.y - self.ymin) / (self.ymax - self.ymin) - 1.0
        z_norm = (self.z - self.zmin) / (self.zmax - self.zmin)
        E_norm = (self.E - 0.001) / (1.001 - 0.001)
        logE_low  = math.log10(self.E_min)
        logE_high = math.log10(self.E_max)
        logE = np.clip(math.log10(max(self.E, self.E_min)), logE_low, logE_high)
        # Assuming self.u, self.v, self.w are unit-vector components.
        u_norm = self.u  # They are already between -1 and 1.
        v_norm = self.v
        w_norm = self.w
        step_frac       = self.steps / self.max_steps
        local_step_norm = (self.steps % self.n_multi) / self.n_multi
        mu        = self.data.mu_total(self.E)
        mfp       = 1.0 / (mu + 1e-12)
        mfp_norm  = np.clip((mfp - 1.4498361e-4) / (1.41654804539e1 - 1.4498361e-4), 0.0, 1.0)
        base_obs = np.array([
            x_norm, y_norm, z_norm,
            E_norm, logE,
            u_norm, v_norm, w_norm,
            step_frac,
            local_step_norm,
            mfp_norm
        ], dtype=np.float32)

        # Normalized cross sections (4 values)
        (coh, inc, pho, ppr, _) = self.data.partial_cs(self.E)
        cs_raw = np.array([coh, inc, pho, ppr], dtype=np.float32)+1e-30
        log_cs = np.log10(cs_raw)
        cs_norm = (log_cs - self.LOG_MIN) / (self.LOG_MAX - self.LOG_MIN)
        cs_norm = np.clip(cs_norm, 0.0, 1.0)   # keep within Box limits
        # Secondary electron history: include energy and direction (θ, φ) for each secondary.
        # We assume each secondary is stored as: ("electron", energy, direction, label)
        sec_features = []
        for s in self.interaction_bank[-self.NsecMax:]:
            if s[0] == "electron":
                # Energy is normalized by mec2; theta and phi are normalized.
                energy = np.clip(s[1], 0.0, 2*mec2)
                theta = np.clip(math.acos(np.clip(s[2][2], -1.0, 1.0))/math.pi, 0.0, 1.0)
                phi = (math.atan2(s[2][1], s[2][0]) % (2*math.pi)) / (2*math.pi)
                sec_features.extend([energy/mec2, theta, phi])
            else:
                sec_features.extend([0.0, 0.0, 0.0])
        while len(sec_features) < 3 * self.NsecMax:
            sec_features.extend([0.0, 0.0, 0.0])

        # Append the shell one-hot vector.
        # If no photoelectric event has happened yet, default to all zeros.
        if hasattr(self, 'last_shell_onehot'):
            shell_info = np.array(self.last_shell_onehot, dtype=np.float32)
        else:
            shell_info = np.zeros(4, dtype=np.float32)
            
        obs = np.concatenate([base_obs, cs_norm, np.array(sec_features, dtype=np.float32), shell_info])
        max_val = np.max(np.abs(obs))
        if max_val > 1e20:
            print("!!! Warning: obs huge:", obs)
        obs = np.clip(obs, -1e6, 1e6)
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"NaN or Inf detected in observation: {obs}")

        obs = np.clip(obs, -1e6, 1e6)
        if not np.all(np.isfinite(obs)):
            # Dump the exact components that are infinite or NaN
            bad = np.where(~np.isfinite(obs))[0]
            print("!!! NaN in obs at indices:", bad, "values:", obs[bad])
            raise ValueError("Non-finite observation detected")

        return obs

    def _normalise(self, val, key):
        lo, hi = self.param_ranges[key]
        return 2*(val - lo)/(hi - lo) - 1.0
        
    def step(self, action):
        self.steps += 1
        self.global_step_count += 1
        info = {}

        # ─────────────────────────────────────────────────────────────
        # 0) quick terminal check
        # ─────────────────────────────────────────────────────────────
        if (not self.alive) or (self.E < self.ecut):
            term, trunc = True, False
            ret = (self._get_obs(), 0.0, term, trunc, info)
            if not self.use_gymnasium_api:            # Gym-style return
                ret = (ret[0], ret[1], term or trunc, ret[-1])
            return ret

        # ─────────────────────────────────────────────────────────────
        # 1) RAW agent output (still in −1 … 1  range)
        # ─────────────────────────────────────────────────────────────
        raw_disc = int(np.clip(round(action[0]), 0, 3))
        orig_choice = raw_disc
        raw_cont = action[1:].copy()
        # ─────────────── μ–pred extraction ───────────────
        mu_pred = float(action[-1])                              # last cont slot
        info["mu_pred"]      = float(mu_pred)
        self.total_physics_steps += 1

        # ─────────────────────────────────────────────────────────────
        # 2)  Monte-Carlo interaction (ground-truth)
        # ─────────────────────────────────────────────────────────────
        mu_real = self.data.mu_total(self.E)
        true_mfp  = 1.0 / (mu_real + 1e-12)
        info["true_mfp"] = true_mfp
        info["mu_real"] = mu_real
        photon_energy_in = self.E
        inc_dir = np.array([self.u, self.v, self.w], dtype=float)

        new_dir, Eout, real_secs, itype, shell_onehot = photon_interact(
            photon_energy_in, inc_dir, self.data
        )
        actual_int = {"rayleigh": 0, "compton": 1,
                      "photo": 2,  "pair":    3}.get(itype.split("_")[0], -1)

        # ─────────────────────────────────────────────────────────────
        # 3) OPTIONAL teacher-forcing of the discrete label
        #    (done *after* we know actual_int)
        # ─────────────────────────────────────────────────────────────
        def _mask_cont(d_idx: int, vec: np.ndarray) -> np.ndarray:
            """
            Apply the same masking the sampler uses, so the stored
            continuous slice matches the forced discrete label.
            """
            vec = vec.copy()
            C = vec.shape[0]

            # --------------------------------------------------------------
            # 1. Photon-energy slot (index 1)
            #    ─ keep it only for Compton (d_idx == 1)
            if C > 1 and d_idx != 1:
                vec[1] = -1.0            # mute Eγ′ prediction

            # zero every secondary block ≥ 1 for Compton / photo
            if C > 4 and d_idx in (1, 2):
                nsec = (C - 4) // 3
                for k in range(1, nsec):
                    base = 4 + 3 * k
                    vec[base:base + 3] = -1.0
            # Pair: mask *positron* polar angle (slot 8); keep electron θ (slot 5)
            if d_idx == 3 and len(vec) > 8:
                vec[8] = -1.0
            
            return vec
            
        cont_masked = raw_cont.copy()
        if self.force_mc_interaction:
            discrete_choice = actual_int
            cont = _mask_cont(discrete_choice, raw_cont)
        else:
            discrete_choice = raw_disc
            cont = raw_cont

        # Make the copy here 
        # get a copy of the continuous action for potential modification
        cont_store = cont.copy()
        # ─────────────────────────────────────────────────────────────
        # 4) Angles extracted from the (possibly-masked) cont (per interaction)
        # Important Note: 
        # +----------+------------------+------------------+------------------------------------------+
        # | I Type   | Scattered Photon | Ejected Electrons| Continuous Parameters Layout & Training  |
        # +-----------------+------------------+------------------+-----------------------------------+
        # | Rayleigh | Yes (elastic)    | None             | Photon E(1)=unchanged,                   |
        # |          |                  |                  | Photon θ,φ (2,3)                         |
        # |          |                  |                  |                                          |
        # | Compton  | Yes (inelastic)  | One electron     | Photon E(1), Photon θ,φ (2,3),           |
        # |          |                  |                  | Electron E(4), Electron θ,φ (5,6)        |
        # |          |                  |                  |                                          |
        # | Photo    | No (absorbed)    | One electron     | Photon E(1)=0 (masked),                  |
        # |          |                  |                  | Photon θ,φ (2,3)=dummy (masked),         |
        # |          |                  |                  | Electron E(4), Electron θ,φ (5,6)        |
        # |          |                  |                  |                                          |
        # | PP       | No               | e- and e+ pair   | Photon E(1)=0 (masked),                  |
        # |          |                  |                  | Photon θ,φ (2,3)=dummy (masked),         |
        # |          |                  |                  | Electron E(4), Electron θ,φ (5,6),       |
        # |          |                  |                  | Positron E(7), Positron θ,φ (8,9)        |
        # +----------+------------------+------------------+------------------------------------------+
        # ─────────────────────────────────────────────────────────────
    
        photon_theta_pred = self._denormalize(cont[2], 'photon_theta')
        photon_phi_pred = self._denormalize(cont[3], 'photon_phi')

        if discrete_choice in (2, 3):  # Photo-electric or Pair production
            # These interactions have no scattered photon
            photon_theta_pred = math.pi/2  # dummy value
            photon_phi_pred = 0.0          # dummy value
        if discrete_choice in (0, 1):  # Rayleigh or Compton - use photon angle
            agent_angle_degrees = math.degrees(photon_theta_pred)
        elif discrete_choice in (2, 3):  # Photo or Pair - use electron angle
            # Define theta_e for these cases
            theta_e = self._denormalize(cont[5], 'theta')
            agent_angle_degrees = math.degrees(theta_e)
        else:
            agent_angle_degrees = 0.0  # fallback
        # ─────────────────────────────────────────────────────────────
        # 5)  Available kinetic energy  (Q subtracted)
        # ─────────────────────────────────────────────────────────────
        Eb_list = [PHOTO_SHELL_BINDINGS[s]*1e-6 for s in ("H_K", "O_K", "O_L1", "O_L2", "O_L3")]
        mec2    = 0.51099895069
        if discrete_choice == 2:          # photoelectric
            Q = sum(h*eb for h, eb in zip(shell_onehot, Eb_list))
        elif discrete_choice == 3:        # pair
            Q = 2*mec2
        else:
            Q = 0.0
        avail_E = max(photon_energy_in - Q, 0.0)

        # ─────────────────────────────────────────────────────────────
        # 6)  Build the agent’s *predicted* outgoing particle set
        # ─────────────────────────────────────────────────────────────
        if discrete_choice == 0:          # Rayleigh
            E_pred, sec_params = photon_energy_in, []

        elif discrete_choice == 1:        # Compton
            # Compton has energy partition between photon and electron
            raw_ph = max(0.0, self._denormalize(cont[1], 'energy'))
            raw_el = max(0.0, self._denormalize(cont[4], 'energy'))
            s = raw_ph + raw_el or 1.0
            # Distribute available energy (no binding energy/Q value)
            E_pred = (raw_ph / s) * avail_E
            Ej     = (raw_el / s) * avail_E
            theta_e = self._denormalize(cont[5], 'theta')
            phi_e   = self._denormalize(cont[6], 'phi')
            dir_e   = rotate_direction(inc_dir, theta_e, phi_e)
            sec_params = [("electron", Ej, dir_e, "compton_e_pred")]

        elif discrete_choice == 2:        # Photo-electric
            # Photo-electric: photon completely absorbed, electron gets all avail_E
            E_pred = 0.0  # No scattered photon
            # Network angles for electron
            theta_e = self._denormalize(cont[5], 'theta')
            phi_e   = self._denormalize(cont[6], 'phi')
            dir_e   = rotate_direction(inc_dir, theta_e, phi_e)
            # Electron must get EXACTLY the available energy (after binding energy)
            sec_params = [("electron", avail_E, dir_e, "photo_pred")]

        else:                             # Pair production
            # Pair: photon completely absorbed, e- and e+ share avail_E
            E_pred = 0.0  # No scattered photon

            # Energy partition between e- and e+ (must sum to avail_E)
            raw_e = max(0.0, self._denormalize(cont[4], 'energy'))
            raw_p = max(0.0, self._denormalize(cont[7], 'energy'))
            s = raw_e + raw_p or 1.0
            Ej_e = (raw_e / s) * avail_E
            Ej_p = (raw_p / s) * avail_E

            # Electron angles — policy predicts independently
            theta_e = self._denormalize(cont[5], 'theta')
            phi_e   = self._denormalize(cont[6], 'phi')
            dir_e   = rotate_direction(inc_dir, theta_e, phi_e)

            # Positron angles — policy also predicts independently
            theta_p = self._denormalize(cont[8], 'theta')
            phi_p   = self._denormalize(cont[9], 'phi')
            dir_p   = rotate_direction(inc_dir, theta_p, phi_p)

            sec_params = [("electron", Ej_e, dir_e, "pair_e_pred"),
                          ("positron", Ej_p, dir_p, "pair_p_pred")]

        # pad sec_params so downstream code is unchanged
        while len(sec_params) < self.NsecMax:
            sec_params.append(("electron", 0.0, inc_dir, None))

        # 4) now force E_pred→0 for PE & pair
        raw_E_corr =  self._denormalize(cont[1], 'energy')  
        r_E_corr = 0
        if discrete_choice in (2, 3):
            if raw_E_corr > 1e-12:
                r_E_corr = -5
                E_pred = 0
        # 4) MC transition
        muT = max(self.data.mu_total(self.E), 1e-15)
        
        r   = max(random.random(), 1e-12)
        dist_real = min(-math.log(r)/muT, self.zmax - self.zmin)
        self.x += dist_real * self.u
        self.y += dist_real * self.v
        self.z += dist_real * self.w
        # out‐of‐bounds?
        if not (self.xmin <= self.x <= self.xmax and
                self.ymin <= self.y <= self.ymax and
                self.zmin <= self.z <= self.zmax):
            self.alive = False
            term, trunc = True, False
            return self._get_obs(), -10.0, term, trunc, {} if self.use_gymnasium_api else (self._get_obs(), -10.0, term or trunc, {})

        # record stats
        dot_val_clamped = float(np.clip(np.dot(inc_dir, new_dir), -1.0, 1.0))
        angle_radians   = math.acos(dot_val_clamped)
        real_phi        = (math.atan2(new_dir[1], new_dir[0]) + 2*math.pi) % (2*math.pi)  
        angle_degrees   = math.degrees(angle_radians)

        # Extract the correct MC angle based on interaction type
        mc_angle_degrees = angle_degrees  # Default to photon scattering angle
        if itype.startswith("photo") or itype.startswith("pair"):
            # For photoelectric and pair production, get electron angle from MC secondaries
            if real_secs and real_secs[0][0] == "electron":
                mc_electron_dir = real_secs[0][2]
                mc_electron_cos = np.clip(np.dot(inc_dir, mc_electron_dir), -1.0, 1.0)
                mc_angle_degrees = math.degrees(math.acos(mc_electron_cos))




        # Update angle histories with correct angles
        interaction_name = ["rayleigh", "compton", "photo", "pair"][discrete_choice]
        self.update_angle_history(itype, mc_angle_degrees, interaction_name, agent_angle_degrees)
        self.interaction_stats.append({
            "interaction": itype,
            "free_path":  dist_real,
            "angle":      angle_degrees,
            "position":   (self.x, self.y, self.z),
            "photon_energy_in":  photon_energy_in,
            "photon_energy_out": Eout,
            "photon_incident_direction": inc_dir,
            "secondaries": real_secs.copy()
        })
        # fill replay‐buffer info
        info.update({
            "phys_fp":   dist_real,
            "phys_ang":  angle_radians,
            "phys_n_sec": len(real_secs),
            "phys_proc":  PROC_NAMES.index(itype),
            "phys_Eout": Eout
        })
        for k in range(self.NsecMax):
            info[f"phys_s{k}_E"]     = 0.0
            info[f"phys_s{k}_theta"] = 0.0
            info[f"phys_s{k}_phi"]   = 0.0
        for idx, sec in enumerate(real_secs[:self.NsecMax]):
            _, sE, sdir, _ = sec
            th = math.acos(np.clip(sdir[2], -1.0, 1.0))
            ph = (math.atan2(sdir[1], sdir[0]) + 2*math.pi)%(2*math.pi)
            info[f"phys_s{idx}_E"]     = sE
            info[f"phys_s{idx}_theta"] = th
            info[f"phys_s{idx}_phi"]   = ph

        # update the state
        self.u, self.v, self.w = new_dir
        self.E                = Eout
        if self.E <= self.ecut:
            self.alive = False
            
        # 7) Discrete‐choice reward
        coh, inc, pho, ppr, _ = self.data.partial_cs(photon_energy_in)
        total_cs = coh+inc+pho+ppr+1e-12
        p_true   = [coh/total_cs, inc/total_cs, pho/total_cs, ppr/total_cs]

  
        # ------------------------------------------------------------------
        #  7-A  Force-MC mechanism for discrete choice and kernels
        #       • Phase 0-1: Override discrete choice if force_mc_interaction is True
        #       • Phase 2-3: Override angle, energy, etc. with MC "truth"
        # ------------------------------------------------------------------
        # Needs to be moved before the first call
        # cont_store = cont.copy()
        # (A) helper, put inside class above override block


        # 1. Handle discrete action override (phases 0-1)
        disc_store = actual_int if self.force_mc_interaction else discrete_choice
        interaction_names = ["rayleigh", "compton", "photo", "pair"]
        # ────────────────────────────────────────────────────────────────
        # 7-B Handle *kernel* override during curriculum phases 2–3
        #     · phase 2 : always override with the Monte-Carlo values
        #     · phase 3 : same schedule that you already use for the
        #                 discrete head – fades out via
        #                 self.force_mc_interaction ∈ {True,False}
        # ────────────────────────────────────────────────────────────────
        if self.phase in (2, 3):
            # phase-dependent gate
            should_override_kernel = (
                True if self.phase == 2 else self.force_mc_interaction
            )

            if should_override_kernel:
                # helper for compactness
                norm = self._normalise     # (value, key) → [-1,1]

                # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
                # MC “truths” that we just calculated a few lines above
                # ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
                #
                #   · photon_energy_in      (MeV)   – incoming photon
                #   · Eout                  (MeV)   – MC scattered photon
                #   · angle_radians                  – polar θ of photon
                #   · real_phi                        – azimuth  φ of photon
                #   · real_secs  list[tuple]         – MC secondaries
                #                                       ("electron", E, dir, label)
                #
                # ---------------------------------------------------------

                # write back into cont_store[•]
                if disc_store == 0:                 # ── Rayleigh
                    # photon keeps its energy – only the angles matter
                    cont_store[2] = norm(angle_radians, 'theta')      # θγ
                    cont_store[3] = norm(real_phi,       'phi')       # φγ

                elif disc_store == 1:               # ── Compton
                    # photon branch
                    cont_store[1] = norm(Eout, 'energy')              # Eγ
                    cont_store[2] = norm(angle_radians, 'theta')      # θγ
                    cont_store[3] = norm(real_phi,       'phi')       # φγ
                    # single recoil electron (slot 0 in secondaries)
                    if real_secs:
                        eE, edir = real_secs[0][1], real_secs[0][2]
                        theta_e  = math.acos(np.clip(np.dot(inc_dir, edir), -1.0, 1.0))
                        phi_e    = (math.atan2(edir[1], edir[0]) + 2*math.pi) % (2*math.pi)
                        cont_store[4] = norm(eE,      'energy')
                        cont_store[5] = norm(theta_e, 'theta')
                        cont_store[6] = norm(phi_e,   'phi')

                elif disc_store == 2:               # ── Photoelectric
                    # no outgoing photon ⇒ Eγ = 0
                    cont_store[1] = norm(0.0, 'energy')
                    cont_store[2] = norm(math.pi/2, 'theta')          # dummy
                    cont_store[3] = norm(0.0,       'phi')
                    # ejected electron in secondary slot 0
                    if real_secs:
                        eE, edir = real_secs[0][1], real_secs[0][2]
                        theta_e  = math.acos(np.clip(np.dot(inc_dir, edir), -1.0, 1.0))
                        phi_e    = (math.atan2(edir[1], edir[0]) + 2*math.pi) % (2*math.pi)
                        cont_store[4] = norm(eE,      'energy')
                        cont_store[5] = norm(theta_e, 'theta')
                        cont_store[6] = norm(phi_e,   'phi')

                else:                               # ── Pair production
                    # no outgoing photon
                    cont_store[1] = norm(0.0, 'energy')
                    cont_store[2] = norm(math.pi/2, 'theta')          
                    cont_store[3] = norm(0.0,       'phi')
                    # electron  (secondary-0)
                    if len(real_secs) >= 1:
                        eE, edir = real_secs[0][1], real_secs[0][2]
                        theta_e  = math.acos(np.clip(np.dot(inc_dir, edir), -1.0, 1.0))
                        phi_e    = (math.atan2(edir[1], edir[0]) + 2*math.pi) % (2*math.pi)
                        cont_store[4] = norm(eE,      'energy')
                        cont_store[5] = norm(theta_e, 'theta')
                        cont_store[6] = norm(phi_e,   'phi')
                    # positron (secondary-1)
                    if len(real_secs) >= 2:
                        pE, pdir = real_secs[1][1], real_secs[1][2]
                        theta_p  = math.acos(np.clip(np.dot(inc_dir, pdir), -1.0, 1.0))
                        phi_p    = (math.atan2(pdir[1], pdir[0]) + 2*math.pi) % (2*math.pi)
                        cont_store[7] = norm(pE,      'energy')
                        cont_store[8] = norm(theta_p, 'theta')
                        cont_store[9] = norm(phi_p,   'phi')

        # 3. Apply proper masking based on the (possibly over-ridden) discrete choice
        cont_masked = _mask_cont(disc_store, cont_store)


        # 4. Store the original agent action for actor updates
        info["agent_action"] = np.concatenate(
            ([float(orig_choice)], cont.copy())
        ).astype(np.float32)

        # 5. Store the (potentially) overridden action for critic/target networks
        info["override_action"] = np.concatenate(
            ([float(disc_store)], cont_masked)
        ).astype(np.float32)
            
        names = ["rayleigh","compton","photo","pair"]
        # ─── 8.  Energy-conditioned JS/L1 reward ─────────────────
        # 1)  Which energy bin does this interaction belong to?
        logE   = math.log10(photon_energy_in + 1e-15)
        bin_idx = np.searchsorted(self.ebin_edges, logE, side="right") - 1
        bin_idx = max(0, min(bin_idx, len(self.ebin_edges) - 2))  # clamp
        info["phys_bin_idx"] = bin_idx

        # Define interaction name for per-bin tracking (used in phases 2+)
        interaction_names = ["rayleigh", "compton", "photo", "pair"]
        current_interaction = interaction_names[discrete_choice]
        # Initialize default values for variables used later
        p_true_all = self.true_prob.mean(axis=0)
        p_true = np.array(p_true_all)
        p_pred = np.zeros_like(p_true)
        js_div = 0.0
        l1_dist = 0.0
        # 2)  Update histograms
        if self.phase < 2:
            self.pred_hist *= self.hist_decay
            choice_agent = info.get("agent_action", info["override_action"])
            self.pred_hist[bin_idx, int(choice_agent[0])] += 1.0
            self.cum_pred_hist[bin_idx, int(choice_agent[0])] += 1.0   # or use actual_int
            # 2b) Update regime-limited histograms for phase-0 curriculum
            self.pred_hist_regime *= self.hist_decay
            self.pred_hist_regime[self.current_regime, int(choice_agent[0])] += 1.0
            self.cum_pred_hist_regime[self.current_regime, int(choice_agent[0])] += 1.0
            # ── debug: recompute summary distributions ───────────────
            counts_recent = self.pred_hist.sum(axis=0).astype(float)
            counts_total  = self.cum_pred_hist.sum(axis=0).astype(float)
            tot_recent = counts_recent.sum() or 1.0
            tot_total  = counts_total.sum() or 1.0
            p_pred_recent = counts_recent / tot_recent
            p_pred_total  = counts_total  / tot_total
            p_true_all = self.true_prob.mean(axis=0)
            bin_counts = self.cum_pred_hist.sum(axis=1).astype(int)
            if self.total_physics_steps % 100 == 0:          # every 1 k photons
                print("Bin populations:", self.cum_pred_hist.sum(axis=1).astype(int))
                print(f"🔍 Current energy range: {self.E_min*1000:.1f} - {self.E_max*1000:.1f} keV, Global step: {self.global_step_count}")
            # 3)  Compute per-bin JS + L1, accumulate over non-sparse bins
            r_dist = 0.0
            # ── aggregate the energy-binned histograms ──────────────
            counts_recent = self.pred_hist.sum(axis=0).astype(float)     # EWMA
            counts_total  = self.cum_pred_hist.sum(axis=0).astype(float) # since t=0

            tot_recent = counts_recent.sum() or 1.0
            tot_total  = counts_total.sum()  or 1.0

            p_pred_recent = counts_recent / tot_recent + 1e-12
            p_pred_total  = counts_total  / tot_total  + 1e-12

            # keep the recent histogram for the reward signal ------------------
            p_pred_all = p_pred_recent          # <- this replaces the old line
            counts_pred = counts_recent         # <-     "
            bin_counts = self.cum_pred_hist.sum(axis=1) 
            valid_mask = (bin_counts >= 10.0).astype(float) 
            weights      = self.pred_hist.sum(axis=1) * valid_mask
            total_w   = weights.sum() or 1.0
            # ensure our weights and true_prob share the same bin count
            assert weights.shape[0] == self.N_EBINS, (
                f"Shape mismatch: weights {weights.shape[0]} vs N_EBINS {self.N_EBINS}"
            )
            p_true_all = (weights[:, None] * self.true_prob).sum(axis=0) / (weights.sum() or 1.0)


            p_pred_all = counts_pred / tot_recent + 1e-12
            p_true = np.array(p_true_all)
            p_pred = np.array(p_pred_all)


            eps = 1e-6
            pt = p_true + eps
            pp = p_pred + eps
            m = 0.5 * (pt + pp)
    
            # Calculate the JS divergence with explicit handling
            js_div = 0.0
            for i in range(len(pt)):
                if pt[i] > eps:  # Only include non-zero terms
                    js_div += 0.5 * pt[i] * np.log(pt[i] / m[i])
                if pp[i] > eps:  # Only include non-zero terms
                    js_div += 0.5 * pp[i] * np.log(pp[i] / m[i])
    
            l1_dist = np.sum(np.abs(p_true_all - p_pred_all))



            baseline = 0.25   
            eps      = 1e-12
            # scale (≈ keep previous magnitude)
            r_p   = math.log((p_true[discrete_choice] + eps) /
                                (baseline                 + eps))
        else:
            r_p = 0.0
        r_disc = r_p 
        
        # KERNEL REWARDS
        Eb_list = [PHOTO_SHELL_BINDINGS[shell]*1e-6 for shell in ("H_K","O_K","O_L1","O_L2","O_L3")]
        r_phi = 0 
        R_phi = 0
        r_phi_e = 0 
        R_phi_e = 0
        r_theta_pair = 0.0
        r_e_comp = 0.0
        r_E_pair = 0.0
        r_ang = 0.0
        r_kernel = 0.0
        r_dist = 0.0
        if self.phase in (2,3):
            # -------------------------------------------------------
            # 1) Dense acceptance reward       r_ang
            # -------------------------------------------------------
            interaction_names = ["rayleigh", "compton", "photo", "pair"]
            current_interaction = interaction_names[discrete_choice]
            r_dist = 0.0
            if discrete_choice == 0:        # Rayleigh
                acc, target_dist = accept_prob(
                        0,
                        photon_energy_in,
                        math.cos(photon_theta_pred),
                        self.data)
                # Distribution matching reward for scattered photon angle
                angle_theta = photon_theta_pred
                # The acceptance probability is proportional to the true probability density
                p_true_angle = acc  # This is already the physics kernel value
                # Baseline: uniform probability over [0, π] gives density 1/π
                baseline_angle = 1.0 / 180.0
                eps_angle = 1e-12
                lambda_E_rayleigh = 0.01
                E_error_rel = abs(E_pred - photon_energy_in) / max(photon_energy_in, 1e-6)
                r_E_rayleigh = - lambda_E_rayleigh * E_error_rel
                # Log-probability ratio reward (similar to discrete actions)
                r_dist = math.log((p_true_angle + eps_angle) / (baseline_angle + eps_angle))
                angle_degrees = math.degrees(photon_theta_pred)

                # Per-bin angle tracking for phases 2+
                if len(self.angle_hist_per_bin[bin_idx][current_interaction]) > 50:
                    angles = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                    keep_count = max(20, int(len(angles) * self.hist_decay))
                    self.angle_hist_per_bin[bin_idx][current_interaction] = deque(
                        angles[-keep_count:], maxlen=200
                    )
                self.angle_hist_per_bin[bin_idx][current_interaction].append(angle_degrees)
                self.angle_target_per_bin[bin_idx][current_interaction] = target_dist
                
                # Compute per-bin KL divergence
                angles_in_bin = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                if len(angles_in_bin) >= 20:
                    hist, _ = np.histogram(angles_in_bin, bins=180, range=(0, 180), density=True)
                    hist = hist + 1e-10
                    hist = hist / np.sum(hist)
                    
                    target_dist_norm = target_dist + 1e-10
                    target_dist_norm = target_dist_norm / np.sum(target_dist_norm)
                    
                    kl_div = np.sum(target_dist_norm * np.log(target_dist_norm / hist))
                    self.angle_kl_per_bin[bin_idx][current_interaction] = kl_div
                    
                    r_angle_bin = -0.05 * np.clip(kl_div, 0, 10.0)
                    r_dist += r_angle_bin
                
                r_dist += r_E_rayleigh
            elif discrete_choice == 1:      # Compton
                acc, target_dist = accept_prob(
                        1,
                        photon_energy_in,
                        E_pred,
                        math.cos(photon_theta_pred),
                        self.comp_sampler_cache[bin_idx])
                # Distribution matching reward for scattered photon angle
                angle_theta = photon_theta_pred
                # The acceptance probability is proportional to the true probability density
                p_true_angle = acc  # This is already the physics kernel value
                # Baseline: uniform probability over [0, π] gives density 1/π
                baseline_angle = 1.0 / 180.0
                eps_angle = 1e-12
    
                # Log-probability ratio reward (similar to discrete actions)
                r_dist = math.log((p_true_angle + eps_angle) / (baseline_angle + eps_angle))
                angle_degrees = math.degrees(photon_theta_pred)

                # Per-bin angle tracking for phases 2+
                if len(self.angle_hist_per_bin[bin_idx][current_interaction]) > 50:
                    angles = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                    keep_count = max(20, int(len(angles) * self.hist_decay))
                    self.angle_hist_per_bin[bin_idx][current_interaction] = deque(
                        angles[-keep_count:], maxlen=200
                    )
                self.angle_hist_per_bin[bin_idx][current_interaction].append(angle_degrees)
                self.angle_target_per_bin[bin_idx][current_interaction] = target_dist
                
                # Compute per-bin KL divergence
                angles_in_bin = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                if len(angles_in_bin) >= 20:
                    hist, _ = np.histogram(angles_in_bin, bins=180, range=(0, 180), density=True)
                    hist = hist + 1e-10
                    hist = hist / np.sum(hist)
                    
                    target_dist_norm = target_dist + 1e-10
                    target_dist_norm = target_dist_norm / np.sum(target_dist_norm)
                    
                    kl_div = np.sum(target_dist_norm * np.log(target_dist_norm / hist))
                    self.angle_kl_per_bin[bin_idx][current_interaction] = kl_div
                    
                    r_angle_bin = -0.05 * np.clip(kl_div, 0, 10.0)
                    r_dist += r_angle_bin
            elif discrete_choice == 2:      # Photo
                shell_idx  = int(np.argmax(shell_onehot))
                shell_name = self.shell_names[shell_idx]
                acc, target_dist = accept_prob(
                        2,
                        photon_energy_in,
                        math.cos(theta_e),
                        shell=shell_name)
                angle_theta = theta_e  # Use electron angle for photoelectric
                p_true_angle = acc
                baseline_angle = 1.0 / 180.0
                eps_angle = 1e-12 
                r_dist = math.log((p_true_angle + eps_angle) / (baseline_angle + eps_angle))
                # Distribution matching reward for ejected electron angle
                predicted_electron_energy = sec_params[0][1]  # This is avail_E from line ~450
                lambda_E_photo = 0.01
                # The electron MUST get exactly the available energy
                E_error_rel = abs(predicted_electron_energy - avail_E) / max(avail_E, 1e-6)
                r_E_photo = -lambda_E_photo * E_error_rel

                angle_degrees = math.degrees(theta_e)
                # Per-bin angle tracking for phases 2+
                if len(self.angle_hist_per_bin[bin_idx][current_interaction]) > 50:
                    angles = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                    keep_count = max(20, int(len(angles) * self.hist_decay))
                    self.angle_hist_per_bin[bin_idx][current_interaction] = deque(
                        angles[-keep_count:], maxlen=200
                    )
                self.angle_hist_per_bin[bin_idx][current_interaction].append(angle_degrees)
                self.angle_target_per_bin[bin_idx][current_interaction] = target_dist
                
                # Compute per-bin KL divergence
                angles_in_bin = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                if len(angles_in_bin) >= 20:
                    hist, _ = np.histogram(angles_in_bin, bins=180, range=(0, 180), density=True)
                    hist = hist + 1e-10
                    hist = hist / np.sum(hist)
                    
                    target_dist_norm = target_dist + 1e-10
                    target_dist_norm = target_dist_norm / np.sum(target_dist_norm)
                    
                    kl_div = np.sum(target_dist_norm * np.log(target_dist_norm / hist))
                    self.angle_kl_per_bin[bin_idx][current_interaction] = kl_div
                    
                    r_angle_bin = -0.05 * np.clip(kl_div, 0, 10.0)
                    r_dist += r_angle_bin
            else:                           # Pair
                acc, target_dist = accept_prob(
                        3,
                        photon_energy_in,
                        math.cos(theta_e))
                angle_theta = theta_e  # Use electron angle for pair production
                p_true_angle = acc
                baseline_angle = 1.0 / 180.0
                eps_angle = 1e-12
                r_dist = math.log((p_true_angle + eps_angle) / (baseline_angle + eps_angle))
                # Distribution matching reward for electron angle in pair production
                angle_degrees = math.degrees(theta_e)
                # Per-bin angle tracking for phases 2+
                if len(self.angle_hist_per_bin[bin_idx][current_interaction]) > 50:
                    angles = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                    keep_count = max(20, int(len(angles) * self.hist_decay))
                    self.angle_hist_per_bin[bin_idx][current_interaction] = deque(
                        angles[-keep_count:], maxlen=200
                    )
                self.angle_hist_per_bin[bin_idx][current_interaction].append(angle_degrees)
                self.angle_target_per_bin[bin_idx][current_interaction] = target_dist
                
                # Compute per-bin KL divergence
                angles_in_bin = list(self.angle_hist_per_bin[bin_idx][current_interaction])
                if len(angles_in_bin) >= 20:
                    hist, _ = np.histogram(angles_in_bin, bins=180, range=(0, 180), density=True)
                    hist = hist + 1e-10
                    hist = hist / np.sum(hist)
                    
                    target_dist_norm = target_dist + 1e-10
                    target_dist_norm = target_dist_norm / np.sum(target_dist_norm)
                    
                    kl_div = np.sum(target_dist_norm * np.log(target_dist_norm / hist))
                    self.angle_kl_per_bin[bin_idx][current_interaction] = kl_div
                    
                    r_angle_bin = -0.05 * np.clip(kl_div, 0, 10.0)
                    r_dist += r_angle_bin
            # ---------------- Photon φ-uniformity reward -------------------
            if discrete_choice in (0, 1):  # Rayleigh or Compton
                self._phi_c_acc += cmath.exp(1j * photon_phi_pred)
                self._phi_c_cnt += 1
                if self._phi_c_cnt >= 32:      # small running batch
                    R_phi  = abs(self._phi_c_acc / self._phi_c_cnt)
                    r_phi  = - self.lambda_phi * R_phi
                    # reset accumulators
                    self._phi_c_acc = 0+0j
                    self._phi_c_cnt = 0
                else:
                    R_phi = 0.0
                    r_phi = 0.0
            else:
                # For photoelectric and pair production, no φ-uniformity reward
                # since there's no scattered photon
                R_phi = 0.0
                r_phi = 0.0
            # ---------------- Electron/positron φ-uniformity reward -------------------
            if discrete_choice in (1, 2, 3):  
                self._phi_e_acc += cmath.exp(1j * phi_e)
                self._phi_e_cnt += 1
                # Applies to Compton, Photo-electric and Pair electrons
                if self._phi_e_cnt >= 32:          # same window length
                    R_phi_e = abs(self._phi_e_acc / self._phi_e_cnt)
                    r_phi_e = - self.lambda_phi * R_phi_e

                    self._phi_e_acc = 0+0j
                    self._phi_e_cnt = 0
                else:
                    R_phi_e = 0.0
                    r_phi_e = 0.0       
            
            else:
                R_phi_e = 0.0
                r_phi_e = 0.0

            if discrete_choice == 3:                      # pair-production positron
                self._phi_p_acc += cmath.exp(1j * phi_p)
                self._phi_p_cnt += 1

                if self._phi_p_cnt >= 32:
                    R_phi_p = abs(self._phi_p_acc / self._phi_p_cnt)
                    r_phi_p = - self.lambda_phi * R_phi_p

                    self._phi_p_acc = 0+0j
                    self._phi_p_cnt = 0
                else:
                    R_phi_p = 0.0
                    r_phi_p = 0.0
            else:
                R_phi_p = 0.0
                r_phi_p = 0.0

            # ------------------------------------------------------------------
            # 2) Compton e⁻ kinematic consistency           r_e_comp
            # ------------------------------------------------------------------
            r_e_comp = 0.0
            if discrete_choice == 1:     # Compton only
                # --- MC expectations from the *photon* part the agent has proposed
                Ee_mc   = photon_energy_in - E_pred                            # MeV
                # protect against division by zero
                if Ee_mc > 1e-15:
                    p_e_mc = math.sqrt(Ee_mc * (Ee_mc + 2*mec2))              # |p_e|
                    cos_th_e_mc = (photon_energy_in - E_pred*math.cos(photon_theta_pred)) \
                                  / max(p_e_mc, 1e-12)
                    cos_th_e_mc = np.clip(cos_th_e_mc, -1.0, 1.0)
                    th_e_mc = math.acos(cos_th_e_mc)
                else:
                    th_e_mc = 0.0                                             # arbitrary

                # --- agent’s electron proposal (already denormalised above)
                Ee_pred = Ej
                th_e_pred = theta_e

                # --- penalties (relative errors, scaled to O(1))
                lambda_E  = 0.05
                lambda_th = 0.05
                dE_rel   = abs(Ee_pred - Ee_mc) / max(Ee_mc, 1e-6)
                dth_rel  = abs(th_e_pred - th_e_mc) / math.pi                 # 0…1

                r_e_comp = - (lambda_E * dE_rel + lambda_th * dth_rel)
            # --------------------------------------------------------------
            # Pair production: energy-share consistency      r_E_pair
            # --------------------------------------------------------------
            r_E_pair = 0.0
            if discrete_choice == 3:      # only for Pair
                avail = photon_energy_in - 2*mec2                  # kinetic energy sum
                if avail > 1e-6:
                    diff = abs(Ej_e + Ej_p - avail) / avail
                    lambda_E_p  = 0.05                                     # tune later
                    r_E_pair = - lambda_E_p * diff
            # --------------------------------------------------------------
            # Pair: electron polar-angle reward          r_theta_pair
            # --------------------------------------------------------------
            r_theta_pair = 0.0
            if  discrete_choice == 3:
                theta_mc   = self._small_angle(photon_energy_in)   # Monte-Carlo reference
                if theta_mc > 1e-6:
                    rel_err = abs(theta_e - theta_mc) / theta_mc
                    lambda_theta_pair = 0.5          # tune (start 0.2-0.5)
                    r_theta_pair = - lambda_theta_pair * rel_err
            #---------------------------------------------------------------
            r_kernel = r_dist + r_phi + r_phi_e + r_phi_p + r_theta_pair + r_E_pair + r_e_comp
            r_kernel = max(min(r_kernel, 50.0), -50.0) 
        else:
            r_kernel = 0.0
            r_e_comp = 0.0
            r_dist = 0.0
            r_phi = 0.0
            r_phi_e = 0.0
            r_theta_pair = 0.0
            r_E_pair = 0.0
            r_e_comp = 0.0
            
            
        
        # 6) Conservation + Q
        Q_photo = sum(h*eb for h,eb in zip(shell_onehot, Eb_list))
        Q_pair  = 2*mec2
        Q = Q_photo if discrete_choice==2 else (Q_pair if discrete_choice==3 else 0.0)

        if discrete_choice == 2:            # photoelectric

            Q_pred = sum(h*eb for h, eb in zip(shell_onehot, Eb_list))
        elif discrete_choice == 3:          # pair production
 
            Q_pred = 2 * mec2
        else:
            Q_pred = 0.0
        pred_sec_sum = sum(p[1] for p in sec_params)

        # Get appropriate KL divergence based on phase
        if self.phase >= 2:
            # Use per-bin KL divergence for phases 2+
            interaction_name = ["rayleigh", "compton", "photo", "pair"][discrete_choice]
            current_kl = self.angle_kl_per_bin.get(bin_idx, {}).get(interaction_name, 0.0)
        else:
            # No KL tracking in phases 0-1 (discrete learning only)
            current_kl = 0.0

   
        info = {
            "phys_fp":   dist_real,
            "phys_ang":  angle_radians,
            "phys_n_sec": len(real_secs),
            "phys_Eout": Eout,
            "phys_proc":  PROC_NAMES.index(itype),
            "r_disc":      r_disc,
            "r_kernel":    r_kernel,
            "r_E_corr":    r_E_corr,
            "r_dist":      r_dist,  
            "kl_div":      current_kl
        }

        # 12) Total reward
        reward = (r_disc + r_kernel+ r_E_corr)

        # 13) Logging
        if self.total_physics_steps % 100 == 0:
            if self.phase in (0,1):
                names = ["rayleigh", "compton", "photo", "pair"]

 
                print(f"\n─ Histogram over ALL energy bins  ")

                print("                Ground truth        Agent choice        Overall")
                for i, name in enumerate(names):
                    pct_true   = 100.0 * p_true_all[i]       # analytic truth
                    pct_recent = 100.0 * p_pred_recent[i]    # last ~window
                    pct_total  = 100.0 * p_pred_total[i]     # since step-0
                    print(f"  {name:9s}:   {pct_true:5.1f}%           {pct_recent:5.1f}%           {pct_total:5.1f}%")   

                print(f"  js_div={js_div:.5f}   L1_dist={l1_dist:.5f}     r_disc={r_disc:.3f}")
                print("-------------------------------------------------")
                self._print_energy_band_stats()
            elif self.phase in (2, 3):
                names = ["rayleigh", "compton", "photo", "pair"]
                print(f"\n─ Per-Bin Angle Learning (Phase {self.phase}) ─")
                print(f"Current regime: {self.current_regime+1}/{self.num_energy_regimes}")
                print(f"Energy range: {self.E_min*1000:.1f} - {self.E_max*1000:.1f} keV")
                
                # Get current regime bins
                e_low = self.energy_regime_boundaries[self.current_regime]
                e_high = self.energy_regime_boundaries[self.current_regime + 1]
                regime_bins = []
                for b_idx in range(len(self.ebin_edges) - 1):
                    e_min = 10**self.ebin_edges[b_idx]
                    e_max = 10**self.ebin_edges[b_idx+1]
                    if (e_min <= e_high and e_max >= e_low):
                        regime_bins.append(b_idx)
                
                print(f"Regime bins: {len(regime_bins)} bins")
                print("Interaction    Bins w/ Data    Avg KL Div    Avg Samples")
                print("-" * 60)
                
                for name in names:
                    bins_with_data = 0
                    total_kl = 0.0
                    total_samples = 0
                    
                    for b_idx in regime_bins:
                        sample_count = len(self.angle_hist_per_bin[b_idx][name])
                        if sample_count >= 10:  # Min samples to be meaningful
                            bins_with_data += 1
                            total_kl += self.angle_kl_per_bin[b_idx][name]
                            total_samples += sample_count
                    
                    avg_kl = total_kl / max(bins_with_data, 1)
                    avg_samples = total_samples / max(bins_with_data, 1)
                    
                    print(f"{name:12s}   {bins_with_data:>3d}/{len(regime_bins):<3d}        "
                          f"{avg_kl:>8.4f}      {avg_samples:>8.1f}")
                
                print(f"\nr_dist: {r_dist:.3f}, r_kernel: {r_kernel:.3f}, "
                      f"r_phi={r_phi:+.4f}, r_phi_e: {r_phi_e:.3f}")
                print(f"r_E_pair: {r_E_pair:.3f}, r_theta_pair: {r_theta_pair:.3f}, r_e_comp: {r_e_comp:.3f}")

                
                # Show per-bin distribution matching metrics for current regime
                print("\n" + "="*70)
                print(f"PER-BIN DISTRIBUTION MATCHING - REGIME {self.current_regime+1}")
                print(f"Energy Range: {self.E_min*1000:.1f} - {self.E_max*1000:.1f} keV")
                print("="*70)
                
                # Get current regime bins
                e_low = self.energy_regime_boundaries[self.current_regime]
                e_high = self.energy_regime_boundaries[self.current_regime + 1]
                regime_bins = []
                for b_idx in range(len(self.ebin_edges) - 1):
                    e_min = 10**self.ebin_edges[b_idx]
                    e_max = 10**self.ebin_edges[b_idx+1]
                    if (e_min <= e_high and e_max >= e_low):
                        regime_bins.append(b_idx)
                
                # Show metrics for bins with data
                bins_with_data = []
                for b_idx in regime_bins:
                    bin_has_data = False
                    for interaction in names:
                        if len(self.angle_hist_per_bin[b_idx][interaction]) >= 10:
                            bin_has_data = True
                            break
                    if bin_has_data:
                        bins_with_data.append(b_idx)
                
                if bins_with_data:
                    for b_idx in bins_with_data[:5]:  # Show first 5 bins to avoid spam
                        e_bin_min = 10**self.ebin_edges[b_idx] * 1000  # Convert to keV
                        e_bin_max = 10**self.ebin_edges[b_idx+1] * 1000
                        
                        print(f"\nBin {b_idx}: {e_bin_min:.1f}-{e_bin_max:.1f} keV")
                        print(f"{'Interaction':<12} | {'Samples':<8} | {'KL Div':<8} | {'Reward':<8}")
                        print(f"{'-'*12} | {'-'*8} | {'-'*8} | {'-'*8}")
                        
                        for interaction in names:
                            sample_count = len(self.angle_hist_per_bin[b_idx][interaction])
                            kl_div = self.angle_kl_per_bin[b_idx][interaction]
                            
                            # Calculate reward (same formula as in step)
                            reward = -0.1 * np.clip(kl_div, 0, 10.0) if sample_count >= 20 else 0.0
                            
                            if sample_count >= 10:  # Only show if meaningful data
                                print(f"{interaction:<12} | {sample_count:>8d} | {kl_div:>8.4f} | {reward:>+8.4f}")
                else:
                    print("No bins with sufficient data yet")
                
                # Show visual histograms
                print("\n" + "="*60)
                print("ANGLE DISTRIBUTION HISTOGRAMS")
                print("="*60)
                self._print_regime_angle_histograms()
            else:
                # Phase 4+: Complete stats
                print(f"r_dist: {r_dist:.3f}, r_kernel: {r_kernel:.3f}," +
                      f"  r_phi={r_phi:+.4f} r_phi_e: {r_phi_e:.3f}," +
                      f"  r_E_pair: {r_E_pair:.3f}, r_theta_pair: {r_theta_pair:.3f}, r_e_comp: {r_e_comp:.3f}. ")

    
            # Always print interaction and energy info
            print(f"  MC interaction  = {names[actual_int]}")
            print(f"  predicted interaction  = {names[discrete_choice]}")
            print(f"  E_in   = {photon_energy_in:.3f}")
            print(f"  Eout   = {Eout:.3f}")
            print(f"  E_pred = {E_pred:.3f}")
            print(f"  μ_real = {mu_real:.5e}   μ_pred = {mu_pred:.5e}")
            # ─────────────── μ log ───────────────
            print(f"r_disc: {r_disc:.3f}, r_dist: {r_dist:.3f}, r_kernel: {r_kernel:.3f}," +
                  f"  r_phi={r_phi:+.4f} r_phi_e: {r_phi_e:.3f}," +
                  f"  r_E_pair: {r_E_pair:.3f}, r_theta_pair: {r_theta_pair:.3f}, r_e_comp: {r_e_comp:.3f}. ")


            # Always print phase
            print(f"  Phase = {self.phase}")
            # secondaries: true vs. predicted
            # inc_dir is the photon incident direction
            inc_dir = np.array([self.u, self.v, self.w], dtype=float)
            for idx in range(self.NsecMax):
                # true secondary
                if idx < len(real_secs):
                    _, tE, tdir, _ = real_secs[idx]
                    ttheta = math.degrees(math.acos(np.clip(np.dot(inc_dir, tdir), -1.0, 1.0)))
                else:
                    tE, ttheta = 0.0, 0.0

                # predicted secondary
                if idx < len(sec_params):
                    _, pE, pdir, _ = sec_params[idx]
                    ptheta = math.degrees(math.acos(np.clip(np.dot(inc_dir, pdir), -1.0, 1.0)))
                else:
                    pE, ptheta = 0.0, 0.0

                print(f"  Sec{idx}: true E={tE:.3f}, true θ={ttheta:.2f}° | "
                     f"pred E={pE:.3f}, pred θ={ptheta:.2f}°")
                print("env.phase =", self.phase)

                    
        # 14) Done / PDD bonus
        term = not self.alive
        trunc = (self.steps >= self.max_steps)
        if term or trunc:
            if np.linalg.norm(self.dose_tally - self.target_pdd) < self.pdd_tol:
                reward += self.episode_bonus
        return self._get_obs(), float(reward), term, trunc, info if self.use_gymnasium_api else (self._get_obs(), float(reward), term or trunc, info)

###############################################################################
#                               PHASING
###############################################################################
class PhasedRewardEnv(WaterPhotonHybridEnvPenelope):
    global PHASE_ENDS
    def __init__(self, *args, phase_ends=None, **kwargs):
        """
        phase_ends: list of 3 integers [end_disc, end_cons, end_shape]
        """
        super().__init__(*args, **kwargs)
        # ─────────────────────────────────────────────────────────────
        #  Scheduled-sampling state
        #    • Phase 0 : always teacher-forcing
        #    • Phase 1 : probability decays 1 → 0 in 2 × 10⁵ steps
        #    • Phase ≥2: always teacher-forcing again
        # ─────────────────────────────────────────────────────────────
        self.t_force_prob        = 1.0          # will be reset when Phase 1 starts
        self.t_force_decay_steps = 30_000      # linear-decay horizon
        self.force_mc_interaction = True        # Phase 0 begins fully teacher-forced
        # default phase boundaries if none provided
        self.phase_ends = PHASE_ENDS
        self.global_step = 0
        self._phase_start_step = 0             # step at which current phase began

    def step(self, action):
        # 1) perform underlying step
        obs_real, rew_full, done, trunc, info = super().step(action)

        base = self.env.unwrapped if hasattr(self, "env") else self          # ← NEW
        self.global_step = base.global_step_count

        # 2) pick which component to return
        if  self.global_step < self.phase_ends[0]:
            reward = info.get("r_disc", 0.0)
        elif self.global_step < self.phase_ends[1]:
            reward = info.get("r_disc", 0.0)
        elif self.global_step < self.phase_ends[2]:
            reward = info.get("r_kernel", 0.0) + info.get("r_E_corr", 0.0) 
#            print(f"r_kernel: {info.get('r_kernel', 0.0)}, final reward: {reward}")
        elif self.global_step < self.phase_ends[3]:
            reward = info.get("r_kernel", 0.0) + info.get("r_E_corr", 0.0)
        elif self.global_step < self.phase_ends[4]:
            reward = info.get("r_kernel", 0.0) + info.get("r_E_corr", 0.0)
        elif self.global_step < self.phase_ends[5]:
            reward = info.get("r_kernel", 0.0) + info.get("r_E_corr", 0.0)
        elif self.global_step < self.phase_ends[6]:
            reward = info.get("r_kernel", 0.0) + info.get("r_E_corr", 0.0)
        elif self.global_step < self.phase_ends[7]:
            reward = info.get("r_kernel", 0.0) + info.get("r_E_corr", 0.0)
        else:
            reward = rew_full - info.get("r_disc", 0.0)  

        # 3) increment step counter
        self.global_step += 1
        # ── scheduled-sampling update for both Phase 1 and Phase 3 ─────────────
        if self.phase in (1, 3):
            decay_steps = self.t_force_decay_steps         # 30 k by default
            steps_in_phase = max(self.global_step - self._phase_start_step, 0)
            if steps_in_phase < decay_steps:
                self.t_force_prob = 1.0 - steps_in_phase / decay_steps
            else:
                self.t_force_prob = 0.0
    
            # decide whether the *next* interaction is teacher‑forced
            # For Phase 1: controls discrete override
            # For Phase 3: controls MFP override
            self.force_mc_interaction = (random.random() < self.t_force_prob)

        # every other phase: always force MC for discrete choice in phases 0,2
        # and always force MFP in phase 2
        elif self.phase not in (1, 3):
            self.force_mc_interaction = True
            
        # 5) build correct return values without crashing in eval vs train
        if self.use_gymnasium_api:
            # Gymnasium expects: obs, reward, terminated, truncated, info
            return obs_real, reward, done, trunc, info
        else:
            # legacy Gym/your eval code expects: obs, reward, done_flag, info
            done_flag = done or trunc
            return obs_real, reward, done_flag, info


###############################################################################
# 1) Big Feature Extractor
###############################################################################
class OptimizedFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor with a deep MLP.
    Input: flattened observation vector (obs_dim).
    Output: a 512-dimensional feature vector.

    Architecture: 
      - Linear(obs_dim -> 1024) -> LayerNorm -> SiLU
      - Linear(1024 -> 512) -> LayerNorm -> SiLU
      - Linear(512 -> 512) -> LayerNorm -> SiLU
      - plus a residual connection from obs -> final 512
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim=512):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            LayerNorm(1024),
            SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            LayerNorm(512),
            SiLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(512, 512),
            LayerNorm(512),
            SiLU()
        )

        # For a skip-connection from input_dim -> final 512
        self.res_conn = nn.Linear(input_dim, 512)


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.block1(obs)
        x = self.block2(x)
        x = self.block3(x)
        # Residual from raw obs -> final 512
        return x + self.res_conn(obs)

###############################################################################
# 2) Distribution for Hybrid Actions (Discrete + Continuous)
###############################################################################
class HybridCategoricalDiagGaussianDistribution(Distribution):
    """
    Merges:
      - 4 logits for the discrete action (Categorical)
      - Exp sample + mean + log_std for mean-free-path neuron
      - means + log_stds for the remainder continuous sub-action (DiagGaussian)
    The final sampled action is a 1D vector: [discrete_float, cont_0, cont_1, ...].
    """
    def __init__(self, n_discrete: int, n_continuous: int):
        super().__init__()
        self.n_discrete = n_discrete
        self.n_continuous = n_continuous
        self.cat_dist = CategoricalDistribution(self.n_discrete)
        self.gauss_dist = DiagGaussianDistribution(self.n_continuous)

        # total = discrete logits + means + log_stds
        self.total_dim = n_discrete + 2*n_continuous + 1

    def proba_distribution(self, params: torch.Tensor) -> "HybridCategoricalDiagGaussianDistribution":
        # params should be [ ..., n_discrete + 2*n_continuous + 1 ]
        assert params.shape[-1] == self.total_dim, (
            f"Got {params.shape[-1]}, expected {self.total_dim}"
        )
        # keep for sampling later
        self._last_params = params

        # split points
        cat_end    = self.n_discrete
        mu_end     = cat_end + self.n_continuous
        logstd_end = mu_end   + self.n_continuous
        theta_end  = logstd_end + 1

        # slice out each piece
        cat_logits = params[...,           :cat_end]
        mu         = params[..., cat_end  :mu_end    ]
        log_std    = params[..., mu_end   :logstd_end]
        # grab the one θ-param as a length-1 slice, then squeeze it to [B,...]
        mu_param  = params[..., logstd_end:theta_end].squeeze(-1)

        # clamp for stability
        cat_logits = torch.clamp(cat_logits, -50, 50)
        log_std    = torch.clamp(log_std,    -6, 2)
        # (optionally you could clamp theta_mfp here if needed)

        # build the underlying dists
        self.cat_dist.proba_distribution(cat_logits)
        self.gauss_dist.proba_distribution(mu, log_std)
        # store the exponential rate for later sampling
        self.mu_param = torch.clamp(mu_param, 7.0e-2, 5.00e7)  # Physical bound

        return self

        
    def proba_distribution_net(
        self, 
        *args, 
        **kwargs
    ) -> "HybridCategoricalDiagGaussianDistribution":
        """
        Since SB3 calls this abstract method,
        we just forward everything to our .proba_distribution() or do nothing if not used.
        """
        return self.proba_distribution(*args, **kwargs)
    def sample(self) -> torch.Tensor:
        # 1) sample discrete & continuous
        cat_sample   = self.cat_dist.sample().float()       # [B]
        gauss_sample = self.gauss_dist.sample()             # [B, n_continuous]
        gauss_sample = torch.tanh(gauss_sample)

        # 2) mask out unwanted continuous slots
        B, C = gauss_sample.shape
        full_mask = torch.zeros((B, C), dtype=torch.bool, device=gauss_sample.device)

        # mask energy slot for photo (2) & pair (3)
        if C > 1:
            mask_photo_pair = (cat_sample == 2) | (cat_sample == 3)  # [B]
            col1 = torch.arange(C, device=gauss_sample.device) == 1
            full_mask |= mask_photo_pair.unsqueeze(1) & col1.unsqueeze(0)
        # mask θ (col 2) and φ (col 3) for the same photo/pair events
        if C > 3:
            col_angles = (torch.arange(C, device=gauss_sample.device) == 2) | \
                         (torch.arange(C, device=gauss_sample.device) == 3)
            full_mask |= mask_photo_pair.unsqueeze(1) & col_angles.unsqueeze(0)
        # mask extra-secondaries for compton (1) & photo (2)
        mask_one_sec = (cat_sample == 1) | (cat_sample == 2)
        if C > 4:
            nsec = (C - 4) // 3
            cols = []
            for i in range(1, nsec):
                cols += [4 + 3*i, 4 + 3*i + 1, 4 + 3*i + 2]
            col_mask = torch.zeros(C, dtype=torch.bool, device=gauss_sample.device)
            col_mask[cols] = True
            full_mask |= mask_one_sec.unsqueeze(1) & col_mask.unsqueeze(0)

        gauss_masked = gauss_sample.masked_fill(full_mask, -1.0)  # [B, C]

        # 3) Use our direct rate prediction for MFP sampling
        # Extract the rate directly from self.mfp_rate which is already calculated in proba_distribution()
        mu_param = self.mu_param # [B]
        
        # Add safety clamping to prevent extreme rate values
        mu_param = torch.clamp(mu_param, 7.0e-2, 5.00e7)
        # Sample from exponential distribution using the clamped rate
        
        mfp_sample = torch.distributions.Exponential(mu_param).sample()  # [B]
        
        
        # 4) build the tail [mfp, theta] both as [B,1]
        mfp_sample = mfp_sample.unsqueeze(-1)   # [B,1]
        mu_param  = mu_param.unsqueeze(-1)      # [B,1]
        tail       = torch.cat([mfp_sample, mu_param], dim=1)  # [

        # 5) prepend the discrete choice and return
        cat_sample = cat_sample.unsqueeze(-1)   # [B,1]
        # return torch.cat([cat_sample, gauss_masked, tail.squeeze(-1)], dim=1)  # [B, 1 + C + 2]
        return torch.cat([cat_sample, gauss_masked, tail], dim=1)

    def mode(self) -> torch.Tensor:
        cat_mode = self.cat_dist.mode().float().unsqueeze(-1)
        gauss_mode = self.gauss_dist.mode()
        return torch.cat([cat_mode, gauss_mode], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # first element is the discrete choice
        discrete_index = actions[..., 0].long()
    
        # FIXED: Only include the actual Gaussian-distributed continuous variables
        # (exclude the MFP sample and theta_mfp at the end)
        continuous_part = actions[..., 1:1+self.n_continuous]
    
        # Separate MFP sample (follows an exponential distribution)
        mfp_sample = actions[..., -2]  # Second-to-last element
        mu_param = actions[..., -1]   # Last element (stored theta)
    
        # Compute log-probs separately
        logp_cat = self.cat_dist.log_prob(discrete_index)
        logp_gauss = self.gauss_dist.log_prob(continuous_part)
    
        # Compute log probability for MFP (Exponential distribution)
        mu_param = torch.clamp(mu_param, 7.0e-2, 5.00e7)
        # log_prob of exponential is log(rate) - rate*x
        logp_mfp = torch.log(mu_param + 1e-12) - mu_param * mfp_sample
    
        # Sum all log probabilities
        return logp_cat + logp_gauss + logp_mfp


    def entropy(self) -> torch.Tensor:
        return self.cat_dist.entropy() + self.gauss_dist.entropy()

    def actions_from_params(self, params: torch.Tensor, deterministic=False) -> torch.Tensor:
        self.proba_distribution(params)
        return self.mode() if deterministic else self.sample()

    def log_prob_from_params(self, params: torch.Tensor, actions: torch.Tensor):
        self.proba_distribution(params)
        return self.log_prob(actions), self.entropy()
        
###############################################################################
# 3) Custom NSTEPSAC
###############################################################################
class NStepSAC(SAC):
    """
    SAC that works with an n‑step replay buffer.
    Only change vs SB3:  use replay_data.discounts (γⁿ) in the TD‑target
    and make entropy‑coef a scalar so shapes match.
    """
    def __init__(self, *args,
                 lambda_phys: float = 1.0,
                 **kwargs):
        log_dir = kwargs.get("tensorboard_log", "runs")
        self.lambda_phys = 1.0  # Default value if not provided
        self.alpha_dual = 0.1    # Default value
        # Add physics loss tracking
        self.physics_losses = {
            "energy_loss": [],
            "angle_loss": [],
            "nsec_loss": [],
            "proc_loss": [],
            "norm_pen": [],
            "total_phys_loss": []
        }
        self.physics_steps = []  # Store timesteps for each recorded loss
        # Let SB3 build everything first
        super().__init__(*args, **kwargs)
        self._n_updates = 0
        self.tb_writer = SummaryWriter(log_dir=log_dir) 
    
    def _update_target_network(self) -> None:
        polyak_update(self.critic.parameters(),
                      self.critic_target.parameters(),
                      self.tau)


    # ------------------------------------------------------------------
    # main training loop
    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:


        if len(self.replay_buffer.infos) < batch_size:
            return        
        for _ in range(gradient_steps):
            if self._n_updates % 100 == 0:
                print(f"\nStep: {self.num_timesteps}")
                print(f"Buffer size: {self.replay_buffer.size()}")
             

                          
            # sample -------------------------------------------------------
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            def _to_dict(x):
                """
                Make sure we always have a plain dict:
                • x is already a dict       -> return it
                • x is [dict] or (dict,)    -> return x[0]
                • anything else             -> return {}
                """
                if isinstance(x, dict):
                    return x
                if isinstance(x, (list, tuple)) and x and isinstance(x[0], dict):
                    return x[0]
                return {}

            infos = [_to_dict(e) for e in replay_data.infos] 
            fp_t   = torch.tensor([e.get("phys_fp",   0.0) for e in infos],
                                  dtype=torch.float32, device=self.device)
            ang_t  = torch.tensor([e.get("phys_ang",  0.0) for e in infos],
                                  dtype=torch.float32, device=self.device)
            Eout_t = torch.tensor([e.get("phys_Eout", 0.0) for e in infos],
                                  dtype=torch.float32, device=self.device)
            nsec_t = torch.tensor([e.get("phys_n_sec",0   ) for e in infos],
                                  dtype=torch.float32, device=self.device)
            proc_t = torch.tensor([e.get("phys_proc", 0) for e in infos],
                                  dtype=torch.long, device=self.device)

            # 5) flat list of (E, θ, φ) for each of NsecMax secondaries
            secs_list = []
            for entry in infos:
                if entry is None:
                    secs_list.append([0.0]*3*self.actor.NsecMax)
                    continue
                row = []
                for k in range(self.actor.NsecMax):
                    e_val = float(entry.get(f"phys_s{k}_E", 0.0))
                    theta_val = float(entry.get(f"phys_s{k}_theta", 0.0))
                    phi_val = float(entry.get(f"phys_s{k}_phi", 0.0))
                    row.extend([e_val, theta_val, phi_val])
                secs_list.append(row)
            secs_t = torch.tensor(
                secs_list,
                dtype=torch.float32,
                device=self.device,
            )



   
            # 2) critic target -----------------------------------------------
            with torch.no_grad():
                a_next, logp_next = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                q1_next, q2_next = self.critic_target(
                    replay_data.next_observations, a_next
                )
                q_next = torch.min(q1_next, q2_next).squeeze(-1)          # (B,)

                # γⁿ from replay buffer:
                if self.log_ent_coef is None:
                    # fixed ent_coef case
                    ent_coef = torch.tensor(self.ent_coef, device=self.device)
                else:
                    ent_coef = torch.exp(self.log_ent_coef.detach())
                disc = replay_data.discounts.squeeze(-1)
                target_q = replay_data.rewards.squeeze(-1) + \
                           (1.0 - replay_data.dones.squeeze(-1)) * \
                           disc * (q_next - ent_coef * logp_next.squeeze(-1))     # (B,)

            # 3) critic update -----------------------------------------------
            q1, q2 = self.critic(replay_data.observations,
                                 replay_data.actions)          # (B,1)
            q1 = q1.squeeze(-1);   q2 = q2.squeeze(-1)          # (B,)

 

            lambda_aux = 0.1  

            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q) 
            self.tb_writer.add_scalar("critic_loss", critic_loss.cpu().item(), self.num_timesteps) 
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic.optimizer.step()

            # 4) actor update -------------------------------------------------
            # 4a) standard SAC actor loss (policy gradient + entropy)
            a_pi, logp_pi = self.actor.action_log_prob(replay_data.observations)
            q1_pi, q2_pi  = self.critic(replay_data.observations, a_pi)
            q_pi          = torch.min(q1_pi, q2_pi).squeeze(-1)

            if self.log_ent_coef is None:
                ent_coef_tensor = torch.tensor(self.ent_coef, device=self.device)
            else:
                ent_coef_tensor = torch.exp(self.log_ent_coef)
            ent_term = ent_coef_tensor * logp_pi
#            ent_term      = float(self.ent_coef) * logp_pi
            actor_loss_rl = (ent_term - q_pi).mean() # typical SAC actor-loss

            # 4b) supervised physics loss (MSE between predictions & true phys)
            feats = self.actor.features_extractor(replay_data.observations)
            phys_feat = self.actor.phys_backbone(feats)
            
            # Get interaction probabilities for conditioning
            proc_logits = self.actor.proc_head(phys_feat)
            proc_probs = torch.softmax(proc_logits, dim=-1)  # (B, 4)
            
            # Get energy bin conditioning (extract from observations)
            E_norm = replay_data.observations[:, 3]  # Energy is at index 3 in obs
            E_phys = E_norm * (1.001 - 0.001) + 0.001
            edges_t = torch.from_numpy(self.actor.ebin_edges).to(E_phys.device)
            logE = torch.log10(E_phys)
            logE = torch.clamp(logE, min=edges_t[0]+1e-6, max=edges_t[-1]-1e-6)
            bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1
            bin_idx = torch.clamp(bin_idx, min=0, max=len(edges_t)-2)
            energy_bin_onehot = torch.zeros((E_phys.shape[0], self.actor.n_energy_bins), 
                                           device=E_phys.device, dtype=torch.float32)
            energy_bin_onehot.scatter_(1, bin_idx.unsqueeze(1), 1.0)
            
            # Create conditioning input for physics heads
            conditioning = torch.cat([phys_feat, proc_probs, energy_bin_onehot], dim=-1)
            
            # Get physics predictions with conditioning
            energy_out_raw = self.actor.energy_head(conditioning)  # (B, n_energies * 4)
            angle_out_raw = self.actor.angle_head(conditioning)    # (B, 2 * n_angles * 4)
            nsec_logits = self.actor.nsec_head(phys_feat)
            
            # Reshape to interaction-specific outputs
            B = energy_out_raw.shape[0]
            energy_p_raw = energy_out_raw.view(B, self.actor.n_energies, self.actor.n_interactions)  # (B, n_energies, 4)
            angle_p_raw = angle_out_raw.view(B, 2 * self.actor.n_angles, self.actor.n_interactions)  # (B, 2*n_angles, 4)

            # Select predictions for the true interaction type
            proc_true_idx = proc_t.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
            # Gather the predictions for the true interaction type
            energy_p_selected = torch.gather(energy_p_raw, 2, 
                                           proc_true_idx.expand(-1, self.actor.n_energies, -1)).squeeze(2)  # (B, n_energies)
            angle_p_selected = torch.gather(angle_p_raw, 2,
                                          proc_true_idx.expand(-1, 2 * self.actor.n_angles, -1)).squeeze(2)  # (B, 2*n_angles)
            
            eps = 1e-12
            energy_p_pos = torch.clamp(energy_p_selected, min=0.0) + eps
            energy_p = torch.log1p(energy_p_pos)
            angle_p = angle_p_selected

            # build matched targets + masks   (helper lives just above train())
            energy_t, angle_t, mask_sec, interaction_mask = build_phys_targets(
                    fp_t, ang_t, Eout_t, nsec_t, secs_t, proc_t,
                    self.actor.NsecMax, self.device)
            
            # --- losses ------------------------------------------------------
            # Interaction-aware losses - only compute loss for the actual interaction type
            
            #   • energy: interaction-specific loss with masking
            energy_diff = (energy_p - energy_t).pow(2)  # (B, n_energies)
            energy_weights = torch.ones_like(energy_diff)
            energy_weights[:, 2:] = mask_sec  # mask secondary energies
            loss_energy = (energy_diff * energy_weights).mean()

            #   • angles: interaction-specific loss  
            angle_diff = (angle_p - angle_t).pow(2)  # (B, 2*n_angles)
            angle_weights = torch.ones_like(angle_diff)
            # Mask secondary angles
            for i in range(1, self.actor.n_angles):
                angle_weights[:, 2*i:2*(i+1)] = mask_sec[:, i-1:i]
            loss_angle = (angle_diff * angle_weights).mean()

            #   • L2-norm regulariser ‖(sin,cos)‖₂→1 for angle pairs
            angle_pairs = angle_p.view(-1, angle_p.shape[1]//2, 2)  # (B, n_angles, 2)
            norm_pen = ((angle_pairs.pow(2).sum(-1) - 1).pow(2)).mean()

            #   • #-of-secondaries (CE over classes {0,1,2})
            loss_nsec = F.cross_entropy(nsec_logits, nsec_t.long().clamp(0, 2))

            #   • interaction type (4-way CE)
            loss_proc = F.cross_entropy(proc_logits, proc_t)

            # combine (you can keep loss_vec if you still print it)
            loss_vec  = torch.stack([loss_energy, loss_angle, loss_nsec, loss_proc])
            loss_phys = loss_energy + loss_angle + 0.4 * norm_pen + loss_nsec + loss_proc


            # Store physics losses for tracking (limit to latest 1000 points to avoid memory issues)
            max_history = 1000
            self.physics_losses["energy_loss"].append(loss_energy.cpu().item())
            self.physics_losses["angle_loss"].append(loss_angle.cpu().item())
            self.physics_losses["nsec_loss"].append(loss_nsec.cpu().item())
            self.physics_losses["proc_loss"].append(loss_proc.cpu().item())
            self.physics_losses["norm_pen"].append(norm_pen.cpu().item())
            self.physics_losses["total_phys_loss"].append(loss_phys.cpu().item())
            self.physics_steps.append(self.num_timesteps)

            # Keep only the most recent values to avoid memory issues
            if len(self.physics_losses["energy_loss"]) > max_history:
                for key in self.physics_losses:
                    self.physics_losses[key] = self.physics_losses[key][-max_history:]
                self.physics_steps = self.physics_steps[-max_history:]
            # 4c) total multitask actor loss
            # ─── physics head now reenabled ────────────────

            actor_loss = actor_loss_rl  + self.lambda_phys * loss_phys #<--- #!!!Warning: physics-head now reenabled
            self.tb_writer.add_scalar("physics/energy_loss", loss_energy.cpu().item(), self.num_timesteps)
            self.tb_writer.add_scalar("physics/angle_loss", loss_angle.cpu().item(), self.num_timesteps)
            self.tb_writer.add_scalar("physics/normalization_penalizer", norm_pen.cpu().item(), self.num_timesteps)
            self.tb_writer.add_scalar("physics/nsec_loss", loss_nsec.cpu().item(), self.num_timesteps)
            self.tb_writer.add_scalar("physics/total_loss", loss_phys.cpu().item(), self.num_timesteps)
            self.tb_writer.add_scalar("actor_loss", actor_loss.cpu().item(), self.num_timesteps)

            self.actor.optimizer.zero_grad()
 

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            
            self.actor.optimizer.step()

            with torch.no_grad():
                total_norm = 0.0
                for p in self.actor.discrete_head.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
            self.tb_writer.add_scalar("grad_norm/discrete_head", total_norm,
                                      self.num_timesteps)
            # 5) entropy‑coef update -----------------------------------------
            
            if self.ent_coef_optimizer is not None:
                # detach logp_pi to avoid second‑order grads
                ent_coef_loss = -(self.log_ent_coef * (logp_pi.detach() + self.target_entropy)).mean()

                self.tb_writer.add_scalar("train/ent_coef_loss", ent_coef_loss.cpu().item(), self.num_timesteps)
                with torch.no_grad():
                    current_ent = torch.exp(self.log_ent_coef)
                self.tb_writer.add_scalar("train/ent_coef", current_ent.cpu().item(), self.num_timesteps)
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.log_ent_coef.data.clamp_(-9.0, 2.0)
                # keep the fresh alpha tensor for the next iteration
                self.ent_coef = torch.exp(self.log_ent_coef.detach())
                # Add entropy component monitoring here
                with torch.no_grad():
                    # Discrete entropy (from the categorical distribution)
                    params, _ = self.actor.forward(replay_data.observations)
                    logits = params[:, :self.actor.n_discrete]
                    p_pol = torch.softmax(logits, dim=1)
                    discrete_entropy = -(p_pol * torch.log(p_pol + 1e-12)).sum(dim=1).mean()
    
                    # Continuous entropy (from the Gaussian components)
                    # This is approximate since we're assuming logp_pi contains both components
                    continuous_entropy = -logp_pi.mean() - discrete_entropy
    
                    # Log to TensorBoard
                    self.tb_writer.add_scalar("entropy/discrete", discrete_entropy.item(), self.num_timesteps)
                    self.tb_writer.add_scalar("entropy/continuous", continuous_entropy.item(), self.num_timesteps)
                    self.tb_writer.add_scalar("entropy/target", self.target_entropy, self.num_timesteps)
    
                    # Print to console every 100 updates
                    if self._n_updates % 1000 == 0:
                        print(f"Alpha: {self.ent_coef.item():.6f}")
                        print(f"Discrete entropy: {discrete_entropy.item():.4f}")
                        print(f"Continuous entropy: {continuous_entropy.item():.4f}")
                        print(f"Target entropy: {self.target_entropy:.4f}")
                        print(f"Entropy gap: {(discrete_entropy + continuous_entropy).item() + self.target_entropy:.4f}")
       

            # --- sanitize infos so we always have a plain dict for each sample ---
            raw_infos = replay_data.infos
            infos = []
            for entry in raw_infos:
                if isinstance(entry, dict):
                    infos.append(entry)
                elif isinstance(entry, (list, tuple)) and len(entry) == 1 and isinstance(entry[0], dict):
                    infos.append(entry[0])
                else:
                    infos.append({})

                
        # 6) Polyak target‑net update ------------------------------------
            self._update_target_network()
            self._n_updates += 1

            if self._n_updates % 100 == 0:
                print(">>> Δ-phys per-col:",
                      loss_vec.detach().cpu().numpy())
    def save(self, path: str, *args, **kwargs) -> None:

        physics_data = {
            "physics_losses": self.physics_losses,
            "physics_steps": self.physics_steps
        }
    

        # Just use SB3’s own save; drop the unsupported save_kwargs.
        super().save(path, exclude=["tb_writer"], **kwargs)
        # Save physics loss data separately
        physics_path = path.replace('.zip', '_physics_losses.pkl')
        with open(physics_path, 'wb') as f:
            pickle.dump(physics_data, f)
            
###############################################################################
# 3) Custom Actor (Discrete + Continuous heads)
###############################################################################
class HybridActor(Actor):
    """
    Actor that produces two sets of outputs from the 512-dim feature:
       - discrete_head => 4 logits for the discrete action
       - continuous_head => (mu + log_std) for the continuous sub-action
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        n_discrete: int,
        n_continuous: int,
        n_interactions: int,  
        features_extractor: nn.Module,
        features_dim: int = 512,
        *,                                   # ← keep everything AFTER this keyword-only
        ebin_edges,                          
        true_prob,                           
        true_mfp_mean,                       
        energy_regime_boundaries,            
        LOG_MIN,
        LOG_MAX,
        activation_fn=nn.SiLU,
        optimizer_class=None,
        optimizer_kwargs=None,
    ):
        # We pass net_arch=[] so the default MlpExtractor is not built
        super().__init__(
            observation_space,
            action_space,
            net_arch=[],
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=False,
            log_std_init=-3,
            full_std=True,
            use_expln=False,
            clip_mean=False,
            normalize_images=False,
        )
        print(f"[DEBUG] In HybridActor ctor: n_discrete={n_discrete}, n_continuous={n_continuous}")

        self.n_discrete = n_discrete
        self.n_continuous = n_continuous
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.prev_logits = {}    # maps energy‐bin index → last logits tensor
        self.freeze_tol = 0.10   # L1‐distance threshold
        self.freeze_eps = 0.05    # max allowable logit change when frozen
        # ───────── per‐bin clamp radius ─────────
        # start every bin at self.freeze_eps, decay over visits
        self.bin_eps = defaultdict(lambda: self.freeze_eps)
        self.locked_bins = set()    
        # Store the physics data
        device = next(self.parameters()).device if next(self.parameters(), None) is not None else torch.device('cpu')
        self.ebin_edges     = np.asarray(ebin_edges,     dtype=np.float32)

        # ----------------------------------------------------------------
        # 2. Register the bin-edge buffer and derive n_bins
        # ----------------------------------------------------------------
        self.register_buffer("bin_edges",
                             torch.from_numpy(self.ebin_edges).to(device))
        n_bins = len(self.ebin_edges) - 1        # guaranteed ≥ 1 now
        print("[HybridActor] n_bins =", n_bins, flush=True)

        self.register_buffer('bin_edges', torch.from_numpy(self.ebin_edges).to(device))
        n_bins = len(self.ebin_edges) - 1
        print(n_bins)
        self.true_prob = np.asarray(true_prob, dtype=np.float32)
        self.true_mfp_mean = np.asarray(true_mfp_mean, dtype=np.float32)
        self.energy_regime_boundaries = np.asarray(energy_regime_boundaries, dtype=np.float32)
        self.LOG_MIN = np.float32(LOG_MIN)
        self.LOG_MAX = np.float32(LOG_MAX)
        print("LOG_MIN:", self.LOG_MIN)
        print("LOG_MAX:", self.LOG_MAX)
#        print("LOG_RANGE:", model.policy.actor.LOG_RANGE)
        # Build heads
        # A) discrete_head
        self.discrete_head = nn.Sequential(
            nn.Linear(features_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, n_discrete)
        )
        # B) continuous_head
        self.continuous_head = nn.Sequential(
            nn.Linear(features_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 2 * n_continuous + 1)     # μ ⧺ log σ
        )
#        nn.init.zeros_(self.continuous_head[-1].weight)
#        nn.init.zeros_(self.continuous_head[-1].bias)
        self.NsecMax = (n_continuous - 4) // 3
        phys_dim  = 5 + 3 * self.NsecMax
        self.phys_backbone = nn.Sequential(
            nn.Linear(features_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        self.n_angles   = 1 + self.NsecMax            # photon + N secs
        self.n_energies = 2 + self.NsecMax            # log fp, log Eout, log E_sec_k
        self.n_interactions = n_interactions

        # ───────── branches ─────────
        self.n_energy_bins = len(self.ebin_edges) - 1

        # ───────── branches ─────────
        self.proc_head = nn.Linear(512, self.n_interactions)
        
        # Interaction + energy-bin aware heads
        # Input: phys_backbone(512) + interaction_probs(4) + energy_bin_onehot(n_bins)
        conditioning_dim = 512 + self.n_interactions + self.n_energy_bins
        
        self.energy_head = nn.Sequential(
            nn.Linear(conditioning_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.n_energies * self.n_interactions)  # interaction-specific outputs
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(conditioning_dim, 256), 
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 2 * self.n_angles * self.n_interactions)  # interaction-specific outputs
        )
        
        self.nsec_head = nn.Linear(512, 3)  # Keep unchanged

        # Create the combined distribution object
        self.action_dist = HybridCategoricalDiagGaussianDistribution(n_discrete, n_continuous)

        # Optional weight init
        for module in self.discrete_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        for module in self.continuous_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        for module in [*self.phys_backbone,
               self.energy_head,
               self.angle_head,
               self.nsec_head]:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        b = self.discrete_head[-1].bias      # shape (4,)
        with torch.no_grad():
            b.zero_()
        # Register the buffers during initialization so they exist when loading state_dict
        self.register_buffer('logits_buffer', torch.zeros((n_bins, self.n_discrete), device=device))
        self.register_buffer('theta_buffer',torch.zeros(n_bins, dtype=torch.float32, device=device)) 
        self.register_buffer('continuous_buffer', torch.zeros((n_bins, 2 * n_continuous + 1), 
                                                        device=device, dtype=torch.float32))
        self.prev_logits   = {}                       # per-bin frozen logits
        self.mu_total = None

    def _initialize_logits_buffer(self, device):
        """
        Initialize the precomputed logits buffer for all energy bins.
        This is called once at the start of phase 2.
        """
        n_bins = len(self.ebin_edges) - 1
        
        # Create buffer tensor of correct size
        logits_tensor = torch.zeros((n_bins, self.n_discrete), device=device)
    
        # Update the bin edges buffer if needed
        if not hasattr(self, 'bin_edges') or self.bin_edges.shape != torch.from_numpy(self.ebin_edges).shape:
            self.register_buffer('bin_edges', torch.from_numpy(self.ebin_edges).to(device))
        else:
            # Just update the existing buffer
            self.bin_edges.copy_(torch.from_numpy(self.ebin_edges).to(device))
    
        # Fill with current values if any bins have been seen
        for bin_idx, logits in self.prev_logits.items():
            if 0 <= bin_idx < n_bins:
                logits_tensor[bin_idx] = logits
    
        # For bins we haven't seen yet, compute the physics-based logits
        for bin_idx in range(n_bins):
            if bin_idx not in self.prev_logits:
                # Get energy for this bin
                e_val = 10**self.ebin_edges[bin_idx]
            
                # Get true physics probabilities
                coh, inc, pho, ppr = self.true_prob[bin_idx]
                tot = coh + inc + pho + ppr + 1e-15
                probs = torch.tensor([coh/tot, inc/tot, pho/tot, ppr/tot], device=device)
            
                # Convert to logits
                logits = torch.log(probs + 1e-12)
            
                # Constrain pair production for low energies
                if e_val < 2 * 0.51099895069:  # 2*mec2
                    logits[3] = -1e9
                
                logits_tensor[bin_idx] = logits
    
        # Update the logits buffer
        if hasattr(self, 'logits_buffer'):
            self.logits_buffer.copy_(logits_tensor)
        else:
            self.register_buffer('logits_buffer', logits_tensor)
    
        # We can now clear the dictionary to free memory
        self.prev_logits = {}
    
        print(f"✅ Initialized logits buffer with shape {logits_tensor.shape}")


    def _lookup_frozen_logits(self, E_phys: torch.Tensor) -> torch.Tensor:
        """
        Fast vectorized lookup for frozen logits using a precomputed buffer.
        """
        # Compute bin indices for the current energies
        edges_t = self.bin_edges if hasattr(self, 'bin_edges') else torch.from_numpy(self.ebin_edges).to(E_phys.device)
        logE = torch.log10(E_phys.view(-1)).clamp(edges_t[0] + 1e-6, edges_t[-1] - 1e-6)
        bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1  # (B,)
        
        # Use precomputed logits buffer if available, otherwise create it
        if not hasattr(self, 'logits_buffer'):
            self._initialize_logits_buffer(E_phys.device)
            
        # Fast vectorized lookup - one operation instead of thousands
        return self.logits_buffer[bin_idx]


    def initialize_all_logits_from_physics(self):
        """
        Initialize all logits from ground truth physics probabilities.
        Should be called at the start of phase 0.
        """
        device = next(self.parameters()).device
        n_bins = len(self.ebin_edges) - 1

        # Clear any existing logits cache
        self.prev_logits = {}

        # Go through each bin and seed logits from physics
        for bin_idx in range(n_bins):
            e_val = 10**self.ebin_edges[bin_idx]

            # Get true physics probabilities for this bin
            coh, inc, pho, ppr = self.true_prob[bin_idx]
            tot = coh + inc + pho + ppr + 1e-15
            probs = torch.tensor([coh/tot, inc/tot, pho/tot, ppr/tot], device=device)

            # Convert to logits
            logits = torch.log(probs + 1e-12)

            # Apply constraints for pair production at low energies
            if e_val < 2 * 0.51099895069:  # 2*mec2
                logits[3] = -1e9

            # Store in prev_logits cache
            self.prev_logits[bin_idx] = logits

        print(f"✅ Initialized all {n_bins} bin logits from ground truth physics")


    def initialize_mu(self):
        """Seed the μ-output neuron with physics values from true_mfp_mean."""
        device = next(self.parameters()).device
        
        # We already have true_mfp_mean from initialization
        # Convert to mu (attenuation coefficient)
        mu_true = 1.0 / (self.true_mfp_mean + 1e-30)     # (N_bins,)
        mu_true_t = torch.as_tensor(mu_true, dtype=torch.float32, device=device)

        # Copy into the buffer
        if not hasattr(self, "theta_buffer"):
            raise RuntimeError("θ-buffer not found – did __init__ run?")
        self.theta_buffer.copy_(mu_true_t)               

        # Print range and sample values
        print(f"✅ Mu initialization range: {mu_true.min():.4e} to {mu_true.max():.4e}")
        
        # Get energy bins in MeV (convert from log)
        e_bins = 10 ** self.ebin_edges  # Convert log bins to linear energy
        
        # Add explicit check for energy-to-bin mapping
        print("\nEnergy-to-bin lookup validation:")
        for e in [0.001, 0.01, 0.1, 0.5, 1.0]:
            # Convert to log space for bin lookup
            log_e = np.log10(e)
            
            # Find bin index
            idx = np.searchsorted(self.ebin_edges, log_e) - 1
            idx = max(0, min(idx, len(mu_true)-1))
            
            # Get mu from buffer
            buffer_mu = mu_true[idx]
            buffer_mfp = self.true_mfp_mean[idx]
            
            print(f"E={e:.4f} MeV: log_E={log_e:.4f}, bin={idx}, "
                  f"mu={buffer_mu:.4e}, mfp={buffer_mfp:.4e}")
        
        # Set up the last linear layer
        with torch.no_grad():
            last_linear = self.continuous_head[-1]       # nn.Linear
            # zero out the weight on features
            last_linear.weight[-1].zero_()
            # bias = mean μ  (so unseen bins get a reasonable default)
            last_linear.bias[-1].zero_()
            last_linear.bias[-1].requires_grad_(False)
            last_linear.weight[-1].requires_grad_(False)

        print(f"✅ μ initialised – range {mu_true.min():.3e} … "
              f"{mu_true.max():.3e} cm⁻¹")

    # ──────────────────────────────────────────────────────────────
    #   diagnostics helper – returns the per-bin μ̂ vector
    # ──────────────────────────────────────────────────────────────
    def current_mu_per_bin(self):
        """
        Detach θ-buffer (one μ̂ per energy bin) and move to CPU.
        Useful for progress monitoring during training.
        """
        return self.theta_buffer.detach().cpu().numpy()
        
    def freeze_mu_residual(self):
        """
        Freeze **only** the μ–residual mean (last output of the means vector),
        so that we continue to import mu(E) exactly but leave every other
        head parameter trainable.
        """
        # continuous_head[-1] is the final Linear mapping to [means | log-σ]
        last_linear = self.continuous_head[-1]

        # 1) Freeze the mean‐residual weight & bias at index -1
        last_linear.weight[-1].requires_grad_(False)
        last_linear.bias[-1].requires_grad_(False)

        # Mark it so we can check later, if desired
        last_linear.weight[-1]._always_frozen = True
        last_linear.bias[-1]._always_frozen   = True

        print("🧊  μ–residual mean is frozen; other continuous_head params remain trainable.")
        
            
    # ──────────────────────────────────────────────────────────────
    #   freeze Gaussian σ for the target dimension (mu)
    # ──────────────────────────────────────────────────────────────
    def freeze_gaussian_sigma(self, target_dim: int = -1,
                              log_std_value: float = -20.0):
        """
        Fix only the log-σ that corresponds to the *target_dim* continuous
        variable (0-based index in the mean vector) to `log_std_value`
        (≈ e⁻²⁰ ≃ 2×10⁻⁹) and stop its gradients.  Every other σ row stays
        trainable.

        By default `target_dim = -1` points to the last coordinate, which
        is the μ-residual neuron.
        """
        last = self.continuous_head[-1]          # Linear that emits means & log-σ
        n_sigma = self.n_continuous                   # number of continuous means

        # convert negative index (e.g. -1) to positive 0…n_cont-1
        d = target_dim % n_sigma
        sigma_row = n_sigma + d                  # row that outputs log-σ_d

        with torch.no_grad():
            last.bias[sigma_row].fill_(log_std_value)
            last.weight[sigma_row].zero_()
            
        # ensure tensors accept gradients (needed for register_hook)
        last.bias.requires_grad_(True)
        last.weight.requires_grad_(True)
        
        # keep means & other sigmas learnable; freeze only this row
        def _zero_row_in_grad(row):
            def hook(grad):
                if grad is not None:
                    grad[row].zero_()          # in-place
                return grad
            return hook

        last.bias.register_hook(_zero_row_in_grad(sigma_row))
        last.weight.register_hook(_zero_row_in_grad(sigma_row))

    def initialize_continuous_buffer(self):
        """Initialize continuous buffer with energy-dependent physics values."""
        device = next(self.parameters()).device
        n_bins = len(self.ebin_edges) - 1

        for bin_idx in range(n_bins):
            # Get energy for this bin
            e_val = 10**self.ebin_edges[bin_idx]

            # Get physics parameters for this energy
            if hasattr(self, 'true_prob') and bin_idx < len(self.true_prob):
                # Use precomputed probabilities from true_prob
                probs = self.true_prob[bin_idx]
                coh, inc, pho, ppr = probs[0], probs[1], probs[2], probs[3]
            else:
                # Fallback values
                coh, inc, pho, ppr = 0.25, 0.35, 0.25, 0.25
            dominant_interaction = np.argmax([coh, inc, pho, ppr])
            def angle_to_normalized(theta_rad):
                return 2.0 * theta_rad / np.pi - 1.0

            # Convert energy fractions to normalized action space
            def energy_frac_to_normalized(frac):
                return 2.0 * frac - 1.0  # [0,1] → [-1,1]
                
            # Initialize means based on physics
            means = torch.zeros(self.n_continuous, device=device)

            # Physics-based mean initialization
            if dominant_interaction == 0:  # Rayleigh dominant
                means[1] = energy_frac_to_normalized(1.0)   # High photon energy retention (normalized to [-1,1])
                means[2] = angle_to_normalized(0.1)
            elif dominant_interaction == 1:  # Compton dominant  
                # Energy-dependent Compton behavior
                alpha = e_val / 0.511  # E/mec2
                typical_eps = 1.0 / (1.0 + alpha)  # Klein-Nishina peak
                means[1] = 2.0 * typical_eps - 1.0  # Map [0,1] → [-1,1]
                # Scattering angle: more forward at high energy
                typical_cos_theta = 1.0 - (1.0 - typical_eps) / (typical_eps * alpha)
                typical_theta = np.arccos(typical_cos_theta)
                means[2] = angle_to_normalized(typical_theta)
                electron_eps = 1.0 - typical_eps
                means[4] = energy_frac_to_normalized(electron_eps)
                
                if typical_theta > 1e-6:  # Avoid division by zero for very small angles
                    tan_half_photon = np.tan(typical_theta / 2.0)
                    tan_electron = 1.0 / ((1.0 + alpha) * tan_half_photon)
                    electron_theta = np.arctan(tan_electron)
                
                else:
                    # For very small photon angles, electron goes forward too
                    electron_theta = 0.1  # Small forward angle
                means[5] = angle_to_normalized(electron_theta) 
                
            elif dominant_interaction == 2:  # Photoelectric dominant
                means[1] = -1.0  # No outgoing photon
                means[4] = 0.999   # Electron gets most energy
                # Angular distribution depends on shell
                if e_val > 0.532e-3:  # Above K-edge
                    theta_k = np.pi/6  # ~30 degrees for K-shell
                    means[5] = angle_to_normalized(theta_k)
                else:
                    theta_l = np.pi/2  # ~90 degrees for L-shell (more isotropic)
                    means[5] = angle_to_normalized(theta_l)
            else:  # Pair production (high energy)
                means[1] = -1.0  # No outgoing photon
                means[4] = 0.5   # e- gets half the available energy  
                means[7] = 0.5   # e+ gets half the available energy
                # Small angle approximation: θ ∝ mec2/E
                typical_angle = 0.511 / e_val
                means[5] = np.cos(typical_angle)  # Forward-peaked
                means[8] = np.cos(typical_angle)

            # Initialize log-stds (small but not zero - allow learning)
            log_stds = torch.full((self.n_continuous,), -2.0, device=device)  # σ ≈ 0.14

            # MFP parameter from physics (last element)
            mu_physics = 1.0 / self.true_mfp_mean[bin_idx] if hasattr(self, 'true_mfp_mean') else 1.0
            mu_param = torch.tensor(mu_physics, device=device)

            # Combine into full parameter vector
            continuous_params = torch.cat([means, log_stds, mu_param.unsqueeze(0)])

            # Store in buffer
            self.continuous_buffer[bin_idx] = continuous_params

        print(f"✅ Initialized continuous buffer with physics-based parameters for {n_bins} energy bins")

    def forward(self, obs: torch.Tensor, deterministic: bool=False) -> torch.Tensor:
        # Before any feature‐extractor or distribution logic:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            
        # Safety check for NaNs in observations
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("⚠️ NaN/Inf in policy input obs:", obs)
            raise ValueError("Non‐finite observation passed to policy")
        
        features = self.extract_features(obs)
        phase = getattr(self, "phase", 0)
        
        # Safety check for NaNs in features
        with torch.no_grad():
            if not torch.all(torch.isfinite(features)):
                print("[DEBUG] Actor forward got non-finite features:", features)
                features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
        # Extract energy information
        E_norm = obs[:, 3].unsqueeze(-1)
        E_phys = E_norm * (1.001 - 0.001) + 0.001

        E_phys_flat = E_phys.view(-1)
        edges_t = torch.from_numpy(self.ebin_edges).to(E_phys_flat.device)
        logE = torch.log10(E_phys_flat)
        logE = torch.clamp(logE, min=edges_t[0]+1e-6, max=edges_t[-1]-1e-6)
        bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1
        bin_idx = torch.clamp(bin_idx, min=0, max=len(edges_t)-2)
        # If we have direct access to mu_total function, use it
        if self.mu_total is not None:

            E_cpu = E_phys.cpu().detach().numpy().flatten()
            # Process each energy value through mu_total
            mu_values = []

        
            for e_val in E_cpu:
                try:
                    mu = self.mu_total(float(e_val))
                    mu_values.append(float(mu))
                except Exception as e:
                    # Fallback in case of error
                    print(f"mu_total calculation error: {e}, using fallback")
                    log_e = math.log10(e_val)
                    idx = np.searchsorted(self.ebin_edges, log_e) - 1
                    idx = max(0, min(idx, len(self.theta_buffer)-1))
                    mu_values.append(float(self.theta_buffer[idx].cpu().item()))
        
            # Convert to tensor and reshape
            mu_tensor = torch.tensor(mu_values, dtype=torch.float32, device=obs.device)
            mu_prior = mu_tensor.view(E_phys.shape) # reshape to [batch_size, 1]
        else:
            logE = torch.log10(torch.clamp(E_phys, min=1e-6)).view(-1)
            edges_t = torch.from_numpy(self.ebin_edges).to(E_phys.device)
            logE = torch.clamp(logE, min=edges_t[0]+1e-6, max=edges_t[-1]-1e-6)
            n_bins = edges_t.shape[0] - 1
            bin_idx_low = torch.searchsorted(edges_t, logE, right=False) - 1
            bin_idx_low = torch.clamp(bin_idx_low, min=0, max=len(edges_t)-2)
    
            # Calculate upper bin index
            bin_idx_high = bin_idx_low + 1
            bin_idx_high = torch.clamp(bin_idx_high, min=0, max=len(edges_t)-2)  

            # Get mu values for both bins
            mu_low = self.theta_buffer[bin_idx_low]
            mu_high = self.theta_buffer[bin_idx_high]
    
            # Calculate interpolation weights
            # How far is logE between the lower and upper bin edges?
            log_e_low = edges_t[bin_idx_low]
            log_e_high = edges_t[bin_idx_high]
    
            # Avoid division by zero
            denominator = torch.clamp(log_e_high - log_e_low, min=1e-6)
            weight = (logE - log_e_low) / denominator
            weight = torch.clamp(weight, min=0.0, max=1.0)
            # Perform log-space interpolation (since mu often varies exponentially with energy)
            # Add small epsilon to avoid log(0)
            log_mu_low = torch.log(mu_low + 1e-12)
            log_mu_high = torch.log(mu_high + 1e-12)
            log_mu_interp = log_mu_low + weight * (log_mu_high - log_mu_low)
            mu_interp = torch.exp(log_mu_interp).to(torch.float32)

            mu_prior = mu_interp.unsqueeze(-1)

        # Get energy-bin-specific base parameters + network residual
        continuous_base = self.continuous_buffer[bin_idx]  # (B, 2*n_continuous+1)
        continuous_residual = self.continuous_head(features)  # (B, 2*n_continuous+1)
        raw_cont = continuous_base + continuous_residual

        
        # Apply tanh to everything except the last element (now for mu)
        params_no_mu = torch.tanh(raw_cont[..., :-1])
        mu_param_raw = raw_cont[..., -1:]  # Raw network output for mu <-- keep perhaps for something?
        mu_param = mu_prior 
        mu_param = torch.clamp(mu_param, 7.0e-2, 5.00e7) 
                   # Bool[B]
        cont_params = torch.cat([params_no_mu, mu_param], dim=-1)
        
        # Get discrete logits - keep existing logic
        if phase >= 2:
            with torch.no_grad():
                logits = self._lookup_frozen_logits(E_phys)
                
            # Increment call counter
            cnt = getattr(self, "_freeze_chk_cnt", 0) + 1
            self._freeze_chk_cnt = cnt

            # Check every 10000 calls
            if cnt % 10000 == 1:
                bad = [n for n, p in self.discrete_head.named_parameters()
                       if p.requires_grad]
                if bad:
                    print(f"[⚠️  HybridActor] WARNING: discrete-head params still trainable: {bad}")
                else:
                    print("[✅ HybridActor] discrete logits confirmed frozen (phase ≥ 2).")
        else:
            logits = self.discrete_head(features)
            
            # All the existing per-energy freeze+clamp logic
            edges_t = torch.from_numpy(self.ebin_edges).to(self.device)
            logE = torch.log10(E_phys.view(-1))
            logE = torch.clamp(logE, edges_t[0] + 1e-6, edges_t[-1] - 1e-6)
            bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1
            
            # Build true-physics probs
            true_np = self.true_prob[bin_idx.cpu().numpy()]
            p_true = torch.from_numpy(true_np).to(logits.device)
            p_true /= p_true.sum(dim=1, keepdim=True)
            
            # Zero-out impossible classes
            zero_mask = (p_true <= 1e-13)
            logits = logits.masked_fill(zero_mask, -1e9)
            p_pred = torch.softmax(logits, dim=1)
            
            # L1 distance per sample
            dist = torch.norm(p_pred - p_true, p=1, dim=1)
            
            # Bin index calculation
            E_phys_flat = E_phys.view(-1)
            edges_t = torch.from_numpy(self.ebin_edges).to(E_phys_flat.device)
            logE = torch.log10(E_phys_flat)
            min_edge = edges_t[0]
            max_edge = edges_t[-1]
            logE = torch.clamp(logE, min=min_edge + 1e-12, max=max_edge - 1e-12)
            
            # Handle new bins
            unique_bins = bin_idx.unique()
            for b in unique_bins:
                b_int = int(b.item())
                if b_int not in self.prev_logits:
                    rep = (bin_idx == b_int).nonzero()[0].item()
                    p0 = p_true[rep]
                    self.prev_logits[b_int] = torch.log(p0 + 1e-12).detach()
                sel = (bin_idx == b_int)
                logits[sel] = self.prev_logits[b_int]
            
            # Clamp logic
            B = logits.shape[0]
            freeze_mask = (dist < self.freeze_tol).unsqueeze(1)
            
            frozen_mask = freeze_mask.squeeze(-1)
            if frozen_mask.any():
                n_frozen = int(frozen_mask.sum().item())
            
            prev_all = torch.zeros_like(logits)
            for idx, prev in self.prev_logits.items():
                sel = (bin_idx == idx)
                if sel.any():
                    prev_all[sel] = prev.unsqueeze(0)
            
            delta = logits - prev_all
            eps_tensor = torch.tensor(
                [self.bin_eps[int(b.item())] for b in bin_idx],
                device=logits.device
            ).unsqueeze(1)
            
            delta_clamped = torch.clamp(delta, -eps_tensor, eps_tensor)
            logits = torch.where(freeze_mask, prev_all + delta_clamped, logits)
            
            # Decay bin epsilon
            for b in unique_bins:
                b_int = int(b.item())
                sel = (bin_idx == b_int)
                if not sel.any():
                    continue
                old_logit = prev_all[sel][0].detach().unsqueeze(0)
                new_logit = logits[sel][0].unsqueeze(0)
                true_prob = p_true[sel][0].unsqueeze(0)
                
                old_p = torch.softmax(old_logit, dim=1)
                new_p = torch.softmax(new_logit, dim=1)
                
                dist_before = torch.norm(old_p - true_prob, p=1)
                dist_after = torch.norm(new_p - true_prob, p=1)
                
                if dist_after < dist_before:
                    self.bin_eps[b_int] *= 0.99
            
            # Update cache
            for i in range(B):
                self.prev_logits[int(bin_idx[i].item())] = logits[i].detach()
        
        # Pair production constraint
        under_thr = (E_phys[:, 0] < 2 * mec2)
        logits[under_thr, 3] -= 1e9
        
        # Skip physics if needed
        if getattr(self, "_skip_phys", False) and not hasattr(self, "_skip_notice"):
            print("[HybridActor] physics heads are DISABLED for this run")
            self._skip_notice = True
            
        if getattr(self, "_skip_phys", False):        

            cont_params = torch.cat([params_no_mu, mu_param], dim=-1)

            return torch.cat([logits, cont_params], dim=-1), {}
        
        # Physics head processing
# Physics head processing
        phys_feat = self.phys_backbone(features)
        
        # Get interaction probabilities (before applying softmax for conditioning)
        proc_logits = self.proc_head(phys_feat)
        proc_probs = torch.softmax(proc_logits, dim=-1)  # (B, 4)
        
        # Get energy bin conditioning
        E_phys_flat = E_phys.view(-1)  # (B,)
        edges_t = torch.from_numpy(self.ebin_edges).to(E_phys_flat.device)
        logE = torch.log10(E_phys_flat)
        logE = torch.clamp(logE, min=edges_t[0]+1e-6, max=edges_t[-1]-1e-6)
        bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1  # (B,)
        bin_idx = torch.clamp(bin_idx, min=0, max=len(edges_t)-2)
        energy_bin_onehot = torch.zeros((E_phys_flat.shape[0], self.n_energy_bins), 
                                       device=E_phys_flat.device, dtype=torch.float32)
        energy_bin_onehot.scatter_(1, bin_idx.unsqueeze(1), 1.0)  # (B, n_bins)
        
        # Condition physics heads on interaction probs + energy bin
        conditioning = torch.cat([phys_feat, proc_probs, energy_bin_onehot], dim=-1)  # (B, 512+4+n_bins)
        
        energy_out_raw = self.energy_head(conditioning)  # (B, n_energies * 4)
        angle_out_raw = self.angle_head(conditioning)    # (B, 2 * n_angles * 4)
        nsec_logits = self.nsec_head(phys_feat)
        
        # Reshape to interaction-specific outputs
        B = energy_out_raw.shape[0]
        energy_out = energy_out_raw.view(B, self.n_energies, self.n_interactions)  # (B, n_energies, 4)
        angle_out = angle_out_raw.view(B, 2 * self.n_angles, self.n_interactions)  # (B, 2*n_angles, 4)
        
        phys_dict = {
            "energy": energy_out,
            "angle": angle_out, 
            "nsec": nsec_logits,
            "proc_logits": proc_logits,
            "proc_probs": proc_probs,
            "bin_idx": bin_idx
        }
        
        # Final check for non-finite values
        with torch.no_grad():
            if not torch.all(torch.isfinite(cont_params)):
                print("[DEBUG] Actor forward produced non-finite params:", cont_params)
                cont_params = torch.nan_to_num(cont_params, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return torch.cat([logits, cont_params], dim=-1), phys_dict

        
    def action_log_prob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override the parent's method, so we return (actions, log_prob)
        from our single-parameter distribution.
        """
        # 1) Build the distribution-parameter vector (shape [..., 20])
        params, _ = self.forward(obs)
        dist = self.get_action_dist_from_params(params)
        # 2) Sample an action
        actions = dist.sample()
        # 3) Compute log-prob
        log_prob = dist.log_prob(actions)
        return actions, log_prob


    def get_action_dist_from_params(self, params: torch.Tensor):
        # 1) Build the distribution exactly as before, with all n_discrete + 2*n_cont + 1 dims
        dist = self.action_dist.proba_distribution(params)

        # 2) Extract the θ_mfp head (it lives after [n_discrete + 2*n_continuous] dims)
        theta_idx = self.n_discrete + 2 * self.n_continuous
        # params shape [..., total_dim], so pick that last head:
        pred_rate = params[..., theta_idx]              # shape (...,)

        dist.pred_rate = torch.clamp(pred_rate, 7.0e-2, 5.00e7).unsqueeze(-1)
        return dist



    def extract_features(
        self,
        obs: torch.Tensor,
        features_extractor: nn.Module = None
    ) -> torch.Tensor:
        """
        Override the parent's method that expects two parameters.
        We can safely ignore the second param if we want to always use self.features_extractor.
        """
        # If SB3 passes in a second argument, we can ignore it:
        return self.features_extractor(obs)
        
    def _get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        params, _ = self.forward(obs, deterministic)
        dist = self.get_action_dist_from_params(params)
        return dist.mode() if deterministic else dist.sample()

    def get_log_prob(self, obs: torch.Tensor):
        params, _ = self.forward(obs)
        dist = self.get_action_dist_from_params(params)
        actions = dist.sample()
        return actions, dist.log_prob(actions)


###############################################################################
# 4) Custom Critic (Twin Q-Networks)
###############################################################################
class CustomCritic(nn.Module):
    def __init__(
        self,
        features_extractor: nn.Module,
        action_dim: int,      # total action length = 1 + n_continuous + n_tail
        n_discrete: int = 4,  # number of discrete choices
        n_tail: int = 2,      # tail dims for [mfp_sample, theta_mfp]
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.n_discrete   = n_discrete
        self.n_tail       = n_tail
        # compute “pure” continuous count
        self.n_continuous = action_dim - self.n_discrete - self.n_tail
        assert self.n_continuous > 0, "action_dim too small for discrete + tail"

        # feature size (e.g. 512)
        feat_dim = features_extractor.features_dim

        # total input dim = feat_dim + discrete_onehot + continuous + tail
        in_dim = feat_dim + self.n_discrete + self.n_continuous + self.n_tail
        hidden_size = 256

        # Q1 network
        self.q1_fc1 = nn.Linear(in_dim, hidden_size)
        self.q1_ln1 = nn.LayerNorm(hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_ln2 = nn.LayerNorm(hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(in_dim, hidden_size)
        self.q2_ln1 = nn.LayerNorm(hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_ln2 = nn.LayerNorm(hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1) 

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        # 1) extract features
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        feats = self.features_extractor(obs)  # [B, feat_dim]

        # 2) unpack actions: [disc_idx | cont_controls | mfp_tail]
        #   discrete index
        disc_idx = actions[:, 0].long()  # [B]
        disc     = F.one_hot(disc_idx, num_classes=self.n_discrete).float()  # [B, n_discrete]

        #   pure continuous controls
        cont = actions[:, 1 : 1 + self.n_continuous]  # [B, n_continuous]

        #   mfp tail = [mfp_sample, mu_params]
        tail = actions[:, 1 + self.n_continuous : 1 + self.n_continuous + self.n_tail]  # [B, 2]

        # 3) Q1
        x1 = torch.cat([feats, disc, cont, tail], dim=-1)  # [B, in_dim]
        x1 = F.silu(self.q1_fc1(x1)); x1 = self.q1_ln1(x1)
        x1 = F.silu(self.q1_fc2(x1)); x1 = self.q1_ln2(x1)
        q1 = self.q1_out(x1).squeeze(-1)

        # 4) Q2
        x2 = torch.cat([feats, disc, cont, tail], dim=-1)
        x2 = F.silu(self.q2_fc1(x2)); x2 = self.q2_ln1(x2)
        x2 = F.silu(self.q2_fc2(x2)); x2 = self.q2_ln2(x2)
        q2 = self.q2_out(x2).squeeze(-1)



        return q1, q2

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)




###############################################################################
# 5) Custom HybridSACPolicy that glues it all together
###############################################################################
class HybridSACPolicy(SACPolicy):
    """
    - No net_arch usage (we set net_arch=[]) 
    - A big OptimizedFeatureExtractor
    - Our custom actor: HybridActor
    - Our custom critic: CustomCritic
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        lr_schedule,
        *args, **kwargs
    ):

        ebin_edges    = np.asarray(kwargs.pop("ebin_edges", []), dtype=np.float32)
        true_prob     = np.asarray(kwargs.pop("true_prob",   []), dtype=np.float32)
        true_mfp_mean = np.asarray(kwargs.pop("true_mfp_mean", []), dtype=np.float32)
        energy_regime_boundaries = np.asarray(kwargs.pop("energy_regime_boundaries", []),dtype=np.float32)
        LOG_MIN = np.float32(kwargs.pop("LOG_MIN", 0.0))
        LOG_MAX = np.float32(kwargs.pop("LOG_MAX", 1.0))
        # We ignore net_arch => set to []
        kwargs["net_arch"] = []
        kwargs["use_sde"] = False
        kwargs["log_std_init"] = -3
        kwargs["use_expln"] = False
        kwargs["normalize_images"] = False
        # Force the features_extractor_class to be our big one
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = OptimizedFeatureExtractor
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = {"features_dim": 512}
        self._lr_schedule = lr_schedule
        self.n_discrete = kwargs.pop("n_discrete", 4)
        self.NsecMax = kwargs.pop("NsecMax", 2)
        self.n_continuous = 4 + 3 * self.NsecMax
        def dummy_schedule(_=None) -> float:
            return 1e-4  # or any constant
        super().__init__(
            observation_space,
            action_space,
            dummy_schedule,
            *args, **kwargs
        )

#        self._build = lambda lr_sched: None
        self.real_action_dim = 1 + self.n_continuous

        # Build the big feature extractor
        self.actor_features_extractor = self.features_extractor_class(
             self.observation_space, 
             **self.features_extractor_kwargs
         ).to(self.device)

        self.critic_features_extractor = self.features_extractor_class(
             self.observation_space, 
             **self.features_extractor_kwargs
         ).to(self.device)

        self.actor_features_dim = self.actor_features_extractor.features_dim  # => 512


        # Build actor
        self.actor = HybridActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_discrete=self.n_discrete,
            n_continuous=self.n_continuous,
            n_interactions=len(PROC_NAMES),     # ← NEW
            features_extractor=self.actor_features_extractor,
            features_dim=self.actor_features_dim,
            # ----------------------------------------------------------------
            ebin_edges    = ebin_edges,
            true_prob     = true_prob,
            true_mfp_mean = true_mfp_mean,
            energy_regime_boundaries=energy_regime_boundaries,  # Pass it to the actor
            LOG_MIN = LOG_MIN,
            LOG_MAX = LOG_MAX,
            # ----------------------------------------------------------------
            activation_fn=nn.SiLU,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
        ).to(self.device)

        # Build critics (twin Q-net)
        self.critic = CustomCritic(
            features_extractor=self.critic_features_extractor,
            action_dim=self.real_action_dim
        ).to(self.device)
        self.critic_target = CustomCritic(
            features_extractor=self.critic_features_extractor,
            action_dim=self.real_action_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Create optimizers
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=self._lr_schedule(1e-5),
            weight_decay=1e-4,              # NEW
            **(self.optimizer_kwargs or {})
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=self._lr_schedule(1e-5),
            weight_decay=1e-4,              # NEW
            **(self.optimizer_kwargs or {})
        )

        # After initializing actor and critic, ensure they are on the correct device


        # Tell SB3 we have manually built everything
        self._setup_model = lambda: None
        self._create_aliases = lambda: None

        obs_dim    = self.observation_space.shape[0]
        act_dim    = self.action_space.shape[0]
        # supervised physics‐MSE weight in actor loss:
        self.alpha_phys     = 1.0
        # intrinsic reward penalty weight in the Env:
        self.beta_intrinsic = 0.1

    def forward(self, obs: torch.Tensor, deterministic: bool=False):
        """ Handy method to get actions from the actor. """
        return self.actor._get_action(obs, deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(observation, deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Called by SAC to compute log_prob(actor), ent, and Q-values for the given (obs, actions).
        """
        # Evaluate Q1, Q2
        q1, q2 = self.critic(obs, actions)

        # Evaluate log prob
        # We'll get the distribution params by feeding obs -> actor
        params, _ = self.actor.forward(obs)
        dist = self.actor.get_action_dist_from_params(params)
        log_prob = dist.log_prob(actions)

        # Approx. entropy
        with torch.no_grad():
            ent = dist.entropy().mean()
        return log_prob, ent, q1, q2

    def _setup_model(self) -> None:
        # NO-OP: skip SB3's default network construction
        pass

    def _create_aliases(self) -> None:
        # Likewise skip alias‐generation if SB3 tries to call it.
        pass
###############################################################################
#                PHYSICS DATASET LOADER FOR PRETRAINING
###############################################################################
class PhysicsDataset(Dataset):
    """
    Holds (obs , fp, ang, Eout, n_sec, flat_sec_tensor) tuples

       flat_sec_tensor  shape = (3*NsecMax,)   [E,θ,φ,E,θ,φ,…]
    """
    def __init__(self, obs_arr, fp, ang, Eout, nsec, secs):
        self.obs  = torch.from_numpy(obs_arr).float()
        self.fp   = torch.from_numpy(fp   ).float()
        self.ang  = torch.from_numpy(ang  ).float()
        self.Eout = torch.from_numpy(Eout ).float()
        self.nsec = torch.from_numpy(nsec ).float()
        self.secs = torch.from_numpy(secs ).float()

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return ( self.obs[idx],
                 self.fp[idx],
                 self.ang[idx],
                 self.Eout[idx],
                 self.nsec[idx],
                 self.secs[idx] )

        
###############################################################################
#                        PHYSICS HEAD PRE-TRAINER
###############################################################################
def pretrain_physics_head(
    actor,
    mc_data_path: str,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-4,
    prompt_every: int = 10,
    min_delta: float = 1e-6,
    save_path: str = "physics_head_pretrained.pth",
):
    """
    Supervisedly pre-train actor.features_extractor + the three physics heads.

    Expected .npz keys:  obs, fp, ang, Eout, nsec, secs, proc
                         secs is flat: [E0,θ0,φ0,  E1,θ1,φ1, …]
    """
    # ──────────────────────────────────────────────────────────
    # 1) LOAD DATA
    # ──────────────────────────────────────────────────────────
    data_np   = np.load(mc_data_path)
    obs_raw   = data_np["obs"]                          # (N, obs_dim)
    fp_raw    = data_np["fp"]   [:, None]               # (N,1)
    ang_raw   = data_np["ang"]  [:, None]               # (N,1)
    Eout_raw  = data_np["Eout"] [:, None]               # (N,1)
    nsec_raw  = data_np["nsec"] [:, None]               # (N,1)
    proc_raw  = data_np["proc"][:, None]              # (N,1)  int 0-3
    secs_raw  = data_np["secs"]                         # (N, 3*NsecMax)

    NsecMax   = actor.NsecMax
    device    = actor.device

    # ──────────────────────────────────────────────────────────
    # 2) z-SCORE OBSERVATIONS
    # ──────────────────────────────────────────────────────────
    obs_mean  = obs_raw.mean(0, keepdims=True)
    obs_std   = obs_raw.std (0, keepdims=True) + 1e-8
    obs_scaled = (obs_raw - obs_mean) / obs_std

    # ──────────────────────────────────────────────────────────
    # 3) BUILD PHYSICS TARGET MATRIX
    # ──────────────────────────────────────────────────────────
    logfp     = np.log1p(fp_raw)
    sin_ang   = np.sin(ang_raw)
    cos_ang   = np.cos(ang_raw)
    logEout   = np.log1p(Eout_raw)

    # secondaries (vectorised)
    logEsecs  = np.log1p(secs_raw[:, 0::3])             # (N,NsecMax)
    sin_secs  = np.sin(   secs_raw[:, 1::3])
    cos_secs  = np.cos(   secs_raw[:, 2::3])

    # flat order =  [logfp sin cos logEout nsec  logEsec*  sinsec* cossec*]
    phys_parts = [logfp, sin_ang, cos_ang, logEout, nsec_raw,
                  logEsecs, sin_secs, cos_secs]
    phys_raw   = np.concatenate(
        [p if p.ndim == 2 else p.reshape(len(p), -1) for p in phys_parts],
        axis=1
    )

    phys_mean  = phys_raw.mean(0, keepdims=True)
    phys_std   = phys_raw.std (0, keepdims=True) + 1e-8
    phys_scaled = (phys_raw - phys_mean) / phys_std

    # ──────────────────────────────────────────────────────────
    # 4) DATASET & DATALOADERS  (-- 5 % validation split)
    # ──────────────────────────────────────────────────────────
    class PhysDS(Dataset):
        def __init__(self, o, p, c): 
            self.o, self.p, self.c = o, p, c  
        def __len__(self):
            return len(self.o)
        def __getitem__(self, i):
            return self.o[i], self.p[i], self.c[i]

    full_ds = PhysDS(obs_scaled.astype(np.float32),
                     phys_scaled.astype(np.float32),
                     proc_raw.astype(np.int64))

    val_frac = 0.05
    n_val    = int(len(full_ds) * val_frac)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # keep mean/std as tensors
    phys_mean_t = torch.from_numpy(phys_mean).float().to(device)
    phys_std_t  = torch.from_numpy(phys_std ).float().to(device)

    # ──────────────────────────────────────────────────────────
    # 5) OPTIMISER, SCHEDULER, AMP
    # ──────────────────────────────────────────────────────────
    params = ( list(actor.features_extractor.parameters())
             + list(actor.phys_backbone.parameters())
             + list(actor.energy_head.parameters())
             + list(actor.angle_head.parameters())
             + list(actor.nsec_head.parameters()) 
             + list(actor.proc_head.parameters()) )

    opt = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20,
        threshold=1e-4
    )

    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_amp = torch.cuda.is_available()
    scaler  = GradScaler(enabled=use_amp)
    autocast_ctx = autocast if use_amp else nullcontext

    actor.train()
    prev_loss = float("inf")

    # weighting you chose
    w_energy, w_angle, w_nsec, w_proc = 0.25, 0.25, 0.25, 0.25

    # ──────────────────────────────────────────────────────────
    # 6) TRAIN LOOP
    # ──────────────────────────────────────────────────────────
    for ep in range(1, epochs + 1):
        total = 0.0
        for obs_b, phys_b_scaled, proc_b in train_loader:
            # --- inputs ---
            obs_b   = obs_b.to(device)
            phys_b  = phys_b_scaled.to(device) * phys_std_t + phys_mean_t
            proc_t = proc_b.to(device).long().squeeze(1).clamp(0, 3)

            # tiny Gaussian noise augmentation
            noise   = torch.randn_like(obs_b) * 0.01
            obs_b   = torch.clamp(obs_b + noise, -3, 3)

            # --- targets (un-scaled) ---
            logfp_t     = phys_b[:, 0]
            sin_t       = phys_b[:, 1]
            cos_t       = phys_b[:, 2]
            logEout_t   = phys_b[:, 3]
            nsec_t      = phys_b[:, 4].long().clamp(0, 2)

            logEsecs_t  = phys_b[:, 5 : 5 + NsecMax]
            sin_secs_t  = phys_b[:, 5 + NsecMax       : 5 + 2*NsecMax]
            cos_secs_t  = phys_b[:, 5 + 2*NsecMax     : 5 + 3*NsecMax]

            m_sec = (torch.arange(NsecMax, device=device)
                     < nsec_t.unsqueeze(1)).float()

            # --- forward ---
            with autocast(device_type=device.type):
                feats        = actor.features_extractor(obs_b)
                latent       = actor.phys_backbone(feats)

                # Get interaction probabilities for conditioning
                proc_logits = actor.proc_head(latent)
                proc_probs = torch.softmax(proc_logits, dim=-1)  # (B, 4)
                
                # Get energy bin conditioning from observations
                E_norm = obs_b[:, 3]  # Energy is at index 3 in obs
                E_phys = E_norm * (1.001 - 0.001) + 0.001
                edges_t = torch.from_numpy(actor.ebin_edges).to(E_phys.device)
                logE = torch.log10(E_phys)
                logE = torch.clamp(logE, min=edges_t[0]+1e-6, max=edges_t[-1]-1e-6)
                bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1
                bin_idx = torch.clamp(bin_idx, min=0, max=len(edges_t)-2)
                energy_bin_onehot = torch.zeros((E_phys.shape[0], actor.n_energy_bins), 
                                               device=E_phys.device, dtype=torch.float32)
                energy_bin_onehot.scatter_(1, bin_idx.unsqueeze(1), 1.0)
                
                # Create conditioning input
                conditioning = torch.cat([latent, proc_probs, energy_bin_onehot], dim=-1)
                
                # Get conditioned physics predictions
                energy_p_raw = actor.energy_head(conditioning)
                angle_p_raw = actor.angle_head(conditioning)
                nsec_logits = actor.nsec_head(latent)  # nsec_head still uses just latent
                
                # Reshape to interaction-specific outputs
                B = energy_p_raw.shape[0]
                energy_p_full = energy_p_raw.view(B, actor.n_energies, actor.n_interactions)
                angle_p_full = angle_p_raw.view(B, 2 * actor.n_angles, actor.n_interactions)
                
                # Select predictions for the true interaction type
                proc_true_idx = proc_t.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
                energy_p_raw = torch.gather(energy_p_full, 2, 
                                          proc_true_idx.expand(-1, actor.n_energies, -1)).squeeze(2)
                angle_p = torch.gather(angle_p_full, 2,
                                     proc_true_idx.expand(-1, 2 * actor.n_angles, -1)).squeeze(2)

                # predictions
                logfp_p     = energy_p_raw[:, 0]
                logEout_p   = energy_p_raw[:, 1]
                logEsecs_p  = energy_p_raw[:, 2 : 2 + NsecMax]

                sin_p       = angle_p[:, 0]
                cos_p       = angle_p[:, 1]
                sin_secs_p  = angle_p[:, 2             : 2 + NsecMax]
                cos_secs_p  = angle_p[:, 2 + NsecMax   : 2 + 2*NsecMax]

                # losses
                loss_energy = mse(logfp_p,   logfp_t) \
                            + mse(logEout_p, logEout_t) \
                            + ((logEsecs_p - logEsecs_t).pow(2) * m_sec).sum() / (m_sec.sum() + 1e-6)

                loss_angle  = mse(sin_p, sin_t) + mse(cos_p, cos_t) \
                            + (((sin_secs_p - sin_secs_t).pow(2) +
                                 (cos_secs_p - cos_secs_t).pow(2)) * m_sec).sum() / (m_sec.sum() + 1e-6)

                loss_nsec   = ce(nsec_logits, nsec_t)

                loss_proc   = ce(proc_logits, proc_t)

                loss = w_energy*loss_energy + w_angle*loss_angle + w_nsec*loss_nsec + w_proc*loss_proc

            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            total += loss.item() * obs_b.size(0)

        train_mse = total / len(train_ds)

        # ── validation ───────────────────────────────────────────
        with torch.no_grad():
            val_sum = 0.0
            for obs_v, phys_v_scaled, proc_v in val_loader:
                obs_v  = obs_v.to(device)
                phys_v = phys_v_scaled.to(device) * phys_std_t + phys_mean_t
                proc_v = proc_v.to(device).long().squeeze(1).clamp(0, 3)

                # targets
                logfp_t     = phys_v[:, 0]
                sin_t       = phys_v[:, 1]
                cos_t       = phys_v[:, 2]
                logEout_t   = phys_v[:, 3]
                nsec_t      = phys_v[:, 4].long().clamp(0, 2)

                logEsecs_t  = phys_v[:, 5 : 5 + NsecMax]
                sin_secs_t  = phys_v[:, 5 + NsecMax       : 5 + 2*NsecMax]
                cos_secs_t  = phys_v[:, 5 + 2*NsecMax     : 5 + 3*NsecMax]

                m_sec = (torch.arange(NsecMax, device=device)
                         < nsec_t.unsqueeze(1)).float()

                # forward

                feats        = actor.features_extractor(obs_v)
                latent       = actor.phys_backbone(feats)
            
                # Get interaction probabilities for conditioning
                proc_logits = actor.proc_head(latent)
                proc_probs = torch.softmax(proc_logits, dim=-1)  # (B, 4)
            
                # Get energy bin conditioning from observations
                E_norm = obs_v[:, 3]  # Energy is at index 3 in obs
                E_phys = E_norm * (1.001 - 0.001) + 0.001
                edges_t = torch.from_numpy(actor.ebin_edges).to(E_phys.device)
                logE = torch.log10(E_phys)
                logE = torch.clamp(logE, min=edges_t[0]+1e-6, max=edges_t[-1]-1e-6)
                bin_idx = torch.searchsorted(edges_t, logE, right=True) - 1
                bin_idx = torch.clamp(bin_idx, min=0, max=len(edges_t)-2)
                energy_bin_onehot = torch.zeros((E_phys.shape[0], actor.n_energy_bins), 
                                               device=E_phys.device, dtype=torch.float32)
                energy_bin_onehot.scatter_(1, bin_idx.unsqueeze(1), 1.0)
            
                # Create conditioning input
                conditioning = torch.cat([latent, proc_probs, energy_bin_onehot], dim=-1)
            
                # Get conditioned physics predictions
                energy_p_raw = actor.energy_head(conditioning)
                angle_p_raw = actor.angle_head(conditioning)
                nsec_logits = actor.nsec_head(latent)
            
                # Reshape to interaction-specific outputs
                B = energy_p_raw.shape[0]
                energy_p_full = energy_p_raw.view(B, actor.n_energies, actor.n_interactions)
                angle_p_full = angle_p_raw.view(B, 2 * actor.n_angles, actor.n_interactions)
            
                # Select predictions for the true interaction type
                proc_true_idx = proc_v.unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
                energy_p_raw = torch.gather(energy_p_full, 2, 
                                          proc_true_idx.expand(-1, actor.n_energies, -1)).squeeze(2)
                angle_p = torch.gather(angle_p_full, 2,
                                         proc_true_idx.expand(-1, 2 * actor.n_angles, -1)).squeeze(2)

                logfp_p     = energy_p_raw[:, 0]
                logEout_p   = energy_p_raw[:, 1]
                logEsecs_p  = energy_p_raw[:, 2 : 2 + NsecMax]

                sin_p       = angle_p[:, 0]
                cos_p       = angle_p[:, 1]
                sin_secs_p  = angle_p[:, 2             : 2 + NsecMax]
                cos_secs_p  = angle_p[:, 2 + NsecMax   : 2 + 2*NsecMax]

                loss_energy = mse(logfp_p,   logfp_t) \
                            + mse(logEout_p, logEout_t) \
                            + ((logEsecs_p - logEsecs_t).pow(2) * m_sec).sum() / (m_sec.sum() + 1e-6)

                loss_angle  = mse(sin_p, sin_t) + mse(cos_p, cos_t) \
                            + (((sin_secs_p - sin_secs_t).pow(2) +
                                 (cos_secs_p - cos_secs_t).pow(2)) * m_sec).sum() / (m_sec.sum() + 1e-6)

                loss_nsec   = ce(nsec_logits, nsec_t)

                loss_proc    = ce(proc_logits, proc_v)

                loss = w_energy*loss_energy + w_angle*loss_angle + w_nsec*loss_nsec + w_proc*loss_proc
                val_sum += loss.item() * obs_v.size(0)

        val_mse = val_sum / len(val_ds)
        scheduler.step(val_mse)

        # ── logging ──────────────────────────────────────────────
        delta = abs(prev_loss - train_mse)
        if ep % prompt_every == 0 or ep == 1:
            print(f"[{ep:4d}/{epochs}]  train={train_mse:9.3e}   "
                  f"val={val_mse:9.3e}   Δ={delta:8.3e}   prev={prev_loss:9.3e}")
        prev_loss = train_mse

        if delta < min_delta:
            ans = input(f"Δ < {min_delta:.1e} — stop pre-training? (y/n) ").strip().lower()
            if ans in {"y", "yes"}:
                print("⏹  Stopping early.")
                break

    # ──────────────────────────────────────────────────────────
    # 7) FREEZE & SAVE
    # ──────────────────────────────────────────────────────────
    for g in (actor.features_extractor,
              actor.phys_backbone,
              actor.energy_head,
              actor.angle_head,
              actor.nsec_head):
        for p in g.parameters():
            p.requires_grad = False

    torch.save({
        "features_extractor": actor.features_extractor.state_dict(),
        "phys_backbone"    : actor.phys_backbone.state_dict(),
        "energy_head"      : actor.energy_head.state_dict(),
        "angle_head"       : actor.angle_head.state_dict(),
        "nsec_head"        : actor.nsec_head.state_dict(),
    }, save_path)
    print(f"✅  Physics head & extractor frozen and saved to '{save_path}'.")



###############################################################################
#                  PHYSICS HELPER
###############################################################################
def build_phys_targets(fp_t, ang_t, Eout_t, nsec_t, secs_t, proc_t, NsecMax, device):
    """
    Returns:
      energy_t  (B, 2+N)   log_fp, log_Eout, logE_sec_k - interaction-specific
      angle_t   (B, 2*(1+N))   sin/cos pairs - interaction-specific  
      mask_sec  (B, N)     1 = real secondary, 0 = padded
      interaction_mask (B, 4) masks for which interactions are valid
    """
    B = fp_t.shape[0]
    log_fp_t   = torch.log1p(fp_t)
    log_Eout_t = torch.log1p(Eout_t)

    secs_t = secs_t.clone()
    logE_secs_t = secs_t[:, ::3]                     # (B,N)

    # Build interaction-specific targets
    energy_t = torch.zeros((B, 2 + NsecMax), device=device, dtype=torch.float32)
    angle_t = torch.zeros((B, 2 * (1 + NsecMax)), device=device, dtype=torch.float32)
    interaction_mask = torch.zeros((B, 4), device=device, dtype=torch.float32)
    
    for b in range(B):
        interaction_type = int(proc_t[b].item())
        interaction_mask[b, interaction_type] = 1.0
        
        if interaction_type == 0:  # Rayleigh
            # Photon keeps energy, scatters
            energy_t[b, 0] = log_fp_t[b]  # free path
            energy_t[b, 1] = log_Eout_t[b]  # should equal input energy
            angle_t[b, 0] = torch.sin(ang_t[b])  # scattered photon angle
            angle_t[b, 1] = torch.cos(ang_t[b])
            
        elif interaction_type == 1:  # Compton  
            # Photon loses energy, electron recoils
            energy_t[b, 0] = log_fp_t[b]
            energy_t[b, 1] = log_Eout_t[b]  # reduced photon energy
            energy_t[b, 2] = logE_secs_t[b, 0]  # electron energy
            angle_t[b, 0] = torch.sin(ang_t[b])  # scattered photon
            angle_t[b, 1] = torch.cos(ang_t[b])
            angle_t[b, 2] = torch.sin(secs_t[b, 1])  # electron angle
            angle_t[b, 3] = torch.cos(secs_t[b, 1])
            
        elif interaction_type == 2:  # Photoelectric
            # Photon absorbed, electron ejected
            energy_t[b, 0] = log_fp_t[b]
            energy_t[b, 1] = torch.log1p(torch.tensor(0.0, device=device))  # no outgoing photon
            energy_t[b, 2] = logE_secs_t[b, 0]  # electron energy
            angle_t[b, 2] = torch.sin(secs_t[b, 1])  # electron angle (no photon angle)
            angle_t[b, 3] = torch.cos(secs_t[b, 1])
            
        elif interaction_type == 3:  # Pair production
            # Photon absorbed, e-/e+ pair created
            energy_t[b, 0] = log_fp_t[b]
            energy_t[b, 1] = torch.log1p(torch.tensor(0.0, device=device))  # no outgoing photon
            energy_t[b, 2] = logE_secs_t[b, 0]  # electron energy
            if NsecMax > 1:
                energy_t[b, 3] = logE_secs_t[b, 1]  # positron energy
            angle_t[b, 2] = torch.sin(secs_t[b, 1])  # electron angle
            angle_t[b, 3] = torch.cos(secs_t[b, 1])
            if NsecMax > 1:
                angle_t[b, 4] = torch.sin(secs_t[b, 4])  # positron angle  
                angle_t[b, 5] = torch.cos(secs_t[b, 4])

    # masks for secondaries
    mask_sec = (torch.arange(NsecMax, device=device)
                < nsec_t.unsqueeze(1)).float()       # (B,N)

    return energy_t, angle_t, mask_sec, interaction_mask
###############################################################################
#                  PHYSICS HEAD PRE-TRAINER LOADER
###############################################################################
def load_physics_head(actor, load_path: str = "physics_head_pretrained.pth"):
    """
    Load a previously-saved physics head + feature extractor
    into actor, then freeze them.
    """
    if not os.path.exists(load_path):
        print(f"⚠️  No checkpoint found at '{load_path}'. Pre-training is required.")
        return
    ckpt = torch.load(load_path, map_location=actor.device)
    actor.features_extractor.load_state_dict(ckpt['features_extractor'])
    actor.phys_backbone     .load_state_dict(ckpt['phys_backbone'])
    actor.energy_head       .load_state_dict(ckpt['energy_head'])
    actor.angle_head        .load_state_dict(ckpt['angle_head'])
    actor.nsec_head         .load_state_dict(ckpt['nsec_head'])
    freeze_groups = [actor.features_extractor,
                 actor.phys_backbone,
                 actor.energy_head,
                 actor.angle_head,
                 actor.nsec_head]
    for g in freeze_groups:
        for p in g.parameters():
            p.requires_grad = False
    print("", flush=True)
    print(f"🔄 Loaded physics head & extractor from '{load_path}'")
###############################################################################
#   MONTE CARLO DATASET GENERATOR FOR PRE-TRAINING THE PHYSICS HEAD
###############################################################################
def generate_mc_dataset(
    mc_data_path: str = "mc_physics_data.npz",
    n_samples: int = 100_000,
    energy_range: tuple[float, float] = (0.001, 1.0),   # (Emin , Emax) MeV
    NsecMax: int = 2,
    ecut: float = 0.001
):
    """
    Runs pure MC to collect (obs, phys_targets) pairs.
    Saves three arrays into mc_data_path:
      - obs:   shape (n_samples, obs_dim)
      - phys:  shape (n_samples, phys_dim)
      - proc:  shape (n_samples,)  int 0–3  (interaction class)
    phys_dim = 5 + 3*NsecMax
    """
    # 1) Build data objects
    data = PenelopeLikeWaterData(
        final_csv_path="Final_cross_sections.csv",
        rayleigh_csv_path="Rayleigh_cross_sections.csv",
        density=1.0
    )
    env = PhasedRewardEnv(
        data,
        ecut=ecut,
        max_steps=1_000_000,
        NsecMax=NsecMax,
        train_mode=False,
        fixed_energy=energy_range[0],   # dummy
        n_multi=N_STEPS_RETURN
    )

    # ensure starting energy is set


    obs_list = []
    phys_list = []
    proc_list = []
    samples_collected = 0

    print(
        f"🎯 Generating {n_samples} MC samples with photon energies "
        f"in [{energy_range[0]}, {energy_range[1]}] MeV…"
    )
    while samples_collected < n_samples:
        # reset environment and get initial observation
        env.fixed_energy = random.uniform(*energy_range)
        obs, _ = env.reset()
        # sample one true free path
        muT = data.mu_total(env.E)
        r = random.random()
        dist_real = -math.log(max(r, 1e-12)) / max(muT, 1e-15)
        # move photon
        old_dir = np.array([env.u, env.v, env.w], dtype=float)
        env.x += dist_real * env.u
        env.y += dist_real * env.v
        env.z += dist_real * env.w
        # skip if exited phantom
        if not (env.xmin <= env.x <= env.xmax and
                env.ymin <= env.y <= env.ymax and
                env.zmin <= env.z <= env.zmax):
            continue

        # perform single MC interaction
        new_dir, Eout, secs, itype, shell = photon_interact(env.E, old_dir, data)
        cos_t = np.clip(np.dot(old_dir, new_dir), -1.0, 1.0)
        angle = math.acos(cos_t)
        sin_ang = math.sin(angle)
        cos_ang = math.cos(angle)
        proc_idx = PROC_NAMES.index(itype) if itype in PROC_NAMES else -1
        # build phys_targets: [dist, angle, Eout, n_sec, sec0_E, θ, φ, ...]
        phys = [dist_real, sin_ang, cos_ang, Eout, len(secs)]
        for k in range(NsecMax):
            if k < len(secs) and secs[k][0] == "electron":
                sE, edir = secs[k][1], secs[k][2]
                s_th  = math.acos(np.clip(edir[2], -1.0, 1.0))
                sE_log = math.log1p(sE)           # match log1p transform
                sin_s  = math.sin(s_th)
                cos_s  = math.cos(s_th)
                phys.extend([sE_log, sin_s, cos_s])
            else:
                phys.extend([0.0, 0.0, 0.0])

        obs_list.append(obs)
        phys_list.append(phys)
        proc_list.append(proc_idx)
        samples_collected += 1
        if samples_collected % 10_000 == 0:
            print(f"  • Collected {samples_collected}")

    # convert & save
    obs_arr  = np.stack(obs_list,  axis=0)
    phys_arr = np.stack(phys_list, axis=0)
    proc_arr = np.asarray(proc_list, dtype=np.int64)
    np.savez(
        mc_data_path,
        obs   = obs_arr,
        fp    = phys_arr[:, 0],     # free-path
        ang   = np.arctan2(phys_arr[:,1], phys_arr[:,2]),   # θ  from sin/cos
        Eout  = phys_arr[:, 3],
        nsec  = phys_arr[:, 4],
        secs  = phys_arr[:, 5:],     # flat list (logE, sin, cos, …)
        proc  = proc_arr 
    )
    print(f"✅ Saved MC dataset: {mc_data_path}")
    


###############################################################################
#       Analysis Functions: Interaction Stats and Dose Tables
###############################################################################
def analyze_interaction_stats(interactions, title=""):
    if not interactions:
        print(f"{title}: No interactions recorded.")
        return
    df = pd.DataFrame(interactions)
    grouped = df.groupby("interaction").agg({
        "free_path": ["mean", "std"],
        "angle": ["mean", "std"],
        "interaction": "count"
    })
    grouped.columns = ["Mean Free Path (cm)", "Std Free Path (cm)",
                       "Mean Angle (deg)", "Std Angle (deg)",
                       "Count"]
    print(f"\n{title} Interaction Statistics:")
    print(grouped)
    return grouped

###############################################################################
#       Generating Photon Showers (MC vs. Agent) + 3D Plot
###############################################################################
def run_mc_shower(n_photons, data: PenelopeLikeWaterData, env, max_steps=100000, ecut=0.001):
    tracks = []
    all_secondaries = []
    mc_interactions = []
    # Reset the environment once and clear the dose tally
    env.reset()
    env.dose_tally[:] = 0.0  # Ensure that dose_tally is initially zero

    for _ in range(n_photons):
        # Manually set the initial state without calling env.reset() each time.
        # This ensures that we do not wipe out the accumulated dose_tally.
        env.x = (random.random() * 10.0) - 5.0
        env.y = (random.random() * 10.0) - 5.0
        env.z = 0.0
        env.u, env.v, env.w = (0.0, 0.0, 1.0)
        # Use the fixed energy provided by the evaluation environment
        env.E = env.fixed_energy  
        env.alive = True
        env.steps = 0

        # Initialize track and other temporary lists for this photon
        single_x = [env.x]
        single_y = [env.y]
        single_z = [env.z]
        photon_interactions = []
        secondaries = []
        alive = True
        steps = 0

        # Start simulating this photon trajectory
        while alive and steps < max_steps and env.E > ecut:
            steps += 1
            mu = data.mu_total(env.E)
            if mu < 1e-30:
                alive = False
                break
            dist = -math.log(random.random()) / mu
            env.x += dist * env.u
            env.y += dist * env.v
            env.z += dist * env.w


            # <-- ADDED: Boundary check
            if (env.x < env.xmin or env.x > env.xmax or
                env.y < env.ymin or env.y > env.ymax or
                env.z < env.zmin or env.z > env.zmax):
                alive = False
                break
            photon_energy_in = env.E
            photon_incident_dir = np.array([env.u, env.v, env.w], dtype=float)
            new_dir, Eout, _secs, itype, _ = photon_interact(env.E, photon_incident_dir, data)
            angle = math.degrees(math.acos(np.clip(np.dot(photon_incident_dir, new_dir), -1, 1)))
            interaction_record = {
                "interaction": itype,
                "free_path": dist,
                "angle": angle,
                "position": (env.x, env.y, env.z),
                "photon_energy_in": photon_energy_in,
                "photon_energy_out": Eout,
                "photon_incident_direction": photon_incident_dir,
                "secondaries": _secs.copy()  # Store a copy of the secondary particles from this interaction
            }
            photon_interactions.append(interaction_record)
            # Handle electron transport for secondary electrons
            for sec in _secs:
                if sec[0] == "electron":
                    stopping_Egrid, stopping_S_col, stopping_S_rad = load_stopping_power("ElectronStoppingPower.csv")
                    transport_electron_csda(
                        sec[1], sec[2], (env.x, env.y, env.z), env.dose_tally, env,
                        n_steps=5, Egrid=stopping_Egrid, S_col_vals=stopping_S_col, S_rad_vals=stopping_S_rad
                    )
 
#                    transport_electron_condensed_history(
#                        sec[1], sec[2], (env.x, env.y, env.z), env.dose_tally, env,
#                        Egrid=stopping_Egrid, S_vals=stopping_S, ecut=0.001, max_steps=2000,   
#                        fraction_of_range=0.05, step_min=0.001, step_max=0.1
#                    )
            secondaries.extend(_secs)
            single_x.append(env.x)
            single_y.append(env.y)
            single_z.append(env.z)
            env.u, env.v, env.w = new_dir
            env.E = Eout
            if env.E < ecut:
                alive = False

        tracks.append({
            "coords": (single_x, single_y, single_z),
            "interactions": photon_interactions,
            "secondaries": secondaries
        })
        all_secondaries.extend(secondaries)
        mc_interactions.extend(photon_interactions)
    return tracks, all_secondaries, env.dose_tally, mc_interactions


def eval_reset(env):
    """
    Resets the photon state for evaluation without clearing the accumulated dose_tally.
    """
    env.steps = 0
    half_side = 5.0
    env.x = (random.random() * 10.0) - half_side
    env.y = (random.random() * 10.0) - half_side
    env.z = 0.0
    env.u, env.v, env.w = (0.0, 0.0, 1.0)
    if env.train_mode:
        env.E = env.fixed_energy
    else:
        env.E = env.fixed_energy
    env.alive = True
    env.interaction_bank = []
    env.interaction_stats = []
    return env._get_obs(), {}



def run_agent_shower(
    n_photons,
    model,
    env,
    batch_size=32,
    E_fixed=None,
    max_steps=100000,
    ecut=0.001
):
    actor = model.policy.actor          # cache a ref; avoids look-ups
    actor._skip_phys = False            # <── Skipping physics head temporarily
    # ——————————————————————————
    # 1) SETUP
    # ——————————————————————————
    # ensure we evaluate, not train
    env.train_mode = False

    # determine our fixed input energy
    if E_fixed is None:
        E_fixed = env.fixed_energy
    else:
        env.fixed_energy = E_fixed

    # ensure dose tally starts at zero
    env.dose_tally[:] = 0.0

    data = env.data
    NsecMax = env.NsecMax
    stopEgrid, stopS_col, stopS_rad = load_stopping_power("ElectronStoppingPower.csv")
    # helper for mapping raw discrete float -> 0..3


    # ——————————————————————————
    # 2) PHOTON STATE INITIALIZATION
    # ——————————————————————————
    def init_photon_states(num):
        half = 5.0
        px = np.random.uniform(-half, half, size=num)
        py = np.random.uniform(-half, half, size=num)
        pz = np.zeros(num)
        ux = np.zeros(num); uy = np.zeros(num); uz = np.ones(num)
        energies = np.full(num, E_fixed, dtype=float)
        alive = np.ones(num, dtype=bool)
        steps = np.zeros(num, dtype=int)
        shell_oh = np.zeros((num,4), dtype=float)
        return px,py,pz,ux,uy,uz,energies,alive,steps,shell_oh

    def init_record_containers(num):
        return (
            [[] for _ in range(num)],
            [[] for _ in range(num)],
            [[] for _ in range(num)],
            [[] for _ in range(num)],
            [[] for _ in range(num)],
            [[] for _ in range(num)],
        )

    # build a single‐photon observation
    def build_obs(i):
        if not alive[i]:
            return np.zeros(10 + 4 + 3*NsecMax + 4, dtype=np.float32)
        # position & direction & energy
        x_norm = 2*(px[i]-env.xmin)/(env.xmax-env.xmin) - 1
        y_norm = 2*(py[i]-env.ymin)/(env.ymax-env.ymin) - 1
        z_norm = (pz[i]-env.zmin)/(env.zmax-env.zmin)
        E_val = max(energies[i], 1e-3)
        E_norm = (E_val - 0.001)/(1.001-0.001)
        logE   = np.clip(math.log10(E_val), -3.0, 1.0)
        u,v,w  = ux[i], uy[i], uz[i]
        stepf  = steps[i]/max_steps
        local_step_norm = (steps[i] % env.n_multi) / env.n_multi
        # --- mean‑free‑path normalised term (matches env._get_obs) ---
        mu        = data.mu_total(E_val)               # linear attenuation coeff
        mfp       = 1.0 / (mu + 1e-12)                 # mean free path (cm)
        mfp_norm  = np.clip((mfp - 1.3333e-4) / (1.428e1 - 1.3333e-4), 0.0, 1.0)

        base   = np.array([x_norm,y_norm,z_norm,
                           E_norm,logE,
                           u,v,w,
                           stepf,
                           local_step_norm,
                           mfp_norm
        ], dtype=np.float32)

        # cross‐sections
        coh,inc,pho,ppr,_ = data.partial_cs(E_val)
        tot = coh+inc+pho+ppr+1e-12
        cs_norm = np.array([coh,inc,pho,ppr],dtype=np.float32)/tot

        # secondary history
        feats = []
        count = 0
        for sec in reversed(recent_secs[i]):
            if count>=NsecMax: break
            if sec[0]=='electron':
                Esec = np.clip(sec[1],0.0,2*mec2)
                theta = math.acos(np.clip(sec[2][2],-1,1))/math.pi
                phi   = (math.atan2(sec[2][1],sec[2][0])%(2*math.pi))/(2*math.pi)
                feats.insert(0,(Esec/mec2,theta,phi))
                count+=1
        flat = [x for triple in feats for x in triple]
        flat += [0.0]* (3*NsecMax - len(flat))
        flat = np.array(flat, dtype=np.float32)

        return np.concatenate([base, cs_norm, flat, shell_oh[i]], dtype=np.float32)

    # ——————————————————————————
    # 3) MAIN BATCH LOOP
    # ——————————————————————————
    tracks      = []
    all_secs    = []
    all_inters  = []

    num_batches = ceil(n_photons/batch_size)
    start_idx   = 0

    for _b in range(num_batches):
        curr = min(batch_size, n_photons - start_idx)
        if curr<=0: break

        px,py,pz,ux,uy,uz,energies,alive,steps,shell_oh = init_photon_states(curr)
        (coords_x, coords_y, coords_z,
         inters, sec_lists, recent_secs) = init_record_containers(curr)

        # seed initial coords
        for i in range(curr):
            coords_x[i].append(px[i])
            coords_y[i].append(py[i])
            coords_z[i].append(pz[i])

        global_step = 0
        while np.any(alive) and global_step<max_steps:
            global_step+=1

            # build batch obs
            idxs  = [i for i in range(curr) if alive[i]]
            obs   = np.vstack([ build_obs(i) for i in idxs ])
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
            # 1) get raw policy params (logits + gauss)
            params, _ = model.policy.actor.forward(obs_tensor)
            shell_onehot = [0, 0, 0, 0]
            # 3) rebuild distribution and sample
            dist    = model.policy.actor.get_action_dist_from_params(params)
            sample  = dist.sample()
            actions = sample.cpu().detach().numpy()
            for batch_i, i in enumerate(idxs):
                steps[i]+=1
                act = actions[batch_i]
                disc = int(act[0])
                process_names = ["rayleigh", "compton", "photo", "pair"]
                itype = process_names[disc]
                cont = act[1:]
                photon_energy_in = energies[i]
                # decode predictions
                mfp_pred = act[-2]
                th_pred   = env._denormalize(cont[2], 'photon_theta')
                ph_pred   = env._denormalize(cont[3], 'photon_phi')
                shell = shell_onehot
                # ——— Build predicted secondary energies & directions ———
                # 1) Compute Q sink and available E
                if disc == 2:  # photoelectric
                    shell_name, _ = data.water_shell_data.pick_shell(energies[i])
                    idx_map = {"H_K": 0, "O_K": 1, "O_L1": 2, "O_L2": 3, "O_L3": 4}
                    shell_onehot[idx_map[shell_name]] = 1
                    Eb = PHOTO_SHELL_BINDINGS[shell_name] * 1e-6
                    Q  = Eb
                elif disc == 3:  # pair production
                    Q = 2 * mec2
                else:
                    Q = 0.0
                avail_E = max(energies[i] - Q, 0.0)

                # ——— Energy‐split with exact masking ———
                if disc == 0:
                    # Rayleigh: photon keeps full energy
                    E_out = photon_energy_in
                    sec_energies = []
                elif disc in (1, 2):
                    # Compton or Photoelectric: one electron
                    raw = [
                        max(0.0, env._denormalize(cont[1], 'energy')),
                        max(0.0, env._denormalize(cont[4], 'energy'))
                    ]
                    total = sum(raw)
                    if total > 0.0:
                        fracs = [r/total for r in raw]
                    else:
                        fracs = [0.5, 0.5]
                    E_out        = fracs[0] * avail_E
                    sec_energies = [fracs[1] * avail_E]
                    if disc in (2,):
                        # photoelectric placeholder
                        E_out  = 0.0
                        th_pred = math.pi/2
                        ph_pred = 0.0
                else:
                    # Pair production: two secondaries
                    raw = [
                        max(0.0, env._denormalize(cont[1], 'energy')),
                        max(0.0, env._denormalize(cont[4], 'energy')),
                        max(0.0, env._denormalize(cont[7], 'energy'))
                    ]
                    total = sum(raw)
                    if total > 0.0:
                        fracs = [r/total for r in raw]
                    else:
                        fracs = [1/3,1/3,1/3]
                    E_out        = 0.0  # no photon
                    sec_energies = [fracs[1]*avail_E, fracs[2]*avail_E]
                    th_pred = math.pi/2
                    ph_pred = 0.0
                # ——— end energy‐split ———

                
                # secondaries
                sec_params = []

                if disc == 0:  # rayleigh
                    # no secondaries at all
                    pass

                elif disc == 1:  # compton
                # one predicted recoil electron
                    Ej    = fracs[1] * photon_energy_in  # or however you allocated it
                    theta = env._denormalize(cont[4+3*0 + 1], 'theta')
                    phi   = env._denormalize(cont[4+3*0 + 2], 'phi')
                    dir_j = rotate_direction((ux[i],uy[i],uz[i]), theta, phi)
                    sec_params.append(("electron", Ej, dir_j, "compton_e"))

                elif disc == 2:  # photoelectric
                    # one predicted shell‐specific electron
                    Ej    = fracs[1] * photon_energy_in
                    theta = env._denormalize(cont[4+3*0 + 1], 'theta')
                    phi   = env._denormalize(cont[4+3*0 + 2], 'phi')
                    dir_j = rotate_direction((ux[i],uy[i],uz[i]), theta, phi)
                    # pick the shell binding exactly as MC
                    shell_name, _ = data.water_shell_data.pick_shell(energies[i])
                    sec_params.append(("electron", Ej, dir_j, f"photo_{shell_name}"))

                elif disc == 3:  # pair production
                    # two secondaries: e– and e+
                    # electron
                    Ej_e  = fracs[1] * avail_E
                    theta = env._denormalize(cont[4+3*0 + 1], 'theta')
                    phi   = env._denormalize(cont[4+3*0 + 2], 'phi')
                    dir_e = rotate_direction((ux[i],uy[i],uz[i]), theta, phi)
                    sec_params.append(("electron", Ej_e, dir_e, "pair_e"))
                    # positron
                    Ej_p  = fracs[2] * avail_E
                    theta = env._denormalize(cont[4+3*1 + 1], 'theta')
                    phi   = env._denormalize(cont[4+3*1 + 2], 'phi')
                    dir_p = rotate_direction((ux[i],uy[i],uz[i]), theta, phi)
                    sec_params.append(("positron", Ej_p, dir_p, "pair_p"))

                # — free path update —
                px[i] += mfp_pred * ux[i]
                py[i] += mfp_pred * uy[i]
                pz[i] += mfp_pred * uz[i]

                # boundary?
                if (px[i]<env.xmin or px[i]>env.xmax or
                    py[i]<env.ymin or py[i]>env.ymax or
                    pz[i]<env.zmin or pz[i]>env.zmax):
                    alive[i]=False
                    continue
                    
                # choose interaction
                inc_dir = np.array([ux[i],uy[i],uz[i]],dtype=float)
                new_dir = rotate_direction(inc_dir, th_pred, ph_pred)

                # record
                dot = np.clip(np.dot(inc_dir,new_dir), -1.0,1.0)
                angle = math.degrees(math.acos(dot))
                rec = {
                    "interaction": itype,
                    "free_path": mfp_pred,
                    "angle": angle,
                    "position": (px[i],py[i],pz[i]),
                    "photon_energy_in": energies[i],
                    "photon_energy_out": E_out,
                    "photon_incident_direction": inc_dir,
                    "secondaries": sec_params.copy()
                }
                for tag,Esec,dir_sec,lbl in sec_params:
                    if Esec>0:
                        transport_electron_csda(
                            Esec, dir_sec,
                            (px[i],py[i],pz[i]),
                            env.dose_tally, env,
                            n_steps=5,
                            Egrid=stopEgrid, S_col_vals=stopS_col, S_rad_vals=stopS_rad
                        )

                inters[i].append(rec)
                shell_oh[i] = shell_onehot

                # advance state
                ux[i],uy[i],uz[i] = new_dir
                energies[i] = E_out



                coords_x[i].append(px[i])
                coords_y[i].append(py[i])
                coords_z[i].append(pz[i])
                sec_lists[i].extend(rec["secondaries"])
                if energies[i]<ecut:
                    alive[i]=False

        # end of one batch
        for i in range(curr):
            tracks.append({
                "coords": (coords_x[i], coords_y[i], coords_z[i]),
                "interactions": inters[i],
                "secondaries": sec_lists[i]
            })
            all_secs   .extend(sec_lists[i])
            all_inters .extend(inters[i])
            recent_secs[i].extend(sec_params)

        start_idx += curr

    return tracks, all_secs, env.dose_tally.copy(), all_inters


def add_phantom_box(ax):
    r = [-50, 50]
    z = [0, 100]
    for zi in z:
        ax.plot([zi, zi], [r[0], r[1]], [r[0], r[0]], color='gray', linestyle='--', lw=0.5)
        ax.plot([zi, zi], [r[0], r[1]], [r[1], r[1]], color='gray', linestyle='--', lw=0.5)
        ax.plot([zi, zi], [r[0], r[0]], [r[0], r[1]], color='gray', linestyle='--', lw=0.5)
        ax.plot([zi, zi], [r[1], r[1]], [r[0], r[1]], color='gray', linestyle='--', lw=0.5)
    for x in r:
        for y in r:
            ax.plot(z, [x, x], [y, y], color='gray', linestyle='--', lw=0.5)

def unnormalize_track(track, env):
    """
    Given a track (tuple of lists: (xs, ys, zs)) where x and y are normalized,
    returns physical coordinates using the environment geometry.
    """
    unnorm_x = [ ((x + 1) / 2) * (env.xmax - env.xmin) + env.xmin for x in track[0] ]
    unnorm_y = [ ((y + 1) / 2) * (env.ymax - env.ymin) + env.ymin for y in track[1] ]
    unnorm_z = [ z * (env.zmax - env.zmin) + env.zmin for z in track[2] ]
    return (unnorm_x, unnorm_y, unnorm_z)
    
def plot_shower_comparison(mc_tracks, agent_tracks, env, max_plot=1000):
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    for track in mc_tracks[:max_plot]:
        xs, ys, zs = track["coords"]
        ax1.plot(zs, xs, ys, linewidth=0.3, alpha=0.3, color='blue')
    ax1.set_xlim(0,100); ax1.set_ylim(-50,50); ax1.set_zlim(-50,50)
    ax1.set_xlabel('Z'); ax1.set_ylabel('X'); ax1.set_zlabel('Y')
    add_phantom_box(ax1)
    ax2 = fig.add_subplot(122, projection='3d')
    for track in agent_tracks[:max_plot]:
        xs, ys, zs = track["coords"]
        ax2.plot(zs, xs, ys, linewidth=0.3, alpha=0.3, color='red')
    ax2.set_xlim(0,100); ax2.set_ylim(-50,50); ax2.set_zlim(-50,50)
    add_phantom_box(ax2)
    plt.tight_layout()
    plt.show()


def extract_xy_at_z(tracks, z_target=50.0):
    xy_points = []
    for track in tracks:
        xs, ys, zs = track["coords"]
        if len(zs) < 2:
            continue
        for i in range(len(zs)-1):
            # Ensure zs[i] is a float; if needed, convert using float(zs[i])
            if (zs[i]-z_target) * (zs[i+1]-z_target) <= 0:
                frac = (z_target - zs[i]) / (zs[i+1]-zs[i])
                x_val = xs[i] + frac*(xs[i+1]-xs[i])
                y_val = ys[i] + frac*(ys[i+1]-ys[i])
                xy_points.append((x_val, y_val))
                break
    return xy_points

def plot_plane_cuts(tracks, title):
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes = axes.flatten()
    z_targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for i, z in enumerate(z_targets):
        xy = extract_xy_at_z(tracks, z_target=z)
        axes[i].hist2d([pt[0] for pt in xy],
                       [pt[1] for pt in xy],
                       bins=50, range=[[-100, 100], [-100, 100]])
        axes[i].set_title(f'{title} XY at z={z}')
    axes[10].hist2d(
        [x for track in tracks for x, z in zip(track["coords"][0], track["coords"][2])],
        [z for track in tracks for x, z in zip(track["coords"][0], track["coords"][2])],
        bins=50, range=[[-50, 50], [0, 100]]
    )
    axes[10].set_title(f'{title} XZ Projection')
    axes[11].hist2d(
        [y for track in tracks for y, z in zip(track["coords"][1], track["coords"][2])],
        [z for track in tracks for y, z in zip(track["coords"][1], track["coords"][2])],
        bins=50, range=[[-50, 50], [0, 100]]
    )
    axes[11].set_title(f'{title} YZ Projection')
    plt.tight_layout()
    plt.show()


def plot_pdd(dose_tally, env):
    print("Depth (cm)\tDose (MeV)")
    z_values = np.linspace(env.zmin+env.dz/2, env.zmax-env.dz/2, env.pdd_bins)
    for z, dose in zip(z_values, dose_tally):
        print(f"{z:.2f}\t\t{dose:.16f}")
    plt.figure(figsize=(8,6))
    plt.plot(z_values, dose_tally, marker='o')
    plt.xlabel("Depth (cm)")
    plt.ylabel("Deposited Energy (MeV)")
    plt.title("PDD (Depth-Dose Curve)")
    plt.grid(True)
    plt.show()

def analyze_secondaries(secondaries):
    data = []
    for s in secondaries:
        if isinstance(s, dict):
            # Check if the dictionary has an 'interaction' key that indicates an electron
            if "interaction" in s and s["interaction"].startswith("electron"):
                # Optionally, if energy is not directly stored, you might need to adjust
                data.append({"interaction": s["interaction"], "energy": s.get("energy", None)})
        elif isinstance(s, (list, tuple)):
            if s[0] == 'electron':
                data.append({"interaction": s[3], "energy": s[1]})
    if not data:
        print("No electron interactions found.")
        return
    df = pd.DataFrame(data)
    print("Unique interaction types:", df["interaction"].unique())
    grouped = df.groupby("interaction").agg({"energy": ["count", "mean"]})
    grouped.columns = ["Count", "Mean Energy (MeV)"]
    print("\nSecondary Electron Stats:")
    print(grouped)
    
def print_tracks_table(tracks, shower_type="MC", num_examples=5):
    import random, math, numpy as np
    # Sample exactly num_examples tracks (if available)
    sample_tracks = tracks if len(tracks) <= num_examples else random.sample(tracks, num_examples)
    
    print("=========================================")
    print(f"{shower_type} Photon Tracks")
    print("=========================================")
    for i, track in enumerate(sample_tracks, start=1):
        interactions = track.get("interactions", [])
        print(f"\nPhoton {i}: {len(interactions)} interactions")
        # Define header with the extra columns:
        header = ("Step", "Type", "Energy (MeV)", "Position (cm)", "Angle (°)", 
                  "Sec Type", "Sec Energy (MeV)", "Ejection Angle (°)")
        print("{:<5} | {:<12} | {:<20} | {:<25} | {:<10} | {:<10} | {:<20} | {:<18}".format(*header))
        print("-" * 120)
        for step, inter in enumerate(interactions, start=1):
            # Interaction type
            inter_type = inter.get("interaction", "N/A")
            # Photon energy info (in → out)
            energy_in = inter.get("photon_energy_in")
            energy_out = inter.get("photon_energy_out")
            if energy_in is not None and energy_out is not None:
                energy_str = f"{energy_in:.3f} → {energy_out:.3f}"
            else:
                energy_str = "N/A"
            # Position formatting
            pos = inter.get("position", ("N/A", "N/A", "N/A"))
            if all(isinstance(p, (int, float, np.number)) for p in pos):
                pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            else:
                pos_str = str(pos)
            # Deflection angle
            angle = inter.get("angle")
            angle_str = f"{angle:.1f}" if angle is not None else "N/A"
            # Secondary particle info:
            secondaries = inter.get("secondaries", [])
            if secondaries:
                sec_type_list = []
                sec_energy_list = []
                sec_ejection_angle_list = []
                incident_dir = inter.get("photon_incident_direction")
                for sec in secondaries:
                    # Each secondary is a tuple: (particle_type, energy, e_dir, label)
                    part_type, sec_energy, sec_e_dir, sec_label = sec
                    sec_type_list.append(sec_label)
                    sec_energy_list.append(f"{sec_energy:.3f}")
                    if incident_dir is not None and np.linalg.norm(sec_e_dir) > 1e-12:
                        dot_val = np.dot(sec_e_dir, incident_dir)
                        dot_val = np.clip(dot_val, -1.0, 1.0)
                        ejection_angle = math.degrees(math.acos(dot_val))
                        sec_ejection_angle_list.append(f"{ejection_angle:.1f}")
                    else:
                        sec_ejection_angle_list.append("N/A")
                sec_type_str = " / ".join(sec_type_list)
                sec_energy_str = " / ".join(sec_energy_list)
                sec_ejection_angle_str = " / ".join(sec_ejection_angle_list)
            else:
                sec_type_str = ""
                sec_energy_str = ""
                sec_ejection_angle_str = ""
            
            # Print the row:
            print("{:<5} | {:<12} | {:<20} | {:<25} | {:<10} | {:<10} | {:<20} | {:<18}".format(
                step, inter_type, energy_str, pos_str, angle_str, 
                sec_type_str, sec_energy_str, sec_ejection_angle_str
            ))
        print()

###############################################################################
#                  PHASE 2 WARMUP CALLBACK
###############################################################################
class Phase2WarmupCallback(BaseCallback):
    """
    • Pause learning + buffer writes for `warmup_env_steps` env steps
      immediately after Phase 2 first starts.
    • Persist a flag file so the pause happens only once across restarts.
    """
    def __init__(
        self,
        warmup_env_steps: int = 500,
        save_dir: str = ".",
        flag_fname: str = "phase2_warmup_done.flag",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.warmup_env_steps = warmup_env_steps
        self.flag_path = os.path.join(save_dir, flag_fname)

        # internal state
        self._armed = False             # currently in warm-up?
        self._env_steps_seen = 0
        self._orig_add = None
        self._skip_already_done = os.path.exists(self.flag_path)

    # ---------- helpers ------------------------------------------------
    def _unwrap(self, env):
        while not hasattr(env, "phase") and hasattr(env, "env"):
            env = env.env
        return env

    def _env0(self):
        return self._unwrap(self.training_env.envs[0])

    # ---------- main hook ---------------------------------------------
    def _on_step(self) -> bool:
        env0   = self._env0()
        n_envs = self.training_env.num_envs

        # (A) Skip everything if the flag file says warm-up already done
        if self._skip_already_done:
            return True
        buf = self.model.replay_buffer           # ← ❶  ADD THIS LINE

        # 🔑 1) Emergency self-heal: if buffer.add is still the stub, restore it
        if (not self._armed) and buf.add.__name__ == "_noop_add_to_buffer":
            buf.add = buf.__class__.add.__get__(buf, buf.__class__)
            
        # (B) Arm once when we first enter Phase 2
        if env0.phase == 2 and not self._armed:
            self._armed = True
            self._env_steps_seen = 0
            self._orig_add = self.model.replay_buffer.add
            self.model.replay_buffer.add = _noop_add_to_buffer
            if self.verbose:
                print(f"[warm-up] Phase 2 detected — pausing learning for "
                      f"{self.warmup_env_steps} env steps")

        # (C) Count env steps while armed
        if self._armed:
            self._env_steps_seen += n_envs
            if self._env_steps_seen >= self.warmup_env_steps:
                # restore buffer writer
                self.model.replay_buffer.add = self._orig_add
                self._orig_add = None
                self._armed = False
                # allow learning to resume immediately
                self.model.learning_starts = self.model.num_timesteps + 1
                # create flag file so we never arm again
                open(self.flag_path, "w").close()
                if self.verbose:
                    print("[warm-up] done. Learning and buffer writes restored.")
        return True



###############################################################################
#                  SAVE EVERYTHING PERIODICALLY
###############################################################################
class SavePKLsCallback(BaseCallback):
    def __init__(self, save_freq: int, env, save_dir: str, prefix: str = "hybrid_sac_model", max_to_keep: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq   = save_freq
        self.env         = env        # still keep this for initialization if you like
        self.save_dir    = save_dir
        self.prefix      = prefix
        self.max_to_keep = max_to_keep
        self._saved_steps: list[int] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _unwrap(self, venv):
        """Drill through any VecEnv wrappers to get at the base env."""
        base = venv
        # e.g. DummyVecEnv / SubprocVecEnv / VecNormalize, etc.
        while hasattr(base, "envs"):
            # for DummyVecEnv, base.envs is the list of wrapped envs
            base = base.envs[0]
        # then strip any Monitor/TimeLimit wrappers
        while hasattr(base, "env"):
            base = base.env
        return base

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            step = self.num_timesteps

            # 1) policy
            zip_path = os.path.join(self.save_dir, f"{self.prefix}_{step}.zip")
            self.model.save(zip_path)

            # 2) replay buffer
            rb_path = os.path.join(self.save_dir, f"replay_buffer_{step}.pkl")
            with open(rb_path, "wb") as f:
                pickle.dump(self.model.replay_buffer, f)

            # 3) phase-dependent persistence  ─────────────────────────────
            base_env = self._unwrap(self.training_env)
            if base_env.phase < 2:
                # keep histograms only for phases 0-1
                hist_path = os.path.join(self.save_dir, f"histograms_{step}.pkl")
                hist_dict = {
                    "pred_hist": base_env.pred_hist.copy(),
                    "cum_hist":  base_env.cum_pred_hist.copy(),
                }
                with open(hist_path, "wb") as f:
                    pickle.dump(hist_dict, f)
            else:
                # keep acceptance statistics for phases 2-3
                dist_path = os.path.join(self.save_dir, f"dist_stats_{step}.pkl")
                dist_stats = {
                    "angle_kl_per_bin": {
                        bin_idx: {k: float(v) for k, v in interactions.items()}
                        for bin_idx, interactions in base_env.angle_kl_per_bin.items()
                    },
                    "angle_hist_per_bin": {
                        bin_idx: {k: list(v) for k, v in interactions.items()}
                        for bin_idx, interactions in base_env.angle_hist_per_bin.items()
                    },
                    "angle_target_per_bin": {
                        bin_idx: {k: v.tolist() for k, v in interactions.items()}
                        for bin_idx, interactions in base_env.angle_target_per_bin.items()
                    },
                    "phase": base_env.phase,
                    "current_regime": getattr(base_env, 'current_regime', 0),
                }
                with open(dist_path, "wb") as f:
                    pickle.dump(dist_stats, f)

            # 4) env metadata
            env0 = self._unwrap(self.training_env.envs[0])
            meta = {
                "phase":             base_env.phase,
                "global_step_count": base_env.global_step_count
            }
            meta_path = os.path.join(self.save_dir, f"env_meta_{step}.pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)
            # 4) Physics loss values

            physics_path = os.path.join(self.save_dir, f"{self.prefix}_{step}_physics_losses.pkl")
            with open(physics_path, 'wb') as f:
                physics_data = {
                    "physics_losses": self.model.physics_losses,
                    "physics_steps": self.model.physics_steps
                }
                pickle.dump(physics_data, f)

            # record and prune old as before…
            self._saved_steps.append(step)
            if len(self._saved_steps) > self.max_to_keep:
                old_step = self._saved_steps.pop(0)
                for fname in [
                    f"{self.prefix}_{old_step}.zip",
                    f"replay_buffer_{old_step}.pkl",
                    f"histograms_{old_step}.pkl",
                    f"acc_stats_{old_step}.pkl", 
                    f"env_meta_{old_step}.pkl",
                    f"{self.prefix}_{old_step}_physics_losses.pkl" 
                ]:
                    path = os.path.join(self.save_dir, fname)
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                if self.verbose:
                    print(f"[SavePKLsCallback] removed old checkpoint: {old_step}")

            if self.verbose:
                print(f"[SavePKLsCallback] saved checkpoint: {step}")
        return True

###############################################################################
#                  PHASE SWITCH
###############################################################################
class PhaseSwitchCallback(BaseCallback):
    def __init__(self, phase_ends, verbose: int = 1):
        super().__init__(verbose)
        self.phase_ends          = phase_ends
        self.current_phase       = 0
        self.last_recorded_phase = None
        self._initial_log_alpha  = None
        self._alpha_reset_phases = set()
        self._last_regime        = None  # track last energy regime

    def _on_training_start(self) -> None:
        # pick up resumed phase
        if hasattr(self.model, "phase"):
            self.current_phase = int(self.model.phase)
        # capture α₀’s log once
        if self._initial_log_alpha is None:
            self._initial_log_alpha = float(self.model.log_ent_coef.detach().item())
            if self.verbose:
                print(f"🔰 Captured initial log-α₀ = {self._initial_log_alpha:.4f}")
        # record starting regime
        env0 = self.training_env.envs[0]
        base = getattr(env0, "env", env0)
        self._last_regime = getattr(base, "current_regime", None)

    def _on_step(self) -> bool:

        # ---- 1) existing phase‐boundary logic ----
        original_phase = self.current_phase  # Capture phase before any changes
        
        while (
            self.current_phase + 1 < len(self.phase_ends)
            and self.num_timesteps >= self.phase_ends[self.current_phase]
        ):
            self.current_phase += 1
            phase = self.current_phase

            _set_phase(self.model, phase)
            # record the step at which this phase started (for decay schedules)
            for v in self.training_env.envs:
                b = getattr(v, "env", v)
                b._phase_start_step = self.num_timesteps

        # Reset histograms if phase actually changed
        if original_phase != self.current_phase:
            print(f"🔄 Phase transition {original_phase} → {self.current_phase}: Resetting histograms and angle tracking")
            for v in self.training_env.envs:
                base = getattr(v, "env", v)
                base.reset_histogram_stats()
                base.current_regime = 0
                
                # Reset per-bin angle tracking
                if hasattr(base, 'angle_hist_per_bin'):
                    for bin_idx in range(base.N_EBINS):
                        for interaction in ["rayleigh", "compton", "photo", "pair"]:
                            base.angle_hist_per_bin[bin_idx][interaction].clear()
                            base.angle_kl_per_bin[bin_idx][interaction] = 0.0
                
                # Reset energy range to first regime
                if self.current_phase in (0, 2):
                    base.E_min = base.energy_regime_boundaries[0]
                    base.E_max = base.energy_regime_boundaries[1]
            for v in self.training_env.envs:
                b = getattr(v, "env", v)
                b.phase = phase

            # Reset energy curriculum when entering Phase 2
            if phase == 2:
                print("🔄 Entering phase 2: Resetting energy curriculum to regime 0")
                for v in self.training_env.envs:
                    base = getattr(v, "env", v)
                    base.current_regime = 0  # Start energy curriculum over
                    # Reset energy range to first regime
                    base.E_min = base.energy_regime_boundaries[0]
                    base.E_max = base.energy_regime_boundaries[1]
                    
            # reset α on first entry into phase ≥ 2
            if phase >= 2 and phase not in self._alpha_reset_phases:
                log0 = self._initial_log_alpha
                with torch.no_grad():
                    self.model.log_ent_coef.data.fill_(log0)
                self.model.ent_coef = math.exp(log0)
                self._alpha_reset_phases.add(phase)
                if self.verbose:
                    print(f"🔄 Reset α → {self.model.ent_coef:.4f} at PHASE {phase}")

        # ---- 2) new: on every regime-bump in phase 0, pre‐seed logits ----
        env0 = self.training_env.envs[0]
        base = getattr(env0, "env", env0)
        cur_reg = getattr(base, "current_regime", None)

        if self.current_phase in (0,2) and cur_reg is not None:
            if self._last_regime is None:
                self._last_regime = cur_reg
            elif cur_reg > self._last_regime:
                # 2a) reset α exactly as before
                log0 = self._initial_log_alpha
                with torch.no_grad():
                    self.model.log_ent_coef.data.fill_(log0)
                self.model.ent_coef = math.exp(log0)
                if self.verbose:
                    print(f"🔄 Reset α → {self.model.ent_coef:.4f} at REGIME {cur_reg}")

                # 2b) pre‐seed ALL discrete‐head logits for this regime’s bins
                actor = self.model.policy.actor
                # regime energy bounds from your env
                e_lo = base.energy_regime_boundaries[cur_reg]
                e_hi = base.energy_regime_boundaries[cur_reg + 1]

                # actor.ebin_edges: array of log₁₀(E) bin boundaries
                edges = actor.ebin_edges
                for b in range(len(edges) - 1):
                    mid_log = 0.5 * (edges[b] + edges[b+1])
                    E_mid   = 10 ** mid_log
                    if e_lo <= E_mid <= e_hi:
                        p_true = base.true_prob[b]  # shape (4,)
                        # store log-p into the cache so forward() will use it
                        actor.prev_logits[b] = (
                            torch.log(torch.from_numpy(p_true).float() + 1e-12)
                                  .to(actor.device)
                        )
                # advance our tracker
                self._last_regime = cur_reg

        return True



###############################################################################
#   SIMPLE CHECKPOINT CALL
###############################################################################
class OverwritingCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        
    def _unwrap(self, env):
        """
        Strip Monitor / TimeLimit wrappers until we reach the
        underlying WaterPhotonHybridEnvPenelope instance.
        """
        while hasattr(env, "env"):
            env = env.env          # unwrap one layer
        return env                 # unwrapped base env
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 1) weights
            self.model.save(self.save_path)

            # 2) replay-buffer
            with open("replay_buffer.pkl", "wb") as f:
                pickle.dump(self.model.replay_buffer, f)

            # 3) histograms (safe even with many wrappers)
            # ─── phase-dependent persistence ─────────────────────────
            env0 = self._unwrap(self.training_env.envs[0])

            if env0.phase < 2:
                hist_dict = {}
                for k, wrapped in enumerate(self.training_env.envs):
                    base = self._unwrap(wrapped)
                    hist_dict[k] = {
                        "pred_hist": base.pred_hist.copy(),
                        "cum_hist":  base.cum_pred_hist.copy(),
                    }
                with open("histograms.pkl", "wb") as f:
                    pickle.dump(hist_dict, f)
            else:
                # Save distribution statistics 
                dist_dict = {}
                for k, wrapped in enumerate(self.training_env.envs):
                    base = self._unwrap(wrapped)
                    dist_dict[k] = {
                        "angle_kl_per_bin": {
                            bin_idx: {itype: float(v) for itype, v in interactions.items()}
                            for bin_idx, interactions in base.angle_kl_per_bin.items()
                        },
                        "angle_hist_per_bin": {
                            bin_idx: {itype: list(v) for itype, v in interactions.items()}
                            for bin_idx, interactions in base.angle_hist_per_bin.items()
                        },
                        "angle_target_per_bin": {
                            bin_idx: {itype: v.tolist() for itype, v in interactions.items()}
                            for bin_idx, interactions in base.angle_target_per_bin.items()
                        },
                        "phase": base.phase,
                        "current_regime": getattr(base, 'current_regime', 0),
                    }
                with open("dist_stats.pkl", "wb") as f:
                    pickle.dump(dist_dict, f)
                    
             # 4) env metadata (unchanged)
            env0 = self._unwrap(self.training_env.envs[0])
            meta = {
                "phase":             env0.phase,
                "global_step_count": env0.global_step_count,
            }
            with open("env_meta.pkl", "wb") as f:
                pickle.dump(meta, f)


            if self.verbose:
                clear_output(wait=True)
                print(f"Saved model + replay buffer + env_meta at timestep {self.num_timesteps}")

        return True


###############################################################################
#  SIMPLE TRAIN DEMO
###############################################################################
def train_hybrid_sac(csv_path="Final_cross_sections.csv", total_timesteps=50000):
    global PHASE_ENDS
    policy_file = "hybrid_sac_model.zip"
    
    data = PenelopeLikeWaterData(final_csv_path="Final_cross_sections.csv",
                rayleigh_csv_path="Rayleigh_cross_sections.csv",density=1.0)
    env = PhasedRewardEnv(data, ecut=0.001, max_steps=100000,
                                   NsecMax=2, train_mode=True, n_multi=N_STEPS_RETURN)
    
    # Setup custom policy parameters
    def lr_schedule(_prog): 
        return 1e-4
    n_discrete = 4
    n_continuous = env.cont_dim
    policy_kwargs = dict(
        n_discrete = n_discrete,
        NsecMax       = env.NsecMax,   
        activation_fn = nn.SiLU,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = dict(eps=1e-5),
        features_extractor_class = OptimizedFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=512),
        use_sde                   = False,              # <— add this
        ebin_edges    = env.ebin_edges,
        true_prob     = env.true_prob,
        true_mfp_mean = env.true_mfp_mean,
        energy_regime_boundaries=env.energy_regime_boundaries, 
        LOG_MIN=env.LOG_MIN,
        LOG_MAX=env.LOG_MAX,
    )
    
    # If the policy file exists, load it. Otherwise, start training from scratch.
    if os.path.exists(policy_file):
        print(f"Existing policy file '{policy_file}' detected. Loading model to continue training.")

        model = NStepSAC.load(
            "hybrid_sac_model.zip",  
            env=env,
            custom_objects={
                "policy_class":HybridSACPolicy,
                "replay_buffer_class" : NStepReplayBuffer,
            },
            replay_buffer_class=NStepReplayBuffer,
            replay_buffer_kwargs=dict(
                n_steps=N_STEPS_RETURN,
                gamma=0.99,
            ),
            tensorboard_log="tb_logs/",
            verbose=1,
            learning_rate=1e-5,
            buffer_size=1000000,
            batch_size=2048,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            gamma=0.99,
            tau=0.001,
            target_entropy=-10,
            force_reset=True,
        )
#        delta = 0.6  # or whatever bump you want
#        current_alpha = math.exp(model.log_ent_coef.detach().item())
#        new_alpha = current_alpha + delta
#        new_log_alpha = math.log(new_alpha)
#        with torch.no_grad():
#             model.log_ent_coef.data.fill_(new_log_alpha)
#        model.ent_coef = new_alpha
#        print(f"🔼 Nudged α → {model.ent_coef:.4f}")
        model.policy.actor.true_mfp_mean = env.true_mfp_mean
        model.policy.actor.ebin_edges = env.ebin_edges
        model.policy.actor.true_prob = env.true_prob
        model.policy.actor.energy_regime_boundaries = env.energy_regime_boundaries 
        model.policy.actor.LOG_MIN = env.LOG_MIN
        model.policy.actor.LOG_MAX = env.LOG_MAX
        model.policy.actor.mu_total = env.data.mu_total
        if os.path.exists("replay_buffer.pkl"):
            with open("replay_buffer.pkl", "rb") as f:
                model.replay_buffer = pickle.load(f)
        else:
            print("⚠️  No 'replay_buffer.pkl' found—starting with an empty replay buffer.")
        # 3) phase-dependent state restoration ───────────────────────────
        if env.phase < 2 and os.path.exists("histograms.pkl"):
            with open("histograms.pkl", "rb") as f:
                hist_dict = pickle.load(f)
            if 0 in hist_dict:
                env.pred_hist     = hist_dict[0]["pred_hist"]
                env.cum_pred_hist = hist_dict[0]["cum_hist"]
            print(f"✅  Histogram state restored "
                  f"(total counts = {int(env.cum_pred_hist.sum())})")
        elif env.phase >= 2 and os.path.exists("acc_stats.pkl"):
            # Also load distribution matching statistics if they exist
            if env.phase >= 2 and os.path.exists("dist_stats.pkl"):
                with open("dist_stats.pkl", "rb") as f:
                    dist_dict = pickle.load(f)
                if "kl_divergences" in dist_dict:
                    env.kl_divergences = dist_dict["kl_divergences"]
                if "dist_rewards" in dist_dict:
                    env.dist_rewards = dist_dict["dist_rewards"]
                if "angle_history" in dist_dict:
                    for k, v in dist_dict["angle_history"].items():
                        env.agent_angle_history[k] = deque(v, maxlen=1000)
                if "target_distributions" in dist_dict:
                    for k, v in dist_dict["target_distributions"].items():
                        env.target_angle_distributions[k] = np.array(v)
                print(f"✅  Distribution matching stats restored")

        else:
            print("⚠️  No saved histogram/distribution data found — starting from zero.")
        if os.path.exists("env_meta.pkl"):
            meta = pickle.load(open("env_meta.pkl","rb"))
            env.phase             = meta["phase"]
            env.global_step_count = meta["global_step_count"]
        else:
            print("⚠️  No phase or global step count meta data found — starting them from zero.")
        env = model.get_env()
        print(type(env), getattr(env, "num_envs", None))
        # 3) Physics loss restoration ───────────────────────────
        physics_path = policy_file.replace('.zip', '_physics_losses.pkl')
        if os.path.exists(physics_path):
            try:
                with open(physics_path, 'rb') as f:
                    physics_data = pickle.load(f)
                    model.physics_losses = physics_data["physics_losses"]
                    model.physics_steps = physics_data["physics_steps"]
                print("✅ Physics loss history loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load physics loss data: {e}")
                # Initialize empty loss tracking if loading fails
                model.physics_losses = {
                    "energy_loss": [],
                    "angle_loss": [],
                    "nsec_loss": [],
                    "proc_loss": [],
                    "norm_pen": [],
                    "total_phys_loss": []
                }
                model.physics_steps = []
        # --- ensure checkpoint uses n‑step buffer -------------------------
        if not isinstance(model.replay_buffer, NStepReplayBuffer):
            model.replay_buffer = NStepReplayBuffer(
                buffer_size            = model.replay_buffer.buffer_size,
                observation_space      = env.observation_space,
                action_space           = env.action_space,
                device                 = model.device,
                n_envs                 = 1,
                gamma                  = 0.995,
                n_steps                = N_STEPS_RETURN,
                optimize_memory_usage  = False,
            )
        phase_ends   = PHASE_ENDS
        # if you stored model.phase last time, just reuse it:
        current_phase = getattr(model, "phase", 0)

        # …otherwise infer it from the accumulated timesteps
        if not hasattr(model, "phase"):
            t = model.num_timesteps
            current_phase = 0
            while (current_phase + 1 < len(phase_ends)
                   and t >= phase_ends[current_phase]):
                current_phase += 1

        # propagate to every env so _mask_cont() etc. read the right value
        _set_phase(model, current_phase)      # copies to model & envs
        print(f"✅  Resuming in phase {current_phase}  "
              f"(num_timesteps={model.num_timesteps})")

# ------------------------------------------------------------------
    else:
        print("No existing policy file found. Starting training from scratch.")
        model = NStepSAC(
            policy = HybridSACPolicy,
            env = env,
            policy_kwargs = policy_kwargs,
            replay_buffer_class  = NStepReplayBuffer,
            replay_buffer_kwargs = dict(
                n_steps = N_STEPS_RETURN,   # 25‑step return
                gamma   = 0.99
            ),
            tensorboard_log="tb_logs/",
            verbose = 1,
            learning_rate = lr_schedule,
            buffer_size = 1000000,
            batch_size = 2048,
            train_freq = 1,
            gradient_steps = 1,
            ent_coef = 'auto',
            gamma = 0.99,
            tau = 0.001,
            target_entropy = -10
            # max_grad_norm = 0.5  # Add gradient clipping - might need tuning
        )
        model.policy.actor.true_mfp_mean = env.true_mfp_mean
        model.policy.actor.ebin_edges = env.ebin_edges
        model.policy.actor.true_prob = env.true_prob
        model.policy.actor.energy_regime_boundaries = env.energy_regime_boundaries
        model.policy.actor.LOG_MIN = env.LOG_MIN
        model.policy.actor.LOG_MAX = env.LOG_MAX
        model.policy.actor.mu_total = env.data.mu_total
        # 3) phase-dependent state restoration ───────────────────────────
        if env.phase < 2 and os.path.exists("histograms.pkl"):
            with open("histograms.pkl", "rb") as f:
                hist_dict = pickle.load(f)
            if 0 in hist_dict:
                env.pred_hist     = hist_dict[0]["pred_hist"]
                env.cum_pred_hist = hist_dict[0]["cum_hist"]
            print(f"✅  Histogram state restored "
                  f"(total counts = {int(env.cum_pred_hist.sum())})")
        elif env.phase >= 2 and os.path.exists("acc_stats.pkl"):
            # Also load distribution matching statistics if they exist
            if env.phase >= 2 and os.path.exists("dist_stats.pkl"):
                with open("dist_stats.pkl", "rb") as f:
                    dist_dict = pickle.load(f)
                if "kl_divergences" in dist_dict:
                    env.kl_divergences = dist_dict["kl_divergences"]
                if "dist_rewards" in dist_dict:
                    env.dist_rewards = dist_dict["dist_rewards"]
                if "angle_history" in dist_dict:
                    for k, v in dist_dict["angle_history"].items():
                        env.agent_angle_history[k] = deque(v, maxlen=1000)
                if "target_distributions" in dist_dict:
                    for k, v in dist_dict["target_distributions"].items():
                        env.target_angle_distributions[k] = np.array(v)
                print(f"✅  Distribution matching stats restored")
        else:
            print("⚠️  No saved histogram/distribution data found — starting from zero.")

        _set_phase(model, 0)      # only done on a brand-new network
    physics_ckpt = "physics_head_pretrained.pth"
    if os.path.exists(physics_ckpt):
        load_physics_head(model.policy.actor, physics_ckpt)
    else:
        print("⚠️  No pre-trained physics head found; training from scratch.") 

    ckpt_cb = OverwritingCheckpointCallback(save_freq=1000, save_path=policy_file, verbose=1)
    # === NEW: full state backup every 10k steps ===
    save_pkls_cb = SavePKLsCallback(
        save_freq=5_000,
        env=env,
        save_dir="older",
        prefix="hybrid_sac_model",
        max_to_keep=5,
        verbose=1
    )
    phase_ends = PHASE_ENDS
    model.policy.actor._skip_phys = False 
    torch.cuda.empty_cache()      # free up reserved but unused memory
    model.learn(
        total_timesteps = total_timesteps, 
        log_interval = 100, 
        callback=[
            ckpt_cb, 
            PhaseSwitchCallback(phase_ends),
            save_pkls_cb,
            BufferPeekCallback(every=500, tail=10, verbose=1)
        ], 
        reset_num_timesteps=False
    )
    print_last_transitions(model, last_k=12)
    
    return model, env


###############################################################################
#   MAIN: Example usage
###############################################################################

def main():
    csv_path = "NIST_WaterCrossSections.csv"
    data = PenelopeLikeWaterData(
        final_csv_path="Final_cross_sections.csv",
        rayleigh_csv_path="Rayleigh_cross_sections.csv",
        density=1.0
    )

    mode = input("Enter 'train' to train a new model or 'eval' to evaluate an existing model: ").strip().lower()
    phase_ends   = PHASE_ENDS          # global list [n₀, n₁, …]

    if mode == "train":
        timesteps_str = input("Enter the total timesteps for training (e.g., 100000): ").strip()
        try:
            total_timesteps = int(timesteps_str)
        except ValueError:
            print("Invalid timesteps value; using default 50000.")
            total_timesteps = 50000
        fixed_energy_str = input("Enter the fixed photon energy (in MeV) to use for evaluation after training (e.g., 3): ").strip()
        try:
            fixed_energy = float(fixed_energy_str)
        except ValueError:
            print("Invalid fixed energy value; using default 3 MeV.")
            fixed_energy = 3
        print(f"Training mode selected. Training for {total_timesteps} timesteps. Evaluation will use fixed energy {fixed_energy} MeV.")
        model, env = train_hybrid_sac(csv_path, total_timesteps=total_timesteps) 
        # After training, you can assign the fixed energy for evaluation runs.
        env.fixed_energy = fixed_energy
        phase_for_eval = getattr(model, "phase", 0)
    elif mode == "eval":
        fixed_energy_str = input("Enter the fixed photon energy (in MeV) to use for evaluation (e.g., 3): ").strip()
        try:
            fixed_energy = float(fixed_energy_str)
        except ValueError:
            print("Invalid fixed energy value; using default 3 MeV.")
            fixed_energy = 3
        print(f"Evaluation mode selected. Loading existing policy with fixed energy {fixed_energy} MeV.")
        dummy_env = WaterPhotonHybridEnvPenelope(data, ecut=0.001, max_steps=100000, NsecMax=2, train_mode=False, n_multi=N_STEPS_RETURN)
        dummy_env.use_gymnasium_api = False
        dummy_env.fixed_energy = fixed_energy
        dummy_env.E = fixed_energy  # Set the starting energy to the fixed value.  
        policy_kwargs = dict(
            n_discrete                = 4,
            NsecMax                   = dummy_env.NsecMax,
            activation_fn             = nn.SiLU,
            optimizer_class           = torch.optim.Adam,
            optimizer_kwargs          = dict(eps=1e-5),
            features_extractor_class  = OptimizedFeatureExtractor,
            features_extractor_kwargs = dict(features_dim=512),
            use_sde                   = False,              # <— add this
        )
        
        # ─────────────────────────────────────────────────────────
        # load checkpoint if it exists, otherwise abort cleanly
        # ─────────────────────────────────────────────────────────
        policy_file = "hybrid_sac_model.zip"
        if not os.path.isfile(policy_file):
            print(f"❌ No trained model found at '{policy_file}'. "
                  f"Run in 'train' mode first.")
            return                    # ← exits main() early

        print(f"✅  Loading trained model from '{policy_file}' …")

        model = NStepSAC.load(
            policy_file,
            env=dummy_env,
            custom_objects={
                "policy_class": HybridSACPolicy,
                "replay_buffer_class": NStepReplayBuffer,
            },
            replay_buffer_class = NStepReplayBuffer,
            replay_buffer_kwargs = dict(
                n_steps = N_STEPS_RETURN,
                gamma   = 0.99,
            ),
            tensorboard_log = "tb_logs/",
            verbose = 1,
            learning_rate = 1e-5,
            buffer_size = 1_000_000,
            batch_size = 2048,
            train_freq = 1,
            gradient_steps = 1,
            ent_coef = "auto",
            gamma = 0.99,
            tau = 0.001,
            target_entropy = -10,
            force_reset = True,
        )
        with open("replay_buffer.pkl", "rb") as f:
            model.replay_buffer = pickle.load(f)
        # ------------------------------------------------------------------
        # Hard-freeze actor for evaluation — no curriculum, no training
        # ------------------------------------------------------------------
        phase_for_eval = 3            # ≥3  ⇒  nothing is masked/frozen
        dummy_env.phase = phase_for_eval

        actor = model.policy.actor
        actor.eval()                  # switch LayerNorm/Dropout to eval

        with torch.no_grad():
            actor.freeze_mu_residual()                        # zero μ-residual
            actor.freeze_gaussian_sigma(target_dim=-1, log_std_value=-4.5)  # lock *all* σ
            actor.logits_buffer.requires_grad_(False)         # logits table const
            actor.theta_buffer .requires_grad_(False)
            # keep physics arrays in sync with env
            actor.true_mfp_mean            = dummy_env.true_mfp_mean
            actor.ebin_edges               = dummy_env.ebin_edges
            actor.true_prob                = dummy_env.true_prob
            actor.energy_regime_boundaries = dummy_env.energy_regime_boundaries
            actor.LOG_MIN                  = dummy_env.LOG_MIN
            actor.LOG_MAX                  = dummy_env.LOG_MAX
            actor.mu_total                 = dummy_env.data.mu_total
        # ------------------------------------------------------------------


        env = dummy_env  # Use this env for subsequent evaluation.
        
    # Create evaluation environment using the fixed energy if in eval mode.
    # In training mode, we use the default (3 MeV) for evaluation.
    def create_eval_env():
        eval_env = WaterPhotonHybridEnvPenelope(
            data=PenelopeLikeWaterData(
                final_csv_path="Final_cross_sections.csv",
                rayleigh_csv_path="Rayleigh_cross_sections.csv",
                density=1.0
            ),
            train_mode = False, 
            ecut=0.001, 
            max_steps=100000, 
            NsecMax=2, 
            n_multi=25
        )
        eval_env.force_mc_interaction = False        #  ← make sure!
        eval_env.use_gymnasium_api = False
        eval_env.xmin = -50.0
        eval_env.xmax = 50.0
        eval_env.ymin = -50.0
        eval_env.ymax = 50.0
        eval_env.zmin = 0.0
        eval_env.zmax = 100.0
        # Use the provided fixed energy in eval mode or default 3 MeV in training mode.
        eval_env.fixed_energy = fixed_energy
        eval_env.E = fixed_energy
        return eval_env

    # Create separate evaluation environments for Monte Carlo (MC) and the agent.
    mc_env = create_eval_env()
    agent_env = create_eval_env()

    mc_env.phase    = phase_for_eval
    agent_env.phase = phase_for_eval
    # Run MC showers (pure simulation)
    n_photons_str = input("Enter the number of photons to run (default: 10000): ").strip()
    if n_photons_str == "":
        n_photons = 10000
    else:
        try:
            n_photons = int(n_photons_str)
        except ValueError:
            print("Invalid input; defaulting to 10000.")
            n_photons = 10000
    print("Running MC showers...")
    start_mc = time.time()
    mc_tracks, mc_secondaries, mc_dose, mc_interactions = run_mc_shower(n_photons, data, env=mc_env, ecut=0.001)
    mc_time = time.time() - start_mc
    print(f"MC simulation time: {mc_time:.2f} seconds")
    
    # Run Agent showers
    print("Running Agent showers...")
    start_agent = time.time()
    
    if hasattr(model, "phase") and hasattr(agent_env, "phase"):
        agent_env.phase = model.phase          # e.g. 2
    actor = model.policy.actor
    actor.ebin_edges = agent_env.ebin_edges
    actor.true_prob  = agent_env.true_prob
    with torch.no_grad():    # no gradients, buffers stay frozen
        agent_tracks, agent_secondaries, agent_dose, agent_interactions = run_agent_shower(
            n_photons, model, env=agent_env, batch_size=512, 
            E_fixed=fixed_energy, max_steps=100000, ecut=0.001
        )

    
    agent_time = time.time() - start_agent
    print(f"Agent simulation time: {agent_time:.2f} seconds")
    agent_dose = agent_env.dose_tally.copy()

    # Plot shower comparisons:
    plot_shower_comparison(mc_tracks, agent_tracks, agent_env)
    plot_plane_cuts(mc_tracks, "MC")
    plot_plane_cuts(agent_tracks, "Agent")
    print("Plotting MC PDD:")
    plot_pdd(mc_dose, mc_env)
    print("Plotting Agent PDD:")
    plot_pdd(agent_dose, agent_env)
    
    # Analyze secondaries:
    print("\nMC Secondary Analysis:")
    analyze_secondaries(mc_secondaries)
    print("\nAgent Secondary Analysis:")
    analyze_secondaries(agent_secondaries)
    
    # New analysis: Compare interaction statistics for MC and Agent:
    mc_stats = analyze_interaction_stats(mc_interactions, title="MC Interaction Analysis")
    agent_stats = analyze_interaction_stats(agent_interactions, title="Agent Interaction Analysis")

    print_tracks_table(mc_tracks, shower_type="Monte Carlo")
    print_tracks_table(agent_tracks, shower_type="Agent")


if __name__=="__main__":
    import os
    required_files = [
        "Final_cross_sections.csv",
        "Rayleigh_cross_sections.csv",
        "WaterPhotoShells.csv",
        "water_fq.csv",
        "water_sq.csv",
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing required files: {missing}")
        print(f"Current directory: {os.getcwd()}")
        exit(1)
    # 1) Dataset generation prompt

    gen_query = input("Do you want to generate MC dataset now? (y/n): ").strip().lower()
    if gen_query in ["y","yes"]:
        # ask number of samples
        try:
            ns = int(input("Enter number of samples (e.g. 100000): ").strip())
        except:
            ns = 100_000
        # ask fixed photon energy
        # ask photon-energy range
        try:
            e_min = float(input("Enter MIN photon energy in MeV (e.g. 0.001): ").strip())
        except:
            e_min = 0.001
        try:
            e_max = float(input("Enter MAX photon energy in MeV (e.g. 1.0): ").strip())
        except:
            e_max = 1.0
        generate_mc_dataset(n_samples=ns, energy_range=(e_min, e_max), NsecMax=2)
        print("Exiting after dataset generation.")

    # 2) Physics‐head pre‐training prompt
    # Dummy energy value – only needed to instantiate a temporary env
    fe = 0.001      # MeV 
    pretrain_query = input("Do you wish to pre-train the physics head with MC data? (y/n): ").strip().lower()
    if pretrain_query in ['y', 'yes']:
        physics_ckpt = "physics_head_pretrained.pth"

        # 1) Build a throw-away env to retrieve obs/action spaces
        data_tmp = PenelopeLikeWaterData(
            final_csv_path="Final_cross_sections.csv",
            rayleigh_csv_path="Rayleigh_cross_sections.csv",
            density=1.0
        )
        env_tmp = WaterPhotonHybridEnvPenelope(
            data_tmp,
            ecut=0.001,
            max_steps=1,     # only need obs_dim
            NsecMax=2,
            train_mode=False,
            fixed_energy=fe,
            n_multi=N_STEPS_RETURN
        )
        env_tmp.E = fe

        env_tmp.initialize_energy_bins()
        LOG_MIN, LOG_MAX = math.log10(0.001), math.log10(1.001)  
        # 2) Instantiate the policy *directly* to get its .actor
        policy_tmp = HybridSACPolicy(
            observation_space=env_tmp.observation_space,
            action_space=env_tmp.action_space,
            lr_schedule=lambda _: 1e-4,  # dummy
            n_discrete=4,
            NsecMax=env_tmp.NsecMax,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
            # ---------- NEW: give the actor its physics-head arrays -----
            ebin_edges        = env_tmp.ebin_edges,
            true_prob         = env_tmp.true_prob,
            true_mfp_mean     = env_tmp.true_mfp_mean,
            energy_regime_boundaries = env_tmp.energy_regime_boundaries,
            LOG_MIN           = LOG_MIN,
            LOG_MAX           = LOG_MAX,
            # -------------------------------------------------------------
        ).to(env_tmp.device)
        actor = policy_tmp.actor

        # 3) Continue training or load from checkpoint
        if os.path.exists(physics_ckpt):
            ans = input(
                f"A pre-trained physics head exists at '{physics_ckpt}'.\n"
                "Do you wish to train more epochs on it? (y/n): "
            ).strip().lower()
            if ans in ('y', 'yes'):
                print("🚀 Continuing physics-head pre-training…")
                pretrain_physics_head(
                    actor,
                    "mc_physics_data.npz",
                    epochs=500,
                    batch_size=2048,
                    lr=5e-5,
                    save_path=physics_ckpt
                )
            else:
                load_physics_head(actor, physics_ckpt)
        else:
            print("🚀 Starting physics-head pre-training…")
            pretrain_physics_head(
                actor,
                "mc_physics_data.npz",
                epochs=500,
                batch_size=2048,
                lr=5e-5,
                save_path=physics_ckpt
            )

        print("✅ Physics head is ready. Exiting.")


        
    profile_query = input("Do you want to profile the script? (y/n): ").strip().lower()
    if profile_query in ['y', 'yes']:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        main()   # Run the main function under profiling
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        pr.dump_stats("profile_stats.prof")
        ps.print_stats()
        print(s.getvalue())
    else:
        main()
