"""Beam Weaver: materials."""

# Copyright (c) 2025–2026 Pedro Teles & João Melo. Apache-2.0.

import math
import numpy as np
import pandas as pd
import random

from .constants import (
    COHERENT_FORM_FACTOR_TABLE,
    HC_KEV_A,
    N_SHELLS,
    PHOTO_SHELL_TABLE,
    SHELL_CROSS_SECTION_FLOOR_CM2_G,
    SHELL_INDEX,
)


class WaterPhotoShellData:
    def __init__(self, csv_path=PHOTO_SHELL_TABLE):
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

    def shell_probs(self, E):
        """Normalised shell probabilities
        p(H|E, photo) from the SAME log-log interpolators sample_shell()
        uses — read-only, consumes NO random numbers.  Order follows
        [H_K, O_K, O_L1, O_L2, O_L3]."""
        vals = [self._loglog_interp(E, self.Egrid, v) for v in
                (self.HKvals, self.OKvals, self.OL1vals,
                 self.OL2vals, self.OL3vals)]
        tot = sum(vals)
        if tot < SHELL_CROSS_SECTION_FLOOR_CM2_G:
            return [0.0] * N_SHELLS
        return [v / tot for v in vals]

    def sample_shell(self, E):
        HK  = self._loglog_interp(E, self.Egrid, self.HKvals)
        OK  = self._loglog_interp(E, self.Egrid, self.OKvals)
        OL1 = self._loglog_interp(E, self.Egrid, self.OL1vals)
        OL2 = self._loglog_interp(E, self.Egrid, self.OL2vals)
        OL3 = self._loglog_interp(E, self.Egrid, self.OL3vals)

        total = HK + OK + OL1 + OL2 + OL3
        if total < SHELL_CROSS_SECTION_FLOOR_CM2_G:
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


class WaterPhotonData:
    def __init__(
        self,
        final_csv_path: str,
        rayleigh_csv_path: str,
        density=1.0,
        water_shell_csv=PHOTO_SHELL_TABLE,
        coherent_ff_csv=COHERENT_FORM_FACTOR_TABLE,
    ):
        # Load Rayleigh (coherent) cross-sections
        df_rayleigh = pd.read_csv(rayleigh_csv_path)
        self.E_coh = df_rayleigh["E"].values
        self.sigma_coh = df_rayleigh["coh"].values
        # Sort Rayleigh data
        sort_idx_coh = np.argsort(self.E_coh)
        self.E_coh = self.E_coh[sort_idx_coh]
        self.sigma_coh = self.sigma_coh[sort_idx_coh]

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
        
        self.HC_KEV_A = HC_KEV_A
        self.F0 = self.coherent_form_factor(0.0)

        # ── Rayleigh q²-CDF for numerical inverse-CDF sampling ────────────
        # Cumulative of ∫ F²(q) dq² expressed in the Hubbell table variable
        # x = q/2 (so dq² = 8x dx):  g(x) = F(x)² · 8x, trapezoid-integrated on
        # a refined grid (≈50 sub-points per table interval, first interval
        # refined linearly to capture the behaviour near x = 0) built from the
        # SAME piecewise-linear F(x) that coherent_form_factor() interpolates —
        # so sampling and angular validation share the same form factor.
        # sample_rayleigh_event
        # inverts this CDF per event via searchsorted-style np.interp.
        _sub = 50
        _xg = [np.array([self.ff_x[0]])]
        for _i in range(len(self.ff_x) - 1):
            _seg = np.linspace(self.ff_x[_i], self.ff_x[_i + 1], _sub + 1)[1:]
            _xg.append(_seg)
        _xg = np.concatenate(_xg)
        _Fg = np.interp(_xg, self.ff_x, self.ff_F,
                        left=self.ff_F[0], right=self.ff_F[-1])
        _g  = (_Fg ** 2) * 8.0 * _xg                       # F²(q) dq²/dx
        _cdf = np.concatenate(([0.0],
                np.cumsum(0.5 * (_g[1:] + _g[:-1]) * np.diff(_xg))))
        self._ray_x_grid = _xg
        self._ray_cdf    = _cdf

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
        


        
    def sample_photo_shell_index(self, E):
        name,_ = self.water_shell_data.sample_shell(E)
        if name is None: 
            return None
        return SHELL_INDEX[name]
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
