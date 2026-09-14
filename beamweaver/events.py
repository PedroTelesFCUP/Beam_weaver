"""Beam Weaver: events."""

# Copyright (c) 2025–2026 Pedro Teles. Apache-2.0.

from dataclasses import dataclass, asdict


@dataclass
class PhotonCollisionEvent:
    process: int          # index into PROCESS_NAMES
    shell: int            # 0..4 for photo, -1 otherwise
    theta_deg: float
    phi_deg: float
    pair: tuple = None    # (frac, e_th, e_ph, p_th, p_ph); frac=T−/(T−+T+)
