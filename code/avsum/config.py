"""Central configuration. Every tunable lives here so run.sh, batch and streaming use identical settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent          # <repo>/code
REPO_DIR = CODE_DIR.parent                                  # <repo>
DATA_DIR = CODE_DIR / "data"
WEIGHTS_DIR = REPO_DIR / "weights"

DEFAULT_MODEL = "google/siglip-so400m-patch14-384"


def _default_team_id() -> str:
    env = os.environ.get("TEAM_ID", "").strip()
    if env:
        return env
    f = REPO_DIR / "team_id.txt"
    if f.exists():
        t = f.read_text(encoding="utf-8").strip()
        if t:
            return t
    return "TEAM_ID_NOT_SET"


@dataclass
class Config:
    # identity
    team_id: str = field(default_factory=_default_team_id)

    # model
    model: str = field(default_factory=lambda: os.environ.get("AVSUM_MODEL", DEFAULT_MODEL))
    device: str = "auto"                 # auto | cuda | cpu
    batch_size: int = 32

    # decoding
    sample_fps: float = 2.0              # frames per second fed to the encoder
    hwaccel: str = "auto"                # auto | cuda | none
    keyframe_mode: str = "auto"          # auto | on | off  (decode only I-frames when they are dense enough)

    # highlights
    budget_ratio: float = 0.15           # official target: total highlight length <= 15% of duration
    budget_margin_sec: float = 0.25      # stay safely under the budget after rounding
    min_seg_sec: float = 4.1             # official minimum is 4.0 s; 4.1 survives 2-decimal rounding
    coverage_seg_sec: float = 4.3        # segment length for action-coverage (kitchen) mode
    f1_seg_lengths: tuple = (4.1, 6.0, 8.0, 10.0, 12.0)
    seg_gap_sec: float = 0.3             # strict non-overlap gap between highlight segments
    kitchen_threshold: float = 0.30      # P(cooking) above which the kitchen/action branch is used

    # actions
    max_actions: int = 2000              # official cap
    action_window_secs: tuple = (1.0, 2.0, 3.5, 6.0)
    action_topk: int = 3                 # verb-noun hypotheses kept per proposal window
    action_nms_iou: float = 0.7
    prior_min_count: int = 2             # keep verb-noun pairs seen >= N times in EPIC-100 *train*
    prior_weight: float = 0.35

    # bookkeeping
    duration_tolerance: float = 0.3      # if probe vs manifest differ more than this, trust manifest
    verbose: bool = True

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "__")
