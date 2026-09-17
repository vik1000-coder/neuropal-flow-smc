r"""Biologically-principled lag-correspondence config  (EDITABLE / PROVISIONAL).

This is the single place to edit the biological assumptions of the lag-metric analysis.
All expected timescales are PROVISIONAL and pending review by neurobiologists — change them
here and the whole analysis + figures update. Nothing biological is hardcoded elsewhere.

Two things live here:
  1. The lag grid (frames <-> seconds) the analysis scans.
  2. The expected timescale / channel for each reference FAMILY, and the BAND scheme used to
     turn "where does the correspondence peak" into a biologically-scored quantity.

Convention: at 4 Hz, lag L frames = L/4 seconds. Signal is deconvolved before fitting.
"""
from __future__ import annotations

FPS = 4.0
LAG_FRAMES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]          # 0.25 .. 10 s
LAG_SECONDS = [l / FPS for l in LAG_FRAMES]

# Denoising-noise ladder for the coupled multi-scale score estimator (new-levers #8). One
# place so the analysis + figures read the same grid.
SIGMA_FRAC_LADDER = [0.0, 0.25, 0.5, 1.0]

# --------------------------------------------------------------------------------------
# EXPECTED BIOLOGY per reference family.  expected_channel: which SID channel theory says
# should carry the correspondence ('mean' = level drive; 'distributional' = gain/tail).
# expected_speed: 'fast' or 'slow' (the ACTIVE 2-band scheme).  band_4 is the provisional
# finer label kept for the follow-up analysis (see BANDS below).  All PROVISIONAL.
# --------------------------------------------------------------------------------------
FAMILIES = {
    "gap": {
        "mechanism": "electrical synapse (gap junction)",
        "expected_speed": "fast",
        "band_4": "instantaneous",
        "expected_channel": "mean",
        "rationale": "electrical coupling is ~instantaneous; on 4 Hz calcium it appears at the "
                     "shortest resolvable lag (level co-fluctuation).",
    },
    "chemical": {
        "mechanism": "chemical synapse (ionotropic, fast)",
        "expected_speed": "fast",
        "band_4": "fast",
        "expected_channel": "mean",
        "rationale": "ionotropic transmission is sub-second; smeared by calcium kinetics to <~1 s.",
    },
    "monoamine": {
        "mechanism": "monoamine (metabotropic GPCR)",
        "expected_speed": "slow",
        "band_4": "intermediate",
        "expected_channel": "distributional",
        "rationale": "GPCR second-messenger cascades act over ~seconds; a gain/excitability effect.",
    },
    "neuropeptide": {
        "mechanism": "neuropeptide (extrasynaptic peptidergic GPCR)",
        "expected_speed": "slow",
        "band_4": "slow",
        "expected_channel": "distributional",
        "rationale": "extrasynaptic diffusion + slow GPCR kinetics act over seconds to tens of "
                     "seconds; the slowest, most distributional modulation.",
    },
    "functional": {
        # Randi/Leifer causal signal-propagation atlas (new-levers #1). MIXED timescale: on
        # 4 Hz calcium it is wired-fast-dominated, so band values are provisional; this target
        # is used mainly as an AUROC target (all channels) and for the per-edge KINETICS test.
        "mechanism": "measured causal functional connectome (optogenetic stim -> response)",
        "expected_speed": "fast",
        "band_4": "fast",
        "expected_channel": "mean",
        "rationale": "volume-transmission-inclusive causal effect; a directed functional edge "
                     "reads most like predictive drive, but gain/tail are scored too (exploratory).",
    },
}

# --------------------------------------------------------------------------------------
# BAND scheme.  ACTIVE = the coarse fast/slow split (what the data can honestly resolve at
# 4 Hz).  FOUR_BAND is documented and ready but INACTIVE — switch ACTIVE_BANDS to "four"
# only after the coarse analysis is shown to work (per project decision, 2026-07-09).
# Bands are in SECONDS; a lag belongs to a band if band_lo <= lag_s <= band_hi.
# --------------------------------------------------------------------------------------
TWO_BAND = {
    "fast": (0.25, 1.0),        # frames 1-4
    "slow": (2.5, 10.0),        # frames 10-40   (1.0-2.5 s is an intentional transition gap)
}
FOUR_BAND = {                   # PROVISIONAL, INACTIVE — the follow-up resolution
    "instantaneous": (0.25, 0.5),
    "fast": (0.5, 1.5),
    "intermediate": (1.5, 5.0),
    "slow": (5.0, 10.0),
}
ACTIVE_BANDS = "two"           # "two" (fast/slow) | "four"


def bands() -> dict:
    return TWO_BAND if ACTIVE_BANDS == "two" else FOUR_BAND


def expected_band_of(family: str) -> str:
    """The band label this family is expected to peak in, under the ACTIVE scheme."""
    if ACTIVE_BANDS == "two":
        return FAMILIES[family]["expected_speed"]
    return FAMILIES[family]["band_4"]


# SID channels analysed, grouped by kind (mean = classical analog; the rest are distributional)
CHANNELS = ["mean", "gain", "tail"]
DISTRIBUTIONAL = ["gain", "tail"]

# -------- pre-registration / scope (from the adversarial metric review, 2026-07-09) --------
# ONE pre-specified confirmatory test; everything else is exploratory (multiplicity honesty).
CONFIRMATORY = {"reference": "neuropeptide:all", "channel": "gain",
                "statistic": "dist_minus_mean", "band": "slow"}

# References whose signalling is biophysically MIXED fast+slow (so NOT a clean slow anchor):
#   serotonin has ionotropic MOD-1 (a serotonin-gated Cl- channel) alongside metabotropic
#   ser-1/4/7; tyramine has ionotropic lgc-55 alongside metabotropic tyra-2/3.
MIXED = {"monoamine:serotonin", "monoamine:tyramine"}

# Caveats surfaced in every report so we do not over-read the metric:
CAVEATS = {
    "kernel_floor": "GCaMP + OASIS AR(1) deconvolution (gamma clipped to 0.98) impose a common "
                    "calcium-decay floor of ~1.5-12 s that convolves EVERY reference identically "
                    "and can exceed the timescale differences being tested. => absolute latencies "
                    "(seconds) are inflated/untrustworthy; only RELATIVE/ORDINAL and WITHIN-FIT "
                    "channel-contrast readouts are defensible. Sub-second (gap vs chemical) bands "
                    "are UNRESOLVABLE at 4 Hz and are folded into one 'fast/wired' prior.",
    "tail_channel": "The tail/burst channel is weakly motivated for graded, mostly non-spiking "
                    "C. elegans neurons and is sensitive to OASIS non-negativity; report gain as "
                    "the primary distributional channel and tail as exploratory only.",
    "global_mode": "Whole-brain C. elegans activity is dominated by a shared global behavioural "
                   "state; the circular-shift null destroys it, so a global-mode-preserving "
                   "control (partial out top global PCs / shared-covariance surrogate) is required "
                   "before any directed-coupling claim.",
    "right_censor": "The neuropeptide expected peak (~7 s) may exceed the 10 s grid; its band "
                    "score / latency are LOWER BOUNDS ('monotone-to-edge'), not a confirmed "
                    "in-band peak.",
    "circularity": "Template centres are PROVISIONAL and partly informed by prior observations on "
                   "this data lineage; require the fast/slow sign + ordering to be invariant under "
                   "a +/-1-octave centre sweep and a template-free coarse fast/slow fallback, and "
                   "de-circularise via cross-strain (OH15500 vs OH16230) replication.",
}

# Provenance note surfaced in every generated report/figure directory.
PROVISIONAL_NOTE = (
    "Expected timescales/bands are PROVISIONAL (set 2026-07-09), pending review by "
    "neurobiologists. Edit sid_elegans/biolag/config.py to update; the analysis and figures "
    "regenerate from it. Active scheme: COARSE fast/slow (the 4-band is a documented follow-up, "
    "and is currently NOT resolvable given the calcium-kernel floor -- see CAVEATS)."
)
