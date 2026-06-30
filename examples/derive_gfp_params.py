"""Derive gfp-mask ``params.json`` from a few intensity readings off your viewer.

The mask pipeline applies flat-field correction *before* global normalization, so the
quantities that matter are not raw intensities but **contrast above local background**::

    corrected = max(raw - background, 0)        # subtract / white top-hat
    normalized = clip(corrected, norm_lower, norm_upper) -> [0, 1]

That means ``norm_lower`` / ``norm_upper`` are in **contrast units** (corrected space),
while the 1st/99th percentiles you read in a viewer are **raw**. This script converts
raw readings -> contrast and turns them into a coherent parameter set, so you don't have
to reason about the conversion by hand for each new dataset.

What you measure in the viewer (raw intensity), large-scale where possible
-------------------------------------------------------------------------
- ``--empty``      : value inside the empty / no-data blocks (often 0 in a fused stack).
- ``--background`` : typical *tissue background* level (the speckle between structures).
                     A robust proxy is the large-scale ~50th percentile of tissue-only
                     regions; the raw 1st percentile is usually background-or-empty.
- ``--p99`` (or ``--bright``) : the bright end of real signal (large-scale 99th
                     percentile, or the value of a clearly-real bright structure).
- ``--dim-signal`` : (optional) the *faintest* real structure you must keep, e.g. a
                     spine neck. Sets ``threshold``. Omit to use a default.
- ``--rim``        : (optional) the intensity of the *false positives* you want gone
                     (boundary rim / tile seams). Sets ``seed_threshold`` above it. Omit
                     to use a default.
- ``--background-noise`` : (optional) std of the tissue background. Sets the contrast
                     floor (``norm_lower``). If omitted, it is **inferred from the raw
                     1st percentile**: a normal background has its 1st percentile ~2.326
                     sigma below the median, so ``sigma ~= (background - p1) / 2.326``.

The mapping (subtract mode), all in contrast C = raw - background
-----------------------------------------------------------------
    norm_upper = p99 - background                  # bright-core contrast -> ~1.0
    background_noise = (background - p1) / 2.326    # inferred when not given
    norm_lower = noise_sigmas * background_noise    # background floor -> clipped to 0
                 (fallback: 5% of norm_upper)
    threshold      = contrast(dim_signal) mapped into [norm_lower, norm_upper]
    seed_threshold = contrast(rim) mapped, then pushed up by --seed-margin so the rim
                     has no seed and hysteresis drops it (real cores keep their seeds)
    flatfield_empty_threshold = between empty and background (no-data detector)

In ``--no-flatfield`` mode there is no subtraction, so the bounds stay raw:
``norm_lower = p1`` (or background), ``norm_upper = p99``, and the thresholds are mapped
in raw-normalized space.

Example
-------
    python examples/derive_gfp_params.py \
        --empty 0 --background 80 --background-noise 8 \
        --p99 400 --dim-signal 120 --rim 140 \
        --out params.json

Then sanity-check the printed raw-equivalent cutoffs against your viewer before running.
"""
import argparse
import json
import sys
from typing import Optional

# Standard-normal z-score of the 1st percentile: it lies ~2.326 sigma below the median.
# Used to infer the background noise sigma from (background median - raw 1st pctile).
Z_P01 = 2.326


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp ``v`` into ``[lo, hi]``."""
    return max(lo, min(hi, v))


def _norm(contrast: float, lo: float, up: float) -> float:
    """Map a contrast value into the normalized [0, 1] range used by the model."""
    return (contrast - lo) / max(up - lo, 1e-9)


def _infer_noise(background: float, p1: Optional[float]) -> Optional[float]:
    """Infer the background noise sigma from the raw 1st percentile.

    The 1st percentile sits ~``Z_P01`` sigma below the background median for a roughly
    normal background, so ``sigma ~= (background - p1) / Z_P01``. Returns ``None`` when
    ``p1`` is missing or not below the background (inference impossible).
    """
    if p1 is None:
        return None
    spread = float(background) - float(p1)
    return spread / Z_P01 if spread > 0 else None


def _norm_lower(noise, c_p1, norm_upper, flatfield, noise_sigmas):
    """The background floor (contrast clipped to 0): noise-based, else a fallback."""
    if noise is not None:
        # A few sigma above background suppresses speckle while keeping faint signal.
        lower = noise_sigmas * float(noise)
    elif not flatfield and c_p1 is not None:
        lower = c_p1  # raw mode: the 1st percentile is the natural lower clip
    else:
        lower = 0.05 * norm_upper  # no noise estimate: a small fraction of the range
    return _clamp(lower, 0.0, 0.9 * norm_upper)


def _grow_threshold(c_dim: Optional[float], lo: float, up: float) -> float:
    """Normalized grow threshold from the faintest signal contrast (or a default)."""
    if c_dim is None:
        return 0.05  # default: catch dim structure, lean on hysteresis to clean up
    if c_dim <= lo:
        print(
            "WARNING: --dim-signal is at/below the background floor (norm_lower); "
            "reduce --background-noise to keep it.",
            file=sys.stderr,
        )
    return _clamp(_norm(c_dim, lo, up), 0.005, 0.5)


def _seed_thr(c_rim: Optional[float], lo: float, up: float, margin: float, grow: float):
    """Normalized seed threshold set above the rim contrast (or a default)."""
    if c_rim is None:
        seed = max(4.0 * grow, 0.2)  # default discriminator
    else:
        rim_norm = _clamp(_norm(c_rim, lo, up), 0.0, 0.95)
        seed = rim_norm * (1.0 + margin)
    return _clamp(seed, grow + 0.01, 0.95)


def derive(args) -> dict:
    """Turn viewer readings into a params dict (see module docstring for mapping)."""
    flatfield = not args.no_flatfield
    bright = args.bright if args.bright is not None else args.p99
    if bright is None:
        sys.exit("Provide --p99 (or --bright): the bright end of real signal.")
    background = args.background if args.background is not None else args.p1
    if background is None:
        sys.exit("Provide --background (tissue level) or --p1.")

    # In subtract mode we work in contrast (raw - background); without flat-field the
    # background is not removed, so "contrast" is just raw intensity (ref = 0).
    ref = float(background) if flatfield else 0.0

    def contrast(raw: Optional[float]) -> Optional[float]:
        return None if raw is None else float(raw) - ref

    norm_upper = contrast(bright)
    if norm_upper <= 0:
        sys.exit("Bright signal must exceed background; check --p99/--background.")

    # Background noise: explicit --background-noise, else inferred from the raw 1st
    # percentile (subtract mode only -- in raw mode p1 is used as the clip directly).
    noise = args.background_noise
    noise_inferred = False
    if noise is None and flatfield:
        noise = _infer_noise(background, args.p1)
        noise_inferred = noise is not None

    norm_lower = _norm_lower(
        noise, contrast(args.p1), norm_upper, flatfield, args.noise_sigmas
    )
    threshold = _grow_threshold(contrast(args.dim_signal), norm_lower, norm_upper)
    seed_threshold = _seed_thr(
        contrast(args.rim), norm_lower, norm_upper, args.seed_margin, threshold
    )
    if args.rim is not None and contrast(args.rim) >= norm_upper:
        print(
            "WARNING: rim is as bright as real signal -- brightness alone cannot "
            "separate them; seed_threshold won't reject it.",
            file=sys.stderr,
        )

    # flatfield_empty_threshold: a no-data detector between empty and background.
    empty = float(args.empty)
    empty_threshold = empty + 0.25 * (float(background) - empty)
    empty_threshold = _clamp(empty_threshold, empty, 0.5 * float(background))

    params = {
        "normalize": "global",
        "norm_lower": round(norm_lower, 2),
        "norm_upper": round(norm_upper, 2),
        "threshold": round(threshold, 4),
        "seed_threshold": round(seed_threshold, 4),
        "smooth_sigma": list(args.smooth_sigma),
        "open_iterations": args.open_iterations,
    }
    if flatfield:
        params.update(
            {
                "flatfield": True,
                "flatfield_mode": args.flatfield_mode,
                "flatfield_level": args.flatfield_level,
                "flatfield_opening_radius": args.flatfield_opening_radius,
                "flatfield_sigma": args.flatfield_sigma,
                "flatfield_empty_threshold": round(empty_threshold, 2),
            }
        )
    info = {
        "ref": ref,
        "flatfield": flatfield,
        "noise": noise,
        "noise_inferred": noise_inferred,
    }
    return params, info


def _print_summary(params, info) -> None:
    """Print the params plus raw-equivalent cutoffs to eyeball against the viewer."""
    ref, flatfield = info["ref"], info["flatfield"]
    lo, up = params["norm_lower"], params["norm_upper"]
    # Raw intensity a normalized cutoff corresponds to: ref + (lo + t*(up-lo)).
    grow_raw = ref + (lo + params["threshold"] * (up - lo))
    seed_raw = ref + (lo + params["seed_threshold"] * (up - lo))
    print("\nDerived parameters:")
    print(json.dumps(params, indent=2))
    if info["noise"] is not None:
        src = "inferred from background - p1" if info["noise_inferred"] else "given"
        print(f"\nBackground noise sigma ~= {info['noise']:.2f} ({src})")
    print("\nSanity check (compare these raw intensities to your viewer):")
    print(f"  reference background (subtracted): {ref:g}")
    print(f"  grow keeps signal with raw value >= ~{grow_raw:.1f}")
    print(f"  seed (kept by hysteresis) needs raw value >= ~{seed_raw:.1f}")
    if flatfield:
        et = params["flatfield_empty_threshold"]
        print(f"  voxels with raw value <= {et:g} are treated as no-data (filled)")
    print(
        "\nReminder: smooth_sigma is in (Z,Y,X) voxels at the segmented level; keep it "
        "small so thin necks survive. Re-derive per level if you change resolution."
    )


def _parse_args(argv):
    """Parse viewer readings + fixed knobs into args (see module docstring)."""
    ap = argparse.ArgumentParser(
        description="Derive gfp-mask params.json from a few intensity readings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--empty", type=float, default=0.0, help="Empty/no-data raw value.")
    ap.add_argument(
        "--background", type=float, default=None, help="Tissue background raw level."
    )
    ap.add_argument(
        "--background-noise",
        type=float,
        default=None,
        help="Std of tissue background (sets norm_lower = noise_sigmas*noise). If "
        "omitted, inferred from --p1 as (background - p1)/2.326.",
    )
    ap.add_argument(
        "--noise-sigmas",
        type=float,
        default=3.0,
        help="How many background-noise sigmas to put norm_lower above background.",
    )
    ap.add_argument(
        "--p1",
        type=float,
        default=None,
        help="Large-scale raw 1st percentile; infers background noise (subtract mode) "
        "or is the lower clip directly (--no-flatfield).",
    )
    ap.add_argument(
        "--p99", type=float, default=None, help="Large-scale raw 99th percentile."
    )
    ap.add_argument(
        "--bright",
        type=float,
        default=None,
        help="Bright real-signal raw value (overrides --p99 for norm_upper).",
    )
    ap.add_argument(
        "--dim-signal",
        type=float,
        default=None,
        help="Faintest real signal to keep (raw); sets threshold.",
    )
    ap.add_argument(
        "--rim",
        type=float,
        default=None,
        help="False-positive rim/seam raw value; seed_threshold is set above it.",
    )
    ap.add_argument(
        "--seed-margin",
        type=float,
        default=0.3,
        help="Fractional safety margin pushing seed_threshold above the rim.",
    )
    ap.add_argument(
        "--no-flatfield",
        action="store_true",
        help="Disable flat-field; norm bounds stay in raw units.",
    )
    ap.add_argument(
        "--flatfield-mode", default="subtract", choices=["subtract", "divide"]
    )
    ap.add_argument("--flatfield-level", type=int, default=4)
    ap.add_argument("--flatfield-opening-radius", type=int, default=5)
    ap.add_argument("--flatfield-sigma", type=float, default=1.0)
    ap.add_argument(
        "--smooth-sigma",
        type=float,
        nargs=3,
        default=[1.0, 1.5, 1.5],
        metavar=("Z", "Y", "X"),
        help="Gaussian sigma in voxels at the segmented level.",
    )
    ap.add_argument("--open-iterations", type=int, default=0)
    ap.add_argument("--out", default="params.json", help="Where to write the JSON.")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    """Derive params, print a sanity summary, and write the JSON."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.flatfield_mode == "divide" and not args.no_flatfield:
        print(
            "NOTE: divide mode rescales around the background mean, not contrast; the "
            "contrast mapping here is approximate. Prefer subtract.",
            file=sys.stderr,
        )
    params, info = derive(args)
    _print_summary(params, info)
    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
