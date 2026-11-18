"""Forecast a big game launch's Twitch concurrent viewers, hour by hour.

Entry for make.fun's gta6-twitch-ccu-forecast-01: given one case (the launch
day's first hour is known, hours 2..24 are to be forecast), write five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each hour's concurrent-viewer
count in the game's Twitch category.

The model is built at run time from the case's own `comparables`: past big
launches with hourly Twitch counts from 14 days before to 3 days after their
release day.  For every comparable we take its release-day profile as ratios
to its own hour-1 print, in log space, and combine them per hour with a
weighted median.  The weights encode two facts that the data shows clearly:

  1. Size similarity.  How a launch day unfolds depends strongly on how big
     hour 1 already is: launches whose hour 1 is huge (a global simultaneous
     unlock, or a launch the whole world watched at once) are already near
     their peak and stay flat, while launches with a small hour 1 (a later
     unlock, a slow build) can grow several-fold through the day.  Each
     comparable is weighted by a Gaussian kernel on log hour-1 size
     (bandwidth 1.0 in log units) around the target's hour-1 print.
  2. Unlock pattern.  A rolling midnight-local launch (consoles unlock at
     midnight in each timezone, the Americas hours after Europe) has a
     different shape from a single simultaneous global unlock: the former
     climbs when the Americas come online, the latter peaks at once.  The
     target's unlock pattern is read from the case's notes, and comparables
     with the same pattern (and a clean, non-early-access launch) are
     boosted; comparables with the opposite pattern are down-weighted, as
     are early-access-contaminated and weak-tier launches, whose release
     days are distorted.

The spread of each hour comes from the same weighted pool: sigma is 1.4826
times the weighted median absolute deviation of that hour's log-ratios,
floored at 0.30 (a launch bigger than anything in the pool deserves honest
uncertainty) and capped at 1.20.  Quantiles are the median ratio times
exp(z_tau * sigma), scaled by the target's hour-1 print.

Validation (local, on the four practice cases against SullyGnome's hourly
averages, the same source the practice is graded on): mean pinball loss
0.215, against the house persistence baseline's 0.387.  Leave-one-out over
the 37 comparables with an hour-1 print: mean 0.396, median 0.258 (baseline
0.930 / 0.325 on the same set, dominated by tiny-hour-1 launches).

Standard library only, no network, deterministic: the same input always
writes the same bytes.
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Size-similarity kernel: bandwidth in natural-log units of the hour-1 print.
SIZE_BANDWIDTH = 1.0
# Unlock-pattern multipliers on a comparable's weight.
SAME_PATTERN_BOOST = 2.0
OPPOSITE_PATTERN_PENALTY = 0.4
UNCLEAN_PENALTY = 0.5
# Spread of an hour's log-ratio, floored and capped.
SPREAD_FLOOR = 0.30
SPREAD_CAP = 1.20
# The persistence fallback's spread, per square-root hour (the house baseline).
BASELINE_SIGMA = 0.15
# The contest's bounds on any single quantile.
Q_MIN = 1.0
Q_MAX = 1_000_000_000.0

_NORMAL = NormalDist()


def round2(x: float) -> float | int:
    """Round x to 2 decimals, half up; a whole number comes back as an int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp (with a Z suffix) as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def classify_unlock(text: str) -> str:
    """A launch's unlock pattern from free text: rolling, simultaneous, unknown.

    "Rolling" is a midnight-local unlock, timezone by timezone; "simultaneous"
    is one global unlock time.  A negated mention ("no simultaneous launch")
    does not count as simultaneous.
    """
    lowered = text.lower()
    stripped = re.sub(
        r"\b(no|not|never|without)\s+(a\s+)?(simultaneous|single global|global)\b",
        "",
        lowered,
    )
    if "simultaneous" in stripped or "single global" in stripped or "global unlock" in stripped:
        return "simultaneous"
    if "midnight local" in stripped or "rolling" in stripped:
        return "rolling"
    return "unknown"


def is_clean_launch(text: str) -> bool:
    """False for launches whose release day is distorted (early access, weak)."""
    lowered = text.lower()
    return "early-access-contaminated" not in lowered and "tier: weak" not in lowered


def release_day_profile(comp: dict) -> dict[int, float] | None:
    """A comparable's release-day hourly viewers, {hour 1..24: viewers}.

    Hours are counted from the comparable's day_start (00:00 UTC of its
    release date).  Returns None when the launch has no hour-1 print, since
    a profile without its anchor cannot be used as a ratio.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    profile: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        try:
            at = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and hour not in profile:
            profile[hour] = float(viewers)
    return profile if 1 in profile else None


def weighted_median(pairs: list[tuple[float, float]]) -> float:
    """The weighted median of (value, weight) pairs; deterministic on ties."""
    ordered = sorted(pairs)
    total = sum(weight for _, weight in ordered)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2:
            return value
    return ordered[-1][0]


def comparable_weight(log_h1: float, target_log_h1: float, pattern: str,
                      clean: bool, target_pattern: str) -> float:
    """One comparable's weight: size similarity times unlock-pattern match."""
    weight = math.exp(-0.5 * ((log_h1 - target_log_h1) / SIZE_BANDWIDTH) ** 2)
    if target_pattern == "rolling":
        if pattern == "rolling" and clean:
            weight *= SAME_PATTERN_BOOST
        elif pattern == "simultaneous":
            weight *= OPPOSITE_PATTERN_PENALTY
        elif not clean:
            weight *= UNCLEAN_PENALTY
    elif target_pattern == "simultaneous":
        if pattern == "simultaneous" and clean:
            weight *= SAME_PATTERN_BOOST
        elif not clean:
            weight *= UNCLEAN_PENALTY
    return weight


def hour_quantiles(p1: float, hour: int, log_ratios: list[tuple[float, float]]) -> list:
    """Five quantiles of one hour's count from that hour's weighted log-ratios.

    With no comparable data for the hour, fall back to the persistence
    baseline: the count stays at hour 1's print, with a random-walk spread.
    """
    if log_ratios:
        median = weighted_median(log_ratios)
        deviations = [(abs(value - median), weight) for value, weight in log_ratios]
        sigma = 1.4826 * weighted_median(deviations)
        sigma = min(SPREAD_CAP, max(SPREAD_FLOOR, sigma))
        raw = [p1 * math.exp(median + _NORMAL.inv_cdf(level) * sigma) for level in LEVELS]
    else:
        spread = BASELINE_SIGMA * math.sqrt(hour - 1)
        raw = [p1 * math.exp(_NORMAL.inv_cdf(level) * spread) for level in LEVELS]
    quantiles: list = []
    previous = 0.0
    for value in raw:
        # Clamp to the contest's bounds, round, and keep the row non-decreasing.
        value = min(Q_MAX, max(Q_MIN, value))
        value = round2(value)
        value = max(value, previous)
        quantiles.append(value)
        previous = value
    return quantiles


def hour_one_print(case: dict) -> float:
    """The case's hour-1 print, with defensive fallbacks (never a crash)."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    for row in reversed(case.get("history_hourly") or []):
        if row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    for point in reversed(case.get("history") or []):
        if point.get("viewers"):
            return max(1.0, float(point["viewers"]))
    return 1.0


def forecast_case(case: dict) -> dict:
    """The forecast for one case: {"hours": [{"hour": h, "q": [...]}]}."""
    p1 = hour_one_print(case)
    target_log_h1 = math.log(p1)
    target_pattern = classify_unlock(str(case.get("notes") or ""))

    # The pool: every comparable with a usable release-day profile, each with
    # its weight against this case's hour-1 print and unlock pattern.
    pool: list[tuple[dict[int, float], float]] = []
    for comp in case.get("comparables") or []:
        profile = release_day_profile(comp)
        if profile is None:
            continue
        unlock_text = str(comp.get("unlock") or "")
        weight = comparable_weight(
            math.log(profile[1]), target_log_h1,
            classify_unlock(unlock_text), is_clean_launch(unlock_text),
            target_pattern,
        )
        pool.append((profile, weight))

    hours = []
    # Always emit hours 2..24, whatever the case's forecast list holds.
    for hour in range(2, 25):
        log_ratios = [
            (math.log(profile[hour] / profile[1]), weight)
            for profile, weight in pool
            if hour in profile
        ]
        hours.append({"hour": hour, "q": hour_quantiles(p1, hour, log_ratios)})
    return {"hours": hours}


def baseline_forecast(p1: float) -> dict:
    """The persistence fallback: every hour stays at hour 1's print."""
    return {
        "hours": [
            {"hour": hour, "q": hour_quantiles(p1, hour, [])}
            for hour in range(2, 25)
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without an output: a missing or malformed
            # output scores as if every quantile were hour 1's print and
            # takes no prize, while this fallback is a valid, scored entry.
            forecast = baseline_forecast(hour_one_print(case))
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
