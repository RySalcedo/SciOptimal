from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics
import math


@dataclass
class LiftHistory:
    name: str
    weekly_sets: List[int]              # last N weeks
    weekly_est_1rm: List[float]         # last N weeks (same length as weekly_sets)
    avg_rpe: float                      # average RPE for main work
    frequency_per_week: float
    exercise_variation_count: int       # main lift changes in last 12 weeks
    accessory_count: int
    sticking_point_location: Optional[str]  # "bottom", "mid", "lockout", "none"
    years_lifting: float
    relative_strength: float            # e.g., squat / bodyweight
    sleep_hours: float                  # average
    stress_level: int                   # 1–5
    soreness_level: int                 # 1–5
    drained_sessions_ratio: float       # 0–1, % of recent sessions feeling drained


# ---------- helpers ----------

def slope(values: List[float]) -> float:
    """Simple linear regression slope over equally spaced points."""
    if len(values) < 2:
        return 0.0
    x = list(range(len(values)))
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(values)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    den = sum((xi - x_mean) ** 2 for xi in x)
    return num / den if den != 0 else 0.0


def no_progress_weeks(weekly_est_1rm: List[float], threshold: float = 0.01) -> int:
    """Approx weeks with no meaningful 1RM change (relative)."""
    if len(weekly_est_1rm) < 2:
        return 0
    start = weekly_est_1rm[0]
    end = weekly_est_1rm[-1]
    if start == 0:
        return len(weekly_est_1rm)
    rel_change = (end - start) / start
    return len(weekly_est_1rm) if abs(rel_change) <= threshold else 0


def fatigue_index(lift: LiftHistory) -> float:
    """Simple composite fatigue score 0–1."""
    score = 0.0
    # sleep: <6.5 bad, 8+ good
    if lift.sleep_hours < 6.5:
        score += 0.3
    elif lift.sleep_hours < 7.0:
        score += 0.15

    # stress 1–5
    if lift.stress_level >= 4:
        score += 0.3
    elif lift.stress_level == 3:
        score += 0.15

    # soreness 1–5
    if lift.soreness_level >= 4:
        score += 0.2
    elif lift.soreness_level == 3:
        score += 0.1

    # drained sessions ratio
    score += min(lift.drained_sessions_ratio, 1.0) * 0.3

    return min(score, 1.0)


# ---------- cause scoring ----------

def score_insufficient_volume(lift: LiftHistory, weeks_no_progress: int) -> int:
    weekly_sets_now = lift.weekly_sets[-1] if lift.weekly_sets else 0
    vol_trend = slope(lift.weekly_sets) if len(lift.weekly_sets) >= 2 else 0.0

    score = 0
    if weekly_sets_now < 8:
        score += 40
    elif weekly_sets_now < 10:
        score += 25

    if vol_trend < 0:
        score += 30

    if weeks_no_progress >= 6:
        score += 30

    return min(score, 100)


def score_insufficient_intensity(lift: LiftHistory, weeks_no_progress: int) -> int:
    weekly_sets_now = lift.weekly_sets[-1] if lift.weekly_sets else 0

    score = 0
    if lift.avg_rpe < 6.5:
        score += 40
    elif lift.avg_rpe < 7.0:
        score += 25

    if weeks_no_progress >= 6:
        score += 30

    if weekly_sets_now >= 10:
        score += 30

    return min(score, 100)


def score_weak_point(lift: LiftHistory, weeks_no_progress: int) -> int:
    score = 0
    if lift.sticking_point_location and lift.sticking_point_location != "none":
        score += 60
    if weeks_no_progress >= 6 and (lift.weekly_sets[-1] if lift.weekly_sets else 0) >= 10:
        score += 40
    return min(score, 100)


def score_lack_variation(lift: LiftHistory, weeks_no_progress: int) -> int:
    # assume exercise_variation_count over ~12 weeks
    score = 0
    if lift.exercise_variation_count == 0 and weeks_no_progress >= 6:
        score += 50
    if weeks_no_progress >= 8:
        score += 20
    # crude: low accessory variety
    if lift.accessory_count <= 1:
        score += 30
    return min(score, 100)


def score_too_much_variation(lift: LiftHistory, weeks_no_progress: int) -> int:
    score = 0
    if lift.exercise_variation_count >= 4 and weeks_no_progress >= 6:
        score += 50
    if lift.frequency_per_week < 1.5:
        score += 30
    return min(score, 100)


def score_neuromuscular_plateau(lift: LiftHistory, weeks_no_progress: int) -> int:
    score = 0
    if lift.years_lifting >= 5:
        score += 40
    elif lift.years_lifting >= 3:
        score += 25

    # crude relative strength thresholds
    if lift.relative_strength >= 1.8:
        score += 30

    if weeks_no_progress >= 12:
        score += 30

    # require at least moderate volume/intensity
    if (lift.weekly_sets[-1] if lift.weekly_sets else 0) >= 10 and lift.avg_rpe >= 7.0:
        score += 10

    return min(score, 100)


def score_fatigue_overreaching(lift: LiftHistory, weeks_no_progress: int) -> int:
    f = fatigue_index(lift)
    score = 0

    if f >= 0.6:
        score += 50
    elif f >= 0.4:
        score += 30

    # if recent performance dipped (last 2 weeks lower than earlier mean)
    if len(lift.weekly_est_1rm) >= 4:
        recent = statistics.mean(lift.weekly_est_1rm[-2:])
        earlier = statistics.mean(lift.weekly_est_1rm[:-2])
        if recent < earlier * 0.98:  # >2% drop
            score += 30

    # high volume or intensity
    weekly_sets_now = lift.weekly_sets[-1] if lift.weekly_sets else 0
    if weekly_sets_now >= 15 or lift.avg_rpe >= 8.0:
        score += 20

    return min(score, 100)


# ---------- main diagnosis ----------

def diagnose_lift(lift: LiftHistory) -> Dict[str, Any]:
    weeks_no_prog = no_progress_weeks(lift.weekly_est_1rm)

    causes = []

    def add_cause(name: str, score: int, explanation: str, key_signals: List[str]):
        if score > 0:
            causes.append({
                "type": name,
                "confidence": score,
                "explanation": explanation,
                "key_signals": key_signals,
            })

    # compute scores
    vol_score = score_insufficient_volume(lift, weeks_no_prog)
    add_cause(
        "Insufficient Volume",
        vol_score,
        "You may not be doing enough weekly sets to keep driving progress.",
        [
            f"Weekly sets (latest): {lift.weekly_sets[-1] if lift.weekly_sets else 0}",
            f"Weeks without clear 1RM progress: {weeks_no_prog}",
        ],
    )

    int_score = score_insufficient_intensity(lift, weeks_no_prog)
    add_cause(
        "Insufficient Intensity / Overload",
        int_score,
        "Your training effort or load may be too low to force further adaptation.",
        [
            f"Average RPE: {lift.avg_rpe:.1f}",
            f"Weeks without clear 1RM progress: {weeks_no_prog}",
        ],
    )

    weak_score = score_weak_point(lift, weeks_no_prog)
    add_cause(
        "Weak Point in Lift",
        weak_score,
        "A specific part of the lift is likely limiting your overall strength.",
        [
            f"Reported sticking point: {lift.sticking_point_location or 'none'}",
            f"Weeks without clear 1RM progress: {weeks_no_prog}",
        ],
    )

    lack_var_score = score_lack_variation(lift, weeks_no_prog)
    add_cause(
        "Lack of Variation",
        lack_var_score,
        "You’ve likely been using the same exercise and rep scheme for too long.",
        [
            f"Main lift changes (last ~12 weeks): {lift.exercise_variation_count}",
            f"Accessory count: {lift.accessory_count}",
        ],
    )

    too_var_score = score_too_much_variation(lift, weeks_no_prog)
    add_cause(
        "Too Much Variation",
        too_var_score,
        "You may be changing exercises or programs too often to build specific strength.",
        [
            f"Main lift changes (last ~12 weeks): {lift.exercise_variation_count}",
            f"Frequency per week: {lift.frequency_per_week}",
        ],
    )

    neuro_score = score_neuromuscular_plateau(lift, weeks_no_prog)
    add_cause(
        "Neuromuscular Adaptation Plateau",
        neuro_score,
        "You’re likely at a more advanced stage where neural gains have slowed.",
        [
            f"Years lifting: {lift.years_lifting}",
            f"Relative strength: {lift.relative_strength:.2f}",
            f"Weeks without clear 1RM progress: {weeks_no_prog}",
        ],
    )

    fatigue_score = score_fatigue_overreaching(lift, weeks_no_prog)
    add_cause(
        "Accumulated Fatigue / Overreaching",
        fatigue_score,
        "Fatigue and recovery issues may be masking your true strength.",
        [
            f"Sleep hours: {lift.sleep_hours}",
            f"Stress level: {lift.stress_level}",
            f"Soreness level: {lift.soreness_level}",
            f"Drained sessions ratio: {lift.drained_sessions_ratio:.2f}",
        ],
    )

    # filter & sort
    causes = [c for c in causes if c["confidence"] >= 40]
    causes.sort(key=lambda c: c["confidence"], reverse=True)

    primary = causes[0] if causes else None
    secondary = causes[1:3] if len(causes) > 1 else []

    return {
        "lift": lift.name,
        "primary_cause": primary,
        "secondary_causes": secondary,
        "weeks_without_progress": weeks_no_prog,
    }


# ---------- example usage ----------

if __name__ == "__main__":
    bench = LiftHistory(
        name="Barbell Bench Press",
        weekly_sets=[8, 8, 9, 9, 8, 7],
        weekly_est_1rm=[120, 121, 121, 120, 120, 119],
        avg_rpe=6.7,
        frequency_per_week=1.5,
        exercise_variation_count=0,
        accessory_count=1,
        sticking_point_location="mid",
        years_lifting=3.5,
        relative_strength=1.3,
        sleep_hours=6.3,
        stress_level=4,
        soreness_level=3,
        drained_sessions_ratio=0.5,
    )

    result = diagnose_lift(bench)
    from pprint import pprint
    pprint(result)
