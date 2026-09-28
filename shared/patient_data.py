"""
shared/patient_data.py
======================
Synthetic CGM (Continuous Glucose Monitor) data generator.

Generates realistic glucose time series for use in demos, notebooks, and
unit tests — no real patient data required.

Key class
---------
CGMSimulator
    Generates 5-minute-interval glucose readings for a simulated patient.

Pre-built scenarios
-------------------
"normal_day"            — well-controlled day with three meals
"hypoglycemia_episode"  — dangerous nocturnal low at ~3 am
"post_meal_spike"       — severe hyperglycemia after a large meal
"dawn_phenomenon"       — early-morning glucose rise (4–8 am)
"exercise_induced_low"  — glucose drop during and after exercise

Usage
-----
    from shared.patient_data import CGMSimulator

    sim = CGMSimulator(patient_id="demo-001", diabetes_type=1, seed=42)
    readings = sim.generate_scenario("hypoglycemia_episode")
    sim.plot_glucose(readings)
    df = sim.to_dataframe(readings)
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — keep the module importable even without optional deps
# ---------------------------------------------------------------------------

def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for to_dataframe(). Run: pip install pandas"
        ) from exc


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_glucose(). "
            "Run: pip install matplotlib"
        ) from exc


# Avoid circular import — import GlucoseReading lazily at runtime
def _glucose_reading_class():
    from shared.gas_state import GlucoseReading
    return GlucoseReading


# ---------------------------------------------------------------------------
# Physiological constants
# ---------------------------------------------------------------------------

#: Fasting baseline glucose (mg/dL) — used as the resting set-point
_FASTING_BASELINE = 100.0

#: Noise standard deviation (mg/dL) — sensor + biological variability
_NOISE_STD = 5.0

#: Interval between readings in minutes
_INTERVAL_MINUTES = 5

#: Number of readings per 24-hour day
_READINGS_PER_DAY = 24 * 60 // _INTERVAL_MINUTES  # 288


# ---------------------------------------------------------------------------
# Glucose dynamics helpers
# ---------------------------------------------------------------------------

def _meal_response(
    t_minutes: float,
    meal_time_minutes: float,
    peak_rise: float,
    rise_duration_min: float = 60.0,
    decay_duration_min: float = 120.0,
) -> float:
    """
    Compute the glucose contribution of a single meal at time *t_minutes*.

    Uses a piecewise linear rise followed by an exponential decay.

    Parameters
    ----------
    t_minutes : float
        Current time in minutes from midnight.
    meal_time_minutes : float
        Meal start time in minutes from midnight.
    peak_rise : float
        Maximum glucose rise above baseline (mg/dL).
    rise_duration_min : float
        Minutes from meal start to peak.
    decay_duration_min : float
        Half-life of the decay phase (minutes).

    Returns
    -------
    float
        Glucose contribution (mg/dL) — always ≥ 0.
    """
    dt = t_minutes - meal_time_minutes
    if dt < 0:
        return 0.0
    if dt <= rise_duration_min:
        # Linear rise to peak
        return peak_rise * (dt / rise_duration_min)
    # Exponential decay after peak
    decay_t = dt - rise_duration_min
    return peak_rise * math.exp(-decay_t / decay_duration_min)


def _exercise_response(
    t_minutes: float,
    exercise_start_min: float,
    duration_min: float = 45.0,
    drop_mgdl: float = 40.0,
    recovery_min: float = 90.0,
) -> float:
    """
    Compute the glucose *reduction* caused by exercise.

    Returns a negative value (glucose drop) during and after exercise.
    """
    dt = t_minutes - exercise_start_min
    if dt < 0:
        return 0.0
    if dt <= duration_min:
        # Gradual drop during exercise
        return -drop_mgdl * (dt / duration_min)
    # Recovery after exercise
    recovery_t = dt - duration_min
    return -drop_mgdl * math.exp(-recovery_t / recovery_min)


def _stress_response(
    t_minutes: float,
    stress_start_min: float,
    duration_min: float = 60.0,
    rise_mgdl: float = 30.0,
) -> float:
    """
    Compute the glucose *rise* caused by a stress event (cortisol spike).
    """
    dt = t_minutes - stress_start_min
    if dt < 0 or dt > duration_min * 3:
        return 0.0
    # Bell-curve shape centred at stress_start + duration/2
    centre = duration_min / 2.0
    sigma = duration_min / 3.0
    return rise_mgdl * math.exp(-((dt - centre) ** 2) / (2 * sigma ** 2))


def _dawn_phenomenon(t_minutes: float, magnitude: float = 25.0) -> float:
    """
    Model the dawn phenomenon: cortisol-driven glucose rise between 4–8 am.
    """
    dawn_start = 4 * 60   # 4:00 am
    dawn_peak  = 6 * 60   # 6:00 am
    dawn_end   = 9 * 60   # 9:00 am

    if t_minutes < dawn_start or t_minutes > dawn_end:
        return 0.0
    if t_minutes <= dawn_peak:
        return magnitude * (t_minutes - dawn_start) / (dawn_peak - dawn_start)
    return magnitude * (dawn_end - t_minutes) / (dawn_end - dawn_peak)


def _nocturnal_drift(t_minutes: float, drift_per_hour: float = -2.0) -> float:
    """
    Slow overnight glucose drift (negative = gradual decline while sleeping).
    Active between midnight and 6 am.
    """
    if t_minutes > 6 * 60:
        return 0.0
    return drift_per_hour * (t_minutes / 60.0)


def _classify_trend(prev: float, curr: float) -> str:
    """Classify glucose trend based on rate of change (mg/dL per 5 min)."""
    delta = curr - prev
    if delta > 3:
        return "rising"
    if delta < -3:
        return "falling"
    return "stable"


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class CGMSimulator:
    """
    Synthetic CGM data generator producing realistic glucose time series.

    Parameters
    ----------
    patient_id : str
        Identifier embedded in generated readings.
    diabetes_type : int
        1 (Type 1) or 2 (Type 2).  Type 1 produces higher variability and
        sharper post-meal spikes; Type 2 has slower, more blunted dynamics.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        patient_id: str = "sim-001",
        diabetes_type: int = 1,
        seed: int = 42,
    ) -> None:
        if diabetes_type not in (1, 2):
            raise ValueError("diabetes_type must be 1 or 2")

        self.patient_id = patient_id
        self.diabetes_type = diabetes_type
        self._rng = random.Random(seed)

        # Type-specific physiological parameters
        if diabetes_type == 1:
            self._baseline = 105.0          # slightly elevated fasting
            self._meal_peak_range = (70, 130)  # large post-meal spikes
            self._rise_duration = 50.0      # fast rise (minutes)
            self._decay_duration = 100.0    # moderate decay
            self._noise_std = 7.0           # higher sensor noise
        else:
            self._baseline = 115.0          # higher fasting (insulin resistance)
            self._meal_peak_range = (50, 90)   # blunted spikes
            self._rise_duration = 75.0      # slower rise
            self._decay_duration = 150.0    # slower decay
            self._noise_std = 4.0           # lower noise

    # ------------------------------------------------------------------
    # Core generation method
    # ------------------------------------------------------------------

    def generate_day(
        self,
        date: date,
        meal_times: list[int] = None,
        stress_events: list[int] = None,
        exercise_events: list[int] = None,
        include_dawn_phenomenon: bool = False,
        hypoglycemia_at: Optional[int] = None,
        hypoglycemia_nadir: float = 48.0,
    ) -> list:
        """
        Generate one day of CGM readings at 5-minute intervals.

        Parameters
        ----------
        date : datetime.date
            Calendar date for the readings.
        meal_times : list[int]
            Hours of day when meals occur (default: [7, 12, 18]).
        stress_events : list[int]
            Hours of day when stress events occur (default: []).
        exercise_events : list[int]
            Hours of day when exercise sessions begin (default: []).
        include_dawn_phenomenon : bool
            If True, add a dawn-phenomenon glucose rise (4–8 am).
        hypoglycemia_at : int | None
            Hour of day to inject a hypoglycemia episode (None = no episode).
        hypoglycemia_nadir : float
            Minimum glucose value during the injected hypoglycemia episode.

        Returns
        -------
        list[GlucoseReading]
            288 readings (one per 5-minute interval) for the full day.
        """
        if meal_times is None:
            meal_times = [7, 12, 18]
        if stress_events is None:
            stress_events = []
        if exercise_events is None:
            exercise_events = []

        GlucoseReading = _glucose_reading_class()

        readings = []
        prev_value: Optional[float] = None

        for i in range(_READINGS_PER_DAY):
            t_min = i * _INTERVAL_MINUTES  # minutes from midnight

            # --- Baseline ---
            glucose = self._baseline

            # --- Meal contributions ---
            for meal_hour in meal_times:
                peak = self._rng.uniform(*self._meal_peak_range)
                glucose += _meal_response(
                    t_min,
                    meal_hour * 60,
                    peak_rise=peak,
                    rise_duration_min=self._rise_duration,
                    decay_duration_min=self._decay_duration,
                )

            # --- Stress contributions ---
            for stress_hour in stress_events:
                glucose += _stress_response(t_min, stress_hour * 60)

            # --- Exercise contributions ---
            for ex_hour in exercise_events:
                glucose += _exercise_response(t_min, ex_hour * 60)

            # --- Dawn phenomenon ---
            if include_dawn_phenomenon:
                glucose += _dawn_phenomenon(t_min)

            # --- Nocturnal drift ---
            glucose += _nocturnal_drift(t_min)

            # --- Injected hypoglycemia episode ---
            if hypoglycemia_at is not None:
                hypo_centre = hypoglycemia_at * 60
                hypo_width = 60.0  # episode lasts ~2h (±60 min)
                dt = t_min - hypo_centre
                if abs(dt) < hypo_width:
                    # Gaussian dip
                    dip = (self._baseline - hypoglycemia_nadir) * math.exp(
                        -(dt ** 2) / (2 * (hypo_width / 2.5) ** 2)
                    )
                    glucose -= dip

            # --- Sensor noise ---
            noise = self._rng.gauss(0, self._noise_std)
            glucose = max(40.0, glucose + noise)  # physiological floor

            # --- Trend ---
            trend = _classify_trend(prev_value, glucose) if prev_value is not None else "stable"
            prev_value = glucose

            # --- Timestamp ---
            ts = datetime(
                date.year, date.month, date.day,
                t_min // 60, t_min % 60, 0,
                tzinfo=timezone.utc,
            )

            readings.append(
                GlucoseReading(
                    timestamp=ts,
                    value_mgdl=round(glucose, 1),
                    trend=trend,
                    source="CGM",
                )
            )

        return readings

    # ------------------------------------------------------------------
    # Pre-built scenarios
    # ------------------------------------------------------------------

    def generate_scenario(self, scenario_name: str) -> list:
        """
        Generate a named clinical scenario.

        Available scenarios
        -------------------
        "normal_day"
            Typical well-controlled day: three meals, glucose stays mostly
            within 70–180 mg/dL.

        "hypoglycemia_episode"
            Dangerous nocturnal hypoglycemia at ~3 am (nadir ≈ 48 mg/dL),
            followed by rebound hyperglycemia (Somogyi effect).

        "post_meal_spike"
            Severe hyperglycemia after a large lunch (missed bolus).
            Peak > 300 mg/dL.

        "dawn_phenomenon"
            Early-morning glucose rise driven by cortisol (4–8 am).
            Common in both T1D and T2D.

        "exercise_induced_low"
            Glucose drop during afternoon exercise session, with risk of
            delayed hypoglycemia 2–4 hours post-exercise.

        Parameters
        ----------
        scenario_name : str
            One of the scenario names listed above.

        Returns
        -------
        list[GlucoseReading]
            288 readings for the scenario day.

        Raises
        ------
        ValueError
            If *scenario_name* is not recognised.
        """
        scenario_date = date(2024, 1, 15)

        scenarios = {
            "normal_day": self._scenario_normal_day,
            "hypoglycemia_episode": self._scenario_hypoglycemia,
            "post_meal_spike": self._scenario_post_meal_spike,
            "dawn_phenomenon": self._scenario_dawn_phenomenon,
            "exercise_induced_low": self._scenario_exercise_low,
        }

        if scenario_name not in scenarios:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {sorted(scenarios)}"
            )

        return scenarios[scenario_name](scenario_date)

    def _scenario_normal_day(self, d: date) -> list:
        """Well-controlled day — three meals, no events."""
        return self.generate_day(
            d,
            meal_times=[7, 12, 18],
            stress_events=[],
            exercise_events=[],
        )

    def _scenario_hypoglycemia(self, d: date) -> list:
        """Nocturnal hypoglycemia at 3 am with Somogyi rebound."""
        readings = self.generate_day(
            d,
            meal_times=[7, 12, 18],
            hypoglycemia_at=3,
            hypoglycemia_nadir=48.0,
        )
        # Inject Somogyi rebound: glucose spikes after the low
        GlucoseReading = _glucose_reading_class()
        rebound_start = 5 * 60  # 5 am
        for i, r in enumerate(readings):
            t_min = i * _INTERVAL_MINUTES
            dt = t_min - rebound_start
            if 0 <= dt <= 180:
                rebound = 80.0 * math.exp(-dt / 90.0)
                readings[i] = GlucoseReading(
                    timestamp=r.timestamp,
                    value_mgdl=round(min(350.0, r.value_mgdl + rebound), 1),
                    trend=r.trend,
                    source="CGM",
                    notes="Somogyi rebound" if dt < 30 else "",
                )
        return readings

    def _scenario_post_meal_spike(self, d: date) -> list:
        """Severe hyperglycemia after large lunch (missed bolus)."""
        GlucoseReading = _glucose_reading_class()
        # Generate with exaggerated lunch spike
        readings = self.generate_day(d, meal_times=[7, 18])  # skip lunch in base

        # Manually inject a massive lunch spike at noon
        lunch_start = 12 * 60
        for i, r in enumerate(readings):
            t_min = i * _INTERVAL_MINUTES
            spike = _meal_response(
                t_min,
                lunch_start,
                peak_rise=200.0,   # very large spike
                rise_duration_min=45.0,
                decay_duration_min=180.0,
            )
            if spike > 0:
                new_val = round(min(450.0, r.value_mgdl + spike), 1)
                trend = _classify_trend(
                    readings[i - 1].value_mgdl if i > 0 else new_val,
                    new_val,
                )
                readings[i] = GlucoseReading(
                    timestamp=r.timestamp,
                    value_mgdl=new_val,
                    trend=trend,
                    source="CGM",
                    notes="Missed bolus" if t_min == lunch_start else "",
                )
        return readings

    def _scenario_dawn_phenomenon(self, d: date) -> list:
        """Early-morning glucose rise (4–8 am)."""
        return self.generate_day(
            d,
            meal_times=[7, 12, 18],
            include_dawn_phenomenon=True,
        )

    def _scenario_exercise_low(self, d: date) -> list:
        """Glucose drop during afternoon exercise with delayed low."""
        return self.generate_day(
            d,
            meal_times=[7, 12, 18],
            exercise_events=[15],  # 3 pm exercise session
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_glucose(
        self,
        readings: list,
        title: str = "CGM Glucose Trace",
        target_low: float = 70.0,
        target_high: float = 180.0,
        figsize: tuple = (14, 5),
        show: bool = True,
    ):
        """
        Plot a glucose trace with target range and danger zones highlighted.

        Parameters
        ----------
        readings : list[GlucoseReading]
            Readings to plot (typically one day = 288 points).
        title : str
            Plot title.
        target_low : float
            Lower bound of the target range (green band).
        target_high : float
            Upper bound of the target range (green band).
        figsize : tuple
            Matplotlib figure size (width, height) in inches.
        show : bool
            If True, call ``plt.show()`` at the end.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object (useful for saving in notebooks).
        """
        plt, mpatches = _import_matplotlib()

        times = [r.timestamp for r in readings]
        values = [r.value_mgdl for r in readings]

        fig, ax = plt.subplots(figsize=figsize)

        # --- Danger zones ---
        ax.axhspan(0, 54, alpha=0.15, color="red", label="Severe hypo (<54)")
        ax.axhspan(54, target_low, alpha=0.10, color="orange", label=f"Hypo (<{target_low:.0f})")
        ax.axhspan(target_high, 300, alpha=0.10, color="orange", label=f"Hyper (>{target_high:.0f})")
        ax.axhspan(300, 500, alpha=0.15, color="red", label="Severe hyper (>300)")

        # --- Target range (green) ---
        ax.axhspan(
            target_low, target_high,
            alpha=0.12, color="green", label=f"Target ({target_low:.0f}–{target_high:.0f})"
        )

        # --- Glucose trace ---
        ax.plot(times, values, color="#1f77b4", linewidth=1.5, label="Glucose")

        # --- Threshold lines ---
        ax.axhline(target_low, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(target_high, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(54, color="red", linestyle=":", linewidth=0.8, alpha=0.7)

        # --- Formatting ---
        ax.set_xlabel("Time")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(title)
        ax.set_ylim(30, max(max(values) + 20, target_high + 40))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Format x-axis as HH:MM
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self, readings: list):
        """
        Convert a list of GlucoseReadings to a pandas DataFrame.

        Columns: timestamp, value_mgdl, trend, source, status, alert, notes

        Parameters
        ----------
        readings : list[GlucoseReading]
            Readings to convert.

        Returns
        -------
        pandas.DataFrame
            One row per reading, indexed by timestamp.
        """
        pd = _import_pandas()
        rows = [r.to_dict() for r in readings]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(self, readings: list) -> dict:
        """
        Compute summary statistics for a list of readings.

        Returns a dict with keys:
        - mean_glucose, std_glucose, min_glucose, max_glucose
        - time_in_range (fraction within 70–180)
        - time_below_70, time_above_180 (fractions)
        - gmi (Glucose Management Indicator, estimated HbA1c proxy)
        - n_readings
        """
        if not readings:
            return {}

        values = [r.value_mgdl for r in readings]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)

        in_range = sum(1 for v in values if 70 <= v <= 180)
        below_70 = sum(1 for v in values if v < 70)
        above_180 = sum(1 for v in values if v > 180)

        # GMI formula: 3.31 + 0.02392 × mean_glucose (Bergenstal et al.)
        gmi = 3.31 + 0.02392 * mean

        return {
            "mean_glucose": round(mean, 1),
            "std_glucose": round(std, 1),
            "min_glucose": round(min(values), 1),
            "max_glucose": round(max(values), 1),
            "time_in_range": round(in_range / n, 3),
            "time_below_70": round(below_70 / n, 3),
            "time_above_180": round(above_180 / n, 3),
            "gmi": round(gmi, 2),
            "n_readings": n,
        }


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date as _date

    print("=== CGMSimulator Demo ===\n")

    sim = CGMSimulator(patient_id="demo-001", diabetes_type=1, seed=42)

    for scenario in [
        "normal_day",
        "hypoglycemia_episode",
        "post_meal_spike",
        "dawn_phenomenon",
        "exercise_induced_low",
    ]:
        readings = sim.generate_scenario(scenario)
        stats = sim.summary_stats(readings)
        print(
            f"{scenario:<25}  "
            f"mean={stats['mean_glucose']:5.1f}  "
            f"TIR={stats['time_in_range']:.1%}  "
            f"low={stats['time_below_70']:.1%}  "
            f"high={stats['time_above_180']:.1%}  "
            f"GMI={stats['gmi']:.2f}"
        )

    print("\nGenerating plot for 'hypoglycemia_episode' ...")
    readings = sim.generate_scenario("hypoglycemia_episode")
    try:
        sim.plot_glucose(readings, title="Hypoglycemia Episode — CGM Trace")
    except ImportError as e:
        print(f"  (skipped: {e})")

    print("\nFirst 5 readings:")
    for r in readings[:5]:
        print(f"  {r}")
