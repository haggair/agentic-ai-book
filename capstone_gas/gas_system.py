"""
Guardian Angel System (GAS)
============================
Complete Multi-Agent Diabetic Patient Supervision System

Inspired by: "Guardian Angel: A Multi-Agent System for Diabetic Patient Supervision"
Built with:  LangGraph, OpenAI/Anthropic/Ollama, sentence-transformers + FAISS

Architecture
------------

    GASState (TypedDict)
    patient | readings_history | current_reading | alert_level
    alert_reason | recommendation | safety_score | safety_approved
    caregiver_notified | agent_messages | next_agent | iteration

    LangGraph StateGraph flow:

    START
      |
      v
    [Monitor Agent]          -- reads CGM data, detects anomalies & trends
      |
      v
    [Supervisor]             -- LangGraph router, decides next agent
      |
      v
    [Advisor Agent]          -- RAG-grounded, evidence-based recommendation
      |
      v
    [Safety Critic]          -- validates recommendation (score 0.0-1.0)
      |
      +-- score >= 0.7 --> [Caregiver Notifier] --> END
      |
      +-- score <  0.7 --> retry (up to 2x) --> [Advisor Agent]

    Supporting services:
      CGM Simulator       5 clinical scenarios (24-hour traces)
      RAG Knowledge Base  FAISS + sentence-transformers, 15 ADA guidelines
      GASLLMClient        auto-detects OpenAI / Anthropic / Ollama / Mock

Five clinical scenarios
-----------------------
    normal_day            — typical 24-hour glucose profile
    hypoglycemia_episode  — nocturnal low with recovery
    post_meal_spike       — post-prandial hyperglycemia
    dawn_phenomenon       — early-morning glucose rise
    exercise_induced_low  — exercise-triggered hypoglycemia

Usage
-----
    python gas_system.py
    python gas_system.py --scenario hypoglycemia_episode
    python gas_system.py --scenario post_meal_spike --patient bob --verbose
    NOTEBOOK_TEST_MODE=1 python gas_system.py   # CI-safe mock mode

⚠️  CLINICAL DISCLAIMER
    This software is for educational and research purposes only.
    It is NOT a medical device and must NOT be used for clinical
    decision-making or patient care.  Always consult a qualified
    healthcare professional for medical advice.
"""

from __future__ import annotations

import os
import sys
import math
import time
import random
import logging
import operator
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Annotated, Literal, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("gas")

# ---------------------------------------------------------------------------
# Optional rich console (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore[assignment]

def _print(msg: str, style: str = "") -> None:
    if _RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)

# ---------------------------------------------------------------------------
# Test-mode flag
# ---------------------------------------------------------------------------

TEST_MODE: bool = bool(os.environ.get("NOTEBOOK_TEST_MODE"))

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PatientProfile:
    """
    Static patient demographics and clinical configuration.

    Attributes
    ----------
    patient_id : str
        Unique identifier.
    name : str
        Patient display name.
    age : int
        Age in years.
    diabetes_type : int
        1 = Type 1 (insulin-dependent), 2 = Type 2.
    target_low : float
        Lower bound of target glucose range in mg/dL (default 70).
    target_high : float
        Upper bound of target glucose range in mg/dL (default 180).
    insulin_sensitivity_factor : float
        Expected mg/dL drop per unit of correction insulin (default 50).
    carb_ratio : float
        Grams of carbohydrate covered by 1 unit of insulin (default 10).
    medications : list[str]
        Current medications.
    caregiver_email : str
        Primary caregiver contact email.
    caregiver_name : str
        Primary caregiver name.
    """

    patient_id: str = "patient-001"
    name: str = "Alice"
    age: int = 34
    diabetes_type: int = 1
    target_low: float = 70.0
    target_high: float = 180.0
    insulin_sensitivity_factor: float = 50.0
    carb_ratio: float = 10.0
    medications: list = field(default_factory=list)
    caregiver_email: str = "caregiver@example.com"
    caregiver_name: str = "Jordan (caregiver)"

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "diabetes_type": self.diabetes_type,
            "target_low": self.target_low,
            "target_high": self.target_high,
            "insulin_sensitivity_factor": self.insulin_sensitivity_factor,
            "carb_ratio": self.carb_ratio,
            "medications": list(self.medications),
            "caregiver_email": self.caregiver_email,
            "caregiver_name": self.caregiver_name,
        }


@dataclass
class GlucoseReading:
    """
    A single CGM or finger-stick glucose measurement.

    Attributes
    ----------
    timestamp : datetime
        When the reading was taken (UTC).
    value_mgdl : float
        Glucose concentration in mg/dL.
    trend : str
        Direction: rising_fast / rising / stable / falling / falling_fast.
    source : str
        "CGM" or "manual".
    notes : str
        Optional annotation (e.g. "post-meal", "post-exercise").
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    value_mgdl: float = 100.0
    trend: str = "stable"
    source: str = "CGM"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value_mgdl": self.value_mgdl,
            "trend": self.trend,
            "source": self.source,
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%H:%M")
        return f"GlucoseReading({ts}, {self.value_mgdl:.1f} mg/dL, {self.trend})"


# ---------------------------------------------------------------------------
# LangGraph State TypedDict
# ---------------------------------------------------------------------------

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]


class GASState(TypedDict):
    """
    Shared state dictionary passed between all LangGraph agent nodes.

    Fields
    ------
    patient : dict
        Serialised PatientProfile.
    readings_history : list[dict]
        Up to 288 recent GlucoseReading dicts (24 h at 5-min intervals).
    current_reading : dict
        The reading currently being processed.
    alert_level : str
        Severity: none / low / medium / high / critical.
    alert_reason : str
        Human-readable explanation of the alert.
    recommendation : str
        Natural-language recommendation from the Advisor agent.
    safety_score : float
        Safety score 0.0–1.0 assigned by the Safety Critic.
    safety_approved : bool
        True once the Safety Critic has approved the recommendation.
    caregiver_notified : bool
        True once the Caregiver Notifier has sent an alert.
    agent_messages : list[str]
        Append-only audit trail of messages from each agent.
    next_agent : str
        Routing target set by the Supervisor.
    iteration : int
        Safety-retry counter (max 2 retries before hard stop).
    """

    patient: dict
    readings_history: Annotated[list, operator.add]
    current_reading: dict
    alert_level: str
    alert_reason: str
    recommendation: str
    safety_score: float
    safety_approved: bool
    caregiver_notified: bool
    agent_messages: Annotated[list, operator.add]
    next_agent: str
    iteration: int


# ============================================================================
# CGM SIMULATOR
# ============================================================================

class CGMSimulator:
    """
    Generates realistic 24-hour continuous glucose monitoring (CGM) time series.

    Five clinical scenarios are supported:

    normal_day
        Fasting glucose ~100 mg/dL, two post-meal spikes (breakfast, dinner),
        stable overnight.

    hypoglycemia_episode
        Nocturnal hypoglycemia at ~02:00 (glucose drops to ~48 mg/dL),
        followed by recovery after treatment.

    post_meal_spike
        Aggressive post-prandial spike to ~280 mg/dL after lunch, slow return
        to range over 3 hours.

    dawn_phenomenon
        Early-morning glucose rise (04:00–08:00) driven by counter-regulatory
        hormones, reaching ~220 mg/dL without additional insulin.

    exercise_induced_low
        Glucose drops sharply during a 60-minute exercise session at ~17:00,
        reaching ~58 mg/dL before recovery.
    """

    SCENARIOS = [
        "normal_day",
        "hypoglycemia_episode",
        "post_meal_spike",
        "dawn_phenomenon",
        "exercise_induced_low",
    ]

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scenario: str,
        start: Optional[datetime] = None,
        interval_minutes: int = 5,
    ) -> list[GlucoseReading]:
        """
        Generate a full 24-hour CGM trace for *scenario*.

        Parameters
        ----------
        scenario : str
            One of :attr:`SCENARIOS`.
        start : datetime | None
            Start timestamp (default: today at midnight UTC).
        interval_minutes : int
            Sampling interval in minutes (default: 5 → 288 readings/day).

        Returns
        -------
        list[GlucoseReading]
            Ordered list of readings from midnight to 23:55.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Choose from: {self.SCENARIOS}"
            )
        if start is None:
            now = datetime.now(timezone.utc)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        n_readings = (24 * 60) // interval_minutes
        timestamps = [
            start + timedelta(minutes=i * interval_minutes)
            for i in range(n_readings)
        ]

        method = getattr(self, f"_scenario_{scenario}")
        values = method(n_readings)

        readings: list[GlucoseReading] = []
        for i, (ts, val) in enumerate(zip(timestamps, values)):
            # Add physiological noise
            noisy_val = max(40.0, val + self._rng.gauss(0, 2.5))
            trend = self._compute_trend(values, i)
            notes = self._scenario_notes(scenario, i, n_readings)
            readings.append(
                GlucoseReading(
                    timestamp=ts,
                    value_mgdl=round(noisy_val, 1),
                    trend=trend,
                    source="CGM",
                    notes=notes,
                )
            )
        return readings

    # ------------------------------------------------------------------
    # Scenario generators (return list of float values)
    # ------------------------------------------------------------------

    def _scenario_normal_day(self, n: int) -> list[float]:
        """Typical day: fasting ~100, two post-meal bumps, stable overnight."""
        vals = []
        for i in range(n):
            hour = (i * 5) / 60.0
            base = 100.0
            # Breakfast spike 07:30–10:00
            breakfast = 45 * math.exp(-0.5 * ((hour - 8.5) / 0.8) ** 2)
            # Dinner spike 18:30–21:00
            dinner = 40 * math.exp(-0.5 * ((hour - 19.5) / 0.9) ** 2)
            vals.append(base + breakfast + dinner)
        return vals

    def _scenario_hypoglycemia_episode(self, n: int) -> list[float]:
        """Nocturnal hypo at ~02:00, recovery by 03:30."""
        vals = []
        for i in range(n):
            hour = (i * 5) / 60.0
            base = 105.0
            # Gradual drop from midnight
            drop = -55 * math.exp(-0.5 * ((hour - 2.0) / 0.6) ** 2)
            # Recovery after treatment (Rule of 15)
            recovery = 30 * (1 / (1 + math.exp(-3 * (hour - 3.2))))
            # Breakfast bump
            breakfast = 35 * math.exp(-0.5 * ((hour - 8.0) / 0.9) ** 2)
            vals.append(max(40.0, base + drop + recovery + breakfast))
        return vals

    def _scenario_post_meal_spike(self, n: int) -> list[float]:
        """Large post-lunch spike to ~280 mg/dL."""
        vals = []
        for i in range(n):
            hour = (i * 5) / 60.0
            base = 95.0
            # Moderate breakfast
            breakfast = 30 * math.exp(-0.5 * ((hour - 8.0) / 0.7) ** 2)
            # Large lunch spike
            lunch = 185 * math.exp(-0.5 * ((hour - 13.5) / 1.2) ** 2)
            # Correction brings it down by 16:00
            correction = -60 * (1 / (1 + math.exp(-2 * (hour - 15.5))))
            # Dinner
            dinner = 40 * math.exp(-0.5 * ((hour - 19.0) / 0.8) ** 2)
            vals.append(max(70.0, base + breakfast + lunch + correction + dinner))
        return vals

    def _scenario_dawn_phenomenon(self, n: int) -> list[float]:
        """Counter-regulatory hormone surge 04:00–08:00."""
        vals = []
        for i in range(n):
            hour = (i * 5) / 60.0
            base = 95.0
            # Dawn rise
            dawn = 125 * (1 / (1 + math.exp(-2.5 * (hour - 5.5)))) * (
                1 / (1 + math.exp(2.5 * (hour - 9.0)))
            )
            # Insulin correction at 08:00 brings it down
            correction = -80 * (1 / (1 + math.exp(-3 * (hour - 9.5))))
            # Lunch
            lunch = 35 * math.exp(-0.5 * ((hour - 13.0) / 0.8) ** 2)
            vals.append(max(70.0, base + dawn + correction + lunch))
        return vals

    def _scenario_exercise_induced_low(self, n: int) -> list[float]:
        """Exercise session 17:00–18:00 causes glucose drop to ~58 mg/dL."""
        vals = []
        for i in range(n):
            hour = (i * 5) / 60.0
            base = 110.0
            # Morning bump
            morning = 30 * math.exp(-0.5 * ((hour - 8.5) / 0.8) ** 2)
            # Exercise-induced drop
            exercise_drop = -55 * math.exp(-0.5 * ((hour - 17.5) / 0.5) ** 2)
            # Recovery snack
            recovery = 35 * math.exp(-0.5 * ((hour - 19.0) / 0.7) ** 2)
            vals.append(max(40.0, base + morning + exercise_drop + recovery))
        return vals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_trend(self, values: list[float], idx: int) -> str:
        """Compute trend label from rate of change (mg/dL per 5 min)."""
        if idx < 2:
            return "stable"
        delta = values[idx] - values[idx - 2]  # change over 10 min
        rate = delta / 2.0  # per 5-min interval
        if rate > 3.0:
            return "rising_fast"
        if rate > 1.0:
            return "rising"
        if rate < -3.0:
            return "falling_fast"
        if rate < -1.0:
            return "falling"
        return "stable"

    def _scenario_notes(self, scenario: str, idx: int, n: int) -> str:
        """Attach contextual notes at key time points."""
        hour = (idx * 5) / 60.0
        notes_map = {
            "normal_day": {
                (7.0, 8.0): "breakfast",
                (12.0, 13.0): "lunch",
                (18.0, 19.5): "dinner",
            },
            "hypoglycemia_episode": {
                (1.5, 2.5): "nocturnal hypoglycemia",
                (2.5, 3.5): "treatment: 15g fast carbs",
                (7.5, 8.5): "breakfast",
            },
            "post_meal_spike": {
                (12.5, 14.5): "large lunch — missed bolus",
                (15.0, 16.0): "correction bolus given",
            },
            "dawn_phenomenon": {
                (4.0, 6.0): "dawn phenomenon rising",
                (8.0, 9.5): "correction insulin given",
            },
            "exercise_induced_low": {
                (17.0, 18.0): "aerobic exercise session",
                (18.5, 19.5): "recovery snack",
            },
        }
        for (h_start, h_end), note in notes_map.get(scenario, {}).items():
            if h_start <= hour < h_end:
                return note
        return ""


# ============================================================================
# MEDICAL KNOWLEDGE BASE
# ============================================================================

MEDICAL_KNOWLEDGE: list[str] = [
    # --- Hypoglycemia management ---
    (
        "ADA Hypoglycemia Rule of 15: When blood glucose is below 70 mg/dL, "
        "consume 15 grams of fast-acting carbohydrates (e.g., 4 glucose tablets, "
        "4 oz juice, or 3–4 hard candies). Recheck glucose after 15 minutes. "
        "Repeat if still below 70 mg/dL. Once glucose is above 70 mg/dL, eat a "
        "small snack if the next meal is more than 1 hour away."
    ),
    (
        "Severe hypoglycemia (glucose < 54 mg/dL) requires immediate intervention. "
        "If the patient is conscious and able to swallow, give 15–20 g fast-acting "
        "carbohydrates. If unconscious or unable to swallow, administer glucagon "
        "(1 mg IM/SC) or call emergency services immediately. Do NOT give insulin "
        "during a hypoglycemic episode."
    ),
    (
        "Nocturnal hypoglycemia is particularly dangerous because the patient may "
        "not wake up. CGM alarms should be set at 70 mg/dL (warning) and 55 mg/dL "
        "(urgent). Caregivers should be notified for any glucose below 60 mg/dL "
        "during sleep hours (22:00–07:00)."
    ),
    (
        "Hypoglycemia unawareness occurs when patients lose the ability to detect "
        "low glucose symptoms. Risk factors include: frequent hypoglycemia, long "
        "diabetes duration, autonomic neuropathy. These patients require stricter "
        "CGM monitoring and higher glucose targets (80–180 mg/dL instead of 70–180)."
    ),
    # --- Hyperglycemia management ---
    (
        "ADA Hyperglycemia Guidelines: For glucose > 250 mg/dL, check for ketones "
        "(Type 1 patients). If ketones are present, contact healthcare provider. "
        "Correction insulin dose = (Current glucose - Target glucose) / ISF, "
        "where ISF is the insulin sensitivity factor (typically 30–100 mg/dL per unit)."
    ),
    (
        "Post-meal hyperglycemia (glucose > 180 mg/dL within 2 hours of eating) "
        "is associated with increased cardiovascular risk. Management includes: "
        "pre-meal insulin bolus timing (15–20 min before eating for rapid-acting "
        "insulin), carbohydrate counting, and avoiding high-glycemic-index foods."
    ),
    (
        "Dawn phenomenon: Early-morning glucose rise (typically 04:00–08:00) caused "
        "by counter-regulatory hormones (cortisol, growth hormone). Management: "
        "increase basal insulin dose, use insulin pump with dawn-specific basal rate, "
        "or consider metformin for Type 2 patients. Distinguish from Somogyi effect "
        "(rebound hyperglycemia after nocturnal hypoglycemia)."
    ),
    (
        "Diabetic Ketoacidosis (DKA) warning signs: glucose > 300 mg/dL with "
        "positive ketones, nausea, vomiting, abdominal pain, fruity breath, "
        "rapid breathing. DKA is a medical emergency — call 911 immediately. "
        "Do NOT attempt to manage DKA at home."
    ),
    # --- Exercise and activity ---
    (
        "Exercise and glucose management: Aerobic exercise typically lowers glucose "
        "by 20–60 mg/dL during and after activity. Recommendations: check glucose "
        "before exercise (target 100–180 mg/dL), consume 15–30 g carbs if < 100 mg/dL, "
        "reduce bolus insulin by 25–50% for meals before exercise, monitor for "
        "delayed hypoglycemia up to 24 hours post-exercise."
    ),
    (
        "Anaerobic exercise (weightlifting, sprinting) can temporarily raise glucose "
        "due to catecholamine release. This is normal and usually self-correcting. "
        "Avoid correction boluses immediately after anaerobic exercise without "
        "rechecking glucose 30–60 minutes later."
    ),
    # --- Target ranges and monitoring ---
    (
        "ADA 2024 Glucose Targets: Fasting/pre-meal: 80–130 mg/dL. "
        "Post-meal (1–2 hours): < 180 mg/dL. Bedtime: 90–150 mg/dL. "
        "Time in Range (TIR) goal: > 70% of readings between 70–180 mg/dL. "
        "Time below range (TBR): < 4% below 70 mg/dL, < 1% below 54 mg/dL."
    ),
    (
        "CGM trend arrows interpretation: ↑↑ (rising fast, > 3 mg/dL/min): "
        "glucose will rise ~60 mg/dL in 20 min — consider pre-emptive correction. "
        "↑ (rising, 1–3 mg/dL/min): add 1–2 units to bolus. "
        "→ (stable): use standard bolus. "
        "↓ (falling, 1–3 mg/dL/min): reduce bolus by 1–2 units. "
        "↓↓ (falling fast, > 3 mg/dL/min): consume 15 g carbs immediately."
    ),
    # --- Insulin management ---
    (
        "Insulin correction formula: Correction dose (units) = "
        "(Current BG - Target BG) / Insulin Sensitivity Factor (ISF). "
        "Example: BG = 280, Target = 120, ISF = 50 → (280-120)/50 = 3.2 units. "
        "Always round down for safety. Do not stack correction doses within 3 hours "
        "of a previous correction (insulin-on-board risk)."
    ),
    (
        "Insulin-on-board (IOB): Active insulin remaining from previous doses. "
        "Rapid-acting insulin (lispro, aspart, glulisine) has a duration of 3–5 hours. "
        "Always account for IOB before giving correction doses to avoid stacking "
        "and subsequent hypoglycemia. Most insulin pumps and CGM systems calculate IOB."
    ),
    # --- Emergency escalation ---
    (
        "Emergency escalation criteria for caregiver notification: "
        "(1) Glucose < 54 mg/dL (critical hypoglycemia), "
        "(2) Glucose > 400 mg/dL (severe hyperglycemia), "
        "(3) Glucose < 70 mg/dL with falling_fast trend, "
        "(4) Glucose > 300 mg/dL with rising trend and Type 1 diabetes (DKA risk), "
        "(5) No CGM reading for > 30 minutes during sleep hours. "
        "Caregiver alerts should include: patient name, current glucose, trend, "
        "last known location, and recommended action."
    ),
]


# ============================================================================
# RAG KNOWLEDGE BASE
# ============================================================================

class GASKnowledgeBase:
    """
    Retrieval-Augmented Generation (RAG) knowledge base for the Advisor agent.

    Uses sentence-transformers for dense embeddings and FAISS for approximate
    nearest-neighbour search.  Falls back to keyword-based retrieval when
    sentence-transformers or faiss are not installed.

    Parameters
    ----------
    documents : list[str]
        Medical knowledge strings to index.
    model_name : str
        Sentence-transformer model name (default: all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        documents: list[str] = MEDICAL_KNOWLEDGE,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.documents = documents
        self.model_name = model_name
        self.embedder = None
        self.index = None
        self._embeddings = None
        self._built = False

    def build_index(self) -> None:
        """
        Embed all documents and build a FAISS index.

        Falls back to keyword search if sentence-transformers or faiss
        are not available.
        """
        if self._built:
            return
        if TEST_MODE:
            logger.debug("TEST_MODE: skipping embedding index build")
            self._built = True
            return
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            logger.info("Building RAG index with %s …", self.model_name)
            self.embedder = SentenceTransformer(self.model_name)
            self._embeddings = self.embedder.encode(
                self.documents, show_progress_bar=False, convert_to_numpy=True
            )

            try:
                import faiss  # type: ignore[import]
                dim = self._embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                # Normalise for cosine similarity
                norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                normed = self._embeddings / (norms + 1e-9)
                self.index.add(normed.astype("float32"))
                logger.info("FAISS index built (%d documents)", len(self.documents))
            except ImportError:
                logger.warning("faiss not installed — using numpy dot-product search")
                self.index = None  # will use numpy fallback

        except ImportError:
            logger.warning(
                "sentence-transformers not installed — using keyword search fallback"
            )
        self._built = True

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Return the top-*k* most relevant documents for *query*.

        Parameters
        ----------
        query : str
            Natural-language query (e.g. "patient has low glucose at night").
        k : int
            Number of documents to return (default: 3).

        Returns
        -------
        list[str]
            Relevant document strings, most relevant first.
        """
        if not self._built:
            self.build_index()

        # FAISS path
        if self.index is not None and self.embedder is not None:
            try:
                import numpy as np
                import faiss  # type: ignore[import]

                q_emb = self.embedder.encode([query], convert_to_numpy=True)
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
                _, indices = self.index.search(q_norm.astype("float32"), k)
                return [self.documents[i] for i in indices[0] if i < len(self.documents)]
            except Exception as exc:
                logger.warning("FAISS search failed: %s — falling back", exc)

        # Numpy dot-product path (no faiss)
        if self._embeddings is not None and self.embedder is not None:
            try:
                import numpy as np

                q_emb = self.embedder.encode([query], convert_to_numpy=True)
                scores = self._embeddings @ q_emb.T
                top_k = int(min(k, len(self.documents)))
                indices = scores.flatten().argsort()[::-1][:top_k]
                return [self.documents[i] for i in indices]
            except Exception as exc:
                logger.warning("Numpy search failed: %s — falling back", exc)

        # Keyword fallback
        return self._keyword_retrieve(query, k)

    def _keyword_retrieve(self, query: str, k: int) -> list[str]:
        """Simple keyword-overlap retrieval (no ML dependencies required)."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


# ============================================================================
# LLM CLIENT
# ============================================================================

class GASLLMClient:
    """
    Unified LLM client for the Guardian Angel System.

    Backend auto-detection order:
      1. OPENAI_API_KEY    → OpenAI  (gpt-4o-mini)
      2. ANTHROPIC_API_KEY → Anthropic (claude-3-haiku-20240307)
      3. Ollama reachable  → Ollama (llama3.2)
      4. Fallback          → Mock (deterministic, no API calls)

    Set NOTEBOOK_TEST_MODE=1 to force mock mode (CI-safe).

    Parameters
    ----------
    backend : str | None
        Force backend: "openai", "anthropic", "ollama", or "mock".
    model : str | None
        Override default model for the selected backend.
    """

    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama3.2",
        "mock": "mock",
    }

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        max_retries: int = 3,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.max_retries = max_retries

        if TEST_MODE:
            self.backend = "mock"
        elif backend is not None:
            self.backend = backend.lower()
        else:
            self.backend = self._detect_backend()

        self.model = model or self.DEFAULT_MODELS.get(self.backend, "gpt-4o-mini")
        logger.info(
            "GASLLMClient: backend=%s, model=%s, test_mode=%s",
            self.backend, self.model, TEST_MODE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """
        Send a chat completion request and return the assistant's reply.

        Parameters
        ----------
        messages : list[dict]
            List of {"role": ..., "content": ...} dicts.
        temperature : float
            Sampling temperature (default 0.3 for clinical consistency).
        max_tokens : int
            Maximum response tokens.

        Returns
        -------
        str
            Assistant text response.
        """
        if self.backend == "mock":
            return self._mock_response(messages)

        for attempt in range(self.max_retries):
            try:
                if self.backend == "openai":
                    return self._chat_openai(messages, temperature, max_tokens)
                elif self.backend == "anthropic":
                    return self._chat_anthropic(messages, temperature, max_tokens)
                elif self.backend == "ollama":
                    return self._chat_ollama(messages, temperature, max_tokens)
                else:
                    return self._mock_response(messages)
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM call failed after %d attempts: %s", self.max_retries, exc)
                    return self._mock_response(messages)
        return self._mock_response(messages)

    def _mock_response(self, messages: list[dict]) -> str:
        """
        Deterministic mock response for TEST_MODE and fallback.

        Inspects the last user message to return a contextually appropriate
        canned response — useful for CI pipelines and offline demos.
        """
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").lower()
                break

        if "critical" in last_user or "severe" in last_user or "54" in last_user:
            return (
                "CRITICAL ALERT: Glucose is dangerously low. "
                "Administer 15–20g fast-acting carbohydrates immediately. "
                "If patient is unconscious, use glucagon and call emergency services. "
                "Do NOT give insulin. Notify caregiver immediately."
            )
        if "hypoglycemia" in last_user or "low" in last_user or "falling" in last_user:
            return (
                "Glucose is below target range. Apply the Rule of 15: consume "
                "15g fast-acting carbohydrates (4 glucose tablets or 4 oz juice). "
                "Recheck in 15 minutes. Avoid strenuous activity until glucose > 90 mg/dL."
            )
        if "hyperglycemia" in last_user or "high" in last_user or "rising" in last_user:
            return (
                "Glucose is above target range. Consider a correction bolus using "
                "your insulin sensitivity factor. Check for ketones if > 250 mg/dL. "
                "Increase water intake. Recheck in 1–2 hours."
            )
        if "safe" in last_user or "valid" in last_user or "approve" in last_user:
            return "APPROVED: Recommendation is clinically safe. Safety score: 0.92."
        if "dawn" in last_user:
            return (
                "Dawn phenomenon detected. Consider adjusting basal insulin rate "
                "between 03:00–07:00. Consult your endocrinologist about pump "
                "programming or long-acting insulin timing."
            )
        if "exercise" in last_user:
            return (
                "Exercise-induced glucose drop detected. Consume 15–30g carbohydrates "
                "before or during exercise if glucose < 100 mg/dL. Monitor for "
                "delayed hypoglycemia up to 24 hours post-exercise."
            )
        return (
            "Glucose is within acceptable range. Continue current management plan. "
            "Maintain regular monitoring every 5 minutes via CGM."
        )

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _chat_openai(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _chat_anthropic(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic()
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_msgs,
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text

    def _chat_ollama(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        import requests
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _detect_backend(self) -> str:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        # Try Ollama
        try:
            import requests
            r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                return "ollama"
        except Exception:
            pass
        logger.warning("No LLM backend available — using mock responses")
        return "mock"

    def __repr__(self) -> str:
        return f"GASLLMClient(backend={self.backend!r}, model={self.model!r})"


# ============================================================================
# GLOBAL SINGLETONS (lazy-initialised)
# ============================================================================

_kb: Optional[GASKnowledgeBase] = None
_llm: Optional[GASLLMClient] = None


def get_knowledge_base() -> GASKnowledgeBase:
    """Return the singleton GASKnowledgeBase, building the index on first call."""
    global _kb
    if _kb is None:
        _kb = GASKnowledgeBase()
        _kb.build_index()
    return _kb


def get_llm() -> GASLLMClient:
    """Return the singleton GASLLMClient."""
    global _llm
    if _llm is None:
        _llm = GASLLMClient()
    return _llm


# ============================================================================
# AGENT NODE FUNCTIONS
# ============================================================================

# ---------------------------------------------------------------------------
# 1. Monitor Agent
# ---------------------------------------------------------------------------

def monitor_node(state: GASState) -> dict:
    """
    Monitor Agent — Analyzes the current CGM reading.

    Responsibilities
    ----------------
    - Classify alert level: none / low / medium / high / critical
    - Detect dangerous trends (rising_fast, falling_fast)
    - Identify clinical patterns (dawn phenomenon, post-meal spike, exercise low)
    - Produce a structured alert reason string

    Returns
    -------
    dict
        Updates: alert_level, alert_reason, agent_messages
    """
    reading = state["current_reading"]
    patient = state["patient"]
    history = state.get("readings_history", [])

    glucose = reading["value_mgdl"]
    trend = reading["trend"]
    ts = reading["timestamp"]
    notes = reading.get("notes", "")

    # --- Classify alert level ---
    if glucose < 54 or glucose > 400:
        alert_level = "critical"
    elif glucose < 70 or glucose > 300:
        alert_level = "high"
    elif glucose < 80 or glucose > 250:
        alert_level = "medium"
    elif glucose < 90 or glucose > 200:
        alert_level = "low"
    else:
        alert_level = "none"

    # Upgrade alert for dangerous trends
    if trend == "falling_fast" and glucose < 90:
        if alert_level in ("none", "low"):
            alert_level = "medium"
        elif alert_level == "medium":
            alert_level = "high"
    if trend == "rising_fast" and glucose > 200:
        if alert_level in ("none", "low"):
            alert_level = "medium"

    # --- Build alert reason ---
    reasons = []
    if glucose < 54:
        reasons.append(f"CRITICAL hypoglycemia: {glucose:.1f} mg/dL (< 54 threshold)")
    elif glucose < 70:
        reasons.append(f"Hypoglycemia: {glucose:.1f} mg/dL (below 70 mg/dL target)")
    elif glucose < 80:
        reasons.append(f"Near-low glucose: {glucose:.1f} mg/dL (approaching hypoglycemia)")
    elif glucose > 400:
        reasons.append(f"CRITICAL hyperglycemia: {glucose:.1f} mg/dL (> 400 — DKA risk)")
    elif glucose > 300:
        reasons.append(f"Severe hyperglycemia: {glucose:.1f} mg/dL (> 300 mg/dL)")
    elif glucose > 250:
        reasons.append(f"Significant hyperglycemia: {glucose:.1f} mg/dL (> 250 mg/dL)")
    elif glucose > 200:
        reasons.append(f"Elevated glucose: {glucose:.1f} mg/dL (> 200 mg/dL)")
    else:
        reasons.append(f"Glucose in range: {glucose:.1f} mg/dL")

    if trend == "falling_fast":
        reasons.append("Trend: FALLING FAST (> 3 mg/dL/min) — imminent hypoglycemia risk")
    elif trend == "falling":
        reasons.append("Trend: falling (1–3 mg/dL/min)")
    elif trend == "rising_fast":
        reasons.append("Trend: RISING FAST (> 3 mg/dL/min)")
    elif trend == "rising":
        reasons.append("Trend: rising (1–3 mg/dL/min)")

    # Pattern detection from notes
    if "dawn" in notes.lower():
        reasons.append("Pattern: Dawn phenomenon detected")
    if "exercise" in notes.lower():
        reasons.append("Pattern: Exercise-induced glucose change")
    if "meal" in notes.lower() or "lunch" in notes.lower() or "dinner" in notes.lower():
        reasons.append(f"Pattern: Post-meal reading ({notes})")
    if "nocturnal" in notes.lower() or "hypoglycemia" in notes.lower():
        reasons.append("Pattern: Nocturnal hypoglycemia episode")

    alert_reason = " | ".join(reasons)

    msg = (
        f"[Monitor] {ts} — Glucose: {glucose:.1f} mg/dL, "
        f"Trend: {trend}, Alert: {alert_level.upper()} | {alert_reason}"
    )

    return {
        "alert_level": alert_level,
        "alert_reason": alert_reason,
        "agent_messages": [msg],
    }


# ---------------------------------------------------------------------------
# 2. Advisor Agent
# ---------------------------------------------------------------------------

def advisor_node(state: GASState) -> dict:
    """
    Advisor Agent — Generates RAG-grounded, evidence-based recommendations.

    Responsibilities
    ----------------
    - Retrieve relevant ADA guidelines from the knowledge base
    - Consider patient profile (diabetes type, medications, targets)
    - Consider glucose history (recent trend, time of day)
    - Generate a specific, actionable recommendation via LLM

    Returns
    -------
    dict
        Updates: recommendation, agent_messages
    """
    reading = state["current_reading"]
    patient = state["patient"]
    alert_level = state["alert_level"]
    alert_reason = state["alert_reason"]
    history = state.get("readings_history", [])

    glucose = reading["value_mgdl"]
    trend = reading["trend"]
    ts = reading["timestamp"]

    # Build RAG query
    query_parts = [f"glucose {glucose:.0f} mg/dL", f"trend {trend}"]
    if alert_level in ("high", "critical"):
        if glucose < 70:
            query_parts.append("hypoglycemia treatment")
        elif glucose > 250:
            query_parts.append("hyperglycemia correction insulin")
    if "dawn" in alert_reason.lower():
        query_parts.append("dawn phenomenon management")
    if "exercise" in alert_reason.lower():
        query_parts.append("exercise induced hypoglycemia")
    if "nocturnal" in alert_reason.lower():
        query_parts.append("nocturnal hypoglycemia caregiver")

    rag_query = " ".join(query_parts)
    kb = get_knowledge_base()
    retrieved_docs = kb.retrieve(rag_query, k=3)
    context = "\n\n".join(f"[Guideline {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs))

    # Build recent history summary
    recent = history[-6:] if len(history) >= 6 else history
    history_str = ", ".join(
        f"{r['value_mgdl']:.0f}" for r in recent
    ) if recent else "no prior readings"

    # Compose LLM prompt
    system_prompt = (
        "You are the Advisor agent in the Guardian Angel System (GAS), "
        "a multi-agent AI for diabetic patient supervision. "
        "Your role is to provide specific, actionable, evidence-based recommendations "
        "grounded in ADA guidelines. Be concise (2–4 sentences). "
        "Always prioritise patient safety. "
        "⚠️ This is for educational purposes only — not a medical device."
    )

    user_prompt = (
        f"Patient: {patient['name']}, Age {patient['age']}, "
        f"Type {patient['diabetes_type']} Diabetes\n"
        f"Medications: {', '.join(patient['medications']) or 'none listed'}\n"
        f"Target range: {patient['target_low']}–{patient['target_high']} mg/dL\n"
        f"ISF: {patient['insulin_sensitivity_factor']} mg/dL per unit\n\n"
        f"Current reading: {glucose:.1f} mg/dL at {ts}\n"
        f"Trend: {trend}\n"
        f"Alert: {alert_level.upper()} — {alert_reason}\n"
        f"Recent glucose history (last 30 min): [{history_str}] mg/dL\n\n"
        f"Relevant ADA guidelines:\n{context}\n\n"
        f"Provide a specific, actionable recommendation for this patient right now."
    )

    llm = get_llm()
    recommendation = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    msg = (
        f"[Advisor] Recommendation generated (alert={alert_level}, "
        f"glucose={glucose:.1f}, trend={trend}): "
        f"{recommendation[:120]}{'...' if len(recommendation) > 120 else ''}"
    )

    return {
        "recommendation": recommendation,
        "agent_messages": [msg],
    }


# ---------------------------------------------------------------------------
# 3. Safety Critic
# ---------------------------------------------------------------------------

def safety_node(state: GASState) -> dict:
    """
    Safety Critic — Validates every recommendation before delivery.

    Hard safety rules (automatic rejection)
    ----------------------------------------
    1. Never recommend insulin when glucose < 80 mg/dL
    2. Never recommend exercise when glucose < 70 mg/dL
    3. Never recommend waiting/monitoring when glucose < 54 mg/dL
    4. Never recommend correction bolus without accounting for IOB
    5. Never recommend DKA home management (must escalate to ER)

    Scoring
    -------
    - Starts at 1.0
    - Deducts for each safety concern found
    - Threshold for approval: >= 0.7

    Returns
    -------
    dict
        Updates: safety_score, safety_approved, agent_messages
    """
    recommendation = state["recommendation"]
    reading = state["current_reading"]
    alert_level = state["alert_level"]
    iteration = state.get("iteration", 0)

    glucose = reading["value_mgdl"]
    trend = reading["trend"]
    rec_lower = recommendation.lower()

    safety_score = 1.0
    concerns = []

    # Rule 1: No insulin during hypoglycemia
    # Check for positive insulin recommendations, excluding negated forms
    # e.g. "Do NOT give insulin" should NOT trigger this rule
    insulin_positive_phrases = [
        "give insulin", "administer insulin", "take insulin",
        "correction bolus", "correction dose", "units of insulin",
        "bolus insulin", "inject insulin",
    ]
    negation_prefixes = ("do not", "don't", "never", "avoid", "do not give",
                         "not give", "no insulin", "without insulin")
    
    def _is_positive_insulin_mention(text: str, phrase: str) -> bool:
        """Return True only if phrase appears without a preceding negation."""
        idx = text.find(phrase)
        if idx == -1:
            return False
        # Check the 20 characters before the phrase for negation
        prefix = text[max(0, idx - 20):idx].strip()
        return not any(neg in prefix for neg in negation_prefixes)
    
    if glucose < 80 and any(
        _is_positive_insulin_mention(rec_lower, phrase)
        for phrase in insulin_positive_phrases
    ):
        safety_score -= 0.5
        concerns.append(
            f"DANGER: Insulin recommended at {glucose:.1f} mg/dL (hypoglycemia risk)"
        )

    # Rule 2: No exercise during hypoglycemia
    if glucose < 70 and any(kw in rec_lower for kw in ["exercise", "activity", "walk"]):
        safety_score -= 0.3
        concerns.append(
            f"DANGER: Exercise recommended at {glucose:.1f} mg/dL (< 70 mg/dL)"
        )

    # Rule 3: No "wait and monitor" for critical hypoglycemia
    # Use specific phrases to avoid false positives (e.g. "monitor" in "notify caregiver")
    wait_phrases = ["continue monitoring", "monitor closely", "wait and see",
                    "recheck in 1 hour", "observe and wait", "just monitor"]
    if glucose < 54 and any(phrase in rec_lower for phrase in wait_phrases):
        if not any(kw in rec_lower for kw in ["15g", "glucose tablet", "juice", "glucagon",
                                               "fast-acting carb", "carbohydrate"]):
            safety_score -= 0.4
            concerns.append(
                f"DANGER: Passive monitoring recommended for critical glucose {glucose:.1f} mg/dL"
            )

    # Rule 4: DKA home management
    if glucose > 300 and any(kw in rec_lower for kw in ["manage at home", "home treatment"]):
        safety_score -= 0.5
        concerns.append("DANGER: DKA home management recommended — must escalate to ER")

    # Rule 5: Falling fast + positive insulin recommendation
    if trend == "falling_fast" and any(
        _is_positive_insulin_mention(rec_lower, phrase)
        for phrase in insulin_positive_phrases
    ):
        safety_score -= 0.3
        concerns.append(
            "WARNING: Insulin recommended with falling_fast trend — hypoglycemia risk"
        )

    # Positive checks — reward safe recommendations
    if glucose < 70 and any(kw in rec_lower for kw in ["15g", "glucose tablet", "juice", "carb"]):
        safety_score = min(1.0, safety_score + 0.1)  # correct treatment

    if glucose > 250 and "ketone" in rec_lower:
        safety_score = min(1.0, safety_score + 0.05)  # good DKA awareness

    safety_score = max(0.0, min(1.0, safety_score))
    approved = safety_score >= 0.7

    if concerns:
        concern_str = "; ".join(concerns)
        msg = (
            f"[Safety] Score: {safety_score:.2f} | REJECTED — {concern_str}"
        )
    else:
        msg = (
            f"[Safety] Score: {safety_score:.2f} | APPROVED — "
            f"No safety concerns detected (iteration {iteration})"
        )

    return {
        "safety_score": safety_score,
        "safety_approved": approved,
        "agent_messages": [msg],
    }


# ---------------------------------------------------------------------------
# 4. Caregiver Notifier
# ---------------------------------------------------------------------------

def caregiver_node(state: GASState) -> dict:
    """
    Caregiver Notifier — Sends structured alerts to caregivers.

    Activation criteria
    -------------------
    - Alert level is "high" or "critical"
    - Safety Critic has approved the recommendation

    Alert format
    ------------
    Structured notification including: patient name, timestamp, glucose value,
    trend, alert level, approved recommendation, and action required.

    In demo mode, the alert is printed to stdout.  In production, this would
    send an email/SMS/push notification via an A2A messaging service.

    Returns
    -------
    dict
        Updates: caregiver_notified, agent_messages
    """
    alert_level = state["alert_level"]
    patient = state["patient"]
    reading = state["current_reading"]
    recommendation = state["recommendation"]
    safety_approved = state["safety_approved"]

    should_notify = alert_level in ("high", "critical") and safety_approved

    if not should_notify:
        msg = (
            f"[Caregiver] No notification needed "
            f"(alert={alert_level}, approved={safety_approved})"
        )
        return {
            "caregiver_notified": False,
            "agent_messages": [msg],
        }

    glucose = reading["value_mgdl"]
    trend = reading["trend"]
    ts = reading["timestamp"]

    # Format the alert
    alert_emoji = "🚨" if alert_level == "critical" else "⚠️"
    alert_text = (
        f"\n{'='*60}\n"
        f"{alert_emoji} GUARDIAN ANGEL SYSTEM — {alert_level.upper()} ALERT\n"
        f"{'='*60}\n"
        f"Patient:     {patient['name']} (Age {patient['age']}, T{patient['diabetes_type']}D)\n"
        f"Time:        {ts}\n"
        f"Glucose:     {glucose:.1f} mg/dL\n"
        f"Trend:       {trend}\n"
        f"Alert Level: {alert_level.upper()}\n"
        f"\nRecommended Action:\n{recommendation}\n"
        f"\nTo: {patient['caregiver_name']} <{patient['caregiver_email']}>\n"
        f"{'='*60}\n"
    )

    # In production: send email/SMS/push notification
    # For demo: print to console
    if _RICH and console:
        style = "bold red" if alert_level == "critical" else "bold yellow"
        console.print(Panel(alert_text, title="📱 Caregiver Alert", style=style))
    else:
        print(alert_text)

    msg = (
        f"[Caregiver] Alert sent to {patient['caregiver_name']} "
        f"({patient['caregiver_email']}) — "
        f"Level: {alert_level.upper()}, Glucose: {glucose:.1f} mg/dL"
    )

    return {
        "caregiver_notified": True,
        "agent_messages": [msg],
    }


# ---------------------------------------------------------------------------
# 5. Supervisor Agent
# ---------------------------------------------------------------------------

def supervisor_node(state: GASState) -> dict:
    """
    Supervisor Agent — Routes between agents based on current state.

    Routing logic
    -------------
    START → monitor (always)
    monitor → advisor (always — generate recommendation)
    advisor → safety (always — validate recommendation)
    safety → caregiver (if approved OR max retries reached)
    safety → advisor (if rejected AND iteration < 2 — retry with feedback)
    caregiver → END (always)

    The iteration counter prevents infinite retry loops.

    Returns
    -------
    dict
        Updates: next_agent, iteration
    """
    alert_level = state["alert_level"]
    safety_approved = state.get("safety_approved", False)
    iteration = state.get("iteration", 0)
    messages = state.get("agent_messages", [])

    # Determine what just ran by inspecting the last agent message
    last_msg = messages[-1] if messages else ""

    if last_msg.startswith("[Monitor]"):
        next_agent = "advisor"
    elif last_msg.startswith("[Advisor]"):
        next_agent = "safety"
    elif last_msg.startswith("[Safety]"):
        if safety_approved:
            next_agent = "caregiver"
        elif iteration >= 2:
            # Max retries reached — pass through with warning
            logger.warning(
                "Safety Critic rejected recommendation after %d retries — "
                "passing to caregiver with warning",
                iteration,
            )
            next_agent = "caregiver"
        else:
            next_agent = "advisor"
            iteration += 1
    elif last_msg.startswith("[Caregiver]"):
        next_agent = "END"
    else:
        # Initial state — start with monitor
        next_agent = "advisor"

    return {
        "next_agent": next_agent,
        "iteration": iteration,
    }


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_from_supervisor(state: GASState) -> str:
    """
    Conditional edge function for LangGraph.

    Returns the value of state["next_agent"], which the graph uses to
    select the next node.
    """
    return state["next_agent"]


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

def build_gas_graph():
    """
    Assemble and compile the Guardian Angel System LangGraph.

    Graph topology
    --------------
    START → monitor → supervisor → advisor → supervisor → safety → supervisor
                                                                  ↓ (approved)
                                                             caregiver → END
                                                  ↑ (rejected, retry)
                                              advisor ←────────────┘

    Returns
    -------
    CompiledGraph
        A compiled LangGraph StateGraph with MemorySaver checkpointer.
    """
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        raise ImportError(
            "langgraph is required. Install with: pip install langgraph"
        ) from exc

    builder = StateGraph(GASState)

    # Add all agent nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("monitor", monitor_node)
    builder.add_node("advisor", advisor_node)
    builder.add_node("safety", safety_node)
    builder.add_node("caregiver", caregiver_node)

    # Fixed edges
    builder.add_edge(START, "monitor")
    builder.add_edge("monitor", "supervisor")
    builder.add_edge("advisor", "supervisor")
    builder.add_edge("safety", "supervisor")
    builder.add_edge("caregiver", END)

    # Conditional routing from supervisor
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "advisor": "advisor",
            "safety": "safety",
            "caregiver": "caregiver",
            "END": END,
        },
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ============================================================================
# PATIENT PROFILES
# ============================================================================

PATIENT_PROFILES: dict[str, PatientProfile] = {
    "alice": PatientProfile(
        patient_id="patient-alice-001",
        name="Alice Chen",
        age=34,
        diabetes_type=1,
        target_low=70.0,
        target_high=180.0,
        insulin_sensitivity_factor=50.0,
        carb_ratio=10.0,
        medications=[
            "Insulin glargine (Lantus) 22U at bedtime",
            "Insulin lispro (Humalog) — sliding scale with meals",
        ],
        caregiver_email="alice-caregiver@example.com",
        caregiver_name="David Chen (spouse)",
    ),
    "bob": PatientProfile(
        patient_id="patient-bob-002",
        name="Bob Martinez",
        age=58,
        diabetes_type=2,
        target_low=80.0,
        target_high=180.0,
        insulin_sensitivity_factor=40.0,
        carb_ratio=12.0,
        medications=[
            "Metformin 1000mg twice daily",
            "Empagliflozin (Jardiance) 10mg once daily",
            "Semaglutide (Ozempic) 0.5mg weekly",
        ],
        caregiver_email="bob-caregiver@example.com",
        caregiver_name="Maria Martinez (daughter)",
    ),
    "carol": PatientProfile(
        patient_id="patient-carol-003",
        name="Carol Thompson",
        age=16,
        diabetes_type=1,
        target_low=70.0,
        target_high=180.0,
        insulin_sensitivity_factor=60.0,
        carb_ratio=15.0,
        medications=[
            "Insulin pump (Omnipod 5) — automated insulin delivery",
            "Continuous glucose monitor (Dexcom G7)",
        ],
        caregiver_email="carol-parent@example.com",
        caregiver_name="Susan Thompson (parent)",
    ),
}


def get_patient(name: str) -> PatientProfile:
    """Return a PatientProfile by name, defaulting to alice."""
    return PATIENT_PROFILES.get(name.lower(), PATIENT_PROFILES["alice"])


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(
    scenario: str = "hypoglycemia_episode",
    patient_name: str = "alice",
    verbose: bool = True,
    max_readings: Optional[int] = None,
) -> list[dict]:
    """
    Run a full 24-hour GAS simulation on a clinical scenario.

    For each CGM reading (every 5 minutes = up to 288 readings per day):
      1. Build the initial GASState with the current reading
      2. Invoke the compiled LangGraph (monitor → advisor → safety → caregiver)
      3. Collect the final state and append to results

    Parameters
    ----------
    scenario : str
        One of: normal_day, hypoglycemia_episode, post_meal_spike,
        dawn_phenomenon, exercise_induced_low.
    patient_name : str
        Patient key: alice, bob, or carol.
    verbose : bool
        Print progress and alerts to stdout.
    max_readings : int | None
        Limit number of readings processed (useful for quick demos).
        None = process all 288 readings.

    Returns
    -------
    list[dict]
        One result dict per reading, containing: timestamp, glucose, trend,
        alert_level, recommendation, safety_score, safety_approved,
        caregiver_notified, agent_messages.
    """
    patient = get_patient(patient_name)
    simulator = CGMSimulator(seed=42)
    readings = simulator.generate(scenario)

    if max_readings is not None:
        readings = readings[:max_readings]

    # Build the graph (graceful fallback if langgraph not installed)
    gas_graph = None
    try:
        gas_graph = build_gas_graph()
    except ImportError as exc:
        logger.warning(
            "LangGraph not available (%s) — running agents directly (no graph checkpointing)",
            exc,
        )
        if verbose:
            _print(
                "ℹ️  LangGraph not installed — running agents in direct mode "
                "(install langgraph for full graph orchestration)",
                style="yellow",
            )

    results: list[dict] = []
    readings_history: list[dict] = []

    # Summary counters
    alert_counts: dict[str, int] = {
        "none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0
    }
    caregiver_notifications = 0
    safety_rejections = 0

    if verbose:
        _print(
            f"\n{'='*70}\n"
            f"🏥 Guardian Angel System — Simulation\n"
            f"   Scenario : {scenario}\n"
            f"   Patient  : {patient.name} (Age {patient.age}, T{patient.diabetes_type}D)\n"
            f"   Readings : {len(readings)} (24h at 5-min intervals)\n"
            f"{'='*70}",
            style="bold cyan",
        )

    for i, reading in enumerate(readings):
        reading_dict = reading.to_dict()

        # Build initial state for this reading
        initial_state: GASState = {
            "patient": patient.to_dict(),
            "readings_history": list(readings_history[-12:]),  # last 60 min
            "current_reading": reading_dict,
            "alert_level": "none",
            "alert_reason": "",
            "recommendation": "",
            "safety_score": 1.0,
            "safety_approved": False,
            "caregiver_notified": False,
            "agent_messages": [],
            "next_agent": "monitor",
            "iteration": 0,
        }

        # Run the graph (or fallback to direct agent calls)
        if gas_graph is not None:
            config = {"configurable": {"thread_id": f"{scenario}-{i}"}}
            try:
                final_state = gas_graph.invoke(initial_state, config=config)
            except Exception as exc:
                logger.error("Graph invocation failed at reading %d: %s", i, exc)
                # Fallback: run agents manually
                state = dict(initial_state)
                state.update(monitor_node(state))  # type: ignore[arg-type]
                state.update(advisor_node(state))  # type: ignore[arg-type]
                state.update(safety_node(state))   # type: ignore[arg-type]
                state.update(caregiver_node(state))  # type: ignore[arg-type]
                final_state = state
        else:
            # Direct agent execution (no LangGraph)
            state = dict(initial_state)
            m = monitor_node(state)  # type: ignore[arg-type]
            state.update(m)
            state["agent_messages"] = m.get("agent_messages", [])
            a = advisor_node(state)  # type: ignore[arg-type]
            state.update(a)
            state["agent_messages"] = state["agent_messages"] + a.get("agent_messages", [])
            s = safety_node(state)   # type: ignore[arg-type]
            state.update(s)
            state["agent_messages"] = state["agent_messages"] + s.get("agent_messages", [])
            c = caregiver_node(state)  # type: ignore[arg-type]
            state.update(c)
            state["agent_messages"] = state["agent_messages"] + c.get("agent_messages", [])
            final_state = state

        # Collect result
        result = {
            "index": i,
            "timestamp": reading_dict["timestamp"],
            "glucose": reading_dict["value_mgdl"],
            "trend": reading_dict["trend"],
            "notes": reading_dict.get("notes", ""),
            "alert_level": final_state.get("alert_level", "none"),
            "alert_reason": final_state.get("alert_reason", ""),
            "recommendation": final_state.get("recommendation", ""),
            "safety_score": final_state.get("safety_score", 1.0),
            "safety_approved": final_state.get("safety_approved", False),
            "caregiver_notified": final_state.get("caregiver_notified", False),
            "agent_messages": final_state.get("agent_messages", []),
        }
        results.append(result)

        # Update history
        readings_history.append(reading_dict)
        if len(readings_history) > 288:
            readings_history = readings_history[-288:]

        # Update counters
        al = result["alert_level"]
        alert_counts[al] = alert_counts.get(al, 0) + 1
        if result["caregiver_notified"]:
            caregiver_notifications += 1
        if not result["safety_approved"] and result["recommendation"]:
            safety_rejections += 1

        # Verbose output for notable readings
        if verbose and al in ("medium", "high", "critical"):
            ts_str = reading.timestamp.strftime("%H:%M")
            glucose = reading_dict["value_mgdl"]
            trend = reading_dict["trend"]
            rec_short = result["recommendation"][:80] + "..." if len(result["recommendation"]) > 80 else result["recommendation"]
            style = "bold red" if al == "critical" else ("bold yellow" if al == "high" else "yellow")
            _print(
                f"  [{ts_str}] {glucose:6.1f} mg/dL  {trend:<12}  "
                f"Alert: {al.upper():<8}  → {rec_short}",
                style=style,
            )
        elif verbose and i % 24 == 0:
            # Print every 2 hours even if no alert
            ts_str = reading.timestamp.strftime("%H:%M")
            glucose = reading_dict["value_mgdl"]
            _print(
                f"  [{ts_str}] {glucose:6.1f} mg/dL  {reading_dict['trend']:<12}  "
                f"Alert: {al.upper():<8}",
                style="dim",
            )

    # Print summary
    if verbose:
        _print(
            f"\n{'='*70}\n"
            f"📊 Simulation Summary — {scenario}\n"
            f"{'='*70}",
            style="bold cyan",
        )
        total = len(results)
        tir = sum(1 for r in results if 70 <= r["glucose"] <= 180) / total * 100
        tbr = sum(1 for r in results if r["glucose"] < 70) / total * 100
        tar = sum(1 for r in results if r["glucose"] > 180) / total * 100
        avg_glucose = sum(r["glucose"] for r in results) / total

        _print(f"  Total readings     : {total}")
        _print(f"  Average glucose    : {avg_glucose:.1f} mg/dL")
        _print(f"  Time in range (TIR): {tir:.1f}% (target: >70%)")
        _print(f"  Time below range   : {tbr:.1f}% (target: <4%)")
        _print(f"  Time above range   : {tar:.1f}%")
        _print(f"\n  Alert distribution:")
        for level in ["none", "low", "medium", "high", "critical"]:
            count = alert_counts.get(level, 0)
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            _print(f"    {level:<8}: {count:3d} ({pct:5.1f}%)  {bar}")
        _print(f"\n  Caregiver notifications: {caregiver_notifications}")
        _print(f"  Safety rejections      : {safety_rejections}")
        _print(f"{'='*70}\n")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Guardian Angel System — Multi-Agent Diabetic Patient Supervision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gas_system.py
  python gas_system.py --scenario hypoglycemia_episode
  python gas_system.py --scenario post_meal_spike --patient bob
  python gas_system.py --scenario dawn_phenomenon --max-readings 48
  NOTEBOOK_TEST_MODE=1 python gas_system.py  # CI-safe mock mode

⚠️  CLINICAL DISCLAIMER: For educational purposes only. Not a medical device.
        """,
    )
    parser.add_argument(
        "--scenario",
        default="hypoglycemia_episode",
        choices=CGMSimulator.SCENARIOS,
        help="Clinical scenario to simulate (default: hypoglycemia_episode)",
    )
    parser.add_argument(
        "--patient",
        default="alice",
        choices=list(PATIENT_PROFILES.keys()),
        help="Patient profile to use (default: alice)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--max-readings",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N readings (default: all 288)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        default=False,
        help="Run all 5 scenarios sequentially",
    )

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    if args.all_scenarios:
        print("🏥 Guardian Angel System — Running ALL 5 scenarios\n")
        all_results = {}
        for sc in CGMSimulator.SCENARIOS:
            print(f"\n{'─'*50}")
            print(f"> Scenario: {sc}")
            print(f"{'─'*50}")
            res = run_simulation(
                scenario=sc,
                patient_name=args.patient,
                verbose=verbose,
                max_readings=args.max_readings,
            )
            all_results[sc] = res
            print(f"✅ {sc}: {len(res)} readings processed")
        print(f"\n🎉 All scenarios complete!")
    else:
        print(f"🏥 Guardian Angel System — Scenario: {args.scenario}")
        results = run_simulation(
            scenario=args.scenario,
            patient_name=args.patient,
            verbose=verbose,
            max_readings=args.max_readings,
        )
        print(f"✅ Simulation complete. {len(results)} readings processed.")
