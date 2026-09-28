"""
shared/gas_state.py
===================
Guardian Angel System (GAS) — shared state schema.

This module defines the data structures that flow through the GAS multi-agent
pipeline across all book chapters.  The schema is intentionally additive: later
chapters extend it without breaking earlier ones.

Key types
---------
PatientProfile   — static patient demographics and medical history
GlucoseReading   — a single CGM or manual glucose measurement
GASMemory        — rolling history: readings, episodes, medications, contacts
GASState         — the LangGraph state TypedDict that agents read/write

Helper functions
----------------
create_default_patient()  → PatientProfile
glucose_status(reading)   → "hypoglycemia" | "normal" | "hyperglycemia" | "severe"
alert_level(reading)      → "none" | "low" | "medium" | "high" | "critical"
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

# TypedDict is available in typing from Python 3.8+
try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Enumerations (as Literal types for LangGraph compatibility)
# ---------------------------------------------------------------------------

DiabetesType = Literal[1, 2]
TrendType = Literal["rising", "falling", "stable"]
SourceType = Literal["CGM", "manual"]
AlertLevelType = Literal["none", "low", "medium", "high", "critical"]
GlucoseStatusType = Literal["hypoglycemia", "normal", "hyperglycemia", "severe"]


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatientProfile:
    """
    Static patient demographics and medical configuration.

    Attributes
    ----------
    patient_id : str
        Unique identifier (UUID by default).
    name : str
        Patient's display name.
    age : int
        Patient's age in years.
    diabetes_type : DiabetesType
        1 for Type 1 (insulin-dependent), 2 for Type 2.
    target_glucose_low : float
        Lower bound of the target glucose range in mg/dL (default 70).
    target_glucose_high : float
        Upper bound of the target glucose range in mg/dL (default 180).
    medications : list[str]
        Current medications (e.g. ["Metformin 500mg", "Insulin glargine 20U"]).
    allergies : list[str]
        Known drug/food allergies.
    weight_kg : float | None
        Body weight in kilograms (optional, used for dosing calculations).
    physician_name : str
        Name of the primary care physician.
    emergency_contact : str
        Emergency contact name and phone number.
    """

    patient_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unknown Patient"
    age: int = 0
    diabetes_type: DiabetesType = 1
    target_glucose_low: float = 70.0
    target_glucose_high: float = 180.0
    medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    weight_kg: Optional[float] = None
    physician_name: str = "Dr. Unknown"
    emergency_contact: str = ""

    def is_in_target_range(self, value_mgdl: float) -> bool:
        """Return True if *value_mgdl* is within the patient's target range."""
        return self.target_glucose_low <= value_mgdl <= self.target_glucose_high

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "diabetes_type": self.diabetes_type,
            "target_glucose_low": self.target_glucose_low,
            "target_glucose_high": self.target_glucose_high,
            "medications": list(self.medications),
            "allergies": list(self.allergies),
            "weight_kg": self.weight_kg,
            "physician_name": self.physician_name,
            "emergency_contact": self.emergency_contact,
        }


@dataclass
class GlucoseReading:
    """
    A single glucose measurement from a CGM sensor or manual finger-stick.

    Attributes
    ----------
    timestamp : datetime
        When the reading was taken (UTC recommended).
    value_mgdl : float
        Glucose concentration in milligrams per decilitre.
    trend : TrendType
        Direction of change: "rising", "falling", or "stable".
    source : SourceType
        "CGM" for continuous sensor, "manual" for finger-stick.
    notes : str
        Optional free-text annotation (e.g. "after lunch", "post-exercise").
    """

    timestamp: datetime = field(default_factory=datetime.utcnow)
    value_mgdl: float = 100.0
    trend: TrendType = "stable"
    source: SourceType = "CGM"
    notes: str = ""

    @property
    def status(self) -> GlucoseStatusType:
        """Convenience property — delegates to :func:`glucose_status`."""
        return glucose_status(self)

    @property
    def alert(self) -> AlertLevelType:
        """Convenience property — delegates to :func:`alert_level`."""
        return alert_level(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value_mgdl": self.value_mgdl,
            "trend": self.trend,
            "source": self.source,
            "notes": self.notes,
            "status": self.status,
            "alert": self.alert,
        }

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return (
            f"GlucoseReading({ts}, {self.value_mgdl:.1f} mg/dL, "
            f"trend={self.trend}, status={self.status})"
        )


@dataclass
class MedicationLogEntry:
    """
    A single medication administration record.

    Attributes
    ----------
    timestamp : datetime
        When the medication was taken.
    medication : str
        Medication name and dose (e.g. "Insulin lispro 4U").
    administered_by : str
        "patient", "caregiver", or "auto-pump".
    notes : str
        Optional notes.
    """

    timestamp: datetime = field(default_factory=datetime.utcnow)
    medication: str = ""
    administered_by: str = "patient"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "medication": self.medication,
            "administered_by": self.administered_by,
            "notes": self.notes,
        }


@dataclass
class Episode:
    """
    A recorded clinical episode (hypoglycemia event, hyperglycemia crisis, etc.).

    Attributes
    ----------
    episode_id : str
        Unique identifier.
    start_time : datetime
        When the episode began.
    end_time : datetime | None
        When the episode resolved (None if ongoing).
    episode_type : str
        E.g. "hypoglycemia", "hyperglycemia", "DKA_risk".
    severity : AlertLevelType
        Severity at peak.
    description : str
        Human-readable summary.
    actions_taken : list[str]
        Interventions applied during the episode.
    """

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    episode_type: str = "unknown"
    severity: AlertLevelType = "low"
    description: str = ""
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "episode_type": self.episode_type,
            "severity": self.severity,
            "description": self.description,
            "actions_taken": list(self.actions_taken),
        }


@dataclass
class CaregiverContact:
    """
    Contact information for a caregiver or healthcare provider.

    Attributes
    ----------
    name : str
        Full name.
    role : str
        E.g. "parent", "spouse", "endocrinologist", "emergency".
    phone : str
        Phone number.
    email : str
        Email address.
    notify_on : list[AlertLevelType]
        Alert levels that should trigger a notification to this contact.
    """

    name: str = ""
    role: str = "caregiver"
    phone: str = ""
    email: str = ""
    notify_on: list[AlertLevelType] = field(
        default_factory=lambda: ["high", "critical"]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "phone": self.phone,
            "email": self.email,
            "notify_on": list(self.notify_on),
        }


@dataclass
class GASMemory:
    """
    Rolling memory store for the Guardian Angel System.

    Attributes
    ----------
    recent_readings : list[GlucoseReading]
        Glucose readings from the last 24 hours (newest last).
    episodes : list[Episode]
        Historical clinical episodes.
    medication_log : list[MedicationLogEntry]
        Medication administration history.
    caregiver_contacts : list[CaregiverContact]
        People to notify in an emergency.
    """

    recent_readings: list[GlucoseReading] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    medication_log: list[MedicationLogEntry] = field(default_factory=list)
    caregiver_contacts: list[CaregiverContact] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_reading(self, reading: GlucoseReading, max_history: int = 288) -> None:
        """
        Append *reading* and trim to the most recent *max_history* entries.

        288 readings = 24 hours at 5-minute intervals (CGM default).
        """
        self.recent_readings.append(reading)
        if len(self.recent_readings) > max_history:
            self.recent_readings = self.recent_readings[-max_history:]

    def latest_reading(self) -> Optional[GlucoseReading]:
        """Return the most recent reading, or None if history is empty."""
        return self.recent_readings[-1] if self.recent_readings else None

    def average_glucose(self) -> Optional[float]:
        """Return the mean glucose over recent_readings, or None if empty."""
        if not self.recent_readings:
            return None
        return sum(r.value_mgdl for r in self.recent_readings) / len(
            self.recent_readings
        )

    def time_in_range(self, low: float = 70.0, high: float = 180.0) -> float:
        """
        Return the fraction of recent readings within [low, high].

        Returns 0.0 if there are no readings.
        """
        if not self.recent_readings:
            return 0.0
        in_range = sum(
            1 for r in self.recent_readings if low <= r.value_mgdl <= high
        )
        return in_range / len(self.recent_readings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_readings": [r.to_dict() for r in self.recent_readings],
            "episodes": [e.to_dict() for e in self.episodes],
            "medication_log": [m.to_dict() for m in self.medication_log],
            "caregiver_contacts": [c.to_dict() for c in self.caregiver_contacts],
        }


# ---------------------------------------------------------------------------
# LangGraph state TypedDict
# ---------------------------------------------------------------------------

class GASState(TypedDict, total=False):
    """
    The shared state dictionary passed between LangGraph nodes.

    All fields are optional (``total=False``) so individual agents can update
    only the fields they own without needing to supply the full state.

    Fields
    ------
    patient : PatientProfile
        Static patient profile.
    current_reading : GlucoseReading | None
        The glucose reading currently being processed.
    memory : GASMemory
        Rolling history and contacts.
    alert_level : AlertLevelType
        Current alert severity computed from *current_reading*.
    recommendation : str
        Natural-language recommendation from the reasoning agent.
    safety_approved : bool
        True once the safety-check agent has approved the recommendation.
    caregiver_notified : bool
        True once the notification agent has alerted caregivers.
    agent_messages : list[str]
        Audit trail of messages produced by each agent node.
    next_agent : str
        Name of the next agent node to route to (used by the router).
    metadata : dict[str, Any]
        Arbitrary extra data for chapter-specific extensions.
    """

    patient: PatientProfile
    current_reading: Optional[GlucoseReading]
    memory: GASMemory
    alert_level: AlertLevelType
    recommendation: str
    safety_approved: bool
    caregiver_notified: bool
    agent_messages: list[str]
    next_agent: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def glucose_status(reading: GlucoseReading) -> GlucoseStatusType:
    """
    Classify a glucose reading into a clinical status category.

    Thresholds (mg/dL)
    ------------------
    - severe hyperglycemia : > 400
    - hyperglycemia        : > 180
    - normal               : 70 – 180
    - hypoglycemia         : < 70

    Parameters
    ----------
    reading : GlucoseReading
        The reading to classify.

    Returns
    -------
    GlucoseStatusType
        One of ``"hypoglycemia"``, ``"normal"``, ``"hyperglycemia"``,
        ``"severe"``.
    """
    v = reading.value_mgdl
    if v > 400:
        return "severe"
    if v > 180:
        return "hyperglycemia"
    if v >= 70:
        return "normal"
    return "hypoglycemia"


def alert_level(reading: GlucoseReading) -> AlertLevelType:
    """
    Compute the alert level for a glucose reading.

    Alert thresholds (mg/dL)
    ------------------------
    - critical : < 54  or  > 400
    - high     : < 70  or  > 300
    - medium   : < 80  or  > 250
    - low      : < 90  or  > 200
    - none     : 90 – 200 (well within range)

    Parameters
    ----------
    reading : GlucoseReading
        The reading to evaluate.

    Returns
    -------
    AlertLevelType
        One of ``"none"``, ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
    """
    v = reading.value_mgdl
    if v < 54 or v > 400:
        return "critical"
    if v < 70 or v > 300:
        return "high"
    if v < 80 or v > 250:
        return "medium"
    if v < 90 or v > 200:
        return "low"
    return "none"


def create_default_patient(
    name: str = "Alex Johnson",
    age: int = 35,
    diabetes_type: DiabetesType = 1,
) -> PatientProfile:
    """
    Create a realistic default patient for demos and testing.

    Parameters
    ----------
    name : str
        Patient name (default: "Alex Johnson").
    age : int
        Patient age (default: 35).
    diabetes_type : DiabetesType
        1 or 2 (default: 1).

    Returns
    -------
    PatientProfile
        A fully populated patient profile with sensible defaults.
    """
    if diabetes_type == 1:
        medications = [
            "Insulin glargine (Lantus) 20U at bedtime",
            "Insulin lispro (Humalog) — sliding scale with meals",
        ]
    else:
        medications = [
            "Metformin 1000mg twice daily",
            "Empagliflozin (Jardiance) 10mg once daily",
        ]

    return PatientProfile(
        patient_id=str(uuid.uuid4()),
        name=name,
        age=age,
        diabetes_type=diabetes_type,
        target_glucose_low=70.0,
        target_glucose_high=180.0,
        medications=medications,
        allergies=["Sulfa drugs"],
        weight_kg=75.0,
        physician_name="Dr. Sarah Chen",
        emergency_contact="Jordan Johnson — +1-555-0100",
    )


def create_initial_gas_state(patient: Optional[PatientProfile] = None) -> GASState:
    """
    Build a fresh, fully-initialised :class:`GASState`.

    Parameters
    ----------
    patient : PatientProfile | None
        Patient to embed in the state.  If *None*, a default patient is created.

    Returns
    -------
    GASState
        Ready-to-use state dict for a new LangGraph run.
    """
    return GASState(
        patient=patient or create_default_patient(),
        current_reading=None,
        memory=GASMemory(),
        alert_level="none",
        recommendation="",
        safety_approved=False,
        caregiver_notified=False,
        agent_messages=[],
        next_agent="monitor",
        metadata={},
    )


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timezone

    print("=== GAS State Demo ===\n")

    patient = create_default_patient()
    print(f"Patient: {patient.name}, age {patient.age}, T{patient.diabetes_type}D")
    print(f"Target range: {patient.target_glucose_low}–{patient.target_glucose_high} mg/dL")
    print(f"Medications: {patient.medications}\n")

    readings = [
        GlucoseReading(
            timestamp=datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc),
            value_mgdl=48.0,
            trend="falling",
            source="CGM",
            notes="Nocturnal hypoglycemia",
        ),
        GlucoseReading(
            timestamp=datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc),
            value_mgdl=145.0,
            trend="stable",
            source="CGM",
            notes="Post-lunch",
        ),
        GlucoseReading(
            timestamp=datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc),
            value_mgdl=310.0,
            trend="rising",
            source="CGM",
            notes="Missed bolus",
        ),
    ]

    for r in readings:
        print(
            f"  {r.timestamp.strftime('%H:%M')}  "
            f"{r.value_mgdl:6.1f} mg/dL  "
            f"status={r.status:<15}  alert={r.alert}"
        )

    print("\nInitial GASState:")
    state = create_initial_gas_state(patient)
    print(f"  alert_level    : {state['alert_level']}")
    print(f"  safety_approved: {state['safety_approved']}")
    print(f"  next_agent     : {state['next_agent']}")
