"""
shared/__init__.py
==================
Public API for the agentic-ai-book shared utilities package.

Import everything you need from a single location::

    from shared import LLMClient, GASState, PatientProfile, CGMSimulator
    from shared import create_default_patient, glucose_status

Modules
-------
llm_client   — Unified LLM wrapper (OpenAI / Anthropic / Ollama)
gas_state    — Guardian Angel System state schema and dataclasses
patient_data — Synthetic CGM data generator
"""

from shared.llm_client import LLMClient, count_tokens

from shared.gas_state import (
    # Dataclasses
    PatientProfile,
    GlucoseReading,
    GASMemory,
    MedicationLogEntry,
    Episode,
    CaregiverContact,
    # TypedDict
    GASState,
    # Helper functions
    create_default_patient,
    create_initial_gas_state,
    glucose_status,
    alert_level,
)

from shared.patient_data import CGMSimulator

__all__ = [
    # llm_client
    "LLMClient",
    "count_tokens",
    # gas_state — dataclasses
    "PatientProfile",
    "GlucoseReading",
    "GASMemory",
    "MedicationLogEntry",
    "Episode",
    "CaregiverContact",
    # gas_state — TypedDict
    "GASState",
    # gas_state — helpers
    "create_default_patient",
    "create_initial_gas_state",
    "glucose_status",
    "alert_level",
    # patient_data
    "CGMSimulator",
]
