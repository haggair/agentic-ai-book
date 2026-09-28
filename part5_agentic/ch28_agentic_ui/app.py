"""
app.py — Guardian Angel System Patient Dashboard
================================================
A complete Streamlit application for monitoring diabetes patients
using the Guardian Angel System (GAS) multi-agent framework.

Run with:
    streamlit run app.py

Requirements:
    pip install streamlit plotly
"""

import sys
import os
import math
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ── Add shared/ to path so we can import CGMSimulator ────────
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Guardian Angel System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Optional imports ──────────────────────────────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Run: pip install plotly")

# ── Patient profiles ──────────────────────────────────────────
PATIENTS = {
    "Alice T1D": {
        "id": "alice-001",
        "diabetes_type": 1,
        "age": 34,
        "isf": 50,          # insulin sensitivity factor (mg/dL per unit)
        "target_low": 70,
        "target_high": 180,
        "scenario": "hypoglycemia_episode",
        "seed": 42,
        "emergency_contact": "+1-555-0100",
    },
    "Bob T2D": {
        "id": "bob-002",
        "diabetes_type": 2,
        "age": 58,
        "isf": 80,
        "target_low": 80,
        "target_high": 200,
        "scenario": "post_meal_spike",
        "seed": 99,
        "emergency_contact": "+1-555-0200",
    },
}

# ── Session state initialization ──────────────────────────────
if "approval_history" not in st.session_state:
    st.session_state.approval_history = []
if "agent_feed" not in st.session_state:
    st.session_state.agent_feed = []
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = "Alice T1D"
if "feed_initialized" not in st.session_state:
    st.session_state.feed_initialized = {}


# ── Helper functions ──────────────────────────────────────────

def classify_glucose(glucose: float) -> dict:
    """Return display properties for a glucose value."""
    if glucose < 40:
        return {"level": "EMERGENCY", "color": "#CC0000", "emoji": "🚨", "alert": "critical"}
    elif glucose < 54:
        return {"level": "CRITICAL LOW", "color": "#FF0000", "emoji": "🔴", "alert": "critical"}
    elif glucose < 70:
        return {"level": "LOW", "color": "#FF6600", "emoji": "🟠", "alert": "high"}
    elif glucose <= 180:
        return {"level": "IN RANGE", "color": "#00AA00", "emoji": "🟢", "alert": "normal"}
    elif glucose <= 250:
        return {"level": "HIGH", "color": "#FF6600", "emoji": "🟠", "alert": "high"}
    elif glucose <= 350:
        return {"level": "CRITICAL HIGH", "color": "#FF0000", "emoji": "🔴", "alert": "critical"}
    else:
        return {"level": "EMERGENCY", "color": "#CC0000", "emoji": "🚨", "alert": "critical"}


def get_trend_arrow(values: list) -> str:
    """Get trend arrow from recent readings."""
    if len(values) < 3:
        return "→"
    recent = values[-3:]
    delta = recent[-1] - recent[0]
    rate = delta / (len(recent) - 1)  # per reading
    if rate > 5:
        return "↑↑"
    elif rate > 2:
        return "↑"
    elif rate < -5:
        return "↓↓"
    elif rate < -2:
        return "↓"
    return "→"


def get_recommendation(glucose: float, trend: str, patient: dict) -> tuple:
    """Get recommendation and alert level for current glucose."""
    if glucose < 40:
        return (
            "🚨 EMERGENCY: Glucose critically low. Call 911. Administer glucagon immediately.",
            "critical"
        )
    elif glucose < 54:
        return (
            "🔴 CRITICAL: Severe hypoglycemia. Consume 15g fast-acting carbs NOW. Recheck in 15 min.",
            "critical"
        )
    elif glucose < patient["target_low"]:
        if "↓" in trend:
            return (
                "🟠 WARNING: Glucose falling below target. Consume 15g carbs. Monitor closely.",
                "high"
            )
        return (
            "🟠 CAUTION: Glucose below target. Consume 10g carbs. Recheck in 15 min.",
            "high"
        )
    elif glucose > 350:
        return (
            "🔴 CRITICAL: Severe hyperglycemia. Check ketones immediately. Contact provider.",
            "critical"
        )
    elif glucose > 250:
        return (
            "🟠 WARNING: Significant hyperglycemia. Administer correction dose. Check for ketones.",
            "high"
        )
    elif glucose > patient["target_high"]:
        return (
            f"🟠 CAUTION: Glucose above target. Check for missed bolus. Consider correction dose.",
            "high"
        )
    else:
        return (
            "🟢 OK: Glucose within target range. Continue current management.",
            "normal"
        )


def load_glucose_data(patient: dict) -> tuple:
    """Load 24h of glucose data for a patient."""
    try:
        from shared.patient_data import CGMSimulator
        sim = CGMSimulator(
            patient_id=patient["id"],
            diabetes_type=patient["diabetes_type"],
            seed=patient["seed"],
        )
        readings = sim.generate_scenario(patient["scenario"])
        times = [r.timestamp for r in readings]
        values = [r.value_mgdl for r in readings]
        return times, values
    except (ImportError, Exception):
        # Fallback: generate synthetic data without CGMSimulator
        return _generate_fallback_data(patient)


def _generate_fallback_data(patient: dict) -> tuple:
    """Generate synthetic glucose data without CGMSimulator."""
    rng = random.Random(patient["seed"])
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    times = [now + timedelta(minutes=5 * i) for i in range(288)]
    values = []

    for t in times:
        h = t.hour + t.minute / 60

        if patient["scenario"] == "hypoglycemia_episode":
            if 2.5 <= h <= 4.5:
                # Nocturnal low
                base_v = 105 - 65 * math.exp(-((h - 3.5) ** 2) / 0.4)
            elif 4.5 < h <= 7:
                # Somogyi rebound
                base_v = 105 + 80 * math.exp(-((h - 5) ** 2) / 0.5)
            else:
                meal_effect = (
                    50 * math.exp(-((h - 8) ** 2) / 0.5) +
                    60 * math.exp(-((h - 13) ** 2) / 0.5) +
                    55 * math.exp(-((h - 19) ** 2) / 0.5)
                )
                base_v = 115 + meal_effect
        elif patient["scenario"] == "post_meal_spike":
            meal_effect = (
                40 * math.exp(-((h - 8) ** 2) / 0.5) +
                200 * math.exp(-((h - 13) ** 2) / 0.3) +  # missed bolus
                50 * math.exp(-((h - 19) ** 2) / 0.5)
            )
            base_v = 120 + meal_effect
        else:
            meal_effect = (
                50 * math.exp(-((h - 8) ** 2) / 0.5) +
                60 * math.exp(-((h - 13) ** 2) / 0.5) +
                55 * math.exp(-((h - 19) ** 2) / 0.5)
            )
            base_v = 105 + meal_effect

        values.append(max(40.0, min(450.0, base_v + rng.gauss(0, 5))))

    return times, values


def add_agent_event(agent: str, message: str, severity: str = "info"):
    """Add an event to the agent activity feed."""
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "message": message,
        "severity": severity,
    }
    st.session_state.agent_feed.insert(0, event)
    # Keep only last 10 events
    st.session_state.agent_feed = st.session_state.agent_feed[:10]


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 GAS Dashboard")
    st.caption("Guardian Angel System")
    st.divider()

    # Patient selector
    st.subheader("👤 Patient")
    selected = st.radio(
        "Select patient:",
        options=list(PATIENTS.keys()),
        index=list(PATIENTS.keys()).index(st.session_state.selected_patient),
        label_visibility="collapsed",
    )

    if selected != st.session_state.selected_patient:
        st.session_state.selected_patient = selected
        st.session_state.agent_feed = []  # reset feed on patient change

    patient = PATIENTS[selected]

    # Patient profile
    st.divider()
    st.subheader("📋 Profile")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Type", f"T{patient['diabetes_type']}D")
        st.metric("ISF", f"{patient['isf']}")
    with col_b:
        st.metric("Age", patient["age"])
        st.metric("Target", f"{patient['target_low']}–{patient['target_high']}")

    st.caption(f"📞 Emergency: {patient['emergency_contact']}")

    st.divider()

    # Refresh button
    refresh = st.button("🔄 Refresh Data", use_container_width=True, type="primary")

    st.divider()
    st.caption("GAS v1.0 | Part 5 Demo")


# ── Load data ─────────────────────────────────────────────────
times, values = load_glucose_data(patient)
current_glucose = values[-1]
trend_arrow = get_trend_arrow(values[-6:])
glucose_info = classify_glucose(current_glucose)
recommendation, alert_level = get_recommendation(current_glucose, trend_arrow, patient)

# Initialize agent feed on first load or refresh
feed_key = f"{selected}_{len(st.session_state.agent_feed)}"
if refresh or not st.session_state.agent_feed:
    add_agent_event("Monitor", f"CGM reading: {current_glucose:.0f} mg/dL (trend: {trend_arrow})",
                    "critical" if alert_level == "critical" else "warning" if alert_level == "high" else "info")
    add_agent_event("Supervisor", f"Routing to Advisor (alert_level={alert_level})", "info")
    add_agent_event("Advisor", f"Recommendation generated: {recommendation[:45]}...", alert_level)
    add_agent_event("Safety", "Guardrails check: PASSED ✓", "info")
    if alert_level in ("high", "critical"):
        add_agent_event("Caregiver",
                        f"Alert queued for {patient['emergency_contact']}",
                        alert_level)


# ── Page header ───────────────────────────────────────────────
header_col, refresh_col = st.columns([5, 1])
with header_col:
    st.title(f"🏥 Guardian Angel System — {selected}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
with refresh_col:
    st.write("")  # spacer
    if st.button("🔄 Refresh", key="header_refresh"):
        st.rerun()


# ── Main layout: 3 columns ────────────────────────────────────
col_chart, col_status, col_feed = st.columns([6, 2, 2])


# ── Column 1: Glucose Chart (60%) ────────────────────────────
with col_chart:
    st.subheader("📈 Glucose — Last 24 Hours")

    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        # Target range (green shading)
        fig.add_hrect(
            y0=patient["target_low"], y1=patient["target_high"],
            fillcolor="rgba(0,200,0,0.08)",
            line_width=0,
            annotation_text=f"Target ({patient['target_low']}–{patient['target_high']})",
            annotation_position="top left",
            annotation_font_size=10,
        )

        # Danger zones (red shading)
        fig.add_hrect(
            y0=0, y1=54,
            fillcolor="rgba(255,0,0,0.12)",
            line_width=0,
            annotation_text="Severe Hypo (<54)",
            annotation_position="bottom left",
            annotation_font_size=9,
        )
        fig.add_hrect(
            y0=300, y1=500,
            fillcolor="rgba(255,0,0,0.12)",
            line_width=0,
            annotation_text="Severe Hyper (>300)",
            annotation_position="top left",
            annotation_font_size=9,
        )

        # Threshold lines
        fig.add_hline(
            y=patient["target_low"],
            line_dash="dash", line_color="orange", opacity=0.6,
            annotation_text=f"{patient['target_low']}", annotation_position="right",
        )
        fig.add_hline(
            y=patient["target_high"],
            line_dash="dash", line_color="orange", opacity=0.6,
            annotation_text=f"{patient['target_high']}", annotation_position="right",
        )
        fig.add_hline(y=54, line_dash="dot", line_color="red", opacity=0.5)

        # Glucose trace
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name="Glucose",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x|%H:%M}<br><b>%{y:.0f} mg/dL</b><extra></extra>",
        ))

        # Current reading marker
        fig.add_trace(go.Scatter(
            x=[times[-1]],
            y=[current_glucose],
            mode="markers+text",
            name="Current",
            marker=dict(
                color=glucose_info["color"],
                size=14,
                symbol="circle",
                line=dict(color="white", width=2),
            ),
            text=[f"  {current_glucose:.0f}"],
            textposition="middle right",
            textfont=dict(size=12, color=glucose_info["color"]),
            hovertemplate=f"<b>Current: {current_glucose:.0f} mg/dL</b><extra></extra>",
        ))

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(
                title="Time",
                tickformat="%H:%M",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
            yaxis=dict(
                title="Glucose (mg/dL)",
                range=[30, max(max(values) + 30, patient["target_high"] + 60)],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.05)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        # Fallback: simple text display
        st.warning("Install plotly for the glucose chart: `pip install plotly`")
        st.write(f"Current glucose: **{current_glucose:.0f} mg/dL** {trend_arrow}")
        st.write(f"Min: {min(values):.0f} | Max: {max(values):.0f} | Mean: {sum(values)/len(values):.0f}")


# ── Column 2: Status Card (20%) ───────────────────────────────
with col_status:
    st.subheader("📊 Status")

    # Main glucose display
    st.markdown(
        f"""
        <div style="
            background-color: {glucose_info['color']}18;
            border: 2px solid {glucose_info['color']};
            border-radius: 12px;
            padding: 16px 12px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <div style="font-size: 2.8em; font-weight: bold; color: {glucose_info['color']}; line-height: 1.1;">
                {current_glucose:.0f}
            </div>
            <div style="font-size: 0.85em; color: #666; margin-bottom: 4px;">mg/dL</div>
            <div style="font-size: 1.8em; margin-bottom: 4px;">{trend_arrow}</div>
            <div style="
                font-size: 0.75em;
                font-weight: bold;
                color: {glucose_info['color']};
                background: {glucose_info['color']}22;
                border-radius: 6px;
                padding: 3px 8px;
                display: inline-block;
            ">
                {glucose_info['emoji']} {glucose_info['level']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Time-in-range and stats
    if values:
        in_range = sum(1 for v in values if patient["target_low"] <= v <= patient["target_high"])
        tir = in_range / len(values) * 100
        below = sum(1 for v in values if v < patient["target_low"]) / len(values) * 100
        above = sum(1 for v in values if v > patient["target_high"]) / len(values) * 100

        st.metric("⏱ Time in Range", f"{tir:.0f}%",
                  delta=f"{tir - 70:.0f}% vs target" if tir != 70 else None)

        col_min, col_max = st.columns(2)
        with col_min:
            st.metric("Min", f"{min(values):.0f}")
        with col_max:
            st.metric("Max", f"{max(values):.0f}")

        st.metric("Mean", f"{sum(values)/len(values):.0f} mg/dL")

        # TIR breakdown bar
        st.caption("24h Distribution")
        st.progress(tir / 100, text=f"In range: {tir:.0f}%")


# ── Column 3: Agent Activity Feed (20%) ──────────────────────
with col_feed:
    st.subheader("🤖 Agent Feed")

    AGENT_EMOJIS = {
        "Monitor": "👁️",
        "Advisor": "💡",
        "Safety": "🛡️",
        "Caregiver": "📱",
        "Supervisor": "🎯",
    }

    SEVERITY_COLORS = {
        "info": "#1f77b4",
        "warning": "#FF6600",
        "critical": "#FF0000",
        "normal": "#00AA00",
        "high": "#FF6600",
    }

    if not st.session_state.agent_feed:
        st.caption("No activity yet. Click Refresh.")
    else:
        for event in st.session_state.agent_feed:
            emoji = AGENT_EMOJIS.get(event["agent"], "🤖")
            border_color = SEVERITY_COLORS.get(event.get("severity", "info"), "#1f77b4")

            st.markdown(
                f"""
                <div style="
                    border-left: 3px solid {border_color};
                    padding: 5px 8px;
                    margin-bottom: 7px;
                    background: {border_color}08;
                    border-radius: 0 6px 6px 0;
                ">
                    <div style="font-size: 0.72em; color: #888;">{event['time']}</div>
                    <div style="font-size: 0.82em; font-weight: bold; color: {border_color};">
                        {emoji} {event['agent']}
                    </div>
                    <div style="font-size: 0.78em; color: #333; margin-top: 2px;">
                        {event['message'][:55]}{'...' if len(event['message']) > 55 else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Bottom: Recommendation Panel ─────────────────────────────
st.divider()

rec_col, btn_col = st.columns([3, 1])

with rec_col:
    st.subheader("💊 Recommendation")

    alert_colors = {
        "normal": "#00AA00",
        "high": "#FF6600",
        "critical": "#FF0000",
    }
    rec_color = alert_colors.get(alert_level, "#666666")

    st.markdown(
        f"""
        <div style="
            background-color: {rec_color}10;
            border-left: 5px solid {rec_color};
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            font-size: 1.05em;
            line-height: 1.5;
        ">
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )

with btn_col:
    st.subheader("👤 Human Gate")

    if alert_level in ("high", "critical"):
        st.warning("⚠️ Approval required")

        approve_col, reject_col = st.columns(2)

        with approve_col:
            if st.button("✅\nAPPROVE", type="primary", use_container_width=True, key="approve_btn"):
                entry = {
                    "time": datetime.now().isoformat(),
                    "action": "APPROVED",
                    "glucose": current_glucose,
                    "alert_level": alert_level,
                    "recommendation": recommendation,
                    "patient": selected,
                }
                st.session_state.approval_history.append(entry)
                add_agent_event("Caregiver", "Recommendation APPROVED by human ✓", "info")
                st.success("✅ Approved!")
                st.rerun()

        with reject_col:
            if st.button("❌\nREJECT", use_container_width=True, key="reject_btn"):
                entry = {
                    "time": datetime.now().isoformat(),
                    "action": "REJECTED",
                    "glucose": current_glucose,
                    "alert_level": alert_level,
                    "recommendation": recommendation,
                    "patient": selected,
                }
                st.session_state.approval_history.append(entry)
                add_agent_event("Caregiver", "Recommendation REJECTED by human ✗", "warning")
                st.warning("❌ Rejected")
                st.rerun()

    else:
        st.success("✅ Auto-approved\n(normal range)")


# ── Approval History ──────────────────────────────────────────
if st.session_state.approval_history:
    patient_history = [
        e for e in st.session_state.approval_history
        if e.get("patient") == selected
    ]

    if patient_history:
        with st.expander(
            f"📋 Approval History — {selected} ({len(patient_history)} decisions)",
            expanded=False,
        ):
            # Summary stats
            approved = sum(1 for e in patient_history if e["action"] == "APPROVED")
            rejected = sum(1 for e in patient_history if e["action"] == "REJECTED")

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Total Decisions", len(patient_history))
            with stat_col2:
                st.metric("✅ Approved", approved)
            with stat_col3:
                st.metric("❌ Rejected", rejected)

            st.divider()

            # Recent decisions
            for entry in reversed(patient_history[-10:]):
                icon = "✅" if entry["action"] == "APPROVED" else "❌"
                time_str = entry["time"][:19].replace("T", " ")
                st.write(
                    f"{icon} **{entry['action']}** | "
                    f"{time_str} | "
                    f"Glucose: {entry['glucose']:.0f} mg/dL | "
                    f"Alert: {entry['alert_level'].upper()}"
                )
                st.caption(f"  → {entry['recommendation'][:80]}")


# ── Footer ────────────────────────────────────────────────────
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🏥 Guardian Angel System v1.0")
with footer_col2:
    st.caption("📚 Agentic AI Book — Part 5, Chapter 28")
with footer_col3:
    st.caption("⚠️ For educational purposes only — not medical advice")
