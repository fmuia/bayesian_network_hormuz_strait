"""Curated catalogue of plausible Strait-of-Hormuz news events.

Each event maps a short headline to a dictionary of node assignments
that the dashboard applies as evidence when the user toggles it on.
Dates are synthetic and roughly ordered for narrative purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

Category = Literal["escalation", "de-escalation", "mixed"]


@dataclass(frozen=True)
class EvidenceEvent:
    """A single news event the user can add as evidence."""

    id: str
    date: str               # ISO date (synthetic)
    headline: str
    category: Category
    assignments: Dict[str, str]


# 18 events spanning the full escalation/de-escalation/mixed spectrum.
# Assignments are deliberately partial — each event sets only the
# nodes it speaks directly to, leaving the rest to inference.
EVENTS: List[EvidenceEvent] = [
    # ---------------- de-escalation ----------------
    EvidenceEvent(
        id="oman_backchannel",
        date="2026-01-12",
        headline="Oman confirms active US–Iran back-channel talks",
        category="de-escalation",
        assignments={"Third_Party_Mediation": "active"},
    ),
    EvidenceEvent(
        id="prisoner_swap",
        date="2026-01-28",
        headline="US–Iran prisoner swap completed via Qatar",
        category="de-escalation",
        assignments={
            "US_Iran_Negotiations": "stalled",
            "Third_Party_Mediation": "active",
        },
    ),
    EvidenceEvent(
        id="sanctions_waiver",
        date="2026-02-09",
        headline="Treasury issues 90-day sanctions waiver for Iranian oil exports",
        category="de-escalation",
        assignments={"Sanctions_Trajectory": "easing"},
    ),
    EvidenceEvent(
        id="iaea_access",
        date="2026-02-21",
        headline="Iran restores IAEA inspector access at Natanz",
        category="de-escalation",
        assignments={
            "US_Iran_Negotiations": "success",
            "Iranian_Regime_Stability": "stable",
        },
    ),
    EvidenceEvent(
        id="ceasefire_signal",
        date="2026-03-04",
        headline="Tehran signals readiness for de-escalation framework",
        category="de-escalation",
        assignments={
            "US_Iran_Negotiations": "success",
            "Diplomatic_Resolution_Path": "open",
        },
    ),
    EvidenceEvent(
        id="strait_reopens_traffic",
        date="2026-03-15",
        headline="Tanker insurers report normal Hormuz transit volumes",
        category="de-escalation",
        assignments={
            "Strait_Operationally_Closed": "no",
            "Tanker_Incidents": "none",
        },
    ),
    # ---------------- mixed ------------------------
    EvidenceEvent(
        id="isolated_attack",
        date="2026-01-19",
        headline="Single tanker reports limpet-mine damage off Fujairah",
        category="mixed",
        assignments={"Tanker_Incidents": "isolated"},
    ),
    EvidenceEvent(
        id="us_carrier_redeploy",
        date="2026-02-02",
        headline="US redeploys carrier strike group to Arabian Sea",
        category="mixed",
        assignments={"US_Military_Response": "limited"},
    ),
    EvidenceEvent(
        id="proxy_drone_intercept",
        date="2026-02-14",
        headline="Houthi drones intercepted in Bab al-Mandeb; no casualties",
        category="mixed",
        assignments={"Iranian_Proxy_Activity": "elevated"},
    ),
    EvidenceEvent(
        id="oil_spike_60",
        date="2026-02-25",
        headline="Brent settles in the $90–120 band on supply jitters",
        category="mixed",
        assignments={"Oil_Price_Regime": "90_to_120"},
    ),
    EvidenceEvent(
        id="negotiations_stall",
        date="2026-03-01",
        headline="Vienna talks adjourn without communiqué",
        category="mixed",
        assignments={"US_Iran_Negotiations": "stalled"},
    ),
    EvidenceEvent(
        id="regime_pressure",
        date="2026-03-08",
        headline="Protests reported in three Iranian provinces",
        category="mixed",
        assignments={"Iranian_Regime_Stability": "pressured"},
    ),
    # ---------------- escalation -------------------
    EvidenceEvent(
        id="frequent_incidents",
        date="2026-03-11",
        headline="Fourth tanker incident in two weeks; insurers raise war-risk premia",
        category="escalation",
        assignments={"Tanker_Incidents": "frequent"},
    ),
    EvidenceEvent(
        id="sanctions_snapback",
        date="2026-03-17",
        headline="UN snapback sanctions reinstated by E3",
        category="escalation",
        assignments={
            "Sanctions_Trajectory": "tightening",
            "Diplomatic_Resolution_Path": "narrowing",
        },
    ),
    EvidenceEvent(
        id="negotiations_breakdown",
        date="2026-03-22",
        headline="Tehran withdraws negotiating team; calls talks 'finished'",
        category="escalation",
        assignments={"US_Iran_Negotiations": "breakdown"},
    ),
    EvidenceEvent(
        id="strait_partial_closure",
        date="2026-03-28",
        headline="IRGC announces 'inspection regime' on Hormuz traffic",
        category="escalation",
        assignments={
            "Strait_Operationally_Closed": "partial",
            "Iranian_Proxy_Activity": "high",
        },
    ),
    EvidenceEvent(
        id="us_strikes",
        date="2026-04-02",
        headline="US conducts strikes against IRGC naval assets",
        category="escalation",
        assignments={
            "US_Military_Response": "major",
            "Diplomatic_Resolution_Path": "narrowing",
        },
    ),
    EvidenceEvent(
        id="terminal_damage",
        date="2026-04-08",
        headline="Major fire at Ras Tanura terminal after missile strike",
        category="escalation",
        assignments={
            "Energy_Infrastructure_Damage": "severe",
            "Strait_Operationally_Closed": "full",
            "Oil_Price_Regime": "above_120",
        },
    ),
]


def events_by_category() -> Dict[str, List[EvidenceEvent]]:
    """Group the catalogue by category for sidebar display."""
    grouped: Dict[str, List[EvidenceEvent]] = {
        "de-escalation": [],
        "mixed": [],
        "escalation": [],
    }
    for ev in EVENTS:
        grouped[ev.category].append(ev)
    return grouped


__all__ = ["EvidenceEvent", "EVENTS", "events_by_category"]
