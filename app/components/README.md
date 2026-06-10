# Dashboard component library & information architecture

`app/dashboard.py` is **orchestration only** (page setup, the engine/caching
layer, header, navigation, view dispatch, footer). Everything renderable lives in
small component modules here, established by the Plan 5 A1 split (commits P1–P5).

## Conventions

- A component exposes one or a few **`render(st, …)`** functions that take the
  Streamlit handle plus the data they need (posterior / CI dicts, `ci_table`,
  `observations`, …). They are **pure rendering**: no inference, no global-state
  mutation beyond the documented session-state writes.
- Components consume the shared palette (`app/theme.py`), session helpers
  (`app/state.py`), and chart helpers (`components/ci_charts.py`) — never the
  dashboard (that would be circular).
- Pure helpers (e.g. `ci_charts._width_category`, everything in `app/state.py`)
  are unit-tested; render functions are covered by `AppTest`.

## Information architecture (the four surface homes)

The dashboard has exactly four kinds of surface. **Top-level *views* register
through the single `st.segmented_control`; persistent surfaces do not.**

| Home | Surface(s) | Why |
|------|------------|-----|
| **Sidebar** (persistent) | the news→evidence **translator** (`translator_stream.render_sidebar`) | always-available evidence entry; drives every view |
| **Pinned top band** (always visible) | scenario cards + probability-evolution chart (`scenario_cards`, `evolution_chart`) | the headline output; the locus of attention |
| **Main-area expander** | the Plan-4 **Elicitation layer** (`elicitation_panel`) | an advanced / occasional CPT-setup surface — deliberately *not* a nav view |
| **Nav views** (one active at a time, via `st.segmented_control`) | Network & model, Observations, Triage, Audit trail, Edge rationale | the analyst's working views |

Nav uses a session-state `segmented_control` (not `st.tabs`) so each view
**re-mounts** on switch — required for the `streamlit-agraph` DAG canvas, which
renders zoomed/blank when kept mounted-but-hidden.

## Icon scheme

One distinct emoji per nav view; **🧪 is reserved for the Elicitation lab**
(the expander), never a nav view:

| Icon | Surface |
|------|---------|
| 🕸️ | Network & model |
| 📝 | Observations |
| ⚖️ | Triage (approve / edit / reject) |
| 🔎 | Audit trail |
| 🧭 | Edge rationale |
| 🧪 | Elicitation layer *(expander, not a view)* |

`tests/test_ia.py` enforces that the nav icons are distinct and that none is 🧪.

## Module map

- `theme.py` — palette + scenario labels (shared by dashboard + components).
- `state.py` *(in `app/`)* — session defaults, evidence merge, named-session
  persistence, HITL review-queue helpers.
- `ci_charts.py` — CI dumbbell / flat bar / robustness badge + `_ci_dataframe`.
- `scenario_cards.py`, `evolution_chart.py` — the pinned band.
- `network_view.py`, `observation_log.py`, `structured_panel.py`,
  `edge_rationale.py`, `triage_view.py`, `audit_view.py` — the nav views.
- `translator_stream.py` — the sidebar translator subsystem.
- `model_explainer.py` — the "How this model works" overview + appendix.
- `elicitation_panel.py` *(in `app/`, Plan 4)` — the elicitation expander.
