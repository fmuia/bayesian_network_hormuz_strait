"""HTML-escaping of user-controlled text in the HTML-rendered surfaces (post-P5
basic-review fix): a headline with angle brackets must render literally, not as
injected markup that corrupts the layout."""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import state  # noqa: E402


def test_headline_html_is_escaped_in_the_observation_log():
    at = AppTest.from_file("app/dashboard.py", default_timeout=90)
    at.session_state["use_fake_translator"] = True
    at.run()
    at.session_state["observations"] = [state.make_observation(
        day=1, headline="<b>boom</b> & <script>x</script>", source="manual",
        assignments={"Tanker_Incidents": "none"}, item_id="x")]
    at.run()
    at.session_state["active_view"] = "📝  Observations"
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "&lt;b&gt;boom&lt;/b&gt;" in md      # rendered escaped (literal)
    assert "<b>boom</b>" not in md              # not injected as raw HTML
