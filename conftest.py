"""Pytest session setup.

Pin the suite to the Hormuz scenario pack. The shared test fixtures (translator
fixtures, golden marginals, viz/scenario assertions) are written against Hormuz,
so the suite must run on it regardless of any ``SCENARIO_PACK`` the developer has
exported. Meridian (and any other pack) is covered by its own tests that import
``packs.<id>`` directly, so they are unaffected by this pin.

Set before any test module imports ``src.scenario`` (the seam resolves the active
pack at import); if it is already imported, reload it.
"""
import os
import sys

os.environ["SCENARIO_PACK"] = "hormuz"

if "src.scenario" in sys.modules:  # already imported (e.g. by a plugin) → re-resolve
    sys.modules["src.scenario"].reload()
