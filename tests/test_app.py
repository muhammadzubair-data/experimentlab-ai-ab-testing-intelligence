"""Streamlit AppTest checks: every page renders, data sources and controls take effect, uploads are used."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from experimentlab.core import prepare
from scripts.make_messy_sample import build

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["", "diagnostics", "segments", "planner", "data", "method"]


def run(page="", **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if page:
        at._page_hash = next(h for h, i in at._registered_pages.items() if i["url_pathname"] == page)
        at.run()
    return at


def text(at):
    return " ".join(m.value for m in at.markdown)


def upload_state():
    clean, rep = prepare(build())
    return dict(upload_df=clean, upload_report=rep, upload_name="export.csv", upload_key="k1", use_upload=True,
                _jump_to_upload=True)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert "Synthetic demonstration data" in text(at)


def test_overview_default_is_honest():
    at = run("")
    t = text(at)
    assert "No statistically significant difference" in t and "does not show there is no effect" in t
    assert "Synthetic check" in t and "only about 28%" in t


@pytest.mark.parametrize("page", ["", "diagnostics", "segments", "planner", "data"])
def test_upload_used_everywhere(page):
    at = run(page, **upload_state())
    assert not at.exception, [e.value for e in at.exception]
    assert "Your uploaded file" in text(at)
    assert at.sidebar.selectbox[0].value == "control" and at.sidebar.selectbox[1].value == "new_checkout"


def test_scenario_builder_srm_and_effect():
    at = run("")
    at.sidebar.radio[0].set_value("Scenario builder").run()
    assert not at.exception and "converted higher" in text(at)
    at.sidebar.slider[2].set_value(.53).run()
    assert not at.exception and "sample ratio mismatch" in text(at)


def test_alpha_and_control_switch():
    at = run("")
    at.sidebar.select_slider[0].set_value(.10).run()
    assert not at.exception and "α = 0.1" in text(at)
    at.sidebar.selectbox[0].set_value("B").run()
    assert not at.exception and at.sidebar.selectbox[1].value == "A"


def test_diagnostics_time_view_with_scenario():
    at = run("diagnostics")
    at.sidebar.radio[0].set_value("Scenario builder").run()
    assert not at.exception and "No sample ratio mismatch" in text(at)


def test_segments_switch_column():
    at = run("segments")
    at.sidebar.radio[0].set_value("Scenario builder").run()
    at.selectbox[0].set_value("country").run()
    assert not at.exception and "Segments compared" in text(at)


def test_planner_inputs():
    at = run("planner")
    at.number_input[1].set_value(5.0).run()
    assert not at.exception and "Simulated power" in text(at)
    at.selectbox[0].set_value(2.0).run()
    assert not at.exception and "ratio 2 : 1" in text(at)


def test_data_filters():
    at = run("data")
    at.selectbox[0].set_value("Converted only").run()
    assert not at.exception
