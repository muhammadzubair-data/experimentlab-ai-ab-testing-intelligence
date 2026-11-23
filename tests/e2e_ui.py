"""Browser end-to-end check of the running ExperimentLab dashboard (Playwright + Chromium).

Not collected by pytest (file name). Usage:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501 [--chrome /path/to/chrome]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_messy_sample import build  # noqa: E402

PAGES = [("", "Experiment Overview", 3), ("diagnostics", "Assignment & Data Quality", 1),
         ("segments", "Segment Explorer", 1), ("planner", "Sample Size Planner", 1),
         ("data", "Data Upload & Exports", 0), ("method", "Methodology & Limitations", 0)]


def idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=180_000)
    time.sleep(1.5)
    prev = -1
    for _ in range(150):
        busy = page.locator("[data-testid='stSpinner']").count() + page.locator("[data-testid='stStatusWidget']").count()
        n = page.locator(".js-plotly-plot").count()
        if busy == 0 and n == prev:
            break
        prev = n
        time.sleep(1)
    time.sleep(extra)


def download(page, label) -> bytes:
    with page.expect_download() as dl:
        page.get_by_role("button", name=label).first.click()
    return Path(dl.value.path()).read_bytes()


def sidebar_go(page, title):
    page.locator("[data-testid='stSidebarNav'] a", has_text=title).first.click()
    idle(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:8501")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    fails: list = []
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if a.chrome:
            launch["executable_path"] = a.chrome
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        print("Pages:")
        for path, title, min_charts in PAGES:
            page.goto(f"{base}/{path}", wait_until="networkidle")
            idle(page)
            exc = page.locator("[data-testid='stException']").count()
            errs = page.locator("[data-testid='stAlertContentError']").count()
            charts = page.locator(".js-plotly-plot").count()
            ok_title = title in page.locator(".el-hero h1").first.inner_text()
            print(f"  {path or 'overview':<12} exceptions={exc} errors={errs} charts={charts} title={'ok' if ok_title else 'MISSING'}")
            if exc or errs or charts < min_charts or not ok_title:
                fails.append((path, exc, errs, charts, ok_title))

        print("Exports:")
        page.goto(f"{base}/", wait_until="networkidle")
        idle(page)
        res = json.loads(download(page, "Results (JSON)"))
        ok = res["units_control"] in (2500, "2500") and abs(float(res["z_test_p_value"]) - 0.8884) < 1e-3
        print(f"  results.json  units={res['units_control']} p={float(res['z_test_p_value']):.4f} ok={ok}")
        if not ok:
            fails.append(("results json", res))
        page.goto(f"{base}/segments", wait_until="networkidle")
        idle(page)
        seg = pd.read_csv(io.BytesIO(download(page, "Download segment table")))
        ok = sorted(seg.segment) == ["Desktop", "Mobile", "Tablet"] and (seg.p_holm >= seg.p_value - 1e-12).all()
        print(f"  segments.csv  rows={len(seg)} holm>=raw={ok}")
        if not ok:
            fails.append(("segments", seg))
        page.goto(f"{base}/planner", wait_until="networkidle")
        idle(page)
        plan = pd.read_csv(io.BytesIO(download(page, "Download plan (CSV)"))).set_index("parameter").value
        sp = float(plan["simulated_power"])
        print(f"  plan.csv      users_control={plan['users_control']} simulated_power={sp:.3f}")
        if abs(sp - .8) > .03:
            fails.append(("plan", sp))

        print("Upload:")
        page.goto(f"{base}/data", wait_until="networkidle")
        idle(page)
        tmp = Path(tempfile.mkdtemp()) / "messy_experiment.csv"
        build().to_csv(tmp, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(tmp))
        idle(page, 4)
        body = page.locator("[data-testid='stMain']").inner_text()
        ok = "messy_experiment.csv" in body and "Group → variant" in body and "excluded" in body
        print(f"  upload        accepted, validation report shown={ok}")
        if not ok:
            fails.append(("upload", body[:300]))
        summ = pd.read_csv(io.BytesIO(download(page, "Variant summary")))
        if sorted(summ.variant) != ["control", "new_checkout"] or summ.units.sum() != 2987:
            fails.append(("variant summary", summ))
        for title in ["Experiment Overview", "Segment Explorer", "Sample Size Planner"]:
            sidebar_go(page, title)
            used = page.locator(".pill.ok").count() > 0
            exc = page.locator("[data-testid='stException']").count()
            print(f"  session       {title:<24} uses upload={used} exceptions={exc}")
            if not used or exc:
                fails.append(("session", title))
        sidebar_go(page, "Data Upload & Exports")
        page.get_by_role("button", name="Remove upload").click()
        idle(page)
        back = page.locator(".pill.warn").count() > 0
        print(f"  upload        removed, back to synthetic={back}")
        if not back:
            fails.append(("remove",))
        bad = tmp.with_name("bad.csv")
        pd.DataFrame({"when": ["2026-01-01"], "value": [1]}).to_csv(bad, index=False)
        page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(str(bad))
        idle(page, 3)
        err = page.locator("[data-testid='stAlertContentError']").all_inner_texts()
        print(f"  upload        bad file error shown={bool(err)}")
        if not err or "Missing required column" not in err[0]:
            fails.append(("bad upload", err))
        browser.close()
    print("FAILURES:" if fails else "All browser checks passed.", *fails, sep="\n  ")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
