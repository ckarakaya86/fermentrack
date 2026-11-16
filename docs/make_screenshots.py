"""Regenerate every screenshot in docs/screenshots/.

    python docs/make_screenshots.py

Starts its own copies of the dashboard and the workbench (ports 8611 / 8612, with a
temporary model registry so your real models are not touched), opens them in the
Microsoft Edge that comes with Windows (via Playwright), waits until each page has
completely finished running, clicks through the workbench where a step needs
results, and saves full-page PNGs.

Needs:  pip install playwright   (no extra browser download: it uses the installed Edge)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
DASH, WB = "http://localhost:8611", "http://localhost:8612"
WIDTH = 1440


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------
def start_server(script: str, port: int, env) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / script), "--server.headless", "true",
           "--server.port", str(port), "--browser.gatherUsageStats", "false",
           "--client.toolbarMode", "minimal", "--theme.base", "light"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"{script} did not start")


# ---------------------------------------------------------------------------
# waiting and capturing
# ---------------------------------------------------------------------------
def wait_ready(page: Page, timeout=180):
    """Wait until Streamlit has finished running (no 'Running...' widget for 2 s in a row)."""
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout * 1000)
    end, quiet_since = time.time() + timeout, None
    while time.time() < end:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        stale = page.locator('[data-stale="true"]').count() > 0
        if running or stale:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since > 2.0:
            break
        time.sleep(0.25)
    page.wait_for_timeout(1500)          # let plotly finish drawing


def shot(page: Page, name: str):
    """Full-height screenshot.

    Streamlit scrolls inside an inner container, so a normal full-page capture only shows
    one screen. Instead: make the window very tall so everything is drawn, measure where
    the content really ends, shrink the window to exactly that height, then capture.
    """
    page.set_viewport_size({"width": WIDTH, "height": 5000})
    page.wait_for_timeout(1500)
    wait_ready(page, 60)
    bottom = page.evaluate("""() => {
        const c = document.querySelector('[data-testid="stMainBlockContainer"]');
        if (!c) return document.body.scrollHeight;
        let b = 0;
        c.querySelectorAll('*').forEach(e => {        // only what is really visible
            if (!e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return;
            const r = e.getBoundingClientRect();
            if (r.height > 0) b = Math.max(b, r.bottom); });
        return Math.ceil(b);
    }""")
    page.set_viewport_size({"width": WIDTH, "height": min(max(bottom + 60, 900), 5000)})
    page.mouse.move(5, 5)                 # keep chart toolbars / tooltips out of the picture
    page.wait_for_timeout(2000)
    wait_ready(page, 60)
    page.screenshot(path=str(OUT / f"{name}.png"))
    page.set_viewport_size({"width": WIDTH, "height": 900})
    print("saved", name, bottom)


def goto(page: Page, url: str):
    page.goto(url)
    wait_ready(page)


def click(page: Page, text: str, exact=True):
    page.get_by_text(text, exact=exact).first.click()
    wait_ready(page)


def button(page: Page, label: str):
    page.get_by_role("button", name=label).first.click()
    wait_ready(page)


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FERMENTRACK_REGISTRY=tempfile.mkdtemp(prefix="ft_registry_"))
    servers = [start_server("app.py", 8611, env), start_server("workbench_app.py", 8612, env)]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge")
            page = browser.new_page(viewport={"width": WIDTH, "height": 900})

            # --- dashboard ---------------------------------------------------------
            goto(page, f"{DASH}/?page=monitor&batch=68&t=36")
            shot(page, "batch_monitor")
            goto(page, f"{DASH}/?page=monitor&batch=62&t=40")
            shot(page, "probe_fouling")
            goto(page, f"{DASH}/?page=overview")
            shot(page, "plant_overview")
            goto(page, f"{DASH}/?page=performance&section=pls")
            shot(page, "pls_calibration")
            goto(page, f"{DASH}/?page=performance&section=yield")
            shot(page, "yield_forecast")

            # --- workbench ---------------------------------------------------------
            goto(page, f"{WB}/")                                   # data step + format help
            shot(page, "wb_data")
            for step in ["spectra", "split"]:
                goto(page, f"{WB}/?demo=1&step={step}")
                shot(page, f"wb_{step}")

            # one user journey in one session: find outliers -> remove them -> look at the
            # cleaned model -> robustness -> save -> check healthy data -> check drifting data
            goto(page, f"{WB}/?demo=1&step=outliers")
            shot(page, "wb_outliers")
            button(page, "Exclude ticked samples and re-fit")
            click(page, "4 Model")
            shot(page, "wb_model")
            click(page, "6 Robustness")
            for b in ["Run perturbation test", "Run resampling test", "Run y-randomisation"]:
                button(page, b)
            shot(page, "wb_robustness")
            click(page, "7 Save & update")
            button(page, "Save as new version")
            click(page, "8 Auto-check")
            click(page, "ftir_new_lab_ok.xlsx")
            button(page, "Run check")
            click(page, "ftir_new_lab_drift.xlsx")
            click(page, "Allow automatic update (saves a better version if action is needed)")
            button(page, "Run check")
            shot(page, "wb_autocheck")
            click(page, "7 Save & update")
            shot(page, "wb_registry")
            browser.close()
    finally:
        for s in servers:
            s.terminate()


if __name__ == "__main__":
    main()
