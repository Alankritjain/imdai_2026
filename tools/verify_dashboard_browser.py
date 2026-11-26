"""Optional real-browser smoke check. Requires Playwright and installed Chrome.

Start Streamlit on localhost:8501 first. This does not train or alter the model.
Playwright is a verification tool, not a runtime project dependency.
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from config import PROCESSED_DIR, REPORTS_DIR


def main():
    with (PROCESSED_DIR / "test.csv").open(newline="", encoding="utf-8") as source:
        record = next(csv.DictReader(source))
    reports = REPORTS_DIR
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.request.get("http://127.0.0.1:8501/_stcore/health")
        assert response.status == 200
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        expect(page.get_by_role("heading", name="Self-Driving Car Behavioral Cloning", exact=True)).to_be_visible(timeout=60000)
        tabs = page.get_by_role("tab")
        expect(tabs).to_have_count(6)
        checked_tabs = []
        for name in ["Project overview", "Dataset overview", "Image & preprocessing explorer",
                     "Steering prediction", "Model performance", "About the project"]:
            page.get_by_role("tab", name=name, exact=True).click()
            checked_tabs.append(name)
        page.get_by_role("tab", name="Image & preprocessing explorer", exact=True).click()
        index_input = page.get_by_role("spinbutton")
        index_input.fill("25")
        index_input.press("Enter")
        expect(page.get_by_text("Source row: 25", exact=False)).to_be_visible()
        page.get_by_role("tab", name="Steering prediction", exact=True).click()
        page.get_by_text("Upload image", exact=True).click()
        page.locator('input[type="file"]').set_input_files(record["image_path"])
        metric = page.get_by_test_id("stMetric").filter(has_text="Predicted Steering")
        expect(metric).to_be_visible(timeout=60000)
        prediction = float(metric.get_by_test_id("stMetricValue").inner_text())
        assert math.isfinite(prediction)
        page.screenshot(path=str(reports / "dashboard_prediction.png"), full_page=True)
        page.get_by_role("tab", name="Model performance", exact=True).click()
        expect(page.get_by_text("CNN MAE", exact=True)).to_be_visible()
        page.screenshot(path=str(reports / "dashboard_performance.png"), full_page=True)
        page.get_by_role("tab", name="Steering prediction", exact=True).click()
        page.locator('input[type="file"]').set_input_files({
            "name": "invalid.jpg", "mimeType": "image/jpeg", "buffer": b"not an image"})
        expect(page.get_by_text("Invalid image:", exact=False)).to_be_visible(timeout=30000)
        page.locator('input[type="file"]').set_input_files(record["image_path"])
        expect(metric).to_be_visible(timeout=30000)
        page.get_by_text("Recorded replay", exact=True).click()
        frame_caption = page.get_by_text(re.compile(r"^Recorded frame \d+ of \d+$"))
        expect(frame_caption).to_be_visible(timeout=30000)
        first_caption = frame_caption.inner_text()
        page.get_by_text("Play recording", exact=True).click()
        expect(frame_caption).not_to_have_text(first_caption, timeout=30000)
        page.get_by_text("Play recording", exact=True).click()
        page.screenshot(path=str(reports / "dashboard_replay.png"), full_page=True)
        page.get_by_text("Built-in examples", exact=True).click()
        expect(metric).to_be_visible(timeout=30000)
        page.screenshot(path=str(reports / "dashboard_examples.png"), full_page=True)
        assert not errors, errors
        browser.close()
    verification = {"browser": "Chrome headless via Playwright", "health_http_status": 200,
                    "tabs_checked": checked_tabs, "sample_selection": "source row 25",
                    "real_uploaded_image": record["image_path"], "actual_steering": float(record["steering"]),
                    "displayed_prediction": prediction, "invalid_upload_handled": True,
                    "built_in_examples": True, "recorded_playback_advanced": True,
                    "page_errors": errors}
    (reports / "dashboard_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

