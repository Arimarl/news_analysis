
from __future__ import annotations

import re
import sys
import time
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
import csv


# ─────────────────────────── BROWSER HELPER ────────────────────────────
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, StaleElementReferenceException
from selenium_stealth import stealth

WAIT_TIME = 15
SEARCH_URL   = "https://bioguide.congress.gov/search"
SEARCH_BOX   = (By.ID, "header-searchbar")
RESULT_CARD  = (
    By.CSS_SELECTOR,
    "#search-results-view .c-card, #search-results-view div[data-testid='record-card']",
)
DATE_SPAN    = (
    By.CSS_SELECTOR,
    "div.u-fz--sm.u-fw--semibold, span.u-fz--sm.u-fw--semibold",
)  # contains "1867 – 1932"
DATE_RE = re.compile(r"(?P<born>\d{4})(?:\s*–\s*(?P<died>\d{4}))?")

PROFILE_DIR = Path("/tmp/chrome-bioguide-profile")

def _detect_chrome_major() -> int | None:
    candidates: list[str] = []
    sys_plat = platform.system()
    if sys_plat == "Darwin":
        candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    elif sys_plat == "Linux":
        candidates = [shutil.which(cmd) for cmd in ("google-chrome", "chromium", "chromium-browser")]
    elif sys_plat == "Windows":
        candidates = [r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                      r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"]
    for exe in filter(None, candidates):
        try:
            out = subprocess.check_output([exe, "--version"], stderr=subprocess.DEVNULL).decode()
            if m := re.search(r"(\d+)\.", out):
                return int(m.group(1))
        except Exception:
            continue
    return None

def launch_browser() -> uc.Chrome:
    opts = uc.ChromeOptions()
    opts.add_argument("--headless=new")               # comment to watch
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument(f"--user-data-dir={PROFILE_DIR}")
    local_major = _detect_chrome_major()
    driver = uc.Chrome(options=opts, version_main=local_major)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="MacIntel",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True)
    return driver

# ─────────────────────────── HELPERS ────────────────────────────

def parse_years(text: str) -> Tuple[str, str]:
    m = DATE_RE.search(text.replace("\u00a0", " "))
    if not m:
        return "", ""
    born = m.group("born")
    died = m.group("died") or ""
    return born, died


def scrape_person(driver: uc.Chrome, bioguide_id: str) -> Tuple[str, str]:
    try:
        driver.get(SEARCH_URL)
        WebDriverWait(driver, WAIT_TIME).until(EC.presence_of_element_located(SEARCH_BOX))
        box = driver.find_element(*SEARCH_BOX)
        box.clear(); box.send_keys(bioguide_id + Keys.ENTER)

        # Bioguide sometimes re‑renders the results list; retry if element goes stale
        for _ in range(3):  # up to 3 attempts
            try:
                WebDriverWait(driver, WAIT_TIME).until(
                    EC.presence_of_element_located(RESULT_CARD)
                )
                card = driver.find_element(*RESULT_CARD)
                date_elem = card.find_element(*DATE_SPAN)
                return parse_years(date_elem.text)
            except StaleElementReferenceException:
                time.sleep(0.3)  # allow DOM to stabilise and retry
    except TimeoutException:
        logger.warning(f"{bioguide_id}: timeout")
    except WebDriverException as e:
        logger.warning(f"{bioguide_id}: webdriver error {e.__class__.__name__}")
    return "", ""

 # How often to checkpoint the CSV (every N processed rows)
CHECKPOINT_EVERY = 100

# ─────────────────────────── MAIN ────────────────────────────
ROOT = Path(__file__).resolve().parent
INPUT_CSV  = ROOT / "politicians_remaining.csv"
OUTPUT_CSV = ROOT / "politicians_with_dates.csv"


def main() -> None:
    if not INPUT_CSV.exists():
        logger.error(f"Input CSV {INPUT_CSV} not found"); sys.exit(1)

    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        try:
            guessed = csv.Sniffer().sniff(sample).delimiter
        except Exception:
            guessed = ","            # fallback to comma

    df = pd.read_csv(INPUT_CSV, dtype=str, sep=guessed)
    # normalise headers
    df.columns = [c.strip().lower() for c in df.columns]

    # try to find the bioguide column automatically
    if "bioguide_id" in df.columns:
        bioguide_col = "bioguide_id"
    else:
        # any column that contains the word 'bioguide'
        hits = [c for c in df.columns if "bioguide" in c]
        if not hits:
            logger.error("No 'bioguide_id' column (case‑insensitive) found in the input CSV.")
            logger.error(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)
        bioguide_col = hits[0]          # take the first match
        logger.warning(f"Using '{bioguide_col}' as bioguide_id column")

    df["born"], df["died"] = "", ""

    driver = launch_browser()
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Bioguide scrape"):
            born, died = scrape_person(driver, row[bioguide_col])
            df.at[idx, "born"], df.at[idx, "died"] = born, died
            time.sleep(0.4)
            # periodic checkpoint to avoid losing progress
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                logger.info(f"Checkpoint: saved first {idx + 1} rows → {OUTPUT_CSV}")
    finally:
        driver.quit()

    df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Saved {len(df)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
