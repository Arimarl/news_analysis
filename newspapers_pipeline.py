"""
Section‑1 output  ➜  data/extracted_paragraphs.csv
Section‑2 output  ➜  data/gpt_results.csv

Designed for **macOS + Chrome**.  Requires:  
    python 3.10+,   chromedriver 120+,   Chrome installed.

A minimal `.env` only needs (for the optional GPT stage):
    OPENAI_API_KEY="sk-..."  
"""


from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTE: pdf2image uses *poppler*.  On macOS install with:
#   brew install poppler
# On Linux:
#   sudo apt-get install poppler-utils
# ---------------------------------------------------------------------------

import os, time, re, tempfile, sys
from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm
from loguru import logger

# ── selenium & browser ────────────────────────────────────────────────────────
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, JavascriptException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import InvalidSessionIdException
from selenium_stealth import stealth

# ── OCR ────────────────────────────────────────────────────────────────────────
import cv2
from pdf2image import convert_from_path
import layoutparser as lp

# ── GPT  (optional) ───────────────────────────────────────────────────────────
import openai

# ── CONFIG & PATHS ────────────────────────────────────────────────────────────
load_dotenv()
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
RAW_PDF  = DATA / "raw_pdf" ; RAW_PDF.mkdir(exist_ok=True)
OCR_TXT   = DATA / "ocr_txt"  ; OCR_TXT.mkdir(exist_ok=True)
LOGS      = DATA / "logs"     ; LOGS.mkdir(exist_ok=True)

CSV_POLITICIANS = ROOT / "sampled_politicians.csv"  # provided by user
OUT_PARAGRAPHS  = DATA / "extracted_paragraphs.csv"
OUT_GPT         = DATA / "gpt_results.csv"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
logger.add(LOGS / "pipeline_{time}.log")

# ── BROWSER ────────────────────────────────────────────────────────────────────
PROFILE_DIR = Path("/tmp/chrome-newspapers-profile")  # persistent login profile


def with_wait(fn):
    """Decorator to catch TimeoutException and log a warning instead of blowing up."""
    @wraps(fn)
    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TimeoutException:
            logger.warning(f"Timeout inside {fn.__name__}")
            return None
    return _inner


def launch_browser() -> uc.Chrome:
    opts = uc.ChromeOptions()
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-print-preview")   # avoid DevTools disconnect on print dialog
    opts.add_argument(f"--user-data-dir={PROFILE_DIR}")

    # downloads
    prefs = {
        "download.default_directory": str(RAW_PDF),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,  # force Chrome to download PDFs
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)

    driver = uc.Chrome(options=opts, version_main=None)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="MacIntel",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
    )
    logger.info("Chrome launched with persistent profile – assuming already logged in.")
    return driver

# ── OCR SETUP ────────────────────────────────────────────────────────────────
LOW_Y = np.array([20, 50, 50])   # HSV range for Newspapers.com yellow highlight
HIGH_Y = np.array([40, 255, 255])
ocr_agent = lp.TesseractAgent(languages="eng")


def extract_paragraph_from_pdf(pdf_path: Path) -> str:
    """Detect yellow highlight bounding‑box and OCR around it; fallback to whole page."""
    try:
        img = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)[0]
    except Exception as e:
        logger.error(f"pdf2image failed on {pdf_path}: {e}")
        return ""

    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOW_Y, HIGH_Y)
    if mask.sum() < 1000:
        crop = img
    else:
        x, y, w, h = cv2.boundingRect(mask)
        pad = 40
        crop = img.crop((max(x-pad,0), max(y-pad,0), x+w+pad, y+h+pad))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        crop.save(tmp.name)
        text = ocr_agent.detect(tmp.name)
    clean = re.sub(r"-\n", "", text)
    clean = re.sub(r"\n+", " ", clean).strip()
    return clean

# ── SCRAPING HELPERS ─────────────────────────────────────────────────────────

@with_wait
def wait_file(tmpdir: Path, suffix: str, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        files = list(tmpdir.glob(f"*{suffix}"))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        time.sleep(0.5)
    raise TimeoutException("Download timed‑out")

# --- core download snippet (adapted from click_and_save.txt) ---
@with_wait
def download_current_page_pdf(driver: uc.Chrome) -> Path | None:
    """
    Minimal, Newspapers.com‑specific logic:

        1. Click the “Print / Download” top‑bar button (matched on visible text).
        2. In the ensuing dialog click the hidden <a><div id="entireP">…</div></a>.
        3. Click the “Save as PDF*” button.
        4. Wait for a *.pdf to appear in RAW_PDF and return its path.

    Returns None if anything fails.
    """
    # ── Pre‑step: close details side‑pane if it’s covering the viewer ─────────
    try:
        # Newspapers.com often shows an obituary / clipping facts pane on the
        # right which blocks the “Entire Page” control.  Close it proactively.
        close_btn = driver.find_element(
            By.CSS_SELECTOR,
            "button[aria-label='Close'][role='button'],"
            "button[aria-label='close'],"
            "button[data-testid='close-button']"
        )
        if close_btn.is_displayed():
            driver.execute_script("arguments[0].click();", close_btn)
            time.sleep(0.4)  # brief pause for the pane to animate away
    except Exception:
        # No side‑pane visible → nothing to do
        pass
    # ── Fast‑path: we might already be in the print dialog ────────────────
    try:
        fast_pdf_btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Save as PDF*']")
            )
        )
        driver.execute_script("arguments[0].click();", fast_pdf_btn)
        pdf = wait_file(RAW_PDF, ".pdf", timeout=90)
        if pdf:
            logger.info(f"Downloaded {pdf.name} (fast‑path)")
            return pdf
    except TimeoutException:
        # Dialog not open yet → fall back to the normal flow below.
        pass
    try:
        # ── Part 1: Top‑bar “Print / Download” button ────────────────────────
        buttons = driver.find_elements(By.TAG_NAME, "button")
        target = next(
            (b for b in buttons
             if "print" in (b.get_attribute("textContent") or "").lower()),
            None
        )
        if not target:
            logger.warning("No visible Print/Download button")
            return None
        driver.execute_script("arguments[0].click();", target)

        # ── Quick‑check: does this sidebar already offer the “Save as PDF*” button? ──
        try:
            pdf_sidebar_btn = WebDriverWait(driver, 6).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[normalize-space()='Save as PDF*']")
                )
            )
            driver.execute_script("arguments[0].click();", pdf_sidebar_btn)
            pdf = wait_file(RAW_PDF, ".pdf", timeout=90)
            if pdf:
                logger.info(f"Downloaded {pdf.name} (sidebar flow)")
                return pdf
        except TimeoutException:
            # Fallback to legacy ‘Entire Page’ flow below.
            pass

        # ── Part 2: choose “Entire Page” (or fallback labels) ────────────────
        # Newspapers.com keeps renaming this control (e.g. “Full Page” in
        # some A/B variants / locales).  First try the historic id="entireP".
        try:
            wait = WebDriverWait(driver, 25)
            print_button = wait.until(
                EC.presence_of_element_located((By.ID, "entireP"))
            )
        except TimeoutException:
            # Fallback: locate by visible text regardless of exact label
            try:
                print_button = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//*[self::div or self::button][contains("
                            "translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                            "'abcdefghijklmnopqrstuvwxyz'),"
                            "'entire page') or "
                            "contains("
                            "translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                            "'abcdefghijklmnopqrstuvwxyz'),'full page')]"
                        )
                    )
                )
            except TimeoutException:
                logger.warning(
                    "'Entire/Full Page' control not found – print dialog failed to load"
                )
                return None

        # The actual clickable element is its ancestor <a>.
        print_link = print_button.find_element(By.XPATH, "./ancestor::a[1]")
        driver.execute_script("arguments[0].click();", print_link)

        # ── Part 3: click “Save as PDF*” in the preview ──────────────────────
        try:
            pdf_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[normalize-space()='Save as PDF*']")
                )
            )
        except TimeoutException:
            logger.warning("'Save as PDF' button not found – preview failed to load")
            return None

        driver.execute_script("arguments[0].click();", pdf_button)

        # ── Part 4: wait for the file to land in RAW_PDF dir ─────────────────
        pdf = wait_file(RAW_PDF, ".pdf", timeout=90)
        if pdf:
            logger.info(f"Downloaded {pdf.name}")
        return pdf
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return None


@with_wait
def scrape_and_ocr(driver: uc.Chrome, df_targets: pd.DataFrame):
    records: list[dict] = []
    for row in tqdm(df_targets.itertuples(), total=len(df_targets)):
        try:
            search_url = (
                "https://www.newspapers.com/search/results/?query="
                f"{row.q}&state={row.state_to_query}"
            )
            driver.get(search_url)
            # give React a moment to render the first result thumbnails before we start waiting
            time.sleep(1.2)

            # ── wait for *a real* clickable result link ────────────────────────
            LINK_SEL = (
                "a[href*='/clip/'],"
                "a[href*='/newspage/'],"
                "a[href*='/image/'],"
                "a[data-testid='result-thumbnail']"
            )
            try:
                first_link = WebDriverWait(driver, 15, poll_frequency=0.3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL))
                )
            except TimeoutException:
                logger.warning(f"No results for {row.q}")
                continue

            if first_link is None:
                # the helper may have swallowed a TimeoutException – skip safely
                logger.warning(f"No clickable results for {row.q}")
                continue

            # ----- collect result links first to avoid stale‑element issues -----
            elements = [first_link] + [el for el in driver.find_elements(By.CSS_SELECTOR, LINK_SEL)[1:5] if el]
            urls = [el.get_attribute("href") for el in elements if el.get_attribute("href")]

            if not urls:
                logger.warning(f"Collected zero URLs for {row.q} – skipping target")
                continue

            # iterate over the captured URLs
            for url in urls:
                driver.get(url)
                driver.execute_script("window.scrollBy(0, 1)")  # poke lazy loader

                # optional: refresh if pay‑wall / upsell overlay injects an iframe
                if driver.find_elements(By.CSS_SELECTOR, "iframe[src*='offer']"):
                    logger.info("offer overlay → refresh")
                    driver.refresh()

                try:
                    # ignore DOM mutations that detach the element while we wait
                    WebDriverWait(
                        driver,
                        15,
                        ignored_exceptions=(StaleElementReferenceException,)
                    ).until(
                        EC.element_to_be_clickable(
                            (By.CSS_SELECTOR,
                             "button#btn-print, button[aria-label*='Print']"))
                    )
                except TimeoutException:
                    logger.warning("Article viewer did not load – skipping card")
                    driver.back()
                    WebDriverWait(
                        driver,
                        10,
                        ignored_exceptions=(StaleElementReferenceException,)
                    ).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL)))
                    continue

                pdf_path = download_current_page_pdf(driver)
                if not pdf_path:
                    driver.back()
                    WebDriverWait(
                        driver,
                        10,
                        ignored_exceptions=(StaleElementReferenceException,)
                    ).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL)))
                    continue

                paragraph = extract_paragraph_from_pdf(pdf_path)
                if not paragraph:
                    logger.warning(f"Empty OCR for {pdf_path.name}")
                    driver.back()
                    WebDriverWait(
                        driver,
                        10,
                        ignored_exceptions=(StaleElementReferenceException,)
                    ).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL)))
                    continue

                records.append({
                    "bioguide_id": row.bioguide_id,
                    "politician": row.name_to_query,
                    "state": row.state_to_query,
                    "source_pdf": pdf_path.name,
                    "paragraph": paragraph,
                })

                driver.back()  # back to search results
                WebDriverWait(
                    driver,
                    10,
                    ignored_exceptions=(StaleElementReferenceException,)
                ).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL)))
                time.sleep(np.random.uniform(0.8, 1.6))
        except InvalidSessionIdException:
            logger.warning("Chrome session lost – relaunching...")
            driver = launch_browser()
            continue

    if records:
        pd.DataFrame(records).to_csv(OUT_PARAGRAPHS, index=False)
        logger.success(f"✍️  Saved {len(records)} paragraphs → {OUT_PARAGRAPHS}")
    else:
        logger.warning("No records extracted!")

# ── GPT CLASSIFICATION (optional) ────────────────────────────────────────────

def run_gpt():
    if not OPENAI_KEY:
        logger.error("OPENAI_API_KEY not set – skipping GPT stage")
        return
    if not OUT_PARAGRAPHS.exists():
        logger.error("Paragraph file missing – run scraper first")
        return

    openai.api_key = OPENAI_KEY
    df = pd.read_csv(OUT_PARAGRAPHS)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = (
            "You are a historian. Read the extract below (within <<<>>>).  "
            "In ≤40 words summarise it, then answer yes/no: Does the author display "
            "populist rhetoric? Explain in ≤30 words.\n<<<" + row.paragraph + ">>>"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini", temperature=0,
                messages=[{"role":"user","content":prompt}]
            )
            content = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT error {e}")
            content = "error"
        results.append({**row.to_dict(), "gpt_response": content})

    pd.DataFrame(results).to_csv(OUT_GPT, index=False)
    logger.success(f"GPT results saved → {OUT_GPT}")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Newspapers.com pipeline")
    parser.add_argument("stage", choices=["scrape","gpt"], help="Which stage to run")
    args = parser.parse_args()

    if args.stage == "scrape":
        if not CSV_POLITICIANS.exists():
            logger.error(f"{CSV_POLITICIANS} not found – aborting")
            sys.exit(1)
        df_targets = pd.read_csv(CSV_POLITICIANS).assign(q=lambda d: '"'+d.name_to_query+'"')
        driver = launch_browser()
        try:
            scrape_and_ocr(driver, df_targets)
        finally:
            driver.quit()
    elif args.stage == "gpt":
        run_gpt()
