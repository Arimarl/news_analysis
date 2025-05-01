"""
REQUIREMENTS:
pandas numpy python-dotenv tqdm loguru
undetected-chromedriver selenium selenium-stealth
opencv-python-headless pdf2image pillow
pytesseract
rapidfuzz
torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
detectron2 (using git-lfs)
layoutparser[ocr]  (pulls in lp + its Detectron2 bridge)
poppler
"""

from __future__ import annotations
import time, re, sys
from pathlib import Path
from functools import wraps
from rapidfuzz.distance import Levenshtein as _levenshtein_distance
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from urllib.parse import quote_plus
from tqdm.auto import tqdm
from collections import Counter
from loguru import logger

# SELENIUM & BROWSER ────────────────────────────────────────────────────────
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, JavascriptException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import InvalidSessionIdException
from selenium_stealth import stealth

# OCR─────────────────────────────────────────────────────────────────────────
import cv2
from pdf2image import convert_from_path
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel


POLITY_WORDS = (
    "congress", "congressman", "congresswoman", "senate", "senator",
    "representative", "rep.", "legislature", "election", "campaign",
    "politician", "congressperson", "interview", "politics", "speech"
)

def contains_polity_keyword(text: str,
                            keywords: tuple[str, ...],
                            max_dist: int = 6) -> bool:
    """
    True iff *text* contains any word that matches one of *keywords*
    within *max_dist* Levenshtein edits (to cope with OCR noise).
    A fast exact‑substring test is tried first; failing that, each
    alphabetic token ≥4 chars is compared fuzzily.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in keywords):
        return True                     # exact hit – fast path
    tokens = re.findall(r"[a-z]{4,}", text_lower)
    for tok in tokens:
        for kw in keywords:
            # cheap length filter then fuzzy distance
            if abs(len(tok) - len(kw)) <= max_dist and \
               _levenshtein_distance(tok, kw) <= max_dist:
                return True
    return False

# ── DATE RANGE HELPER ────────────────────────────────────────────────────
def _year_range(born: float | int | str | None, died: float | int | str | None) -> str:
   #Format if search is YYYY-YYYY
    try:
        y1 = int(str(born)[:4])
        y2 = int(str(died)[:4]) - 1
        if y1 <= 0 or y2 < y1:
            return ""
        return f"{y1}-{y2}"
    except (TypeError, ValueError):
        return ""

# single‑year parser (first 4 consecutive digits)
def _year(val) -> int | None:
    try:
        y = int(re.findall(r"\d{4}", str(val))[0])
        return y if y > 0 else None
    except (IndexError, ValueError, TypeError):
        return None

# Helper: death year minus one (returns None if year missing)
def _death_year_minus1(val) -> int | None:
    y = _year(val)
    return y - 1 if y else None

NAME_RE_FLAGS = re.I | re.MULTILINE

#Detectron 2 label map ──────────────────────────────────────────────────────────

LABEL_MAP = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}

# ── OCR ENGINE PATH───────────────────────────────────────────────────────────
import shutil, pytesseract
_TESS_PATH = shutil.which("tesseract")
if _TESS_PATH:
    pytesseract.pytesseract.tesseract_cmd = _TESS_PATH
else:  # remove crash error inside pytesseract
    logger.error(
        "Tesseract not found"
    )

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Runtime switches
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# If STRICT_FILTER  False = NO CHECK ON 
STRICT_FILTER: bool = True
STATS: Counter = Counter()       # collect rejection reasons
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# CONFIG & PATHS ────────────────────────────────────────────────────────────
load_dotenv()
ROOT = Path(__file__).resolve().parent
MODEL_WEIGHTS = ROOT / "PubLayNet-faster_rcnn_R_50_FPN_3x" / "model_final.pth"
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
RAW_PDF  = DATA / "raw_pdf" ; RAW_PDF.mkdir(exist_ok=True)
OCR_TXT   = DATA / "ocr_txt"  ; OCR_TXT.mkdir(exist_ok=True)
POPPLER_PATH = "/opt/homebrew/bin"
LOGS      = DATA / "logs"     ; LOGS.mkdir(exist_ok=True)

CSV_POLITICIANS = ROOT / "sampled_politicians_3.csv"  # provided by user
OUT_PARAGRAPHS  = DATA / "extracted_paragraphs.csv"

logger.add(LOGS / "pipeline_{time}.log")

# ── BROWSER ────────────────────────────────────────────────────────────────────
PROFILE_DIR = Path("/tmp/chrome-newspapers-profile")  # persistent login profile

# -----------------------------------------------------------------------
# Browser version helper - in case of Chrome version issue
# -----------------------------------------------------------------------
import platform, subprocess, shutil

def _detect_chrome_major() -> int | None:
    """
    Return the locally installed Chrome major version (e.g. 135) or None.
    Works on macOS, Linux, Windows; relies on chrome executable being on disk.
    """
    candidates: list[str] = []
    sys_plat = platform.system()
    if sys_plat == "Darwin":
        candidates = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    elif sys_plat == "Linux":
        candidates = [shutil.which(cmd) for cmd in ("google-chrome", "chromium", "chromium-browser")]
    elif sys_plat == "Windows":
        candidates = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                      r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    for exe in filter(None, candidates):
        try:
            out = subprocess.check_output([exe, "--version"], stderr=subprocess.DEVNULL).decode()
            m = re.search(r"(\d+)\.", out)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None

def with_wait(fn):
    """Decorator to catch TimeoutException and log a warning instead of blowing up."""
    @wraps(fn)
    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TimeoutException:
            logger.warning(f"Timeout in {fn.__name__}")
            return None
    return _inner


def launch_browser() -> uc.Chrome:
    opts = uc.ChromeOptions()
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-print-preview")   # avoid DevTools
    opts.add_argument(f"--user-data-dir={PROFILE_DIR}")

    # downloads
    prefs = {
        "download.default_directory": str(RAW_PDF),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,  #CHROME MUST DOWNLOAD PDF
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)

    local_major = _detect_chrome_major()
    driver = uc.Chrome(options=opts, version_main=local_major)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="MacIntel",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
    )
    logger.info("Chrome launched with persistent profile. (need newspapercom login)")
    return driver

# ── OCR SETUP ────────────────────────────────────────────────────────────────
ocr_agent = lp.TesseractAgent(languages="eng")

# ── Detectron2 layout model (loaded once) ────────────────────────────
DETECTOR = Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path=str(MODEL_WEIGHTS),     # ← use local copy, skip re‑download
    label_map=LABEL_MAP,
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
)

def extract_paragraph_from_pdf(
    pdf_path: Path,
    full_name: str,
    keywords: tuple[str, ...] = POLITY_WORDS,
    strict: bool = STRICT_FILTER,
) -> str:
    """
    Return the paragraph that (a) contains the politician’s name and
    (b) at least one political keyword; otherwise return ''.
    """
    try:
        page = convert_from_path(
            pdf_path,
            dpi=300, first_page=1, last_page=1,
            poppler_path=POPPLER_PATH,
        )[0]                        # PIL.Image
    except Exception as e:
        logger.error(f"pdf2image failed on {pdf_path.name}: {e}")
        return ""

    img_rgb = np.array(page)[:, :, ::-1]          # PIL RGB → BGR for cv2
    layout = DETECTOR.detect(img_rgb)

    # if OCR blanket fallback ever needed
    def _ocr_full_page() -> str:
        return ocr_agent.detect(img_rgb)

    surname = full_name.split()[-1]
    name_pat = re.compile(rf"\b{re.escape(surname)}\b", NAME_RE_FLAGS)
    kw_pat   = re.compile("|".join(map(re.escape, keywords)), NAME_RE_FLAGS)

    hit_blocks, hit_texts = [], []
    for blk in layout:
        txt = ocr_agent.detect(blk.crop_image(img_rgb))
        if name_pat.search(txt):
            hit_blocks.append(blk)
            hit_texts.append(txt)

    if not hit_blocks:
        STATS["no_name"] += 1
        return "" if strict else _ocr_full_page()

    # Fusion OCR blocks that are in the same paragraph ───────────────────────────
    try:
        para = lp.Layout(hit_blocks).union()
    except AttributeError:
        # Manually done if old version of layoutparser (I still encounter some errors sometimes)
        xs = [b.block.x_1 for b in hit_blocks] + [b.block.x_2 for b in hit_blocks]
        ys = [b.block.y_1 for b in hit_blocks] + [b.block.y_2 for b in hit_blocks]
        merged = lp.TextBlock(block=lp.Rectangle(min(xs), min(ys), max(xs), max(ys)))
        para = [merged]

    full_txt = ocr_agent.detect(para[0].crop_image(img_rgb))
    full_txt = re.sub(r"-\n", "", full_txt)        # de-hyphenate
    full_txt = re.sub(r"\s+\n?", " ", full_txt).strip()

    # Accept paragraph if an exact OR fuzzy keyword match is found
    if kw_pat.search(full_txt) or contains_polity_keyword(full_txt, keywords):
        return full_txt                 # OK – keep the paragraph
    STATS["no_keyword"] += 1
    return "" if strict else full_txt


#SCRAPING HELPERS ───────────────────────────────────────────────────────────────────────

@with_wait
def wait_file(tmpdir: Path, suffix: str, timeout: float = 90) -> Path:
#wait file writing
    end = time.time() + timeout
    last_size = -1
    latest: Path | None = None

    while time.time() < end:
        # newest finished-download candidate
        done = [p for p in tmpdir.glob(f"*{suffix}") if not p.with_suffix(p.suffix + ".crdownload").exists()]
        if done:
            latest = max(done, key=lambda p: p.stat().st_mtime)
            size = latest.stat().st_size
            if size and size == last_size:            # Size check (to remove)
                return latest
            last_size = size
        time.sleep(0.5)

    raise TimeoutException("Download never completed (wait_file)")

# --- core download snippet (adapted from click_and_save.txt) ---
@with_wait
def download_current_page_pdf(driver: uc.Chrome) -> Path | None:
    """
    Minimal, Newspapers.com‑specific logic:

        1. Click the “Print / Download” top‑bar button (matched on visible text).
        2. In the ensuing dialog click the hidden <a><div id="entireP">…</div></a>.
        3. Click the “Save as PDF” button.
        4. Wait for a .pdf to appear in RAW_PDF and return its path.
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
        # No side‑pane visible = OK nothing to do
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
        # Dialog not open yet = fall back to normal flow below.
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
            logger.warning("No Print/Download button")
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
                    "'Entire/Full Page' control not found"
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
            logger.warning("'Save as PDF' not found")
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
    visited_urls: set[str] = set()
    for row in tqdm(df_targets.itertuples(), total=len(df_targets)):
        got_paragraph = False      # stop after FIRST successful OCR for this politician
        try:
            base = "https://www.newspapers.com/search/results/?"
            parts = [f"keyword={quote_plus(row.q)}"]
            if isinstance(row.state_to_query, str) and row.state_to_query.strip():
                parts.append(f"state={row.state_to_query}")
            if row.date_start and row.date_end:
                parts.append(f"date-start={row.date_start}")
                parts.append(f"date-end={row.date_end}")
            search_url = base + "&".join(parts)
            driver.get(search_url)
            logger.debug(f"Search URL: {search_url}")
            # give React a moment to render the first result thumbnails before we start waiting
            time.sleep(1.2)

            #wait for a real clickable result link ────────────────────────
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
                # the helper may have swallowed a TimeoutException = skip
                logger.warning(f"No clickable results for {row.q}")
                continue

            # ----- collect result links first to avoid stale‑element issues -----
            elements = [first_link] + [el for el in driver.find_elements(By.CSS_SELECTOR, LINK_SEL)[1:5] if el]
            urls = [el.get_attribute("href") for el in elements if el.get_attribute("href")]

            if not urls:
                logger.warning(f"Zero URLs for {row.q} – skipping")
                continue

            attempts = 0            # reset count of attemps
            # iterate over the captured URLs
            for url in urls:
                if attempts >= 3:                       # 3 tries max per person
                    logger.info("Reached 3 attempts = NEXT person")
                    break
                attempts += 1
                if url in visited_urls:
                    continue
                visited_urls.add(url)
                driver.get(url)
                driver.execute_script("window.scrollBy(0, 1)")  #poke lazy loader

                # refresh if pay‑wall / upsell overlay injects an iframe
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
                    logger.warning("Article viewer did not load = Skipping card")
                    driver.back()
                    WebDriverWait(
                        driver,
                        10,
                        ignored_exceptions=(StaleElementReferenceException,)
                    ).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL)))
                    continue

                pdf_path = download_current_page_pdf(driver)
                if pdf_path and pdf_path.name in {r["source_pdf"] for r in records}:
                    logger.info(f"Skipping duplicate PDF {pdf_path.name}")
                    driver.back()
                    continue
                if not pdf_path:
                    driver.back()
                    WebDriverWait(
                        driver,
                        10,
                        ignored_exceptions=(StaleElementReferenceException,)
                    ).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, LINK_SEL)))
                    continue

                paragraph = extract_paragraph_from_pdf(
                    pdf_path,
                    row.name_to_query,        # full name
                    POLITY_WORDS,
                    STRICT_FILTER,
                )
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
                break   # exit URL loop – outer loop will move to next person
        except InvalidSessionIdException:
            logger.warning("Chrome session lost – relaunching")
            driver = launch_browser()
            continue

    if STATS:
        logger.info(f"Rejection stats: {dict(STATS)}")
    if records:
        pd.DataFrame(records).to_csv(OUT_PARAGRAPHS, index=False)
        logger.success(f"Saved {len(records)} paragraphs -> {OUT_PARAGRAPHS}")
    else:
        logger.warning("No records extracted")

# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not CSV_POLITICIANS.exists():
        logger.error(f"{CSV_POLITICIANS} not found")
        sys.exit(1)

    df_targets = (
        pd.read_csv(CSV_POLITICIANS)
          .assign(
              q=lambda d: '"' + d.name_to_query + '"',
              date_start=lambda d: d.born.apply(_year),
              date_end=lambda d: d.died.apply(_death_year_minus1)
          )
    )
    driver = launch_browser()
    try:
        scrape_and_ocr(driver, df_targets)
    finally:
        driver.quit()

