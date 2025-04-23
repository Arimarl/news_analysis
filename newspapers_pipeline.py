"""
End‑to‑end pipeline for the
    * Newspapers.com  scraping (Chrome on macOS)
    * OCR paragraph extraction with LayoutParser / Tesseract
    * (optional) GPT‑4 classification step

Section‑1 output  ➜  data/extracted_paragraphs.csv
Section‑2 output  ➜  data/gpt_results.csv

Designed for **macOS + Chrome**.  Requires:  
    python 3.10+,   chromedriver 120+,   Chrome installed.

Dependencies (install once):
    pip install selenium==4.20.0 undetected-chromedriver selenium-stealth \
                layoutparser[tesseract] pdf2image pillow opencv-python \
                pandas numpy loguru tqdm python-dotenv openai
Plus system packages:  
    brew install tesseract  poppler  # (poppler provides `pdftoppm` for pdf2image)

Create `.env` in project root with:
    NP_USER="your_email"        # Newspapers.com login
    NP_PASS="your_password"
    OPENAI_API_KEY="sk-..."     # for section‑2
"""

import os, time, json, pickle, shutil, hashlib, re, tempfile
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from functools import wraps

import pandas as pd
from tqdm.auto import tqdm

# ---- selenium & browser ----
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium_stealth import stealth

# Helper to robustly close the OneTrust/Ancestry cookie banner on newspapers.com

# ---- OCR ----
import cv2, numpy as np
from pdf2image import convert_from_path
import layoutparser as lp

# ---- GPT ----
import openai

# -----------------------------------------------------------------------------
#  CONFIG & UTILS
# -----------------------------------------------------------------------------

load_dotenv()
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
RAW_PDF  = DATA / "raw_pdf" ; RAW_PDF.mkdir(exist_ok=True)
OCR_TXT   = DATA / "ocr_txt"  ; OCR_TXT.mkdir(exist_ok=True)
LOGS      = DATA / "logs"     ; LOGS.mkdir(exist_ok=True)

CSV_POLITICIANS = ROOT / "sampled_politicians.csv"  # provided by user
OUT_PARAGRAPHS  = DATA / "extracted_paragraphs.csv"
OUT_GPT         = DATA / "gpt_results.csv"

NP_USER = os.getenv("NP_USER")
NP_PASS = os.getenv("NP_PASS")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

assert NP_USER and NP_PASS, "Newspapers.com credentials must be in .env"

logger.add(LOGS / "pipeline_{time}.log")

# -----------------------------------------------------------------------------
#  BROWSER  (one instance reused for entire run)
# -----------------------------------------------------------------------------

def launch_browser() -> uc.Chrome:
    opts = uc.ChromeOptions()
    opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    opts.add_argument("--user-data-dir=/tmp/chrome-persist")
    # downloads
    prefs = {
        "download.default_directory": str(RAW_PDF),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
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

    driver.get("https://www.newspapers.com/")
    time.sleep(2)
    try:
        login(driver)
    except Exception as e:
        logger.error(f"Login failed: {e}")
        input("Manually log in, then press ENTER to continue...")

    return driver


def login(driver):
    """Perform interactive login once."""
    logger.info("Logging in to Newspapers.com…")
    # use an explicit wait throughout the login process
    wait = WebDriverWait(driver, 20)

    # the sign‑in form now lives inside a dynamically inserted iframe
    iframe = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "iframe[src*='identity']"))
    )
    driver.switch_to.frame(iframe)
    driver.find_element(By.XPATH, "//a[contains(@href,'/signin')]").click()
    WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.ID,"email")))
    driver.find_element(By.ID,"email").send_keys(NP_USER)
    driver.find_element(By.ID,"password").send_keys(NP_PASS)
    submit_btn = wait.until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR,
             "button[title='Sign in with Newspapers.com']"))
    )
    submit_btn.click()

    # back to the main DOM for the rest of the pipeline
    driver.switch_to.default_content()
    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,"//a[contains(@href,'/account')]")))
    logger.success("Login successful")

# -----------------------------------------------------------------------------
#  SCRAPING HELPERS
# -----------------------------------------------------------------------------

def human_wait(min_sec=1.2, max_sec=2.8):
    time.sleep(np.random.uniform(min_sec, max_sec))





def wait_file(tmpdir: Path, suffix: str, timeout=30):
    """Wait until a new file with given suffix appears in tmpdir."""
    start = time.time()
    while time.time() - start < timeout:
        pdfs = list(tmpdir.glob(f"*{suffix}"))
        if pdfs:
            return max(pdfs, key=lambda p: p.stat().st_mtime)
        time.sleep(0.5)
    raise TimeoutException("Download timed‑out")

# bounding yellow highlight detection constants (HSV range)
LOW_Y = np.array([20, 50, 50])
HIGH_Y = np.array([40, 255, 255])

ocr_agent = lp.TesseractAgent(languages="eng")


def extract_paragraph_from_pdf(pdf_path: Path) -> str:
    # convert first page to image
    img = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)[0]
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOW_Y, HIGH_Y)
    if mask.sum() < 1000:  # if highlight not found, fallback to whole image
        crop = img
    else:
        x, y, w, h = cv2.boundingRect(mask)
        pad = 40
        crop = img.crop((max(x-pad,0), max(y-pad,0), x+w+pad, y+h+pad))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        crop.save(tmp.name)
        text = ocr_agent.detect(tmp.name)
    clean = re.sub(r"-\n", "", text)  # fix hyphen linebreaks
    clean = re.sub(r"\n+", " ", clean).strip()
    return clean

# -----------------------------------------------------------------------------
#  MAIN SCRAPER ROUTINE
# -----------------------------------------------------------------------------

def scrape_and_ocr(driver, df_targets: pd.DataFrame):
    records = []
    for row in tqdm(df_targets.itertuples(), total=len(df_targets)):
        search_url = f"https://www.newspapers.com/search/results/?query={row.q}&state={row.state_to_query}"
        driver.get(search_url)
        human_wait()
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "a.searchResults__result")
        except NoSuchElementException:
            logger.warning(f"No results for {row.q}")
            continue
        for card in cards[:5]:  # cap per politician to avoid explosion
            driver.execute_script("arguments[0].click();", card)
            WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.ID,"viewerContainer")))
            # open print / download menu
            try:
                # >>> Print / Download logic START
                # Step‑1: open the Print / Download dialog
                buttons = driver.find_elements(By.TAG_NAME, "button")
                target = next(
                    (b for b in buttons
                     if "print" in (b.get_attribute("textContent") or "").lower()),
                    None
                )
                if not target:
                    raise RuntimeError("No button whose textContent contains 'Print'.")
                driver.execute_script("arguments[0].click();", target)

                # Step‑2: click the hidden “Entire Page” radio option inside the dialog
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "entireP"))
                )
                print_button = driver.find_element(By.ID, "entireP")
                print_link = print_button.find_element(By.XPATH, "./ancestor::a")
                driver.execute_script("arguments[0].click();", print_link)

                # Step‑3: trigger the “Save as PDF*” download
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[normalize-space()='Save as PDF*']")
                    )
                ).click()
                # >>> Print / Download logic END
                pdf_file = wait_file(RAW_PDF, ".pdf", timeout=40)
            except TimeoutException:
                logger.error("Download failed – skipping")
                driver.back(); continue
            paragraph = extract_paragraph_from_pdf(pdf_file)
            if not paragraph:
                driver.back(); continue
            rec = {
                "bioguide_id": row.bioguide_id,
                "politician": row.name_to_query,
                "state": row.state_to_query,
                "source_pdf": pdf_file.name,
                "paragraph": paragraph,
            }
            records.append(rec)
            driver.back()
            human_wait(0.8,1.6)
    if records:
        pd.DataFrame(records).to_csv(OUT_PARAGRAPHS, index=False)
        logger.success(f"✍️  Saved {len(records)} paragraphs → {OUT_PARAGRAPHS}")
    else:
        logger.warning("No records extracted!")

# -----------------------------------------------------------------------------
#  GPT CLASSIFICATION  (section‑2 – optional)
# -----------------------------------------------------------------------------

if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def run_gpt():
    if not OPENAI_KEY:
        logger.error("OPENAI_API_KEY not set – skipping GPT stage")
        return
    if not OUT_PARAGRAPHS.exists():
        logger.error("Paragraph file missing – run scraper first")
        return
    df = pd.read_csv(OUT_PARAGRAPHS)
    results = []
    for idx,row in tqdm(df.iterrows(), total=len(df)):
        prompt = f"""You are a historian. Read the extract below (within <<<>>>).  In ≤40 words summarise it, then answer yes/no: Does the author display populist rhetoric? Explain in ≤30 words.\n<<<{row.paragraph}>>>"""
        try:
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", temperature=0, messages=[{"role":"user","content":prompt}])
            content = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT error {e}")
            content = "error"
        results.append({**row.to_dict(), "gpt_response": content})
    pd.DataFrame(results).to_csv(OUT_GPT, index=False)
    logger.success(f"GPT results saved → {OUT_GPT}")

# -----------------------------------------------------------------------------
#  CLI ENTRYPOINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, numpy as np
    parser = argparse.ArgumentParser(description="Newspapers.com pipeline")
    parser.add_argument("stage", choices=["scrape","gpt"], help="Which stage to run")
    args = parser.parse_args()

    if args.stage == "scrape":
        df_targets = pd.read_csv(CSV_POLITICIANS).assign(q=lambda d: '"'+d.name_to_query+'"')
        driver = launch_browser()
        try:
            scrape_and_ocr(driver, df_targets)
        finally:
            driver.quit()
    elif args.stage == "gpt":
        run_gpt()