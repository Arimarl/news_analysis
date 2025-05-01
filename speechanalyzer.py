from pathlib import Path
import os
import json
import time

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate
import tiktoken

# Configuration
# --------------------------------------------------------------------------------------
HERE                 = Path(__file__).resolve().parent
DATA_DIR             = HERE
PARAGRAPH_CSV        = DATA_DIR / "extracted_paragraphs.csv"
OUT_CSV              = DATA_DIR / "paragraph_results.csv"

MODEL_NAME           = "gpt-3.5-turbo"
TEMPERATURE          = 0
MAX_ALLOWED_TOKENS_CTX = 4_096       # keep within context window
RATE_LIMIT_SECONDS   = 1.2           # simple rate‑limit buffer

PROMPT_TEMPLATE = (
    "Read the paragraph. Decide if {candidate} supports {domain}, opposes it, or if the stance is unclear. "
    "Respond **exactly** as: 'YES || reason', 'NO || reason', or 'NA'. "
    "Use only the paragraph.\n\nParagraph:\n{paragraph}"
)

# Helpers
#--------------------------------------------------------------------------------------
def num_tokens(text: str, model: str = MODEL_NAME) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def build_columns() -> list[str]:
    domains = ["labor_unions", "immigration", "anti_elite_sentiment", "economic_redistribution", "us_vs_them_rhetoric", "free_trade", "xenophobia"]
    cols = []
    for domain in domains:
        cols += [f"{domain}_yn", f"{domain}_explanation"]
    cols += ["bioguide_id", "politician", "state",
             "source_pdf", "paragraph_id", "using_web", "num_tokens"]
    return cols


def parse_model_output(raw: str) -> list[str]:
    """
    Expect output exactly in the ‘|| … || …’ format requested by the prompt.
    Extract the pieces; fall back to 'unknown' if something is missing.
    """
    # remove leading / trailing whitespace and split on '||'
    pieces = [p.strip() for p in raw.strip().split("||")]
    # after split we expect ( '', q1_yn, q1_expl, q1_src, q2_yn, … , q6_src )
    needed = 1 + 6 * 3      # 19 pieces including the leading empty string
    if len(pieces) < needed:
        pieces.extend(["insufficient information"] * (needed - len(pieces)))
    return pieces[1:needed]     # drop leading empty slot


# Main classification loop
#--------------------------------------------------------------------------------------
def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing – check your .env")

    # load artefacts
    domains = ["labor_unions", "immigration", "anti_elite_sentiment", "economic_redistribution", "us_vs_them_rhetoric", "free_trade", "xenophobia"]
    df = pd.read_csv(PARAGRAPH_CSV)

    # safety check
    expected_cols = {"bioguide_id", "politician", "state", "source_pdf", "paragraph"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"paragraphs.csv must contain {expected_cols}")

    # init model
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

    # prepare results frame
    cols = build_columns()
    results = []

    # iterate
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Classifying paragraphs"):
        domain_results = []
        for domain in domains:
            prompt = PROMPT_TEMPLATE.format(
                domain=domain.replace('_', ' '),
                candidate=row["politician"],
                paragraph=row["paragraph"]
            )

            if num_tokens(prompt) > MAX_ALLOWED_TOKENS_CTX:
                available = MAX_ALLOWED_TOKENS_CTX - num_tokens(prompt.replace(row['paragraph'], ""))
                paragraph_cut = row["paragraph"][:available*4]
                prompt = prompt.replace(row["paragraph"], paragraph_cut)

            response = llm.invoke(prompt)
            raw = response.content.strip()
            if raw == "NA":
                yn, explanation = "NA", "NA"
            elif "||" in raw:
                yn, explanation = [p.strip() for p in raw.split("||", 1)]
            else:
                # Fallback: try first word as Y/N, rest as explanation
                parts = raw.split(None, 1)
                yn = parts[0].strip() if parts else "unknown"
                explanation = parts[1].strip() if len(parts) > 1 and yn.upper() != "NA" else "NA"
            yn = yn.lower()
            domain_results += [yn, explanation]
            time.sleep(RATE_LIMIT_SECONDS)

        meta = [
            row["bioguide_id"], row["politician"], row["state"],
            row["source_pdf"], idx, False,
            num_tokens(prompt) + num_tokens(response.content)
        ]
        results.append(domain_results + meta)

    # save
    out_df = pd.DataFrame(results, columns=cols)
    # normalise yn fields
    for domain in domains:
        out_df[f"{domain}_yn"] = out_df[f"{domain}_yn"].str.lower().str.strip()
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Finished.  Saved {out_df.shape[0]} rows to {OUT_CSV.relative_to(HERE)}")


if __name__ == "__main__":
    main()
