from pathlib import Path
import os
import json
import time

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI          # pip install langchain‑openai
from langchain.prompts import PromptTemplate
import tiktoken

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
HERE          = Path(__file__).resolve().parent
DATA_DIR      = HERE
PARAGRAPH_CSV = DATA_DIR / "extracted_paragraphs.csv"
OUT_CSV       = DATA_DIR / "paragraph_results.csv"

MODEL_NAME              = "gpt-3.5-turbo"
TEMPERATURE             = 0
# Budget-conscious variant: using GPT-3.5-turbo ("4o mini")
MAX_ALLOWED_TOKENS_CTX  = 4_096          # sanity limit so we do not exceed context window
RATE_LIMIT_SECONDS      = 1.2            # naïve sleep to stay clear of OpenAI rate‑limits

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Main classification loop
# --------------------------------------------------------------------------------------
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
            prompt = (
                f"You are a political scientist. Using the news article paragraph given, determine the views on {domain.replace('_', ' ')} "
                f"of the candidate {row['politician']} mentioned in the text? "
                f"If the paragraph does not make the candidate’s opinion clear, respond only with 'NA' and nothing else. "
                f"Limit your answer to 300 tokens. Begin your answer with a simple yes or no to the question of whether the candidate was in support of this idea.\n\n"
                f"Paragraph: {row['paragraph']}"
            )

            if num_tokens(prompt) > MAX_ALLOWED_TOKENS_CTX:
                available = MAX_ALLOWED_TOKENS_CTX - num_tokens(prompt.replace(row['paragraph'], ""))
                paragraph_cut = row["paragraph"][:available*4]
                prompt = prompt.replace(row["paragraph"], paragraph_cut)

            response = llm.invoke(prompt)
            answer = response.content.strip().split("\n", 1)
            yn = answer[0].strip() if answer else "unknown"
            explanation = answer[1].strip() if len(answer) > 1 and answer[0].strip().upper() != "NA" else "NA"
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
