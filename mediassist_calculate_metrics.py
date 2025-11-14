import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
import argparse
import os
import json

from config import Config

try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')


def calculate_metrics(predictions_csv_path):

    print(f"Loading predictions from: {predictions_csv_path}")
    if not os.path.exists(predictions_csv_path):
        print(f"🚨 Error: Predictions file not found at '{predictions_csv_path}'!")
        return

    try:
        df = pd.read_csv(predictions_csv_path)
        if df.empty:
            print(f"🚨 Error: Predictions file '{predictions_csv_path}' is empty!")
            return
        if 'ground_truth' not in df.columns or 'generated' not in df.columns:
            print("🚨 Error: CSV must contain 'ground_truth' and 'generated' columns.")
            return
    except Exception as e:
        print(f"🚨 Error reading CSV: {e}")
        return

    # --- Preprocessing & Tokenization ---
    print("Preprocessing and tokenizing text...")
    references = []
    candidates = []

    df['ground_truth'] = df['ground_truth'].fillna('').astype(str)
    df['generated'] = df['generated'].fillna('').astype(str)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        ref_tokens = nltk.word_tokenize(row['ground_truth'].lower())
        references.append([ref_tokens])

        cand_tokens = nltk.word_tokenize(row['generated'].lower())
        candidates.append(cand_tokens)

    if not candidates or not references:
        print("🚨 Error: No valid reference/candidate pairs found after tokenization.")
        return

    # --- Calculate BLEU Scores ---
    print("\nCalculating BLEU scores...")
    chencherry = SmoothingFunction()
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    try:
        bleu1 = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1)
        bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
        bleu3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
        bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

        print(f"  BLEU-1: {bleu1:.4f}")
        print(f"  BLEU-2: {bleu2:.4f}")
        print(f"  BLEU-3: {bleu3:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
    except Exception as e:
        print(f"🚨 Error calculating BLEU: {e}")

    # --- Calculate ROUGE Scores ---
    print("\nCalculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    total_rouge1_f = 0
    total_rouge2_f = 0
    total_rougeL_f = 0
    valid_rouge_scores = 0
    avg_rouge1_f, avg_rouge2_f, avg_rougeL_f = 0, 0, 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="ROUGE Scoring"):
        ref_text = row['ground_truth']
        cand_text = row['generated']

        if not ref_text.strip() or not cand_text.strip():
            continue

        try:
            scores = scorer.score(ref_text, cand_text)
            total_rouge1_f += scores['rouge1'].fmeasure
            total_rouge2_f += scores['rouge2'].fmeasure
            total_rougeL_f += scores['rougeL'].fmeasure
            valid_rouge_scores += 1
        except Exception as e:
            print(f"🚨 Error calculating ROUGE for row {index}: {e}")

    if valid_rouge_scores > 0:
        avg_rouge1_f = total_rouge1_f / valid_rouge_scores
        avg_rouge2_f = total_rouge2_f / valid_rouge_scores
        avg_rougeL_f = total_rougeL_f / valid_rouge_scores

        print(f"  ROUGE-1 (F1): {avg_rouge1_f:.4f}")
        print(f"  ROUGE-2 (F1): {avg_rouge2_f:.4f}")
        print(f"  ROUGE-L (F1): {avg_rougeL_f:.4f}")
    else:
        print("  Could not calculate average ROUGE scores (no valid pairs found).")

    # Create metrics dictionary
    metrics = {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "ROUGE-1": avg_rouge1_f,
        "ROUGE-2": avg_rouge2_f,
        "ROUGE-L": avg_rougeL_f
    }
    
    # Save metrics to JSON file
    try:
        cfg = Config()
        metrics_path = cfg.METRICS_FILE

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Metrics saved to '{metrics_path}'")
    except Exception as e:
        print(f"🚨 Error saving metrics to JSON: {e}")
    # ---------------

# --- Main Execution Block ---
if __name__ == "__main__":
    cfg = Config() 
    
    parser = argparse.ArgumentParser(description="Calculate BLEU and ROUGE metrics from a predictions CSV file.")
    
    parser.add_argument("--predictions_csv", default=cfg.PREDICTIONS_CSV_PATH, help="Path to the CSV file containing 'ground_truth' and 'generated' columns.")

    args = parser.parse_args()

    calculate_metrics(args.predictions_csv)