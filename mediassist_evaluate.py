import torch
import pandas as pd
from transformers import AutoTokenizer
from PIL import Image
import argparse
import os
from tqdm import tqdm

try:
    from mediassist_dataset import image_transform
    from mediassist_model import TienetReportGenerator
    from config import Config
except ImportError as e:
    print(f"🚨 Error importing modules: {e}")
    exit()

def run_evaluation(cfg, test_csv_path, model_path, output_csv_path, device, batch_size=1):

    # 1. Load Tokenizer
    print(f"Loading tokenizer: {cfg.TOKENIZER_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        VOCAB_SIZE = tokenizer.vocab_size
        if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
            START_TOKEN_ID = tokenizer.cls_token_id
        elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            START_TOKEN_ID = tokenizer.bos_token_id
        else:
            START_TOKEN_ID = tokenizer.pad_token_id
        
        if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
            END_TOKEN_ID = tokenizer.sep_token_id
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            END_TOKEN_ID = tokenizer.eos_token_id
        else:
            END_TOKEN_ID = tokenizer.pad_token_id
    except Exception as e:
        print(f"🚨 Error loading tokenizer: {e}")
        return

    # 2. Load Model Architecture
    print("Initializing model architecture...")
    try:
        model = TienetReportGenerator(
            vocab_size=VOCAB_SIZE,
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEAD,
            num_decoder_layers=cfg.NUM_DECODER_LAYERS,
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
            freeze_epochs=0  # Not training, so no freezing needed
        ).to(device)
    except Exception as e:
        print(f"🚨 Error initializing model architecture: {e}")
        return

    # 3. Load Trained Weights
    print(f"Loading trained weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"🚨 Error: Model weights file not found at '{model_path}'!")
        return
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Trained weights loaded.")
    except Exception as e:
        print(f"🚨 Error loading model weights: {e}")
        return

    # 4. Load Test Data Manifest
    print(f"Loading test data manifest from: {test_csv_path}")
    print(f"Using test CSV file: {test_csv_path}")
    try:
        test_df = pd.read_csv(test_csv_path)
        if test_df.empty:
            print(f"🚨 Error: Test CSV file '{test_csv_path}' is empty!")
            return
        print(f"Found {len(test_df)} test samples.")
    except FileNotFoundError:
        print(f"🚨 Error: Test CSV file not found at '{test_csv_path}'!")
        return
    except Exception as e:
        print(f"🚨 Error reading test CSV: {e}")
        return

    # 5. Generate Predictions
    results = []
    print("Generating predictions for the test set...")
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        image_path = row['image_path']
        ground_truth_report = str(row['report_text']) if pd.notna(row['report_text']) else ""
        image_id = row.get('id', index)

        generated_report_text = "GENERATION_FAILED"
        try:
            # Load and preprocess image
            img_pil = Image.open(image_path).convert("RGB")
            img = image_transform(img_pil).unsqueeze(0).to(device)

            # Generate report using beam search
            with torch.no_grad():
                
                generated_report_text, _ = model.generate(
                    img=img,
                    tokenizer=tokenizer,
                    max_len=cfg.MAX_TEXT_LENGTH,
                    beam_width=cfg.BEAM_WIDTH,
                    start_token_id=START_TOKEN_ID,
                    end_token_id=END_TOKEN_ID
                )

        except FileNotFoundError:
            print(f"⚠️ Warning: Image not found for ID {image_id}. Skipping.")
            generated_report_text = "IMAGE_NOT_FOUND"
        except Exception as e:
            print(f"🚨 Error generating report for ID {image_id}: {e}")

        results.append({
            'id': image_id,
            'ground_truth': ground_truth_report,
            'generated': generated_report_text
        })

    # 6. Save Results
    print(f"Saving generation results to: {output_csv_path}")
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_csv_path, index=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"🚨 Error saving results CSV: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    cfg = Config()
    
    parser = argparse.ArgumentParser(description="Evaluate report generation model on the test set.")
    parser.add_argument("--test_csv", default=cfg.TEST_CSV_FILE, help="Path to the test set CSV file.")
    parser.add_argument("--model", default=cfg.BEST_MODEL_PATH, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--output_csv", default=cfg.PREDICTIONS_CSV_PATH, help="Path to save the generated reports.")

    args = parser.parse_args()

    current_device = torch.device(cfg.DEVICE)
    print(f"Using device: {current_device} {'🚀' if str(current_device) == 'cuda' else '🐌'}")

    run_evaluation(cfg, args.test_csv, args.model, args.output_csv, current_device)