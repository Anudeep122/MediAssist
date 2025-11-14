import torch
from transformers import AutoTokenizer
from PIL import Image
import argparse
import os

try:
    from mediassist_dataset import image_transform
    from mediassist_model import TienetReportGenerator
    from config import Config
except ImportError:
    print("🚨 Error: Make sure 'mediassist_dataset.py', 'mediassist_model.py', and 'config.py' are in the same folder.")
    exit()


def predict_report(cfg, image_path, model_path, device):

    # 1. Load Tokenizer
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
        print(f"Tokenizer loaded: {cfg.TOKENIZER_NAME}")
    except Exception as e:
        print(f"🚨 Error loading tokenizer: {e}")
        return None

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
            freeze_epochs=0
        ).to(device)
        print("Model architecture initialized.")
    except Exception as e:
        print(f"🚨 Error initializing model architecture: {e}")
        return None

    # 3. Load Trained Weights
    print(f"Loading trained weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"🚨 Error: Model weights file not found at '{model_path}'!")
        return None
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Trained weights loaded successfully.")
    except Exception as e:
        print(f"🚨 Error loading model weights: {e}")
        return None

    # 4. Load and Preprocess Image
    print("Loading and preprocessing image...")
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img = image_transform(img_pil).unsqueeze(0).to(device)
        print("Image preprocessed.")
    except FileNotFoundError:
        print(f"🚨 Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"🚨 Error processing image: {e}")
        return None

    # 5. Generate Report
    print("Generating report...")
    try:
        with torch.no_grad():
            generated_text, _ = model.generate(
                img=img,
                tokenizer=tokenizer,
                max_len=cfg.MAX_TEXT_LENGTH,
                beam_width=cfg.BEAM_WIDTH,
                start_token_id=START_TOKEN_ID,
                end_token_id=END_TOKEN_ID
            )
        print("Report generated.")
        return generated_text

    except Exception as e:
        print(f"🚨 Error during report generation: {e}")
        return None


if __name__ == "__main__":
    cfg = Config()
    
    parser = argparse.ArgumentParser(description="Generate a medical report from chest X-ray image.")
    parser.add_argument("--image", required=True, help="Path to the X-ray image.")
    parser.add_argument("--model", default=cfg.BEST_MODEL_PATH, help="Path to the trained model checkpoint (.pth file).")

    args = parser.parse_args()

    current_device = torch.device(cfg.DEVICE)
    print(f"Using device: {current_device}")

    generated_report_result = predict_report(cfg, args.image, args.model, current_device)

    if generated_report_result:
        print("\n--- Generated Report ---")
        print(generated_report_result)
        print("------------------------")
    else:
        print("\n🚨 Report generation failed or returned None.")