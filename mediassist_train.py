import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json

try:
    from mediassist_dataset import XRayReportDataset, collate_fn_filter_invalid
    from mediassist_model import TienetReportGenerator
    from config import Config
except ImportError as e:
    print(f"🚨 Error importing modules: {e}")
    exit()

# --- Validation Function ---
def validate_model(model, loader, criterion, device, scaler=None, vocab_size=None, pad_token_id=None):

    if loader is None:
        return float('inf')

    model.eval()
    total_val_loss = 0.0
    val_batches = 0

    progress_bar = tqdm(loader, desc="Validating", leave=False, unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            if not batch or batch['input_ids'].nelement() == 0:
                continue

            img = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)

            tgt_input = input_ids[:, :-1]
            tgt_expected_output = input_ids[:, 1:]

            tgt_seq_len = tgt_input.shape[1]
            if tgt_seq_len <= 0:
                continue

            tgt_mask = model.text_decoder.generate_square_subsequent_mask(tgt_seq_len).to(device)
            tgt_padding_mask = (tgt_input == pad_token_id).to(device)

            try:
                if Config.USE_MIXED_PRECISION and scaler is not None:
                    with autocast():
                        logits, _, _ = model(img, tgt_input, tgt_mask, tgt_padding_mask)
                        loss = criterion(logits.reshape(-1, vocab_size), tgt_expected_output.reshape(-1))
                else:
                    logits, _, _ = model(img, tgt_input, tgt_mask, tgt_padding_mask)
                    loss = criterion(logits.reshape(-1, vocab_size), tgt_expected_output.reshape(-1))

                if torch.isnan(loss):
                    continue

                total_val_loss += loss.item()
                val_batches += 1
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\n🚨 CUDA OOM in validation!")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\n🚨 Runtime error during validation: {e}")
                    continue
            except Exception as e:
                print(f"\n🚨 Error during validation: {e}")
                continue

    model.train()

    if val_batches == 0:
        return float('inf')

    avg_val_loss = total_val_loss / val_batches
    return avg_val_loss

# --- Training Loop ---
if __name__ == '__main__':
    
    cfg = Config()
    
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device} {'🚀' if str(device) == 'cuda' else '🐌'}")
    if str(device) == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\nChecking data files...")
    print(f"  Training CSV: {cfg.TRAIN_CSV_FILE}")
    print(f"  Validation CSV: {cfg.VALID_CSV_FILE}")
    if not os.path.exists(cfg.TRAIN_CSV_FILE):
        print(f"🚨 Error: Training CSV file not found at '{cfg.TRAIN_CSV_FILE}'!")
        exit()
    if not os.path.exists(cfg.VALID_CSV_FILE):
        print(f"🚨 Error: Validation CSV file not found at '{cfg.VALID_CSV_FILE}'!")
        exit()
    print("  ✓ Data files found.")

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        VOCAB_SIZE = tokenizer.vocab_size
        PAD_TOKEN_ID = tokenizer.pad_token_id
        print(f"Tokenizer loaded: {tokenizer.name_or_path} (Vocab size: {VOCAB_SIZE})")
    except Exception as e:
        print(f"🚨 Error loading tokenizer: {e}")
        exit()

    # Setup Dataset and DataLoader
    print("Loading datasets...")
    print(f"Training CSV: {cfg.TRAIN_CSV_FILE}")
    print(f"Validation CSV: {cfg.VALID_CSV_FILE}")
    try:
        train_dataset = XRayReportDataset(
            csv_file=cfg.TRAIN_CSV_FILE,
            tokenizer=tokenizer,
            max_length=cfg.MAX_TEXT_LENGTH
        )
        if len(train_dataset) == 0:
            print(f"🚨 Error: Training dataset '{cfg.TRAIN_CSV_FILE}' is empty!")
            exit()
        
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_DATA_WORKERS, pin_memory=True if str(device) == 'cuda' else False,
            collate_fn=collate_fn_filter_invalid
        )
        print(f"Training dataset loaded: {len(train_dataset)} samples.")
        if len(train_loader) == 0:
            print("🚨 Error: Training DataLoader created zero batches.")
            exit()

        valid_dataset = XRayReportDataset(
            csv_file=cfg.VALID_CSV_FILE,
            tokenizer=tokenizer,
            max_length=cfg.MAX_TEXT_LENGTH
        )
        if len(valid_dataset) == 0:
            print(f"🚨 Error: Validation dataset '{cfg.VALID_CSV_FILE}' is empty!")
            valid_loader = None
        else:
            valid_loader = DataLoader(
                valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                num_workers=cfg.NUM_DATA_WORKERS, pin_memory=True if str(device) == 'cuda' else False,
                collate_fn=collate_fn_filter_invalid
            )
            print(f"Validation dataset loaded: {len(valid_dataset)} samples.")
            if len(valid_loader) == 0:
                valid_loader = None

    except Exception as e:
        print(f"🚨 Error setting up dataset/dataloader: {e}")
        exit()

    # --- Initialize Model ---
    print("Initializing model...")
    try:
        model = TienetReportGenerator(
            vocab_size=VOCAB_SIZE,
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEAD,
            num_decoder_layers=cfg.NUM_DECODER_LAYERS,
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
            freeze_epochs=cfg.FREEZE_EPOCHS
        ).to(device)
        
        if cfg.USE_GRADIENT_CHECKPOINTING:
            print("Gradient checkpointing enabled (handled by model architecture)")
        
        print("Model initialized successfully.")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    except Exception as e:
        print(f"🚨 Error initializing model: {e}")
        exit()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    print(f"Loss function: CrossEntropyLoss (ignoring PAD token ID: {PAD_TOKEN_ID})")

    print("Setting up optimizer with differential learning rates...")
    try:
        image_encoder_param_ids = set(id(p) for p in model.image_encoder.parameters())
        
        other_params = [
            p for p in model.parameters() 
            if id(p) not in image_encoder_param_ids and p.requires_grad
        ]
        
        params_to_optimize = [
            {"params": model.image_encoder.parameters(), "lr": cfg.LR_IMAGE_ENCODER},
            {"params": other_params, "lr": cfg.LEARNING_RATE}, 
        ]
        
        optimizer = optim.AdamW(params_to_optimize)
        print(f"Optimizer: AdamW (Image LR: {cfg.LR_IMAGE_ENCODER}, Text/Other LR: {cfg.LEARNING_RATE})")

    except AttributeError:
         print("🚨 Error: Could not find 'model.image_encoder'. Using single learning rate.")
         optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
         print(f"Optimizer: AdamW (Learning Rate: {cfg.LEARNING_RATE})")
   

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    print("Learning rate scheduler (ReduceLROnPlateau) added.")

    scaler = GradScaler() if cfg.USE_MIXED_PRECISION else None
    if cfg.USE_MIXED_PRECISION:
        print("Mixed precision training (FP16) enabled.")

    print("-" * 50)
    print("Starting training... 🏋️‍♀️")
    print(f"Config => Epochs: {cfg.NUM_EPOCHS}, Batch Size: {cfg.BATCH_SIZE}")
    print(f"Mixed Precision: {cfg.USE_MIXED_PRECISION}, Gradient Checkpointing: {cfg.USE_GRADIENT_CHECKPOINTING}")
    print("-" * 50)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {
        "train_losses": [],
        "val_losses": []
    }
    for epoch in range(cfg.NUM_EPOCHS):
        model.set_epoch(epoch)
        
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} Training", leave=True, unit="batch")

        for i, batch in enumerate(progress_bar):
            if not batch or batch['input_ids'].nelement() == 0:
                continue

            img = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)

            tgt_input = input_ids[:, :-1]
            tgt_expected_output = input_ids[:, 1:]

            tgt_seq_len = tgt_input.shape[1]
            if tgt_seq_len <= 0:
                continue

            tgt_mask = model.text_decoder.generate_square_subsequent_mask(tgt_seq_len).to(device)
            tgt_padding_mask = (tgt_input == PAD_TOKEN_ID).to(device)

            optimizer.zero_grad()

            try:
                if cfg.USE_MIXED_PRECISION and scaler is not None:
                    with autocast():
                        logits, _, _ = model(img, tgt_input, tgt_mask, tgt_padding_mask)
                        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_expected_output.reshape(-1))

                    if torch.isnan(loss):
                        print(f"\n🚨 NaN loss detected during training batch {i+1}! Skipping update.")
                        optimizer.zero_grad()
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP_VALUE)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _, _ = model(img, tgt_input, tgt_mask, tgt_padding_mask)
                    loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_expected_output.reshape(-1))

                    if torch.isnan(loss):
                        print(f"\n🚨 NaN loss detected during training batch {i+1}! Skipping update.")
                        optimizer.zero_grad()
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP_VALUE)
                    optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n🚨 CUDA out of memory during training batch {i+1}!")
                    torch.cuda.empty_cache()
                    if scaler is not None:
                        scaler.update() # Must update scaler even on OOM
                    continue
                else:
                    print(f"\n🚨 Runtime error during training: {e}")
                    if scaler is not None:
                        scaler.update()
                    continue
            except Exception as e:
                print(f"\n🚨 Unexpected error during training: {e}")
                if scaler is not None:
                    scaler.update()
                continue

            current_loss = loss.item()
            epoch_train_loss += current_loss
            train_batches += 1

            if len(optimizer.param_groups) > 1:
                progress_bar.set_postfix(
                    loss=f'{current_loss:.4f}',
                    lr_img=f'{optimizer.param_groups[0]["lr"]:.1e}',
                    lr_txt=f'{optimizer.param_groups[1]["lr"]:.1e}'
                )
            else:
                progress_bar.set_postfix(
                    loss=f'{current_loss:.4f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.1e}'
                )

        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
        print(f"\n✅ End of Epoch {epoch+1} | Average Training Loss: {avg_train_loss:.4f}")

        # Run Validation
        avg_val_loss = validate_model(model, valid_loader, criterion, device, scaler, VOCAB_SIZE, PAD_TOKEN_ID)
        if avg_val_loss != float('inf'):
            print(f"      | Average Validation Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            training_history["train_losses"].append(avg_train_loss)
            training_history["val_losses"].append(avg_val_loss)

            # Save Best Model
            if avg_val_loss < best_val_loss:
                print(f"   ✨ Validation Loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving best model...")
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_path = cfg.BEST_MODEL_PATH 
                try:
                    torch.save(model.state_dict(), best_model_path)
                    print(f"      Best model saved to '{best_model_path}'")
                except Exception as e:
                    print(f"      🚨 Error saving best model: {e}")
            else:
                epochs_without_improvement += 1
                print(f"   Validation Loss did not improve from {best_val_loss:.4f} ({epochs_without_improvement} epochs)")

        print("-" * 50)

    print("🎉 Training finished! 🎉")
    if best_val_loss != float('inf'):
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        print(f"Best model state saved to '{cfg.BEST_MODEL_PATH}'")
    else:
        print("Training complete, but validation did not run or produced invalid results.")
        
    # Save the history to a JSON file
    try:
        history_path = cfg.TRAINING_HISTORY_FILE
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        print(f"Training history saved to '{history_path}'")
    except Exception as e:
        print(f"🚨 Error saving training history: {e}")