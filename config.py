import torch
import os

class Config:
    
    # --- Core Infrastructure ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Directory Paths ---
    CHECKPOINT_DIR = "./checkpoints"
    VISUALIZATIONS_DIR = "./visualizations"
    RESULTS_DIR = "./results"

    # --- File Paths ---
    TRAIN_CSV_FILE = 'train_split.csv'
    VALID_CSV_FILE = 'valid_split.csv'
    TEST_CSV_FILE = 'model_test.csv' 
    
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'tienet_report_transformer_best.pth')
    
    PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, 'mediassist_test_predictions.csv')
    
    METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.json') 
    TRAINING_HISTORY_FILE = os.path.join(CHECKPOINT_DIR, 'training_history.json')

    # --- Model Hyperparameters ---
    D_MODEL = 512
    N_HEAD = 4
    NUM_DECODER_LAYERS = 2
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    FREEZE_EPOCHS = 5  

    # --- Training Hyperparameters ---
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    LR_IMAGE_ENCODER = 1e-5  # <-- ADDED THIS
    BATCH_SIZE = 8
    GRAD_CLIP_VALUE = 1.0
    NUM_DATA_WORKERS = 0  
    
    # Optimization settings
    USE_MIXED_PRECISION = True
    USE_GRADIENT_CHECKPOINTING = True

    TOKENIZER_NAME = 'bert-base-uncased'
    MAX_TEXT_LENGTH = 128
    
    IMAGE_SIZE = (224, 224)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    # --- Evaluation Configuration ---
    BEAM_WIDTH = 5

    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.VISUALIZATIONS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("--- Model Configuration ---")
    config = Config()
    
    attrs = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and k != '__init__'}
    for key, value in attrs.items():
        print(f"{key:<30} = {value}")
    
    print("\nDirectories created (if they didn't exist):")
    print(f" - {config.CHECKPOINT_DIR}")
    print(f" - {config.VISUALIZATIONS_DIR}")
    print(f" - {config.RESULTS_DIR}")