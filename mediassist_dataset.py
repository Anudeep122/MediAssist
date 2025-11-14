import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
import torchvision.transforms as T

# --- Image Transformations ---
image_transform = T.Compose(
    [
        T.Resize((224, 224)),  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def collate_fn_filter_invalid(batch):
    
    valid_batch = [item for item in batch if item and item.get("is_valid", False)]

    if not valid_batch:
        return {
            "image": torch.empty(0, 3, 224, 224),
            "input_ids": torch.empty(0, 128, dtype=torch.long),
            "attention_mask": torch.empty(0, 128, dtype=torch.long),
            "is_valid": torch.empty(0, dtype=torch.bool),
        }

    return torch.utils.data.dataloader.default_collate(valid_batch)


class XRayReportDataset(Dataset):
    
    def __init__(self, csv_file, tokenizer, max_length=128):
        
        try:
            self.report_data = pd.read_csv(csv_file)
            print(f"Loaded {len(self.report_data)} rows from {csv_file}")
        except FileNotFoundError:
            print(f"🚨 Error: CSV file not found at {csv_file}")
            self.report_data = pd.DataFrame()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.report_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.report_data.empty:
            return None

        try:
            row = self.report_data.iloc[idx]
            image_path = row["image_path"]
            report_text = row["report_text"]
            report_text = str(report_text) if pd.notna(report_text) else ""

        except IndexError:
            msg = (
                f"🚨 Error: Index {idx} out of bounds "
                f"for the dataset (size {len(self.report_data)})."
            )
            print(msg)
            return None
        except KeyError as e:
            msg = (
                f"🚨 Error: Missing column in CSV: {e}. "
                f"Check '{self.report_data.columns}'."
            )
            print(msg)
            return None

        # --- Load and Transform Image ---
        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            msg = f"⚠️ Warning: Image not found at {image_path}. Skipping sample {idx}."
            print(msg)
            return None
        except Exception as e:
            print(f"🚨 Error loading image for sample {idx}: {e}")
            return None

        img = image_transform(img)

        # --- Tokenize Report Text ---
        text = f"[CLS] {report_text} [SEP]"

        try:
            tokenized_output = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"🚨 Error tokenizing text for sample {idx}: {e}")
            return None

        input_ids = tokenized_output["input_ids"].squeeze()
        attention_mask = tokenized_output["attention_mask"].squeeze()

        return {
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_valid": True,
        }


# --- Example Usage ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    print("Testing dataset loading...")
    test_csv = "train_split.csv" 
    dataset = XRayReportDataset(csv_file=test_csv, tokenizer=tokenizer, max_length=128)

    if len(dataset) > 0:
        print(f"Dataset test loaded successfully with {len(dataset)} samples.")
        sample = dataset[0]

        if sample and sample.get("is_valid", False):
            print("\n--- Sample 0 ---")
            print("Image shape:", sample["image"].shape)
            print("Report input_ids shape:", sample["input_ids"].shape)
            print("Report attention_mask shape:", sample["attention_mask"].shape)
            print("Sample input IDs (first 20):", sample["input_ids"][:20].tolist())
            decoded = tokenizer.decode(sample["input_ids"][:50])
            print("Decoded sample text (first part):", decoded)

        print("\nTesting DataLoader...")
        from torch.utils.data import DataLoader

        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_filter_invalid,
        )

        try:
            print(f"DataLoader contains {len(data_loader)} batches.")
            first_batch = next(iter(data_loader))
            if first_batch["input_ids"].nelement() > 0:
                print("\n--- First Batch ---")
                print("Image batch shape:", first_batch["image"].shape)
                print("Report input_ids batch shape:", first_batch["input_ids"].shape)
                batch_size = first_batch["input_ids"].shape[0]
                print(f"Batch size retrieved: {batch_size}")
            else:
                print("🚨 DataLoader retrieved an empty first batch.")

        except StopIteration:
            print("🚨 DataLoader returned no batches.")
        except Exception as e:
            print(f"🚨 Error getting batch from DataLoader: {e}")

    else:
        print(f"🚨 Dataset failed to load or is empty ({test_csv}).")