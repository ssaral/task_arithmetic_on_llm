import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Load the SST-2 dataset from Hugging Face Hub
def load_sst2_data():
    dataset = load_dataset("stanfordnlp/sst2")
    return dataset

# Process and tokenize the dataset
def preprocess_data(dataset, tokenizer, max_length=128):
    processed_data = []

    for split in ["train", "validation", "test"]:
        split_data = []
        for entry in dataset[split]:
            text = entry["sentence"]
            label = entry["label"]

            # Tokenize the text
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Create a dictionary for each example
            data_entry = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
            }
            split_data.append(data_entry)

        processed_data.append(split_data)

    return processed_data

# Save the processed data to .pt files
def save_data(processed_data, save_dir="data/sst2"):
    import os

    os.makedirs(save_dir, exist_ok=True)
    
    splits = ["train", "validation", "test"]
    for split, data in zip(splits, processed_data):
        save_path = os.path.join(save_dir, f"{split}_data.pt")
        torch.save(data, save_path)
        print(f"Saved {split} data to {save_path}")

# Main function to execute the data preparation
def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading dataset...")
    dataset = load_sst2_data()

    print("Processing dataset...")
    processed_data = preprocess_data(dataset, tokenizer)

    print("Saving processed data...")
    save_data(processed_data)

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
