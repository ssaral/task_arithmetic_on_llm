import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# Initialize WandB
wandb.init(project="llama-lora", entity="ssaral-india")  # Replace 'your-username' with your WandB username

# Load tokenizer
model_name = "gpt2"  # Switched to GPT-2 small model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Dataset preparation
class SentimentDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as file:
            self.data = json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }

# Tokenization function
def tokenize_function(jsonl_file, output_file):
    tokenized_data = []
    with open(jsonl_file, "r") as file:
        for line in file:
            data = json.loads(line)
            tokens = tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=512)
            token_dict = tokens.data  # Convert BatchEncoding to dictionary
            token_dict["label"] = data["label"]  # Add the label to tokenized data
            tokenized_data.append(token_dict)
    with open(output_file, "w") as outfile:
        json.dump(tokenized_data, outfile)

# Tokenize and save train dataset
tokenize_function("data/sst2/train.jsonl", "sst2_tokenized_train_gpt2.json")

# Load dataset and DataLoader
train_dataset = SentimentDataset("sst2_tokenized_train_gpt2.json")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define model with LoRA
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Set num_labels based on task
model.config.pad_token_id = tokenizer.pad_token_id

zero_shot_path = "./results_gpt2/zeroshot_model.pt"
torch.save(model.state_dict(), zero_shot_path)
print(f"Zero-shot model saved to {zero_shot_path}")

# LoRA configuration
lora_config = LoraConfig(
    task_type="SEQ_CLS",  # Sequence Classification task
    inference_mode=False,
    r=8,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0.1
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
print(f"Using LoRA. Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_gpt2",
    eval_strategy="no",  # Disable evaluation
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    gradient_accumulation_steps=2,
    fp16=True,  # Enable mixed precision
    logging_dir="./logs",
    logging_steps=100,
    report_to="wandb",  # Integrate with WandB
    run_name="lora-finetune-gpt2",  # Updated run name
    ddp_find_unused_parameters=False,  # Required for multi-GPU training with LoRA
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save LoRA fine-tuned model
model.save_pretrained("./lora_finetuned_model_bkp")
torch.save(model.state_dict(), "./results_gpt2/nonlinear_lora_sentiment_model.pt")  # Save as .pt file
print("LoRA fine-tuned model saved!")
