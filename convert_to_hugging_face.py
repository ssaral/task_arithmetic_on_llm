import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)

def load_model(pt_model_path, task_type):
    """
    Load a PyTorch model saved in .pt format.
    
    Args:
        pt_model_path (str): Path to the .pt model file.
        task_type (str): Task type, e.g., 'classification', 'causal-lm', 'token-classification'.
    
    Returns:
        model (torch.nn.Module): Loaded PyTorch model.
    """
    model = torch.load(pt_model_path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The provided .pt file does not contain a valid PyTorch model.")
    return model

def convert_to_huggingface(model, save_dir, base_model_name, task_type, num_labels=None):
    """
    Convert a PyTorch model to Hugging Face format.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        save_dir (str): Directory to save the converted Hugging Face model.
        base_model_name (str): Name of the base model (e.g., 'bert-base-uncased').
        task_type (str): Task type, e.g., 'classification', 'causal-lm', 'token-classification'.
        num_labels (int, optional): Number of labels for classification tasks.
    """
    # Load configuration and tokenizer for the base model
    config = AutoConfig.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Adjust config based on task type
    if task_type == "classification":
        config.num_labels = num_labels
        hf_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)
    elif task_type == "causal-lm":
        hf_model = AutoModelForCausalLM.from_pretrained(base_model_name, config=config)
    elif task_type == "token-classification":
        config.num_labels = num_labels
        hf_model = AutoModelForTokenClassification.from_pretrained(base_model_name, config=config)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Transfer weights from PyTorch model to Hugging Face model
    hf_model.load_state_dict(model.state_dict())

    # Save the Hugging Face model and tokenizer
    hf_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Hugging Face model saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert .pt model to Hugging Face format.")
    parser.add_argument("--pt_model_path", type=str, required=True, help="Path to the .pt model file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the Hugging Face model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base model (e.g., 'bert-base-uncased').")
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "causal-lm", "token-classification"], help="Task type for the model.")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labels for classification tasks (required for 'classification' or 'token-classification').")
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load PyTorch model
    model = load_model(args.pt_model_path, args.task_type)

    # Convert and save as Hugging Face format
    convert_to_huggingface(
        model=model,
        save_dir=args.save_dir,
        base_model_name=args.base_model_name,
        task_type=args.task_type,
        num_labels=args.num_labels,
    )

if __name__ == "__main__":
    main()
