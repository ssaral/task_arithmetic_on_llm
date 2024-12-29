import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import json
import numpy as np

def eval_single_dataset(model, tokenizer, dataset_name, split, args):
    """
    Evaluate the model on a single text dataset.

    Args:
        model: The transformer model to evaluate.
        tokenizer: The tokenizer for the model.
        dataset_name: The name of the dataset to evaluate on.
        args: Arguments containing evaluation configurations.

    Returns:
        dict: Evaluation metrics (e.g., accuracy).
    """
    # Load the dataset
    dataset = load_dataset("glue", dataset_name, split=split)
    # dataset = load_dataset("glue", dataset_name, split="test")
    
    # Tokenize the dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

    # Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)

            # Calculate accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return {"accuracy": accuracy}


# def eval_pipeline(task_list, args):
#     """
#     Evaluation pipeline for text-based models.

#     Args:
#         task_list (list): List of tasks to evaluate on.
#         args: Arguments containing evaluation configurations.

#     Returns:
#         None
#     """
#     accuracies = {}

#     print("*" * 100)
#     if args.finetuning_mode == "none":
#         print("Evaluating pretrained models.")
#     elif args.finetuning_mode == "standard":
#         print("Evaluating non-linear FT models.")
#     elif args.finetuning_mode == "linear":
#         print("Evaluating linear FT models.")
#     elif args.finetuning_mode == "posthoc":
#         print("Evaluating post-hoc linearized models.")

#     # Iterate over each task
#     for task in task_list:
#         print("*" * 100)
#         print(f"Evaluating on {task}")

#         pretrained_checkpoint = f"{args.save}/{task}/zeroshot.pt"
#         finetuned_checkpoint = (
#             f"{args.save}/{task}/linear_finetuned.pt"
#             if args.finetuning_mode == "linear"
#             else f"{args.save}/{task}/finetuned.pt"
#         )

#         # Load task vector
#         try:
#             task_vector = (
#                 LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
#                 if args.finetuning_mode == "linear"
#                 else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
#             )
#         except FileNotFoundError:
#             print(f"Error: Could not find {finetuned_checkpoint}.")
#             continue

#         # Prepare the model
#         if args.finetuning_mode == "none":
#             text_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
#         elif args.finetuning_mode in ["standard", "linear"]:
#             text_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
#         elif args.finetuning_mode == "posthoc":
#             zs_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
#             ft_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
#             text_model = LinearizedLM(init_model=zs_model, lm_model=ft_model, args=args)

#         # Load tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(args.model)

#         # Evaluate
#         accuracies[task] = eval_single_dataset(text_model, tokenizer, task, args)["accuracy"]

#     # Save results
#     if args.finetuning_mode == "none":
#         save_path = f"{args.save}/zeroshot_accuracies.json"
#     elif args.finetuning_mode == "standard":
#         save_path = f"{args.save}/ft_accuracies.json"
#     elif args.finetuning_mode == "linear":
#         save_path = f"{args.save}/linear_ft_accuracies.json"
#     elif args.finetuning_mode == "posthoc":
#         save_path = f"{args.save}/posthoc_ft_accuracies.json"

#     with open(save_path, "w") as f:
#         json.dump(accuracies, f)

#     print(f"Results saved to {save_path}")


# if __name__ == "__main__":
#     from src.args import parse_arguments
#     from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
#     from src.linearize import LinearizedLM

#     # Parse arguments
#     args = parse_arguments()

#     # Define tasks for evaluation
#     tasks = ["sst2", "mnli", "qnli", "cola"]  # Example GLUE tasks

#     # Run evaluation pipeline
#     eval_pipeline(tasks, args)
