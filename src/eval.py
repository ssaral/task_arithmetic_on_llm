import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import json
import numpy as np
from src.linearize import LinearizedLM
from src.eval import eval_single_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer


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
            outputs = model(inputs)#, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)

            # Calculate accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return {"accuracy": accuracy}

def evaluate(model, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        for split in ["test", "validation"]:
            results = eval_single_dataset(model, tokenizer, dataset_name, split, args)
            print(f"{dataset_name} Top-1 accuracy: {results['accuracy']:.4f}")
            per_dataset_results[dataset_name + ":top1"] = results["accuracy"]

    return per_dataset_results

def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, scaling_coef, posthoc_linearization=False
):
    model = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    if posthoc_linearization:
        pretrained_model = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=0.0
        )
        model = LinearizedLM(
            init_model=pretrained_model, model=model, args=args
        )
    coef_info = evaluate(model, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info

def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args, posthoc_linearization=False
):
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
            posthoc_linearization,
        )

    return info

def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results
