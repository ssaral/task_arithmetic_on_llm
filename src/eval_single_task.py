import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.linearize import LinearizedLM, LinearizedModel


# Parse arguments
args = parse_arguments()

# Define checkpoint save path
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

# Initialize results dictionary
accuracies = {}

# Print evaluation mode
print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")

# Define tasks for evaluation
for task in ["sst2", "qnli", "cola"]: #, "mnli_matched"]:  
    print("*" * 100)
    print(f"Evaluating on {task}")

    # Define paths for pretrained and finetuned checkpoints
    pretrained_checkpoint = f"{args.save}/zeroshot_full_model.pt"
    finetuned_checkpoint = (
        f"{args.save}/linear_finetuned.pt"
        if args.finetuning_mode == "linear"
        else f"{args.save}/finetuned_full_model.pt"
    )

    # Load task vector
    try:
        task_vector = (
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    # Prepare the model based on finetuning mode
    if args.finetuning_mode == "none":
        text_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif args.finetuning_mode in ["standard", "linear"]:
        text_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    elif args.finetuning_mode == "posthoc":
        zs_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        ft_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        text_model = LinearizedModel(init_model=zs_model, lm_model=ft_model, args=args)

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token


    # Evaluate on test and validation splits
    for split in ["test", "validation"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_task = task #if split == "test" else f"validation"

        accuracies[eval_task] = eval_single_dataset(
            text_model, tokenizer, eval_task, split,  args
        )["accuracy"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_accuracies.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
