import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import itertools

def load_model_and_tokenizer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_probabilities(model, tokenizer, dataset, device):
    model.eval()
    probabilities = []
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Processing Data"):
            inputs = tokenizer(sample["sentence"], return_tensors="pt", padding=True, truncation=True).to(device)  # Move inputs to GPU
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            probabilities.append(probs)
    
    return probabilities


def compute_similarity_score(probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A, device):
    n = len(probs_A_on_B)
    m = len(probs_B_on_A)

    print("n: ", n, ", m:", m)

    # Convert lists to tensors and move to GPU
    probs_A_on_B = torch.tensor(probs_A_on_B, device=device)
    probs_B_on_B = torch.tensor(probs_B_on_B, device=device)
    probs_B_on_A = torch.tensor(probs_B_on_A, device=device)
    probs_A_on_A = torch.tensor(probs_A_on_A, device=device)

    # Compute log probability ratios across all labels
    term1 = torch.sum(torch.log(probs_A_on_B / probs_B_on_B), dim=1).mean()
    term2 = torch.sum(torch.log(probs_B_on_A / probs_A_on_A), dim=1).mean()
    
    S_AB = 0.5 * (term1 + term2)
    return S_AB.item() 

tasks = [
    "axb", "axg", "boolq", "cola", "copa", "cr", "mpqa", "mr", "mrpc", "pawsx", 
    "qnli", "qqp", "rte", "sst2", "subj", "wnli", "xwinograd"
]

model_paths = {
    "axb": "nonlinear_checkpoints/axb/",
    "axg": "nonlinear_checkpoints/axg/",
    "boolq": "nonlinear_checkpoints/boolq/",
    # "cb": "nonlinear_checkpoints/cb/",
    "cola": "nonlinear_checkpoints/cola/",
    "copa": "nonlinear_checkpoints/copa/",
    "cr": "nonlinear_checkpoints/cr/",
    # "mnli": "nonlinear_checkpoints/mnli/",
    "mpqa": "nonlinear_checkpoints/mpqa/",
    "mr": "nonlinear_checkpoints/mr/",
    "mrpc": "nonlinear_checkpoints/mrpc/",
    "pawsx": "nonlinear_checkpoints/pawsx/",
    "qnli": "nonlinear_checkpoints/qnli/",
    "qqp": "nonlinear_checkpoints/qqp/",
    "rte": "nonlinear_checkpoints/rte/",
    # "snli": "nonlinear_checkpoints/snli/",
    "sst2": "nonlinear_checkpoints/sst2/",
    # "sst5": "nonlinear_checkpoints/sst5/",
    "subj": "nonlinear_checkpoints/subj/",
    # "trec": "nonlinear_checkpoints/trec/",
    "wnli": "nonlinear_checkpoints/wnli/",
    # "xnli": "nonlinear_checkpoints/xnli/",
    "xwinograd": "nonlinear_checkpoints/xwinograd/"
}

dataset_paths = {
    "axb": "data_clubbed/axb_clubbed.jsonl",
    "axg": "data_clubbed/axg_clubbed.jsonl",
    "boolq": "data_clubbed/boolq_clubbed.jsonl",
    # "cb": "data_clubbed/cb_clubbed.jsonl",
    "cola": "data_clubbed/cola_clubbed.jsonl",
    "copa": "data_clubbed/copa_clubbed.jsonl",
    "cr": "data_clubbed/cr_clubbed.jsonl",
    # "mnli": "data_clubbed/mnli_clubbed.jsonl",
    "mpqa": "data_clubbed/mpqa_clubbed.jsonl",
    "mr": "data_clubbed/mr_clubbed.jsonl",
    "mrpc": "data_clubbed/mrpc_clubbed.jsonl",
    "pawsx": "data_clubbed/pawsx_clubbed.jsonl",
    "qnli": "data_clubbed/qnli_clubbed.jsonl",
    "qqp": "data_clubbed/qqp_clubbed.jsonl",
    "rte": "data_clubbed/rte_clubbed.jsonl",
    # "snli": "data_clubbed/snli_clubbed.jsonl",
    "sst2": "data_clubbed/sst2_clubbed.jsonl",
    # "sst5": "data_clubbed/sst5_clubbed.jsonl",
    "subj": "data_clubbed/subj_clubbed.jsonl",
    # "trec": "data_clubbed/trec_clubbed.jsonl",
    "wnli": "data_clubbed/wnli_clubbed.jsonl",
    # "xnli": "data_clubbed/xnli_clubbed.jsonl",
    "xwinograd": "data_clubbed/xwinograd_clubbed.jsonl"
}


device = "cuda" if torch.cuda.is_available() else "cpu"

# generating all unique pairs of tasks (task_A, task_B) such that task_A != task_B
task_pairs = list(itertools.permutations(tasks, 2))


for task_A, task_B in task_pairs:
    print(f"Processing pair: {task_A} - {task_B}")

    # Load models and tokenizers for task_A and task_B
    model_path_A = model_paths.get(task_A)
    model_path_B = model_paths.get(task_B)
    dataset_path_A = dataset_paths.get(task_A)
    dataset_path_B = dataset_paths.get(task_B)

    if model_path_A and model_path_B and dataset_path_A and dataset_path_B:
        # loading model A and tokenizer A
        model_A, tokenizer_A = load_model_and_tokenizer(model_path_A)
        model_A.to(device)
        # loading dataset A
        dataset_A = load_dataset(dataset_path_A)

        # loadin model B and tokenizer B
        model_B, tokenizer_B = load_model_and_tokenizer(model_path_B)
        model_B.to(device)
        # loading dataset B
        dataset_B = load_dataset(dataset_path_B)

        # computing probabilities for task A and task B
        probs_A_on_A = compute_probabilities(model_A, tokenizer_A, dataset_A, device)
        probs_A_on_B = compute_probabilities(model_A, tokenizer_A, dataset_B, device)

        probs_B_on_A = compute_probabilities(model_B, tokenizer_B, dataset_A, device)
        probs_B_on_B = compute_probabilities(model_B, tokenizer_B, dataset_B, device)

        # computing similarity score for the pair (task_A, task_B)
        S_AB = compute_similarity_score(probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A, device)
        print(f"Model Similarity Score (S_AB) for pair {task_A} - {task_B}: {S_AB}")
    else:
        print(f"Missing paths for pair {task_A} - {task_B}")
