import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import itertools

def load_model_and_tokenizer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Use the EOS token as pad if needed
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_label_probabilities(model, tokenizer, dataset, device, label_field="label", text_field="sentence", epsilon=1e-12):
    """
    For a given dataset, compute for each sample the probability assigned by the model
    to the ground-truth label.
    """
    model.eval()
    label_probs = []
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Computing label probabilities"):
            inputs = tokenizer(sample[text_field], return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(**inputs).logits  # shape: [1, num_classes]
            probs = F.softmax(logits, dim=-1)  # probability distribution over classes
            # Extract the ground truth label probability.
            label = sample[label_field]
            if not isinstance(label, int):
                label = int(label)
            # Prevent a zero probability (thus log(0)) by clamping with epsilon.
            prob = probs[0, label].clamp(min=epsilon)
            label_probs.append(prob.item())
            
    return label_probs

def compute_similarity_score(probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A):
    """
    Computes the PMI similarity score between tasks using the probabilities for the ground-truth labels.
    
    For samples from dataset D_Tj:
      term1 = (1/n) sum[ log( p_A(y|x) / p_B(y|x) ) ] 
      
    For samples from dataset D_Ti:
      term2 = (1/m) sum[ log( p_B(y|x) / p_A(y|x) ) ]
      
    Finally:
      S_AB = 0.5 * (term1 + term2)
    """
    n = len(probs_A_on_B)
    m = len(probs_B_on_A)

    term1 = sum(torch.log(torch.tensor(pA) / torch.tensor(pB)) 
                for pA, pB in zip(probs_A_on_B, probs_B_on_B)) / n
    term2 = sum(torch.log(torch.tensor(pB) / torch.tensor(pA)) 
                for pA, pB in zip(probs_A_on_A, probs_B_on_A)) / m

    S_AB = 0.5 * (term1 + term2)
    return S_AB.item()

# Define the tasks, model paths, and dataset paths.
tasks = [
    "axb", "axg", "boolq", "cola", "copa", "cr", "mpqa", "mr", "mrpc", "pawsx", 
    "qnli", "qqp", "rte", "sst2", "subj", "wnli", "xwinograd"
]

model_paths = {
    "axb": "nonlinear_checkpoints/axb/",
    "axg": "nonlinear_checkpoints/axg/",
    "boolq": "nonlinear_checkpoints/boolq/",
    "cola": "nonlinear_checkpoints/cola/",
    "copa": "nonlinear_checkpoints/copa/",
    "cr": "nonlinear_checkpoints/cr/",
    "mpqa": "nonlinear_checkpoints/mpqa/",
    "mr": "nonlinear_checkpoints/mr/",
    "mrpc": "nonlinear_checkpoints/mrpc/",
    "pawsx": "nonlinear_checkpoints/pawsx/",
    "qnli": "nonlinear_checkpoints/qnli/",
    "qqp": "nonlinear_checkpoints/qqp/",
    "rte": "nonlinear_checkpoints/rte/",
    "sst2": "nonlinear_checkpoints/sst2/",
    "subj": "nonlinear_checkpoints/subj/",
    "wnli": "nonlinear_checkpoints/wnli/",
    "xwinograd": "nonlinear_checkpoints/xwinograd/"
}

dataset_paths = {
    "axb": "data_clubbed/axb_clubbed.jsonl",
    "axg": "data_clubbed/axg_clubbed.jsonl",
    "boolq": "data_clubbed/boolq_clubbed.jsonl",
    "cola": "data_clubbed/cola_clubbed.jsonl",
    "copa": "data_clubbed/copa_clubbed.jsonl",
    "cr": "data_clubbed/cr_clubbed.jsonl",
    "mpqa": "data_clubbed/mpqa_clubbed.jsonl",
    "mr": "data_clubbed/mr_clubbed.jsonl",
    "mrpc": "data_clubbed/mrpc_clubbed.jsonl",
    "pawsx": "data_clubbed/pawsx_clubbed.jsonl",
    "qnli": "data_clubbed/qnli_clubbed.jsonl",
    "qqp": "data_clubbed/qqp_clubbed.jsonl",
    "rte": "data_clubbed/rte_clubbed.jsonl",
    "sst2": "data_clubbed/sst2_clubbed.jsonl",
    "subj": "data_clubbed/subj_clubbed.jsonl",
    "wnli": "data_clubbed/wnli_clubbed.jsonl",
    "xwinograd": "data_clubbed/xwinograd_clubbed.jsonl"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate all unique ordered pairs of tasks (task_A, task_B) where task_A != task_B.
task_pairs = list(itertools.permutations(tasks, 2))

for task_A, task_B in task_pairs:
    print(f"\nProcessing pair: {task_A} - {task_B}")

    # Retrieve corresponding paths for models and datasets.
    model_path_A = model_paths.get(task_A)
    model_path_B = model_paths.get(task_B)
    dataset_path_A = dataset_paths.get(task_A)
    dataset_path_B = dataset_paths.get(task_B)

    if model_path_A and model_path_B and dataset_path_A and dataset_path_B:
        # Load model A and its tokenizer, then its dataset.
        model_A, tokenizer_A = load_model_and_tokenizer(model_path_A)
        model_A.to(device)
        dataset_A = load_dataset(dataset_path_A)

        # Load model B and its tokenizer, then its dataset.
        model_B, tokenizer_B = load_model_and_tokenizer(model_path_B)
        model_B.to(device)
        dataset_B = load_dataset(dataset_path_B)

        # For dataset_B (task T_j), compute:
        #   p_A(y^Tj|x) using model A and p_B(y^Tj|x) using model B.
        probs_A_on_B = compute_label_probabilities(model_A, tokenizer_A, dataset_B, device)
        probs_B_on_B = compute_label_probabilities(model_B, tokenizer_B, dataset_B, device)

        # For dataset_A (task T_i), compute:
        #   p_A(y^Ti|x) using model A and p_B(y^Ti|x) using model B.
        probs_A_on_A = compute_label_probabilities(model_A, tokenizer_A, dataset_A, device)
        probs_B_on_A = compute_label_probabilities(model_B, tokenizer_B, dataset_A, device)

        # Compute the PMI similarity score following the defined equation.
        S_AB = compute_similarity_score(probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A)
        print(f"PMI Similarity Score (S_AB) for pair {task_A} - {task_B}: {S_AB:.4f}")
    else:
        print(f"Missing paths for pair {task_A} - {task_B}")
