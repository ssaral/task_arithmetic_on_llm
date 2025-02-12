import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from src.task_vectors_similarity_2 import LinearizedTaskVector, NonLinearTaskVector
from src.args import parse_arguments
from sklearn.random_projection import GaussianRandomProjection

args = parse_arguments()

def apply_random_projection(task_vector, target_dim):
    """
    Reduces the dimensionality of the task vector using random projection.

    Args:
        task_vector (np.ndarray): The high-dimensional task vector.
        target_dim (int): The target dimension for the reduced vector.

    Returns:
        np.ndarray: The reduced task vector.
    """
    # Initialize the random projection transformer
    transformer = GaussianRandomProjection(n_components=target_dim)
    
    # Reshape task vector to 2D (required by sklearn)
    task_vector_reshaped = task_vector.reshape(1, -1)
    print("task vector ka shape", task_vector_reshaped.shape)
    # Apply random projection
    reduced_vector = transformer.fit_transform(task_vector_reshaped)
    print("reduced vector ka shape", reduced_vector.shape)
    return reduced_vector.flatten()

# Function to calculate cosine similarity
def compute_cosine_similarity(vector_1, vector_2):
    similarity = cosine_similarity(vector_1.reshape(1, -1), vector_2.reshape(1, -1))
    # similarity = cosine_similarity(vector_1, vector_2)
    return similarity[0][0]

# Function to calculate Euclidean distance
def compute_euclidean_distance(vector_1, vector_2):
    return euclidean(vector_1, vector_2)

# Function to calculate dot product
def compute_dot_product(vector_1, vector_2):
    return np.dot(vector_1, vector_2)

# Function to convert task vector to numpy array
def task_vector_to_numpy(task_vector):
    return np.concatenate([v.flatten().numpy() for v in task_vector.vector.values()])

# Main function to perform the analysis
def analyze_task_vectors(task_vector_1, task_vector_2):
    # Convert task vectors to numpy arrays
    vector_1 = task_vector_to_numpy(task_vector_1)
    vector_2 = task_vector_to_numpy(task_vector_2)

    reduced_vector_1 = apply_random_projection(vector_1, target_dim=2048)
    reduced_vector_2 = apply_random_projection(vector_2, target_dim=2048)

    print(f"Original Dimension: {vector_1.shape}, Reduced Dimension: {reduced_vector_1.shape}")
    print(f"Original Dimension: {vector_2.shape}, Reduced Dimension: {reduced_vector_2.shape}")

    # Cosine Similarity
    cosine_sim = compute_cosine_similarity(reduced_vector_1, reduced_vector_2)
    print(f"Cosine similarity: {cosine_sim}")

    # Euclidean Distance
    # euclidean_dist = compute_euclidean_distance(vector_1, vector_2)
    # print(f"Euclidean distance: {euclidean_dist}")

    # Dot Product
    dot_prod = compute_dot_product(vector_1, vector_2)
    print(f"Dot product: {dot_prod}")

    # Symmetry Check (Difference Vector)
    # difference_vector = vector_1 - vector_2
    # print(f"Difference vector norm: {np.linalg.norm(difference_vector)}")

    # Cosine similarity of the difference vector to zero
    # diff_cosine_sim = compute_cosine_similarity(difference_vector, np.zeros_like(difference_vector))
    # print(f"Difference vector cosine similarity to zero: {diff_cosine_sim}")


# For task vector 1
flag1 = 0
tasks1 = [
    # {"task": "cola", "pretrained": "checkpoints/gpt2_linear_classification_cola/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_cola/linear_finetuned.pt", "labels": "2"},
    # {"task": "cr", "pretrained": "checkpoints/gpt2_linear_classification_cr/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_cr/linear_finetuned.pt", "labels": "2"},
    # {"task": "mpqa", "pretrained": "checkpoints/gpt2_linear_classification_mpqa/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mpqa/linear_finetuned.pt", "labels": "2"},
    # {"task": "mr", "pretrained": "checkpoints/gpt2_linear_classification_mr/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mr/linear_finetuned.pt", "labels": "2"},
    # {"task": "mrpc", "pretrained": "checkpoints/gpt2_linear_classification_mrpc/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mrpc/linear_finetuned.pt", "labels": "2"},
    # {"task": "qnli", "pretrained": "checkpoints/gpt2_linear_classification_qnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_qnli/linear_finetuned.pt", "labels": "2"},
    # {"task": "snli", "pretrained": "checkpoints/gpt2_linear_classification_snli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_snli/linear_finetuned.pt", "labels": "3"},
    # {"task": "wnli", "pretrained": "checkpoints/gpt2_linear_classification_wnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_wnli/linear_finetuned.pt", "labels": "2"},
    # {"task": "qqp", "pretrained": "checkpoints/gpt2_linear_classification_qqp/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_qqp/linear_finetuned.pt", "labels": "2"},
    # {"task": "rte", "pretrained": "checkpoints/gpt2_linear_classification_rte/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_rte/linear_finetuned.pt", "labels": "2"}, 
    # {"task": "sst2", "pretrained": "checkpoints/gpt2_linear_classification_sst2/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_sst2/linear_finetuned.pt", "labels": "2"}, 
    # {"task": "sst5", "pretrained": "checkpoints/gpt2_linear_classification_sst5/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_sst5/linear_finetuned.pt", "labels": "5"}, 
    # {"task": "trec", "pretrained": "checkpoints/gpt2_linear_classification_trec/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_trec/linear_finetuned.pt", "labels": "6"}, 
    # {"task": "subj", "pretrained": "checkpoints/gpt2_linear_classification_subj/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_subj/linear_finetuned.pt", "labels": "2"}, 
    # {"task": "mnli", "pretrained": "checkpoints/gpt2_linear_classification_mnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mnli/linear_finetuned.pt", "labels": "3"},
    # {"task": "axb", "pretrained": "checkpoints/gpt2_linear_classification_axb/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_axb/linear_finetuned.pt", "labels": "2"},  
    # {"task": "axg", "pretrained": "checkpoints/gpt2_linear_classification_axg/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_axg/linear_finetuned.pt", "labels": "2"},
    # {"task": "boolq", "pretrained": "checkpoints/gpt2_linear_classification_boolq/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_boolq/linear_finetuned.pt", "labels": "2"},
    # {"task": "cb", "pretrained": "checkpoints/gpt2_linear_classification_cb/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_cb/linear_finetuned.pt", "labels": "3"},
    # {"task": "copa", "pretrained": "checkpoints/gpt2_linear_classification_copa/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_copa/linear_finetuned.pt", "labels": "2"},
    # {"task": "pawsx", "pretrained": "checkpoints/gpt2_linear_classification_pawsx/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_pawsx/linear_finetuned.pt", "labels": "2"},
    # {"task": "xnli", "pretrained": "checkpoints/gpt2_linear_classification_xnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_xnli/linear_finetuned.pt", "labels": "3"},
    # {"task": "xwinograd", "pretrained": "checkpoints/gpt2_linear_classification_xwinograd/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_xwinograd/linear_finetuned.pt", "labels": "2"}
    {"task": "sst5", "pretrained": "nonlinear_checkpoints/sst5/gpt2_standard_classification_sst5/zeroshot_full_model.pt", "finetuned": "nonlinear_checkpoints/sst5/gpt2_standard_classification_sst5/epoch_checkpoint_full_model_1_1067.pt", "labels": "5"}
]

flag2 = 0
tasks2 = [
    # {"task": "cola", "pretrained": "checkpoints/gpt2_linear_classification_cola/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_cola/linear_finetuned.pt", "labels": "2"},
    # {"task": "cr", "pretrained": "checkpoints/gpt2_linear_classification_cr/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_cr/linear_finetuned.pt", "labels": "2"},
    # {"task": "mpqa", "pretrained": "checkpoints/gpt2_linear_classification_mpqa/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mpqa/linear_finetuned.pt", "labels": "2"},
    # {"task": "mr", "pretrained": "checkpoints/gpt2_linear_classification_mr/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mr/linear_finetuned.pt", "labels": "2"},
    # {"task": "mrpc", "pretrained": "checkpoints/gpt2_linear_classification_mrpc/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mrpc/linear_finetuned.pt", "labels": "2"},
    # {"task": "qnli", "pretrained": "checkpoints/gpt2_linear_classification_qnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_qnli/linear_finetuned.pt", "labels": "2"},
    # {"task": "snli", "pretrained": "checkpoints/gpt2_linear_classification_snli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_snli/linear_finetuned.pt", "labels": "3"},
    # {"task": "wnli", "pretrained": "checkpoints/gpt2_linear_classification_wnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_wnli/linear_finetuned.pt", "labels": "2"},
    # {"task": "qqp", "pretrained": "checkpoints/gpt2_linear_classification_qqp/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_qqp/linear_finetuned.pt", "labels": "2"},
    # {"task": "rte", "pretrained": "checkpoints/gpt2_linear_classification_rte/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_rte/linear_finetuned.pt", "labels": "2"}, 
    # {"task": "sst2", "pretrained": "checkpoints/gpt2_linear_classification_sst2/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_sst2/linear_finetuned.pt", "labels": "2"}, 
    # {"task": "sst5", "pretrained": "checkpoints/gpt2_linear_classification_sst5/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_sst5/linear_finetuned.pt", "labels": "5"}, 
    # {"task": "trec", "pretrained": "checkpoints/gpt2_linear_classification_trec/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_trec/linear_finetuned.pt", "labels": "6"}, 
    # {"task": "subj", "pretrained": "checkpoints/gpt2_linear_classification_subj/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_subj/linear_finetuned.pt", "labels": "2"}, 
    # {"task": "mnli", "pretrained": "checkpoints/gpt2_linear_classification_mnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_mnli/linear_finetuned.pt", "labels": "3"},
    # {"task": "axb", "pretrained": "checkpoints/gpt2_linear_classification_axb/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_axb/linear_finetuned.pt", "labels": "2"},  
    # {"task": "axg", "pretrained": "checkpoints/gpt2_linear_classification_axg/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_axg/linear_finetuned.pt", "labels": "2"},
    # {"task": "boolq", "pretrained": "checkpoints/gpt2_linear_classification_boolq/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_boolq/linear_finetuned.pt", "labels": "2"},
    # {"task": "cb", "pretrained": "checkpoints/gpt2_linear_classification_cb/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_cb/linear_finetuned.pt", "labels": "3"},
    # {"task": "copa", "pretrained": "checkpoints/gpt2_linear_classification_copa/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_copa/linear_finetuned.pt", "labels": "2"},
    # {"task": "pawsx", "pretrained": "checkpoints/gpt2_linear_classification_pawsx/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_pawsx/linear_finetuned.pt", "labels": "2"},
    # {"task": "xnli", "pretrained": "checkpoints/gpt2_linear_classification_xnli/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_xnli/linear_finetuned.pt", "labels": "3"},
    # {"task": "xwinograd", "pretrained": "checkpoints/gpt2_linear_classification_xwinograd/linear_zeroshot.pt", "finetuned": "checkpoints/gpt2_linear_classification_xwinograd/linear_finetuned.pt", "labels": "2"}
    {"task": "sst2", "pretrained": "nonlinear_checkpoints/sst2/gpt2_standard_classification_sst2/zeroshot_full_model.pt", "finetuned": "nonlinear_checkpoints/sst2/gpt2_standard_classification_sst2/checkpoint_full_model_2_0.pt", "labels": "2"}
]

for t1 in tasks1:
    task1 = t1["task"]
    pretrained_checkpoint_1 = t1["pretrained"]  
    finetuned_checkpoint_1 = t1["finetuned"]
    for t2 in tasks2:
        task2 = t2["task"]
        pretrained_checkpoint_2 = t2["pretrained"]
        finetuned_checkpoint_2 = t2["finetuned"]

        print(f"\nGetting {t1['task']} task vector similarity with task {t2['task']}")
        # Load task vector 1
        labels_1 = t1["labels"]
        try:
            task_vector_1 = (
                LinearizedTaskVector(pretrained_checkpoint_1, finetuned_checkpoint_1, labels_1)
                if args.finetuning_mode == "linear" and flag1 == 1
                else NonLinearTaskVector(pretrained_checkpoint_1, finetuned_checkpoint_1, labels_1)
            )
            # print(task_vector_1)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for task vector 1.")

        # Load task vector 2
        labels_2 = t2["labels"]
        try:
            task_vector_2 = (
                LinearizedTaskVector(pretrained_checkpoint_2, finetuned_checkpoint_2, labels_2)
                if args.finetuning_mode == "linear" and flag2 == 1
                else NonLinearTaskVector(pretrained_checkpoint_2, finetuned_checkpoint_2, labels_2)
            )
        except FileNotFoundError:
            print(f"Error: Could not find checkpoints for task vector 2.")

        # Run the analysis if both vectors are loaded
        if 'task_vector_1' in locals() and 'task_vector_2' in locals():
            analyze_task_vectors(task_vector_1, task_vector_2)
        else:
            print("Unable to perform analysis. Ensure both task vectors are properly loaded.")
    print("\n", "===============================================================================================================", "\n")
