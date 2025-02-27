import os
import csv
import re

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.args import parse_arguments
from src.task_vectors_similarity_2 import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()


def apply_random_projection(task_vector, target_dim, device="cuda:0"):
    """
    Reduces the dimensionality of the task vector using random projection.

    Args:
        task_vector (np.ndarray): The high-dimensional task vector.
        target_dim (int): The target dimension for the reduced vector.

    Returns:
        np.ndarray: The reduced task vector.
    """
    # Initialize the random projection transformer
    # transformer = GaussianRandomProjection(n_components=target_dim)

    rng = torch.Generator(device=device)

    # rng = check_random_state(None)

    # Reshape task vector to 2D 
    task_vector_reshaped = task_vector.reshape(1, -1)
    torch_task_vector = torch.from_numpy(task_vector_reshaped).float().to(device)

    print("task vector ka shape", task_vector_reshaped.shape)
    burst_factor = 32

    projected_components = []

    for dim in tqdm(range(target_dim // burst_factor)):
        projection_matrix = torch.normal(
            0.0,
            1.0 / np.sqrt(target_dim),
            size=(task_vector_reshaped.shape[1], burst_factor),
            generator=rng,
            device=device,
        )
        projected_components.append(torch.matmul(torch_task_vector, projection_matrix).squeeze(0))

    torch.cuda.synchronize()

    final_projection = torch.empty(target_dim)
    """
    proj_comp = [ [burst_factor] ]
    """

    for burst in range(target_dim // burst_factor):
        final_projection[burst * burst_factor : (burst + 1) * burst_factor].copy_(
            projected_components[burst]
        )

    # reduced_vector = (projected_components).T

    return final_projection.numpy()


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
def analyze_task_vectors(task_vector_1, task_vector_2, target_dim=1 << 13):
    # Convert task vectors to numpy arrays
    vector_1 = task_vector_to_numpy(task_vector_1)
    vector_2 = task_vector_to_numpy(task_vector_2)

    # Add explicit GPU device to distribute compute load
    reduced_vector_1 = apply_random_projection(vector_1, target_dim=target_dim)
    reduced_vector_2 = apply_random_projection(vector_2, target_dim=target_dim)

    print(f"Original Dimension: {vector_1.shape}, Reduced Dimension: {reduced_vector_1.shape}")
    print(f"Original Dimension: {vector_2.shape}, Reduced Dimension: {reduced_vector_2.shape}")

    # Cosine Similarity
    cosine_sim = compute_cosine_similarity(reduced_vector_1, reduced_vector_2)
    print(f"Cosine similarity: {cosine_sim}")

    # Euclidean Distance
    # euclidean_dist = compute_euclidean_distance(vector_1, vector_2)
    # print(f"Euclidean distance: {euclidean_dist}")

    # Dot Product
    dot_prod = compute_dot_product(reduced_vector_1, reduced_vector_2)
    print(f"Dot product: {dot_prod}")

    # Symmetry Check (Difference Vector)
    # difference_vector = vector_1 - vector_2
    # print(f"Difference vector norm: {np.linalg.norm(difference_vector)}")

    # Cosine similarity of the difference vector to zero
    # diff_cosine_sim = compute_cosine_similarity(difference_vector, np.zeros_like(difference_vector))
    # print(f"Difference vector cosine similarity to zero: {diff_cosine_sim}")

    return cosine_sim, dot_prod


# For task vector 1
#set these flags to 1, when working with linearFT saved models.
flag1 = 0
flag2 = 0


def extract_task_checkpoints(
    rx_type, rx_pretraining, rx_finetuning, checkpoint_folder="./checkpoints/"
):
    tasks = []
    pretraining = []
    finetuning = []

    for task in os.listdir(checkpoint_folder):
        if re.search(rx_type, task):
            tasks.append(task)

    for task in tasks:
        pt_lex_large = None
        ft_lex_large = None

        for t_pretraining in os.listdir(os.path.join(checkpoint_folder, task)):
            if re.search(rx_pretraining, t_pretraining):
                pt_lex_large = t_pretraining

        for t_finetuning in os.listdir(os.path.join(checkpoint_folder, task)):
            if re.search(rx_finetuning, t_finetuning):
                ft_lex_large = t_finetuning

        pretraining.append(pt_lex_large)
        if ft_lex_large == None:
            print(task)
        if pt_lex_large == None:
            print(task)
        finetuning.append(ft_lex_large)
    labels = [2] * len(tasks)
    task_meta = [
        {
            "task": tasks[i][::-1][0 : tasks[i][::-1].find("_")],
            "pretrained": os.path.join(checkpoint_folder, tasks[i], pretraining[i]),
            "finetuned": os.path.join(checkpoint_folder, tasks[i], finetuning[i]),
            "labels": labels[i],
        }
        for i in range(len(tasks))
    ]

    for meta in task_meta:
        if None in meta.values():
            raise Exception("One of more fields are empty")

    return task_meta


tasks1 = extract_task_checkpoints(
    "standard_classification_", "zeroshot_full_model.pt", "epoch_checkpoint_full_model"
)

n = len(tasks1)

# Define split to process on current GPU device
# tasks1 = tasks1[: n // 3] 

tasks2 = tasks1

data = [["Task-1", "Task-2", "Cosine-Similarity", "Dot-Product"]]

target_dim_list = [2048, 4096, 8192, 16384, 32768, 65536]

for target_dim_curr in target_dim_list:
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
            # print("LOAD task vector 1")
            labels_1 = t1["labels"]
            # print(args.num_labels)
            try:
                task_vector_1 = (
                    LinearizedTaskVector(pretrained_checkpoint_1, finetuned_checkpoint_1, labels_1)
                    if args.finetuning_mode == "linear" and flag1 == 1
                    else NonLinearTaskVector(
                        pretrained_checkpoint_1, finetuned_checkpoint_1, labels_1
                    )
                )
                # print(task_vector_1)
            except FileNotFoundError:
                print("Error: Could not find checkpoints for task vector 1.")

            # Load task vector 2
            # print("LOAD task vector 2")
            labels_2 = t2["labels"]
            # print(args.num_labels)
            try:
                task_vector_2 = (
                    LinearizedTaskVector(pretrained_checkpoint_2, finetuned_checkpoint_2, labels_2)
                    if args.finetuning_mode == "linear" and flag2 == 1
                    else NonLinearTaskVector(
                        pretrained_checkpoint_2, finetuned_checkpoint_2, labels_2
                    )
                )
            except FileNotFoundError:
                print("Error: Could not find checkpoints for task vector 2.")

            # Run the analysis if both vectors are loaded
            if "task_vector_1" in locals() and "task_vector_2" in locals():
                cs, dp = analyze_task_vectors(task_vector_1, task_vector_2, target_dim_curr)
                data.append([task1, task2, cs, dp])
            else:
                print("Unable to perform analysis. Ensure both task vectors are properly loaded.")
        print(
            "\n",
            "===============================================================================================================",
            "\n",
        )

    with open(
        f"projections_standard_nonlinear_similarity_results/target_dim_0_{target_dim_curr}.csv", "w"
    ) as f:
        csv.writer(f).writerows(data)
