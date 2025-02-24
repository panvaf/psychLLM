# generate_heatmaps.py

import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add the src directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from utils import get_project_root, load_json, list_files

def extract_time_step(blank_latent_id):
    """
    Extracts the numerical part from 'blank_latent_###'.
    E.g., 'blank_latent_001' -> 1
    """
    match = re.search(r'blank_latent_(\d+)', blank_latent_id)
    if match:
        return int(match.group(1))
    else:
        return None

def get_sorted_blank_latent_ids(filled_latent_dict):
    """
    Returns a sorted list of blank_latent_ids based on their numerical suffix.
    """
    sorted_ids = sorted(
        filled_latent_dict.keys(),
        key=lambda x: extract_time_step(x) if extract_time_step(x) is not None else -1
    )
    return sorted_ids

def generate_heatmap(user_id, user_data, output_dir):
    """
    Generates and saves a heatmap for a single user.
    """
    filled_latents = user_data.get('filled_latent', {})
    if not filled_latents:
        print(f"No filled_latent data for User: {user_id}. Skipping.")
        return

    # Sort blank_latent_ids
    sorted_blank_latent_ids = get_sorted_blank_latent_ids(filled_latents)
    if not sorted_blank_latent_ids:
        print(f"No valid blank_latent_ids for User: {user_id}. Skipping.")
        return

    # Collect all questions across all question banks
    question_banks = user_data.get('question_banks', {})
    if not question_banks:
        print(f"No question_banks found for User: {user_id}. Skipping.")
        return

    # Assuming questions are consistent across question banks
    # Initialize a set to store unique question IDs
    question_ids = set()
    for bank in question_banks.values():
        questions = bank.get('questions', {})
        question_ids.update(questions.keys())

    question_ids = sorted(question_ids)  # Sort for consistent ordering

    # Initialize a DataFrame
    kl_data = pd.DataFrame(index=question_ids, columns=sorted_blank_latent_ids)

    # Populate the DataFrame with KL divergence values
    for blank_latent_id in sorted_blank_latent_ids:
        for bank in question_banks.values():
            questions = bank.get('questions', {})
            for question_id in question_ids:
                question_data = questions.get(question_id, {})
                latents = question_data.get('latents', {})
                kl_divergence = latents.get(blank_latent_id, {}).get('kl_divergence', None)
                kl_data.at[question_id, blank_latent_id] = kl_divergence

    # Convert all KL divergence values to float (handle None)
    kl_data = kl_data.astype(float)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        kl_data,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        cbar_kws={'label': 'KL Divergence'},
        linewidths=.5,
        linecolor='gray'
    )
    plt.title(f'KL Divergence Over Time for {user_id}')
    plt.xlabel('Blank Latent ID (Time Step)')
    plt.ylabel('Question ID')
    plt.tight_layout()

    # Save the plot
    user_output_dir = os.path.join(output_dir)
    os.makedirs(user_output_dir, exist_ok=True)
    heatmap_path = os.path.join(user_output_dir, f'{user_id}_kl_divergence_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved for User: {user_id} at {heatmap_path}")

def main():
    # Define paths
    project_root = get_project_root()
    users_dir = os.path.join(project_root, "data", "users")
    output_dir = os.path.join(project_root, "data", "heatmaps", "linear")
    os.makedirs(output_dir, exist_ok=True)

    # List all user JSON files
    user_files = list_files(users_dir, extension=".json")
    if not user_files:
        print("No user JSON files found in 'data/users/'. Exiting.")
        return

    for user_file in user_files:
        user_id_full = os.path.splitext(user_file)[0]  # e.g., 'user_001'
        user_id = user_id_full.split('_')[1]  # Extract '001' from 'user_001'
        user_json_path = os.path.join(users_dir, user_file)

        # Load the user data
        user_data = load_json(user_json_path)
        if not user_data:
            print(f"User data for {user_id_full} could not be loaded. Skipping.")
            continue

        # Generate heatmap for the user
        generate_heatmap(user_id_full, user_data, output_dir)

    print("All heatmaps generated successfully.")

if __name__ == "__main__":
    main()
