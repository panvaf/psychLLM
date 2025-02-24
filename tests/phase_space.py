# generate_phase_space.py

import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np

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

def prepare_phase_space_data(users_dir):
    """
    Prepare data for phase space visualization.
    Extracts pairs of consecutive KL divergence values (KL(t), KL(t+1)) for each question/user,
    along with the initial KL(t=0) value for color encoding.
    """
    data = []

    # List all user JSON files
    user_files = list_files(users_dir, extension=".json")
    if not user_files:
        print("No user JSON files found in 'data/users/'. Exiting.")
        return pd.DataFrame()

    for user_file in user_files:
        user_id_full = os.path.splitext(user_file)[0]  # e.g., 'user_001'
        user_id = user_id_full.split('_')[1]  # Extract '001' from 'user_001'
        user_json_path = os.path.join(users_dir, user_file)

        # Load the user data
        user_data = load_json(user_json_path)
        if not user_data:
            print(f"User data for {user_id_full} could not be loaded. Skipping.")
            continue

        # Extract KL divergence data
        question_banks = user_data.get('question_banks', {})
        if not question_banks:
            continue

        filled_latents = user_data.get('filled_latent', {})
        if not filled_latents:
            continue

        # Get sorted blank_latent_ids
        sorted_blank_latent_ids = get_sorted_blank_latent_ids(filled_latents)
        if len(sorted_blank_latent_ids) < 2:
            print(f"Not enough time steps for User: {user_id_full}. Skipping.")
            continue

        # Iterate over each question in all question banks
        for bank in question_banks.values():
            questions = bank.get('questions', {})
            for question_id, question_data in questions.items():
                latents = question_data.get('latents', {})
                if len(latents) < 2:
                    continue  # Need at least two time steps to form a pair

                # Sort the latent_ids to ensure chronological order
                sorted_latent_ids = sorted(
                    latents.keys(),
                    key=lambda x: extract_time_step(x) if extract_time_step(x) is not None else -1
                )

                # Extract initial KL(t=0) for color encoding
                initial_latent_id = sorted_latent_ids[0]
                kl_initial = latents.get(initial_latent_id, {}).get('kl_divergence', None)
                if kl_initial is None:
                    print(f"Missing initial KL divergence for User: {user_id_full}, Question: {question_id}. Skipping.")
                    continue

                try:
                    kl_initial_log = np.log(kl_initial)
                except ValueError:
                    print(f"Non-positive initial KL divergence for User: {user_id_full}, Question: {question_id}. Skipping.")
                    continue

                # Iterate through consecutive pairs
                for i in range(len(sorted_latent_ids) - 1):
                    kl_t = latents.get(sorted_latent_ids[i], {}).get('kl_divergence', None)
                    kl_t_plus_1 = latents.get(sorted_latent_ids[i + 1], {}).get('kl_divergence', None)

                    if kl_t is not None and kl_t_plus_1 is not None:
                        # Apply logarithmic transformation
                        try:
                            kl_t_log = np.log(kl_t)
                            kl_t_plus_1_log = np.log(kl_t_plus_1)
                        except ValueError:
                            # Handle log(0) or negative values if any
                            print(f"Non-positive KL divergence encountered for User: {user_id_full}, Question: {question_id} at time steps {sorted_latent_ids[i]} and {sorted_latent_ids[i + 1]}. Skipping this pair.")
                            continue

                        data.append({
                            'user_id': user_id_full,
                            'question_id': question_id,
                            'blank_latent_t': sorted_latent_ids[i],
                            'blank_latent_t_plus_1': sorted_latent_ids[i + 1],
                            'kl_t_log': kl_t_log,
                            'kl_t_plus_1_log': kl_t_plus_1_log,
                            'kl_initial_log': kl_initial_log
                        })

    return pd.DataFrame(data)

def plot_phase_space(data, output_dir):
    """
    Plot the phase space graph showing KL(t) vs KL(t+1), colored by initial KL(t=0).
    The legend is a continuous color bar representing the initial KL(t=0) values.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(style="whitegrid")

    # Create scatter plot without a legend
    scatter = sns.scatterplot(
        data=data,
        x='kl_t_log',
        y='kl_t_plus_1_log',
        hue='kl_initial_log',
        palette='viridis',
        alpha=0.6,
        edgecolor=None,
        legend=False,  # Disable the default legend
        ax=ax
    )

    # Reference line y = x
    max_val = max(data['kl_t_log'].max(), data['kl_t_plus_1_log'].max())
    min_val = min(data['kl_t_log'].min(), data['kl_t_plus_1_log'].min())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x (No Change)')

    ax.set_xlabel('Log(KL Divergence) at Time t')
    ax.set_ylabel('Log(KL Divergence) at Time t+1')
    ax.set_title('Phase Space Plot: KL Divergence Transition')

    # Create a ScalarMappable for the color bar
    norm = plt.Normalize(data['kl_initial_log'].min(), data['kl_initial_log'].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib

    # Add the color bar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Log(KL Divergence) at t=0')

    # Add the reference line to the legend
    ax.legend(loc='best')

    plt.tight_layout()

    # Save the plot
    phase_space_path = os.path.join(output_dir, 'phase_space_plot.png')
    plt.savefig(phase_space_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Phase space plot saved at {phase_space_path}")

def plot_phase_space_per_user(data, output_dir):
    """
    Plot separate phase space graphs for each user, colored by initial KL(t=0).
    The legend is a continuous color bar representing the initial KL(t=0) values.
    """
    users = data['user_id'].unique()
    for user in users:
        user_data = data[data['user_id'] == user]
        if user_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set(style="whitegrid")

        # Create scatter plot without a legend
        scatter = sns.scatterplot(
            data=user_data,
            x='kl_t_log',
            y='kl_t_plus_1_log',
            hue='kl_initial_log',
            palette='viridis',
            alpha=0.6,
            edgecolor=None,
            legend=False,  # Disable the default legend
            ax=ax
        )

        # Reference line y = x
        max_val = max(user_data['kl_t_log'].max(), user_data['kl_t_plus_1_log'].max())
        min_val = min(user_data['kl_t_log'].min(), user_data['kl_t_plus_1_log'].min())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x (No Change)')

        ax.set_xlabel('Log(KL Divergence) at Time t')
        ax.set_ylabel('Log(KL Divergence) at Time t+1')
        ax.set_title(f'Phase Space Plot for {user}')

        # Create a ScalarMappable for the color bar
        norm = plt.Normalize(user_data['kl_initial_log'].min(), user_data['kl_initial_log'].max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib

        # Add the color bar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Log(KL Divergence) at t=0')

        # Add the reference line to the legend
        ax.legend(loc='best')

        plt.tight_layout()

        # Save the plot
        phase_space_path = os.path.join(output_dir, f'phase_space_{user}.png')
        plt.savefig(phase_space_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Phase space plot saved for {user} at {phase_space_path}")

def main():
    # Define paths
    project_root = get_project_root()
    users_dir = os.path.join(project_root, "data", "users")
    output_dir = os.path.join(project_root, "data", "phase_space_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    phase_space_data = prepare_phase_space_data(users_dir)
    if phase_space_data.empty:
        print("No data available for phase space visualization. Exiting.")
        return

    # Plot overall phase space
    plot_phase_space(phase_space_data, output_dir)

    # Optionally, plot per-user phase space graphs
    plot_phase_space_per_user(phase_space_data, output_dir)

    print("All phase space plots generated successfully.")

if __name__ == "__main__":
    main()
