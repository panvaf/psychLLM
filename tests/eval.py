# Evaluate different ways to extract NEO scores from LLMs

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

# Whether to save figures
SAVE_FIGS = True

# Load data files
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define directory containing user JSON files
users_dir = os.path.abspath(os.path.join(script_dir, "..", "data", 'users', "NEO"))

# Initialize lists to store extracted data
traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

data = {trait: {"ground_truth": [], "rubric_pred": [], "direct_pred": []} for trait in traits}

# Load and extract data
for user_file in os.listdir(users_dir):
    if user_file.endswith(".json"):
        user_path = os.path.join(users_dir, user_file)
        with open(user_path, "r") as f:
            user_data = json.load(f)

        # Extract ground truth scores for wave 1
        if "neo_scores" in user_data:
            for trait in traits:
                trait_key = f"NEO_{trait.capitalize()}"
                wave_1_score = next(
                    (entry["response"] for entry in user_data["neo_scores"].get(trait_key, []) if entry["wave"] == "1.0"),
                    None
                )
                if wave_1_score is not None:
                    data[trait]["ground_truth"].append(wave_1_score)

        # Extract rubric-based LLM predictions (blank_latent_NEO_noexp)
        filled_latent = user_data.get("filled_latent", {})
        rubric_prediction = filled_latent.get("blank_latent_NEO_noexp", {}).get("wave_1", {}).get("neo_scores", {})
        for trait in traits:
            rubric_score = rubric_prediction.get(trait, None)
            if rubric_score is not None:
                data[trait]["rubric_pred"].append(rubric_score)

        # Extract direct LLM predictions (blank_latent_NEO)
        direct_prediction = filled_latent.get("blank_latent_NEO", {}).get("wave_1", {}).get("neo_scores", {})
        for trait in traits:
            direct_score = direct_prediction.get(trait, None)
            if direct_score is not None:
                data[trait]["direct_pred"].append(direct_score)

# Function to plot scatterplots with regression line
def plot_scatter(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(6, 6))
    sns.regplot(x=x, y=y, scatter_kws={"s": 10, "alpha": 0.3}, line_kws={"color": "red"})  # 👈 Smaller & transparent
    
    # Fit a linear model and compute regression coefficient
    model = LinearRegression()
    x_np = np.array(x).reshape(-1, 1)
    y_np = np.array(y)
    model.fit(x_np, y_np)
    r2_score = model.score(x_np, y_np)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nR² = {r2_score:.2f}")
    
    # Save the figure
    if SAVE_FIGS:
        plt.savefig(filename, dpi=300)  # High-resolution saving
    else:
        plt.show()  # Show plot instead of saving
    plt.close()

# Generate scatterplots for each trait
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for trait in traits:
    if len(data[trait]["ground_truth"]) > 0 and len(data[trait]["rubric_pred"]) > 0:
        plot_scatter(
            x=data[trait]["ground_truth"],
            y=data[trait]["rubric_pred"],
            title=f"Ground Truth vs Rubric-Based Prediction ({trait.capitalize()})",
            xlabel="Ground Truth Score",
            ylabel="Rubric-Based Prediction",
            filename=os.path.join(output_dir, f"{trait}_rubric_vs_ground.png")
        )

    if len(data[trait]["ground_truth"]) > 0 and len(data[trait]["direct_pred"]) > 0:
        plot_scatter(
            x=data[trait]["ground_truth"],
            y=data[trait]["direct_pred"],
            title=f"Ground Truth vs Direct LLM Prediction ({trait.capitalize()})",
            xlabel="Ground Truth Score",
            ylabel="Direct LLM Prediction",
            filename=os.path.join(output_dir, f"{trait}_direct_vs_ground.png")
        )

print(f"Scatterplots saved in {output_dir}/")