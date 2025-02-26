# utils.py

import os
import json
import logging
import re

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)

def list_files(directory, extension=None):
    try:
        if extension:
            return [f for f in os.listdir(directory) if f.endswith(extension)]
        return [f for f in os.listdir(directory)]
    except Exception as e:
        logging.error(f"Error listing files in {directory}: {e}")
        return []

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return {}

def save_json(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")

def load_user_data(user_id):
    user_file = os.path.join(get_project_root(), 'data', 'users', 'NEO', f'user_{user_id}.json')
    return load_json(user_file)

def save_user_data(user_id, data):
    user_file = os.path.join(get_project_root(), 'data', 'users', 'NEO', f'user_{user_id}.json')
    save_json(data, user_file)

def get_max_latent_key(filled_latents):
    """
    Given a dictionary of filled_latents, return the key with the largest numerical ID.
    E.g., if keys are 'blank_latent_000', 'blank_latent_001', returns 'blank_latent_001'.
    """
    max_id = -1
    max_key = None
    for key in filled_latents.keys():
        match = re.search(r'(\d+)$', key)
        if match:
            num_id = int(match.group(1))
            if num_id > max_id:
                max_id = num_id
                max_key = key
    return max_key


def extract_neo_scores(llm_response, offset=0, wave=1):
    """
    Extracts NEO scores from the LLM response.

    Args:
        llm_response (str): The LLM-generated text containing NEO scores.
        wave (int): The wave number.

    Returns:
        dict: Dictionary of extracted scores in the required format.
    """
    trait_mapping = {
        "openness": "Your score in openness is",
        "conscientiousness": "Your score in conscientiousness is",
        "extraversion": "Your score in extraversion is",
        "agreeableness": "Your score in agreeableness is",
        "neuroticism": "Your score in neuroticism is"
    }

    scores = {}

    # Regex pattern to extract numbers after each trait phrase
    for trait, phrase in trait_mapping.items():
        match = re.search(rf"{re.escape(phrase)}\s+(\d+)", llm_response)
        if match:
            scores[trait] = int(match.group(1)) - offset # Adjusting to scale scores to e.g. 0-48

    return scores
