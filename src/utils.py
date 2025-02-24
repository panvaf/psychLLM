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
    user_file = os.path.join(get_project_root(), 'data', 'users', f'user_{user_id}.json')
    return load_json(user_file)

def save_user_data(user_id, data):
    user_file = os.path.join(get_project_root(), 'data', 'users', f'user_{user_id}.json')
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
