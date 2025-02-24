# evolooption.py

import os
import subprocess
import re
import logging
import sys
from utils import get_project_root, list_files, load_json, get_max_latent_key, load_user_data

def get_max_blank_latent_key_dir(blank_latents_dir):
    """
    Scans the blank_latents_dir for files named 'blank_latent_###.json',
    extracts the numeric parts, and returns the key with the highest number.
    
    Args:
        blank_latents_dir (str): Path to the blank_latents directory.
    
    Returns:
        str: The highest latent key in the format 'blank_latent_###'.
             Defaults to 'blank_latent_000' if no valid files are found.
    """
    # List all JSON files in the directory matching the pattern
    pattern = re.compile(r'blank_latent_(\d+)\.json$')
    blank_latent_files = list_files(blank_latents_dir, extension=".json")
    
    max_num = -1
    max_key = 'blank_latent_000'  # Default value
    
    for file in blank_latent_files:
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                max_key = f'blank_latent_{str(num).zfill(3)}'
    
    if max_num == -1:
        logging.info("No existing blank_latent files found. Starting with 'blank_latent_000'.")
    else:
        logging.info(f"Found max latent key: {max_key}")
    
    return max_key

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Get project root
    try:
        project_root = get_project_root()
        src_path = os.path.join(project_root, 'src')
    except Exception as e:
        logging.error(f"Error determining project root: {e}")
        sys.exit(1)

    # Define paths to scripts and directories
    blank_latents_dir = os.path.join(project_root, 'data', 'blank_latents')
    users_dir = os.path.join(project_root, 'data', 'users')  # Assuming user data is in data/users/
    fill_script = os.path.join(src_path, 'fill_latents.py')
    compute_script = os.path.join(src_path, 'compute_latent_logits.py')
    gen_script = os.path.join(src_path, 'gen_blank_latent.py')

    # Verify that the scripts exist
    for script_path in [fill_script, compute_script, gen_script]:
        if not os.path.isfile(script_path):
            logging.error(f"Required script not found: {script_path}")
            sys.exit(1)

    # Verify that the blank_latents and users directories exist
    if not os.path.isdir(blank_latents_dir):
        logging.error(f"Blank latents directory does not exist: {blank_latents_dir}")
        sys.exit(1)

    if not os.path.isdir(users_dir):
        logging.error(f"Users directory does not exist: {users_dir}")
        sys.exit(1)

    # Define the number of iterations (e.g., 10 times)
    num_iterations = 3  # Change to 10 for production

    # Repeat the process for the specified number of iterations
    for iteration in range(1, num_iterations + 1):
        logging.info(f"--- Iteration {iteration}/{num_iterations} ---")

        # Determine the current max_latent_key by scanning blank_latents_dir
        current_latent_key = get_max_blank_latent_key_dir(blank_latents_dir)

        logging.info(f"Using latent key: {current_latent_key}")

        # Run fill_latents.py with current_latent_key
        try:
            logging.info(f"Running fill_latents.py with key: {current_latent_key}")
            subprocess.run(['python', fill_script, current_latent_key], check=True)
            logging.info("fill_latents.py executed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing fill_latents.py: {e}")
            sys.exit(1)

        # Run compute_latent_logits.py with current_latent_key
        try:
            logging.info(f"Running compute_latent_logits.py with key: {current_latent_key}")
            subprocess.run(['python', compute_script, current_latent_key], check=True)
            logging.info("compute_latent_logits.py executed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing compute_latent_logits.py: {e}")
            sys.exit(1)

        # Run gen_blank_latent.py
        try:
            logging.info("Running gen_blank_latent.py")
            subprocess.run(['python', gen_script], check=True)
            logging.info("gen_blank_latent.py executed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing gen_blank_latent.py: {e}")
            sys.exit(1)

        logging.info(f"--- Iteration {iteration}/{num_iterations} Completed ---\n")

    logging.info(f"evolooption.py script completed successfully after {num_iterations} iterations.")

if __name__ == "__main__":
    main()
