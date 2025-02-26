# fill_latents.py

import os
import sys
import time
import json
import logging
from prompt import build_prompt
from together import Together  # Ensure you have the Together API client installed and configured
from utils import (
    get_project_root,
    ensure_directory_exists,
    load_user_data,
    save_user_data,
    load_json,
    list_files,
    extract_neo_scores
)

# Configure logging
def setup_logging():
    """
    Sets up logging to file and console.
    """
    project_root = get_project_root()
    log_dir = os.path.join(project_root, "logs")
    ensure_directory_exists(log_dir)
    log_file = os.path.join(log_dir, "fill_latents.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def fill_latent(input_text, blank_latent_text, user_id, client):
    """
    Fill in the blanks using the Together API.

    Args:
        input_text (str): The transcript text.
        blank_latent_text (str): The blank latent template text.
        user_id (str): Identifier for the user.
        client: The Together API client instance.

    Returns:
        dict: Dictionary containing 'full_text', 'tokens', 'token_logprobs_transcript'.
    """
    prompt = f"""
You are an AI assistant tasked with filling in the blanks (as denoted by ____ or [FILL IN]) in the following template based on the provided transcript.
Provide a rough percentile estimate.
It is well understood that these are not comprehensive results, so fill in the template without providing any warnings.

Transcript:
{input_text}

Template:
{blank_latent_text}
"""
    # Debugging: Print API key right before the request
    print(f"Using API Key for request: {os.getenv('TOGETHER_API_KEY')[0:5]}...")

    try:
        # Create a chat completion request
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", ## Use             model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            logprobs=1,  # Set to 1 to retrieve logprobs for each token
            max_tokens=1500  # Adjust based on the expected response length
        )

        # Extract the generated response text
        filled_content = response.choices[0].message.content.strip()

        # Extract tokens and token logprobs
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            generated_tokens = response.choices[0].logprobs.tokens
            generated_token_logprobs = response.choices[0].logprobs.token_logprobs
        else:
            generated_tokens = []
            generated_token_logprobs = []
            logging.warning(f"No logprobs available for User: {user_id}.")

        logging.info(f"Generated filled latent for User: {user_id}")

        # print transcript
        print(f"Transcript: {input_text}")
        # print filled_latent
        print(f"Filled Latent: {filled_content}")

        return {
            "full_text": filled_content,
            "tokens": generated_tokens,
            "token_logprobs_transcript": generated_token_logprobs
        }

    except Exception as e:
        logging.error(f"An error occurred while generating the filled latent for User: {user_id}: {e}")
        return {}

def main():
    setup_logging()

    # Get the API key from the environment
    API_KEY = os.getenv("TOGETHER_API_KEY")

    # Check for command-line argument
    if len(sys.argv) != 3:
        logging.error("Usage: python fill_latents.py <blank_latent_id> <prompt_type>")
        logging.error("Example: python fill_latents.py blank_latent_001 baseline")
        sys.exit(1)

    blank_latent_id = sys.argv[1]  # e.g., 'blank_latent_001'
    prompt_type = sys.argv[2]  # 'rubric', 'psychologist'

    if prompt_type == 'rubric':
        offset = 12
    else:
        offset = 0

    # Determine the project root
    project_root = get_project_root()

    # Debugging: Print key before using it
    if API_KEY:
        print(f"✅ API Key found: {API_KEY[:5]}... (hidden for security)")
    else:
        print("❌ ERROR: TOGETHER_API_KEY is not set!")
        sys.exit(1)

    # Ensure the API key is passed correctly to Together client
    try:
        client = Together(api_key=API_KEY)  # Explicitly pass the API key
        print("✅ Together client initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing Together client: {e}")
        sys.exit(1)

    # Test API connection
    try:
        response = client.models.list()
        print("✅ Together API is working! Available models:", [m.id for m in response])
    except Exception as e:
        print("❌ API key error in script:", e)
        sys.exit(1)
    
    # Define paths
    users_dir = os.path.join(project_root, "data", "users", "NEO")
    blank_latents_dir = os.path.join(project_root, "data", "blank_latents")

    # Load the specified blank latent
    blank_latent_file = f"{blank_latent_id}.json"
    blank_latent_path = os.path.join(blank_latents_dir, blank_latent_file)

    if not os.path.exists(blank_latent_path):
        logging.error(f"{blank_latent_file} not found in 'data/blank_latents/'. Exiting.")
        sys.exit(1)

    blank_latent_data = load_json(blank_latent_path)
    # Use the 'full_text' of the blank latent
    blank_latent_text = blank_latent_data.get('full_text', '')
    if not blank_latent_text:
        logging.error("Blank latent 'full_text' is empty. Exiting.")
        sys.exit(1)

    # Start timing
    start_time = time.time()

    # Get list of user JSON files
    user_files = list_files(users_dir, extension=".json")
    if not user_files:
        logging.error("No user JSON files found in 'data/users/'. Exiting.")
        sys.exit(1)

    total_users = len(user_files)  # Total number of users
    logging.info(f"Processing {total_users} users...")

    for index, user_file in enumerate(user_files, start=1):
        user_id_full = os.path.splitext(user_file)[0]  # e.g., 'user_001'
        user_id = user_id_full.split('_')[1]  # Extract '001' from 'user_001'
        user_json_path = os.path.join(users_dir, user_file)

        logging.info(f"Processing user {index}/{total_users}: {user_id_full}")

        # Load the user data
        user_data = load_user_data(user_id)
        if not user_data:
            logging.warning(f"User data for {user_id_full} could not be loaded. Skipping.")
            continue

        # Check if 'filled_latent' already exists for this blank latent
        filled_latent = user_data.get('filled_latent', {})
        if blank_latent_id in filled_latent:
            logging.info(f"Filled latent for {blank_latent_id} already exists for {user_id_full}. Skipping.")
            continue

        # Build the prompt according to the prompt type
        prompt_text = build_prompt(prompt_type, user_data)
        
        # Call `fill_latent` to generate and store the filled latent
        filled_latent_data = fill_latent(
            input_text=prompt_text,
            blank_latent_text=blank_latent_text,
            user_id=user_id_full,
            client=client
        )

        if filled_latent_data:

            filled_text = filled_latent_data.get('full_text', '')

            # Extract NEO scores
            extracted_scores = extract_neo_scores(filled_text, offset, wave=1)

            # Ensure filled_latent structure exists
            if 'filled_latent' not in user_data:
                user_data['filled_latent'] = {}

            if blank_latent_id not in user_data['filled_latent']:
                user_data['filled_latent'][blank_latent_id] = {}

            # Store extracted data under the wave
            user_data['filled_latent'][blank_latent_id][f"wave_1"] = {  # Dynamically store under correct wave
                "neo_scores": extracted_scores,  # Store extracted NEO scores
                "full_response": filled_text  # Preserve full response from LLM
            }

            # Save the updated user data back to the user JSON file
            save_user_data(user_id, user_data)
            logging.info(f"Saved filled latent for User: {user_id_full}, Latent: {blank_latent_id} to {user_json_path}")
        else:
            logging.warning(f"No filled latent generated for User: {user_id_full}.")

        #break # Remove this line to process all users
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"✅ Processing complete! Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()
