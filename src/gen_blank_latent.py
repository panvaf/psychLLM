# gen_blank_latent.py

import os
import sys
import json
import logging
import re
import time  # For measuring time taken by certain operations
from together import Together  # Ensure you have the Together API client installed and configured
from utils import (
    get_project_root,
    ensure_directory_exists,
    load_user_data,
    load_json,
    list_files,
    get_max_latent_key
)

# Configure logging
def setup_logging():
    """
    Sets up logging to file and console.
    """
    project_root = get_project_root()
    log_dir = os.path.join(project_root, "logs")
    ensure_directory_exists(log_dir)
    log_file = os.path.join(log_dir, "gen_blank_latent.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def construct_aggregated_prompt(users_data, k=5):
    """
    Constructs a comprehensive system prompt for all users.

    Args:
        users_data (dict): Dictionary where each key is a user_id_full and value is another dict containing:
                           - filled_latent
                           - top_k_questions (list of tuples)
                           - bottom_k_questions (list of tuples)
        k (int): Number of top and bottom questions to include per user.

    Returns:
        str: The constructed aggregated system prompt.
    """
    logging.info("Constructing aggregated prompt for all users.")
    prompt = f"""
You are an AI assistant specialized in template generation aimed at minimizing KL Divergence between user responses and latent templates when filled out.

Your task is to infer and generate a new blank latent template that better captures the attributes necessary to explain user answers effectively. The goal is to reduce the KL Divergence, indicating a better fit between the template and the user responses.

You have been provided with data from multiple users. Analyze the provided information to generate a revised blank latent template that addresses the shortcomings identified by high KL divergence questions and reinforces strengths indicated by low KL divergence questions.

**Guidelines:**
1. For each user:
   a. Analyze the top {k} poorly predicted questions to identify missing or underrepresented attributes in the current template.
   b. Review the bottom {k} well-predicted questions to reinforce effective attributes.
   c. Synthesize the insights to draft a revised blank latent template.
   d. Ensure the template is concise, comprehensive, and structured in a way that facilitates accurate filling.

**Desired Output:**
Provide a single template that may be filled out by future users that isolates latent variables most relevant to the data being analyzed.
---

**New Blank Latent Template:**
Your name is ____ and
1. You are ___ percentile in [LATENT]. This is exemplified by: [FILL IN]
2. You are ___ percentile in [LATENT]. This is exemplified by: [FILL IN]
3. You are ___ percentile in [LATENT]. This is exemplified by: [FILL IN]
4. You are ___ percentile in [LATENT]. This is exemplified by: [FILL IN]
5. You are ___ percentile in [LATENT]. This is exemplified by: [FILL IN]
...
---

You may only modify the [LATENT] token. You may not modify the other text. Do not justify your response.
**Example Response:**

---
Your name is ____ and
1. You are ___ percentile in openness. This is exemplified by: [FILL IN]
2. You are ___ percentile in extraversion. This is exemplified by: [FILL IN]
3. You are ___ percentile in conscientiousness. This is exemplified by: [FILL IN]
4. You are ___ percentile in agreeableness. This is exemplified by: [FILL IN]
5. You are ___ percentile in neuroticism. This is exemplified by: [FILL IN]
...
---
"""

    for user_id_full, data in users_data.items():
        filled_latent = data['filled_latent']
        top_k = data['top_k_questions']
        bottom_k = data['bottom_k_questions']

        user_prompt = f"""
**Filled Latent for {user_id_full}:**
{filled_latent}

**Poorly Predicted Questions and KL Divergences:**
"""
        for idx, (question, kl_div) in enumerate(top_k, 1):
            user_prompt += f"{idx}. \"{question}\" - KL Divergence: {kl_div}\n"

        user_prompt += "\n**Best Predicted Questions and KL Divergences:**\n"
        for idx, (question, kl_div) in enumerate(bottom_k, 1):
            user_prompt += f"{idx}. \"{question}\" - KL Divergence: {kl_div}\n"

        # Insert user-specific prompt into the aggregated prompt
        prompt += user_prompt

    logging.info("Aggregated prompt construction completed.")
    return prompt


def gen_blank_latent_aggregated(users_data, client, k=5):
    """
    Generate a new blank latent template for all users using the Together API in a single aggregated call.

    Args:
        users_data (dict): Dictionary where each key is a user_id_full and value is another dict containing:
                           - filled_latent
                           - top_k_questions (list of tuples)
                           - bottom_k_questions (list of tuples)
        client (Together): The Together API client instance.
        k (int): Number of top and bottom questions to include per user.

    Returns:
        str: The generated blank latent template.
    """
    logging.info("Starting the generation of a new blank latent template for all users.")
    prompt = construct_aggregated_prompt(users_data, k)

    # Optionally, save the prompt to a file for debugging
    # with open("aggregated_prompt.txt", "w") as f:
    #     f.write(prompt)

    try:
        logging.info("Making API call to Together API for template generation.")
        start_time = time.time()
        # Create a chat completion request
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Use the specified model
            messages=[
                {"role": "user", "content": prompt}
            ],
            logprobs=1,  # Retrieve logprobs for each token
            max_tokens=5000  # Adjust based on the expected response length and API limits
        )
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"API Prompt: {prompt[:500]}...")  # Log first 500 chars for brevity

        # Extract the generated response text
        response_text = response.choices[0].message.content.strip()
        logging.info(f"AI Response: {response_text[:500]}...")  # Log first 500 chars for brevity

        return response_text

    except Exception as e:
        logging.error(f"An error occurred while generating blank latent: {e}")
        return ""


def main():
    setup_logging()

    logging.info("Starting gen_blank_latent.py script.")

    # Determine the project root
    project_root = get_project_root()

    # Initialize the Together API client
    try:
        client = Together()  # Ensure your Together API client is correctly configured
        logging.info("Together API client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Together API client: {e}")
        sys.exit(1)

    # Define paths
    users_dir = os.path.join(project_root, "data", "users")

    # Get list of user JSON files
    user_files = list_files(users_dir, extension=".json")
    if not user_files:
        logging.error("No user JSON files found in 'data/users/'. Exiting.")
        return
    logging.info(f"Found {len(user_files)} user JSON files to process.")

    users_data = {}  # Dictionary to hold data for all users

    for user_file in user_files:
        user_id_full = os.path.splitext(user_file)[0]  # e.g., 'user_001'
        user_id_parts = user_id_full.split('_')
        if len(user_id_parts) != 2:
            logging.warning(f"Invalid user_id format in filename: {user_file}. Skipping.")
            continue
        user_id_num = user_id_parts[1]  # Extract '001' from 'user_001'
        user_json_path = os.path.join(users_dir, user_file)

        logging.info(f"Processing User: {user_id_full} ({user_json_path})")

        # Load the user data
        user_data = load_user_data(user_id_num)
        if not user_data:
            logging.warning(f"User data for {user_id_full} could not be loaded. Skipping.")
            continue

        # Get the filled_latent data
        filled_latents = user_data.get('filled_latent', {})
        if not filled_latents:
            logging.warning(f"No filled_latent found for user {user_id_full}. Skipping user.")
            continue

        # Find the filled_latent with the largest key ID
        max_latent_key = get_max_latent_key(filled_latents)
        if not max_latent_key:
            logging.warning(f"No valid filled_latent keys found for user {user_id_full}. Skipping user.")
            continue

        # Get the latent content
        latent_content = filled_latents[max_latent_key].get('full_text', '')
        if not latent_content:
            logging.warning(f"Latent content is empty for user {user_id_full}, latent {max_latent_key}. Skipping user.")
            continue

        logging.info(f"Collected filled_latent for user {user_id_full}.")

        # Iterate over each question bank
        question_banks = user_data.get('question_banks', {})
        if not question_banks:
            logging.warning(f"No question banks found for user {user_id_full}. Skipping user.")
            continue

        # Collect all questions and their KL divergences across all question banks
        kl_divergences = []
        for question_bank_name, question_bank in question_banks.items():
            questions = question_bank.get('questions', {})
            for question_id, question_data in questions.items():
                # Get the response from the 'transcript' section
                transcript_data = question_data.get('transcript', {})
                kl_divergence = question_data.get('latents', {}).get(max_latent_key, {}).get('kl_divergence', None)

                if kl_divergence is not None:
                    question_text = question_data.get('question', '')
                    kl_divergences.append((question_text, kl_divergence))
                else:
                    logging.warning(f"No KL divergence found for User: {user_id_full}, Question: {question_id}. Skipping question.")

        if not kl_divergences:
            logging.warning(f"No KL divergence data available for user {user_id_full}. Skipping user.")
            continue

        logging.info(f"Collected {len(kl_divergences)} questions with KL divergence for user {user_id_full}.")

        # Sort questions based on KL divergence
        kl_divergences_sorted = sorted(kl_divergences, key=lambda x: x[1], reverse=True)

        # Select top 5 poorly predicted and bottom 5 well-predicted questions
        top_k = 5
        bottom_k = 5
        top_questions = kl_divergences_sorted[:top_k]
        bottom_questions = kl_divergences_sorted[-bottom_k:]

        logging.info(f"Selected top {len(top_questions)} poorly predicted and bottom {len(bottom_questions)} well-predicted questions for user {user_id_full}.")

        # Aggregate data for the user
        users_data[user_id_full] = {
            "filled_latent": latent_content,
            "top_k_questions": top_questions,
            "bottom_k_questions": bottom_questions
        }

    if not users_data:
        logging.error("No valid user data found to process. Exiting.")
        return

    logging.info(f"Aggregating data for {len(users_data)} users.")

    # Generate a new blank latent template for all users in a single API call
    new_template = gen_blank_latent_aggregated(users_data, client, k=5)

    if not new_template:
        logging.error("Failed to generate new blank latent template. Exiting.")
        return

    logging.info("Assigning new blank latent template.")

    # Prepare the data to be saved
    new_blank_latent_data = {
        "full_text": new_template,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Define the output path

    # Ensure max_latent_key is a string and formatted as expected
    print(f"max latent key: {max_latent_key}")
    max_latent_key_number = max_latent_key.split('_')[-1]

    if not isinstance(max_latent_key_number, str) or not max_latent_key_number.isdigit():
        raise ValueError(f"Invalid max_latent_key_number: {max_latent_key_number}. Must be a string of digits like '000', '001', etc.")

    # Increment the key and preserve zero-padding
    new_latent_key = str(int(max_latent_key_number) + 1).zfill(len(max_latent_key_number))

    # Construct the output path
    output_dir = os.path.join(project_root, "data", "blank_latents")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Create the file name and path
    output_path = os.path.join(output_dir, f"blank_latent_{new_latent_key}.json")

    # Save the aggregated response to a single JSON file
    try:
        with open(output_path, "w") as outfile:
            json.dump(new_blank_latent_data, outfile, indent=4)
        logging.info(f"Saved new blank latent template to '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save new blank latent template: {e}")

    logging.info("gen_blank_latent.py script completed successfully.")


if __name__ == "__main__":
    main()
