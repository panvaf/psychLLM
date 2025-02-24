# get_answers.py

import os
import json
import logging
from together import Together
from utils import (
    get_project_root,
    ensure_directory_exists,
    load_user_data,
    save_user_data,
    list_files
)

# Configure logging
def setup_logging():
    project_root = get_project_root()
    log_dir = os.path.join(project_root, "logs")
    ensure_directory_exists(log_dir)
    log_file = os.path.join(log_dir, "get_answers.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_answers(input_text, instructions, question, identifier1, identifier2, client):
    """
    Generate answers using the Together API for a given input and set of questions.
    """
    prompt = f"""
Your task is to parse the following transcript and pretend to be the patient described. After you do this, you will read the questionnaire instructions and follow them exactly.

Transcript:
{input_text}

Questionnaire Instructions:
{instructions}

Question:
{question}
"""

    try:
        # Create a chat completion request
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            logprobs=1,
            max_tokens=150
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
            logging.warning(f"No logprobs available for {identifier1} & {identifier2}.")

        logging.info(f"Generated response for User: {identifier1}, Question Bank: {identifier2}")
        return {
            "response": filled_content,
            "tokens": generated_tokens,
            "token_logprobs": generated_token_logprobs
        }

    except Exception as e:
        logging.error(f"An error occurred while generating the response for User: {identifier1}, Question Bank: {identifier2}: {e}")
        return {}

def main():
    setup_logging()

    # Determine the project root
    project_root = get_project_root()

    # Initialize the Together API client
    client = Together()

    # Define paths
    users_dir = os.path.join(project_root, "data", "users")

    # Get list of user JSON files
    user_files = list_files(users_dir, extension=".json")
    if not user_files:
        logging.error("No user JSON files found in 'data/users/'. Exiting.")
        return

    for user_file in user_files:
        user_id = os.path.splitext(user_file)[0]  # e.g., 'user_001'
        user_json_path = os.path.join(users_dir, user_file)

        # Load the user data
        user_data = load_user_data(user_id.split('_')[1])  # Extract '001' from 'user_001'
        if not user_data:
            logging.warning(f"User data for {user_id} could not be loaded. Skipping.")
            continue

        # Load and aggregate all transcripts for the current user
        transcripts = user_data.get('transcripts', [])
        if not transcripts:
            logging.warning(f"No transcripts found for user {user_id}. Skipping user.")
            continue

        total_transcript = "\n".join(transcripts)

        # Ensure 'question_banks' in user_data
        if 'question_banks' not in user_data:
            logging.warning(f"No question banks found in user data for {user_id}. Skipping.")
            continue

        # Iterate over each question bank
        for question_bank_name, question_bank in user_data['question_banks'].items():
            instructions = question_bank["instructions"]
            questions_dict = question_bank["questions"]

            logging.info(f"Processing Question Bank: {question_bank_name} for User: {user_id}")

            # Iterate over each question in the question list
            for question_id, question_data in questions_dict.items():
                # Generate identifiers
                identifier1 = user_id  # e.g., 'user_001'
                identifier2 = question_bank_name  # e.g., 'question_bank_001'

                # Check if the question already has a response under 'transcript'
                if 'transcript' in question_data and 'response' in question_data['transcript'] and question_data['transcript']['response']:
                    logging.info(f"Transcript response already exists for {identifier1}, {identifier2}, {question_id}. Skipping.")
                    continue

                question_text = question_data['question']

                # Call the get_answers function to get the response from the LLM
                response_data = get_answers(
                   input_text=total_transcript,
                   instructions=instructions,
                   question=question_text,
                   identifier1=identifier1,
                   identifier2=identifier2,
                   client=client
                )

                if response_data:
                    # Ensure 'transcript' dictionary exists
                    if 'transcript' not in question_data:
                        question_data['transcript'] = {}
                    # Store the response in the 'transcript' section
                    question_data['transcript']['response'] = response_data.get("response", "")
                    question_data['transcript']['tokens'] = response_data.get("tokens", [])
                    question_data['transcript']['token_logprobs_transcript'] = response_data.get("token_logprobs", [])
                else:
                    logging.warning(f"No response data for User: {identifier1}, Question Bank: {identifier2}, Question: {question_text}")

            # Save the updated user data back to the user JSON file
            save_user_data(user_id.split('_')[1], user_data)
            logging.info(f"Saved responses for Question Bank: {question_bank_name} to {user_json_path}")

if __name__ == "__main__":
    main()
