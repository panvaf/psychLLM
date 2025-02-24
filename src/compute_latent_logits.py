# compute_latent_logits.py

import os
import sys
import json
import logging
from scipy.stats import entropy
from math import exp
from together import Together  # Ensure you have the Together API client installed and configured
from utils import (
    get_project_root,
    ensure_directory_exists,
    load_user_data,
    save_user_data,
    list_files,
    
)

# Configure logging
def setup_logging():
    """
    Sets up logging to file and console.
    """
    project_root = get_project_root()
    log_dir = os.path.join(project_root, "logs")
    ensure_directory_exists(log_dir)
    log_file = os.path.join(log_dir, "compute_latent_logits.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def convert_logprobs_to_probs(logprobs, epsilon=1e-10):
    """
    Convert a list of log probabilities to a normalized probability distribution,
    adding epsilon to prevent zero probabilities.

    Args:
        logprobs (list): List of log probabilities.
        epsilon (float): Small value to add to probabilities to prevent zeros.

    Returns:
        list: List of normalized probabilities, or None if normalization fails.
    """
    try:
        probs = [exp(lp) for lp in logprobs]
        probs = [p + epsilon for p in probs]  # Add epsilon to each probability
        total = sum(probs)
        if total == 0:
            logging.error("Sum of probabilities is zero after adding epsilon. Cannot normalize.")
            return None
        normalized_probs = [p / total for p in probs]
        return normalized_probs
    except Exception as e:
        logging.error(f"Error converting logprobs to probs: {e}")
        return None

def compute_kl_divergence_logprobs(p_logp, q_logp, epsilon=1e-10):
    """
    Compute the KL divergence between two probability distributions given their log probabilities.

    Args:
        p_logp (list): Log probabilities from the transcript (true distribution).
        q_logp (list): Log probabilities from the latent (approximate distribution).
        epsilon (float): Small value to add to probabilities to prevent zeros.

    Returns:
        float: KL divergence value, or None if computation fails.
    """
    try:
        # Convert log probabilities to probabilities with epsilon
        p = convert_logprobs_to_probs(p_logp, epsilon)
        q = convert_logprobs_to_probs(q_logp, epsilon)
        
        if p is None or q is None:
            logging.error("Probability conversion failed. Cannot compute KL divergence.")
            return None

        # Compute KL divergence: KL(p || q)
        kl_div = entropy(p, q)
        return kl_div
    except Exception as e:
        logging.error(f"Error computing KL divergence: {e}")
        return None
    

def compute_latent_logits(latent_content, answer, instructions, question, question_id, question_tokens_length, identifier1, identifier2, client):
    """
    Compute log probabilities for the provided answer using the Together API.

    Args:
        latent_content (str): The filled latent content containing the 'full_text'.
        answer (str): The answer to compute log probabilities for.
        instructions (str): Instructions for the questionnaire.
        question (str): A single questionnaire question.
        question_tokens_length (int): The number of tokens in the existing response's tokens.
        identifier1 (str): Identifier for the user.
        identifier2 (str): Identifier for the filled latent.
        client (Together): Initialized Together API client.

    Returns:
        dict: Dictionary containing 'tokens', 'token_logprobs'.
    """
    prompt = f"""
Your task is to parse the following filled latent and pretend to be the patient described.
After assuming the role of the patient, you will take the questionnaire that is provided 
and respond to each question as accurately as possible. Do not add additional details or speculation beyond the instructions in the questionnaire. 

Filled Latent:
{latent_content}

Questionnaire Instructions:
{instructions}

Question:
{question}
"""

    try:
        # Create a chat completion request
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Absolutely DO NOT MODIFY THE MODEL!
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ],
            logprobs=1,      # Retrieve logprobs for each token
            max_tokens=0,    # Do not modify this code; this makes the LLM not generate anything except the end token
            echo=True        # Do not modify this code; this makes the LLM believe it generated the content we force fed it
        )

        # Extract the assistant's response text
        # assistant_response = response.prompt[0].message.content.strip()  # Do not modify this section; we swapped out choice for .prompt on purpose

        logging.debug(f"API Response: {response}")

        ## Get the end token and add it.
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            tokens_end = response.choices[0].logprobs.tokens  
            token_logprobs_end = response.choices[0].logprobs.token_logprobs  
        else:
            logging.warning(f"No end token available for User: {identifier1}, Filled Latent: {identifier2}.")
            return {}
        # Extract tokens and token logprobs
        if hasattr(response.prompt[0], 'logprobs') and response.prompt[0].logprobs:  # Do not modify this section; we did this to ensure that we strip the probabilities of the forcefed prompt.
            tokens = response.prompt[0].logprobs.tokens
            token_logprobs = response.prompt[0].logprobs.token_logprobs
            tokens = tokens + tokens_end
            token_logprobs = token_logprobs + token_logprobs_end
            # Cuts off the tokens so we only see the tokens of the answer as desired.
            # We need to get the last N tokens, where N is question_tokens_length
            tokens = tokens[-question_tokens_length:]  # Adjusted code
            token_logprobs = token_logprobs[-question_tokens_length:]  # Adjusted code

            logging.info(f"Computed log probabilities for User: {identifier1}, Filled Latent: {identifier2}, Question: {question_id}")
            return {
                # "response": filled_content,
                "tokens": tokens,
                "token_logprobs": token_logprobs
            }
        else:
            logging.warning(f"No logprobs available for User: {identifier1}, Filled Latent: {identifier2}, Question: {question_id}.")
            return {}

    except Exception as e:
        logging.error(f"An error occurred while computing logprobs for User: {identifier1}, Filled Latent: {identifier2}, Question: {question_id}: {e}")
        return {}

def main():
    setup_logging()

    # Check for command-line argument
    if len(sys.argv) != 2:
        logging.error("Usage: python compute_latent_logits.py <blank_latent_id>")
        logging.error("Example: python compute_latent_logits.py blank_latent_001")
        sys.exit(1)

    blank_latent_id = sys.argv[1]  # e.g., 'blank_latent_001'

    # Determine the project root
    project_root = get_project_root()

    # Initialize the Together API client
    client = Together()  # Ensure your Together API client is correctly configured

    # Define paths
    users_dir = os.path.join(project_root, "data", "users")

    # Get list of user JSON files
    user_files = list_files(users_dir, extension=".json")
    if not user_files:
        logging.error("No user JSON files found in 'data/users/'. Exiting.")
        return

    for user_file in user_files:
        user_id_full = os.path.splitext(user_file)[0]  # e.g., 'user_001'
        user_id = user_id_full.split('_')[1]  # Extract '001' from 'user_001'
        user_json_path = os.path.join(users_dir, user_file)

        # Load the user data
        user_data = load_user_data(user_id)
        if not user_data:
            logging.warning(f"User data for {user_id_full} could not be loaded. Skipping.")
            continue

        # Get the filled_latent data
        filled_latents = user_data.get('filled_latent', {})
        if not filled_latents:
            logging.warning(f"No filled_latent found for user {user_id_full}. Skipping user.")
            continue


        # Get the latent content
        latent_content = filled_latents[blank_latent_id].get('full_text', '')
        if not latent_content:
            logging.warning(f"Latent content is empty for user {user_id_full}, latent {blank_latent_id}. Skipping user.")
            continue

        # Iterate over each question bank
        question_banks = user_data.get('question_banks', {})
        if not question_banks:
            logging.warning(f"No question banks found for user {user_id_full}. Skipping user.")
            continue

        for question_bank_name, question_bank in question_banks.items():
            instructions = question_bank.get('instructions', '')
            questions = question_bank.get('questions', {})

            if not questions:
                logging.warning(f"No questions found in question bank {question_bank_name} for user {user_id_full}. Skipping question bank.")
                continue

            logging.info(f"Processing Question Bank: {question_bank_name} for User: {user_id_full}")

            # Iterate over each question
            for question_id, question_data in questions.items():
                question_text = question_data.get('question', '')
                # Get the response from the 'transcript' section
                transcript_data = question_data.get('transcript', {})
                existing_response = transcript_data.get('response', '')
                existing_tokens = transcript_data.get('tokens', [])
                existing_logprob_tokens = transcript_data.get('token_logprobs_transcript', [])

                if not existing_response:
                    logging.warning(f"No existing transcript response found for question {question_id} in question bank {question_bank_name} for user {user_id_full}. Skipping question.")
                    continue

                logging.info(f"Processing Question: {question_id} for User: {user_id_full}")

                # Compute log probabilities for the response
                result = compute_latent_logits(
                    latent_content=latent_content,
                    answer=existing_response,
                    instructions=instructions,
                    question=question_text,
                    question_id = question_id,
                    question_tokens_length=len(existing_tokens),
                    identifier1=user_id_full,
                    identifier2=blank_latent_id,
                    client=client
                )

                if result:
                    # Initialize 'latents' dictionary if it doesn't exist
                    if 'latents' not in question_data:
                        question_data['latents'] = {}

                    # Define saved variables 
                    result_tokens = result.get('tokens', [])
                    result_token_logprobs = result.get('token_logprobs', [])

                    # Add the results under the blank_latent_id
                    question_data['latents'][blank_latent_id] = {
                        "tokens": result_tokens,
                        "token_logprobs_answer": result_token_logprobs
                    }
                    # Compute KL divergence and add it

                    # Tons of checks first. These should never be activated, but really, we do this for sanity.
                    if len(existing_logprob_tokens) != len(result_token_logprobs):
                        logging.warning(f"Logprobs length mismatch for User: {user_id_full}, Question: {question_id}. Skipping KL divergence computation.")
                        continue

                    # Compute KL divergence: KL(p || q) where p is transcript logprobs and q is answer logprobs
                    kl_divergence = compute_kl_divergence_logprobs(existing_logprob_tokens, result_token_logprobs)

                    if kl_divergence is not None:
                        # Store the KL divergence in the latents section
                        question_data['latents'][blank_latent_id]["kl_divergence"] = kl_divergence
                        logging.info(f"Computed KL divergence for User: {user_id_full}, Question: {question_id}: {kl_divergence}")
                    else:
                        logging.warning(f"KL divergence could not be computed for User: {user_id_full}, Question: {question_id}.")
                else:
                    logging.warning(f"No logprobs computed for User: {user_id_full}, Latent: {blank_latent_id}, Question: {question_id}")

            # Save the updated user data back to the user JSON file
            save_user_data(user_id, user_data)
            logging.info(f"Saved updated data with latent log probabilities for Question Bank: {question_bank_name} to {user_json_path}")

if __name__ == "__main__":
    main()
