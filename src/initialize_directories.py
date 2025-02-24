# initialize_directories.py

import os
import logging
import shutil
import json

def process_question_banks(question_banks_txt_dir):
    """
    Processes question bank text files and returns a dictionary of question banks.
    """
    question_banks = {}
    files_processed = 0
    if not os.path.exists(question_banks_txt_dir):
        logging.error(f"Question banks directory does not exist: {question_banks_txt_dir}")
        return question_banks

    for filename in os.listdir(question_banks_txt_dir):
        if filename.startswith('question_bank_') and filename.endswith('.txt'):
            input_path = os.path.join(question_banks_txt_dir, filename)
            try:
                with open(input_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                if len(lines) < 2:
                    logging.warning(f"Skipping {filename}: File does not contain enough lines.")
                    continue
                # Extract instructions and questions
                instructions = lines[0].strip()
                questions_list = [line.strip() for line in lines[1:] if line.strip()]
                # Get question bank name without extension
                question_bank_name = os.path.splitext(filename)[0]
                # Create the question bank dictionary
                question_bank = {
                    "instructions": instructions,
                    "questions": {}
                }
                # Populate the questions dictionary
                for idx, question_text in enumerate(questions_list):
                    question_id = f"question_{idx}"
                    question_bank['questions'][question_id] = {
                        "question": question_text
                    }
                # Add to question_banks dictionary
                question_banks[question_bank_name] = question_bank
                logging.info(f"Processed question bank: {filename}")
                files_processed += 1
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
    if files_processed == 0:
        logging.info(f"No valid question bank files were processed in {question_banks_txt_dir}.")
    else:
        logging.info(f"Successfully processed {files_processed} question bank(s).")
    return question_banks

def process_transcripts(transcripts_dir, users_dir, question_banks):
    """
    Processes transcript files and updates user JSON files with transcripts and question banks.
    """
    if not os.path.exists(transcripts_dir):
        logging.error(f"Transcripts directory does not exist: {transcripts_dir}")
        return

    transcripts = [
        f for f in os.listdir(transcripts_dir)
        if os.path.isfile(os.path.join(transcripts_dir, f)) and f.startswith('transcript_') and f.endswith('.txt')
    ]

    if not transcripts:
        logging.warning(f"No transcript files found in {transcripts_dir}.")
        return

    files_processed = 0

    for transcript_file in transcripts:
        transcript_path = os.path.join(transcripts_dir, transcript_file)
        logging.debug(f"Processing file: {transcript_path}")

        # Extract user ID
        try:
            base_name = os.path.splitext(transcript_file)[0]
            user_id = base_name.split('_')[1]
            if not user_id.isdigit():
                raise ValueError(f"Invalid user ID in filename: {transcript_file}")
        except (IndexError, ValueError) as e:
            logging.warning(f"Skipping file '{transcript_file}': {e}")
            continue

        # Read transcript content
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read()

        # Create or update user JSON file
        user_json_path = os.path.join(users_dir, f"user_{user_id}.json")
        if os.path.exists(user_json_path):
            # Load existing user data
            with open(user_json_path, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            # Append transcript content if not already present
            if 'transcripts' in user_data:
                if transcript_content not in user_data['transcripts']:
                    user_data['transcripts'].append(transcript_content)
            else:
                user_data['transcripts'] = [transcript_content]
        else:
            # Create new user data
            user_data = {"transcripts": [transcript_content]}

        # Merge question banks into user data
        if 'question_banks' not in user_data:
            user_data['question_banks'] = {}

        for qb_name, qb_data in question_banks.items():
            if qb_name not in user_data['question_banks']:
                # Add the question bank to the user's data
                user_data['question_banks'][qb_name] = qb_data
            else:
                # Update the instructions
                user_data['question_banks'][qb_name]['instructions'] = qb_data['instructions']
                # Merge questions
                for q_id, q_data in qb_data['questions'].items():
                    if q_id not in user_data['question_banks'][qb_name]['questions']:
                        user_data['question_banks'][qb_name]['questions'][q_id] = q_data
                    else:
                        # Do not overwrite existing question data (may contain responses)
                        pass  # Or handle as needed

        # Save user JSON
        with open(user_json_path, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=4)

        logging.info(f"Created/Updated user JSON file: {user_json_path}")
        files_processed += 1

    if files_processed == 0:
        logging.info(f"No valid transcripts were processed in {transcripts_dir}.")
    else:
        logging.info(f"Successfully processed {files_processed} transcript(s).")

def process_blank_latent(blank_latent_init_dir, blank_latents_dir):
    """
    Processes the blank latent file, converts it to JSON format, and saves it.
    """
    if not os.path.exists(blank_latent_init_dir):
        logging.error(f"Blank latent initialization directory does not exist: {blank_latent_init_dir}")
        return

    blank_latent_files = [
        f for f in os.listdir(blank_latent_init_dir)
        if os.path.isfile(os.path.join(blank_latent_init_dir, f))
    ]

    if len(blank_latent_files) != 1:
        logging.error(f"Ensure 'blank_latent_init' contains exactly one file.")
        return

    initial_blank_latent = blank_latent_files[0]
    src_path = os.path.join(blank_latent_init_dir, initial_blank_latent)
    dest_json_path = os.path.join(blank_latents_dir, 'blank_latent_000.json')

    try:
        with open(src_path, 'r', encoding='utf-8') as file:
            content = file.read()
        lines = content.strip().split('\n')
        # Create the JSON structure
        blank_latent_data = {
            "full_text": content.strip(),
            "lines": {}
        }
        for idx, line in enumerate(lines):
            line_id = f"line_{idx}"
            blank_latent_data["lines"][line_id] = line
        # Save the JSON file
        with open(dest_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(blank_latent_data, json_file, indent=4)
        logging.info(f"Processed and saved blank latent as JSON: {dest_json_path}")
    except Exception as e:
        logging.error(f"Error processing blank latent file '{initial_blank_latent}': {e}")

def initialize_directories():
    """
    Main function to initialize directories and process files.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Change to logging.DEBUG for detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logging.debug(f"Determined project root: {project_root}")

    # Define the required directories
    data_dir = os.path.join(project_root, 'data')
    users_dir = os.path.join(data_dir, 'users')
    blank_latents_dir = os.path.join(data_dir, 'blank_latents')

    transcripts_dir = os.path.join(project_root, 'input', 'transcripts')
    blank_latent_init_dir = os.path.join(project_root, 'input', 'blank_latent_init')
    question_banks_txt_dir = os.path.join(project_root, 'input', 'question_banks_txt')

    # Ensure data directories exist
    os.makedirs(users_dir, exist_ok=True)
    os.makedirs(blank_latents_dir, exist_ok=True)
    logging.info(f"Ensured directories exist: {users_dir}, {blank_latents_dir}")

    # Process question banks
    question_banks = process_question_banks(question_banks_txt_dir)

    # Process transcripts and update user JSON files
    process_transcripts(transcripts_dir, users_dir, question_banks)

    # Process and convert blank latent to JSON
    process_blank_latent(blank_latent_init_dir, blank_latents_dir)

if __name__ == "__main__":
    initialize_directories()
