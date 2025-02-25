import sys

# Prompt templates

def build_prompt(user_data, prompt_type):
    """
    Main function to build prompts dynamically based on the specified type.
    This function acts as a switch and calls the appropriate prompt builder.
    """
    PROMPT_FUNCTIONS = {
        "direct": score_from_rubric,
        "NEO": predict_NEO
    }

    if prompt_type not in PROMPT_FUNCTIONS:
        print(f"⚠️ Warning: Unrecognized prompt type '{prompt_type}'. Defaulting to 'simple_question'.")
        sys.exit(1)


    return PROMPT_FUNCTIONS[prompt_type](user_data)




def score_from_rubric(user_data, wave=1):
    """
    Build a prompt for the LLM to calculate NEO scores based on rubric-specified questions.
    
    Args:
        user_data (dict): The user JSON data.
        wave (int): The wave number to use (default: 1).
    
    Returns:
        str: The generated prompt for the LLM.
    """

    rubric = {
        "neuroticism": ["NEO_1", "NEO_11", "NEO_16", "NEO_31", "NEO_46", "NEO_6", "NEO_21", "NEO_26", "NEO_36", "NEO_41", "NEO_51", "NEO_56"],
        "extraversion": ["NEO_7", "NEO_12", "NEO_37", "NEO_42", "NEO_2", "NEO_17", "NEO_27", "NEO_57", "NEO_22", "NEO_32", "NEO_47", "NEO_52"],
        "openness": ["NEO_13", "NEO_23", "NEO_43", "NEO_48", "NEO_53", "NEO_58", "NEO_3", "NEO_8", "NEO_18", "NEO_38", "NEO_28", "NEO_33"],
        "agreeableness": ["NEO_9", "NEO_14", "NEO_19", "NEO_24", "NEO_29", "NEO_44", "NEO_54", "NEO_59", "NEO_4", "NEO_34", "NEO_39", "NEO_49"],
        "conscientiousness": ["NEO_5", "NEO_10", "NEO_15", "NEO_30", "NEO_55", "NEO_25", "NEO_35", "NEO_60", "NEO_20", "NEO_40", "NEO_45", "NEO_50"]
    }

    responses_by_latent = {latent: [] for latent in rubric}  # Initialize lists

    for latent, questions in rubric.items():
        for question in questions:
            if question in user_data["questions"]:  # Ensure question exists
                for answer in user_data["questions"][question]["answers"]:
                    if answer["wave"] == wave:  # Check the correct wave
                        # Reverse the response if needed
                        response_value = 6 - answer["response"] if user_data["questions"][question].get("reverse_scored", False) else answer["response"]
                        responses_by_latent[latent].append(response_value)

    # Generate the formatted prompt
    prompt = f"""You are an assistant evaluating NEO scores based on user responses. Each response is rated from 1 to 5.

Here are the responses grouped by personality traits and wave {wave}:
"""

    for latent, responses in responses_by_latent.items():
        formatted_responses = ", ".join(map(str, responses)) if responses else 0
        prompt += f"\n**{latent.capitalize()}**\n{formatted_responses}\n"

    prompt += "\nPlease sum the responses for each trait to arrive at the NEO score for that trait."

    return prompt

