# Preprocess NEO data and save to json files
import os
import pandas as pd
import json
import re

# Function to transform column names
def transform_neo_column(col_name):
    match = re.match(r"NEO(\d)_(\d+)", col_name)
    if match:
        first_num, second_num = map(int, match.groups())
        new_num = (first_num - 1) * 10 + second_num
        return f"NEO_{new_num}"
    return col_name  # Return unchanged if not matching

# Load data files
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move one folder up
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "NEO data"))

# Construct the paths
neo_data_path = os.path.join(data_dir, "NEO_data.csv")
data_dict_path = os.path.join(data_dir, "DataDict.csv")

neo_df = pd.read_csv(neo_data_path)
dictionary_df = pd.read_csv(data_dict_path)

# Extract relevant columns from the dictionary
question_mapping = dictionary_df.set_index("Column Name")["Question Text / Score Description"].to_dict()

# Extract only NEO<num>_<num> columns
# Create a dictionary for renaming only relevant columns
rename_dict = {col: transform_neo_column(col) for col in neo_df.columns if re.match(r"NEO\d+_\d+", col)}

# Update question_mapping keys to match the new column names
question_mapping = {rename_dict.get(k, k): v for k, v in question_mapping.items()}

# Reverse columns (for manually grading NEO)
reverse_columns = ['NEO_1', 'NEO_16', 'NEO_31', 'NEO_46', 'NEO_12', 'NEO_42', 'NEO_27', 'NEO_57', 'NEO_23', 'NEO_48',
                    'NEO_3', 'NEO_8', 'NEO_18', 'NEO_38', 'NEO_33', 'NEO_9', 'NEO_14', 'NEO_24', 'NEO_29',
                      'NEO_44', 'NEO_54', 'NEO_59', 'NEO_39', 'NEO_15', 'NEO_30', 'NEO_55', 'NEO_45']

# Rename the selected columns in the dataset
neo_df = neo_df.rename(columns=rename_dict)

# Extract relevant columns
neo_columns = list(rename_dict.values())

# Define the columns representing NEO scores
neo_score_columns = ["NEO_Openness", "NEO_Conscientiousness", "NEO_Extraversion", "NEO_Agreeableness", "NEO_Neuroticism"]

# Create output directory
output_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "users", "NEO"))
os.makedirs(output_dir, exist_ok=True)

# Process data per user
for cvdid, user_data in neo_df.groupby("CVDID"):

    formatted_user_id = f"user_{int(cvdid):04d}"  # Ensures zero-padded format (e.g., user_0001)

    user_json = {
        "user_id": cvdid,
        "transcript": None,  # Placeholder for now (can be updated later)
        "questions": {},
        "neo_scores": {}
    }
    
    # Process each NEO question
    for col in neo_columns:
        if col in question_mapping:
            question_text = question_mapping[col]
            responses = user_data[["wave", col]].dropna().to_dict("records")
            
            # Format responses properly
            answers = [
                {"wave": row['wave'], "response": int(row[col])}
                for row in responses if not pd.isna(row[col])
            ]
            
            if answers:
                if col not in user_json["questions"]:
                    user_json["questions"][col] = {
                        "question": question_text,
                        "reverse_scored": col in reverse_columns,  # ✅ Only checked ONCE when initializing
                        "answers": []
                }
                user_json["questions"][col]["answers"].extend(answers)
    
    # Process NEO aggregate scores
    for col in neo_score_columns:
        responses = user_data[["wave", col]].dropna()

        if not responses.empty:
            user_json["neo_scores"][col] = [
                {"wave": row["wave"], "response": float(row[col])}
                for _, row in responses.iterrows() if pd.notna(row[col])
            ]
    
    # Save JSON file
    output_path = os.path.join(output_dir, f"{formatted_user_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(user_json, f, indent=4)

    # Print completion message
    print(f"Done with user {str(cvdid).zfill(4)}")

print(f"JSON files saved in {output_dir}")
