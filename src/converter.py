import json
import pandas as pd

# 1. Load the source CSV file
csv_file_path = "merged_dataset_logging_cluster.csv"
df = pd.read_csv(csv_file_path)

# 2. Convert each row into an array of [id, abstract, question]
json_data = []
for index, row in df.iterrows():
    # Extract the values from each column
    #data_id = row["question_id"]
    abstract = row["abstract"]
    question = row["question"]
    
    # Pack them into a single list (Array) representing one record
    # Index 0 = abstract, Index 1 = question
    record = [abstract, question]
    
    # Append the record into the main list (creating a List of Lists)
    json_data.append(record)

# 3. Save the structured data into a JSON file
output_file_path = "data/acl-dataset/source_inputs/acl_source_inputs.json"
with open(output_file_path, "w", encoding="utf-8") as f:
    # ensure_ascii=False allows non-English characters to be saved properly instead of \uXXXX
    # indent=2 formats the JSON beautifully for human readability
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Conversion completed successfully! Source Input File saved at: {output_file_path}")
