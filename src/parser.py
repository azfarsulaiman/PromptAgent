import os
import re
import csv

# Base directory containing the subdirectories with log files
base_directory = '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/train_log'

# Regular expression to capture child node and reward information
pattern = re.compile(r"child_node (\d+) \(reward:([\d\.]+)")

# List to store all records
all_records = []

# Loop through each subdirectory in the base directory
for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)
    if os.path.isdir(folder_path):  # Confirm it's a directory
        for filename in os.listdir(folder_path):
            if filename.endswith(".log"):  # Filter for log files
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()

                # Remove prefix 'log_' and suffix '.log-train-*'
                base_filename = os.path.splitext(filename)[0]  # Remove the file extension
                base_filename = base_filename.replace('log_', '')  # Remove the 'log_' prefix
                base_filename = re.sub(r'\.log-train.*$', '', base_filename)  # Remove the '.log-train-*' pattern

                # Extract and store child node and reward data with formatted filename
                for line in lines:
                    match = pattern.search(line)
                    if match:
                        all_records.append((base_filename, match.group(1), match.group(2)))

# Write all collected data to a single CSV file
csv_file = os.path.join(base_directory, 'all_child_nodes_rewards.csv')
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Log_File', 'Child_Node', 'Reward'])  # Column headers
    writer.writerows(all_records)  # Write all rows at once

print(f"All data has been consolidated and written to {csv_file}")
