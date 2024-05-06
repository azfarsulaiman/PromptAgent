import os
import re
import pandas as pd

# Base directory containing the subdirectories with log files
base_directory = '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/train_log'

# Regular expression to capture child node and reward information
pattern = re.compile(r"child_node (\d+) \(reward:([\d\.]+)")

# Create a Pandas Excel writer using XlsxWriter as the engine
excel_file = os.path.join(base_directory, 'All_Rewards_Tabs.xlsx')
writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

print("Starting to process directories...")

# Loop through each subdirectory in the base directory
for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)
    if os.path.isdir(folder_path):  # Confirm it's a directory
        print(f"Processing folder: {folder}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".log"):  # Filter for log files
                filepath = os.path.join(folder_path, filename)
                print(f"Processing file: {filename}")
                with open(filepath, 'r') as file:
                    lines = file.readlines()

                # Clean the filename for display and extract percentage
                base_filename = os.path.splitext(filename)[0]
                cleaned_filename = re.sub(r'log_|\.log-train.*$', '', base_filename)
                percentage_match = re.search(r'_(\d+)$', cleaned_filename)
                dataset_percentage = percentage_match.group(1) if percentage_match else 'Unknown'

                # Extract and store child node and reward data along with filename and percentage
                records = []
                for line in lines:
                    match = pattern.search(line)
                    if match:
                        records.append((cleaned_filename, dataset_percentage, match.group(1), match.group(2)))

                # Write to Excel if records found
                if records:
                    df = pd.DataFrame(records, columns=['File Name', 'Dataset_Percentage', 'Child_Node', 'Reward'])
                    df.to_excel(writer, sheet_name=cleaned_filename, index=False)
                    print(f"Data written to tab: {cleaned_filename}")

# Close the Excel writer to save the file
writer.close()
print(f"All data has been consolidated and written to {excel_file}")
