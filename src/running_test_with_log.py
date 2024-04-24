import os
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

def get_optimized_prompt(train_log_file):
    with open('/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/optimized_prompts_dict.json', 'r') as file:
        data = json.load(file)
        try:
            best_prompt = data[train_log_file]['prompt']
        except KeyError:
            print(f"Warning: No entry found for '{train_log_file}' in JSON data.")
            return None  # Return from inside the except block if key error occurs
        return best_prompt  # Return from inside the with block, aligned with try-except



directory = '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/train_log'  # Replace with the actual directory path
files = os.listdir(directory)
for file in files:
    optimized_prompt = get_optimized_prompt(file)
    print(optimized_prompt)

    file_elements = file.split('_')
    this_train_size = file_elements[-2]
    this_eval_size = file_elements[-1]
    dataset = '_'.join(file_elements[1:-2])
    if(dataset == 'counting'):
        dataset = 'object_counting'
    dataset_path = f"../datasets/{dataset}.json"

    test_log_dir = "test_log/" + dataset + '_' + this_train_size + '_' + this_eval_size
    test_log_file = dataset + '_' + this_train_size + '_' + this_eval_size

    print(f"Dataset path: {dataset_path}, Train size: {this_train_size}, Eval size: {this_eval_size}, Optimized prompt: {optimized_prompt}")
    subprocess.run(
        f"python test.py --task_name bigbench --eval_prompt '{optimized_prompt}' "
        f"--train_size {this_train_size} --eval_size {this_eval_size} --test_size 0 --seed 42 --pred_model 'gpt-3.5-turbo' --api_key OPENAI_API_KEY --log_file {test_log_file} "
        f"--log_dir {test_log_dir} --data_dir '{dataset_path}'", shell=True)
    


    # if os.path.exists(train_log_file):
    #     train_rewards = parse_log_file(train_log_file)
    # else:
    #     print(f"Training log file not found: {train_log_file}")
    #     continue

    # if os.path.exists(test_log_file):
    #     test_metric = parse_test_log_file(test_log_file)
    # else:
    #     print(f"Test log file not found: {test_log_file}")
    #     continue

    # train_rewards = parse_log_file(train_log_file)
    # test_metric = parse_test_log_file(test_log_file)

    # plot_convergence_for_size(train_size, train_rewards, test_metric)
