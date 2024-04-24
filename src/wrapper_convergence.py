import os
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


def parse_log_file(log_file_path):
    rewards = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'path_reward' in line:
                rewards_str = line.split(': ')[-1].strip()
                rewards = [float(x) for x in rewards_str.strip('[]').split(',')]
                return rewards
    return []

def parse_test_log_file(file_path):
    metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Parse lines with metrics, assuming format "key: value"
            parts = line.strip().split(':', 1)  # Split at the first colon
            if len(parts) == 2:
                key, value = parts
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
    return metrics

# def plot_convergence_for_size(train_size, train_rewards, test_metric):
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     ax1.set_xlabel('Number of Iterations')
#     ax1.set_ylabel('Reward', color='tab:blue')
#     ax1.plot(np.arange(len(train_rewards)), train_rewards, marker='o', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Test Metric', color='tab:orange')
#     ax2.bar(train_size, test_metric, color='tab:orange')
#     ax2.tick_params(axis='y', labelcolor='tab:orange')
#     plt.title(f'Convergence Curve for Training Size {train_size}')
#     fig.tight_layout()
#     plt.show()

#Original
# def get_optimized_prompt(data_json_path):
#      with open(data_json_path, 'r') as file:
#          data = json.load(file)
#          best_prompt = data['best_reward_path_selected_node'][0]['prompt'] if data['best_reward_path_selected_node'] else ''
#      return best_prompt

def get_optimized_prompt(train_log_file):
     with open('/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/optimized_prompts_dict.json', 'r') as file:
         data = json.load(file)
         best_prompt = data[train_log_file]['prompt']
     return best_prompt

data_splits = {
    "penguins_in_a_table": {"train": 70, "test": 79},
    "geometric_shapes": {"train": 150, "test": 200},
    "epistemic": {"train": 500, "test": 500},
    "object_counting": {"train": 300, "test": 500},
    "temporal_sequences": {"train": 300, "test": 500},
    "causal_judgement": {"train": 90, "test": 100}
}

# data_splits = {
#     "penguins_in_a_table": {"train": 70, "test": 79},
#     "geometric_shapes": {"train": 150, "test": 200},
#     "epistemic": {"train": 500, "test": 500},
#     "object_counting": {"train": 300, "test": 500},
#     "temporal_sequences": {"train": 300, "test": 500},
# }

datasets = ["causal_judgement.json", "epistemic.json", "geometric_shapes.json", "penguins_in_a_table.json", "temporal_sequences.json"]

# datasets = ["epistemic.json", "geometric_shapes.json",
#             "object_counting.json", "penguins_in_a_table.json", "temporal_sequences.json"]

# datasets = ["object_counting.json"]
# datasets = [ "penguins_in_a_table.json"]
# datasets = ["temporal_sequences.json"]
# datasets = ["geometric_shapes.json"]

descriptions = [ "Name geometric shapes from their SVG paths", 
                "Answer questions about causal attribution", 
                "Determine whether one sentence entails the next", 
                "Questions that involve enumerating objects of different types and asking the model to count them", 
                "Answer questions about penguins in a table", 
                "Answer questions about temporal sequences"]

#make a dictionary here based on the paper and run the train size multiple for each respective dataset
#train_size = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
eval_size = 50
training_percentages = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
#training_percentages = [15, 10, 5, 1]
#training_percentages = [10, 5, 1]

log_dir = "../train_log/"
log_test_dir = "../test_log/"

# Running subprocesses and collecting data for both training and testing
for i, dataset in enumerate(datasets):
    description = descriptions[i]
    # dataset_key = dataset.replace('.json', '').replace('_', ' ').lower()
    total_train_size = data_splits[dataset.split('.')[0]]['train']
    total_test_size = data_splits[dataset.split('.')[0]]['test']
    # total_train_size = data_splits.get(dataset_key, {}).get("train", 0)
    # total_test_size = data_splits.get(dataset_key, {}).get("test", 0)

    for percent in training_percentages:
        train_size = int((percent / 100) * total_train_size)
        test_size = total_test_size
        
        train_log_file = f"log_{os.path.splitext(dataset)[0]}_{train_size}.log"
        test_log_file = f"log_test_{os.path.splitext(dataset)[0]}_{test_size}.log"
        json_file_path = os.path.join(log_dir, f"data_{dataset}_{train_size}.json")
        data_path = f"../datasets/{dataset}" 

        dictionary_element_name = f"bigbench_{os.path.splitext(dataset)[0]}_{train_size}_{eval_size}"
        print(dictionary_element_name)

#    for train, eval in zip(train_size, eval_size):
#     data_path = f"../datasets/{dataset}"
#     train_log_file = f"log_{os.path.splitext(dataset)[0]}_{train}.log" #Check the file names here and the name format is different here
#     test_log_file = f"log_test_{os.path.splitext(dataset)[0]}_{train}.log"
#     json_file_path = os.path.join(log_dir, f"data_{dataset}_{train}.json")

        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            continue

        ## Changing task name to object_counting for object_counting dataset

        # subprocess.run(
        #     f"python main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 "
        #     f"--train_size {train_size} --eval_size {eval_size} --test_size 0 --seed 42 --train_shuffle True "
        #     f"--iteration_num 10 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo "
        #     f"--optim_model gpt-4 --log_dir {log_dir} --log_file {train_log_file} --data_dir {data_path} --init_prompt '{description}' --api_key OPENAI_API_KEY", # Make sure the main.py writes the trainiing log from main.py. Check for each dataset whether it writes to the intended log. If there are any empty log files. 
        #     shell=True)
#
        #Extract optimized prompt from the specific data.json file for each training set
        # if os.path.exists(json_file_path):
        directory = '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/train_log'  # Replace with the actual directory path
        files = os.listdir(directory)
        for file in files:
            print(file)
            
        optimized_prompt = get_optimized_prompt(dictionary_element_name)
        # else:
        #      print(f"JSON file not found: {json_file_path}")
        #      continue
        
        print(optimized_prompt)




        # subprocess.run(
        #     f"python test.py --task_name bigbench --eval_prompt '{optimized_prompt}' "
        #     f"--train_size {train_size} --eval_size {eval_size} --test_size {test_size} --seed 42 --pred_model 'gpt-3.5-turbo' --api_key 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 --log_file {test_log_file} "
        #     f"--log_dir {log_test_dir} --data_dir '{data_path}'", shell=True)
        


        if os.path.exists(train_log_file):
            train_rewards = parse_log_file(train_log_file)
        else:
            print(f"Training log file not found: {train_log_file}")
        #     continue

        # if os.path.exists(test_log_file):
        #     test_metric = parse_test_log_file(test_log_file)
        # else:
        #     print(f"Test log file not found: {test_log_file}")
        #     continue

        # train_rewards = parse_log_file(train_log_file)
        # test_metric = parse_test_log_file(test_log_file)

        # plot_convergence_for_size(train_size, train_rewards, test_metric)
