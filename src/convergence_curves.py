
"""
import os
import json
import subprocess

datasets = ["causal_judgement.json", "epistemic.json", "geometric_shapes.json","object_counting.json", "penguins_in_a_table.jsoxn","temporal_sequences.json"]
descriptions = ["Answer questions about causal attribution", "Determine whether one sentence entails the next", "Name geometric shapes from their SVG paths", "Questions that involve enumerating objects of different types and asking the model to count them", "Answer questions about penguins in a table", "Answer questions about temporal sequences"]

train_size = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
eval_size = [50, 50, 50, 50, 50, 50, 50]

print(datasets)

# Outer loop: iterate over datasets and descriptions
for i, dataset in enumerate(datasets):
    description = descriptions[i]

    # Inner loop: iterate over each train_size
    for train in train_size:
        subprocess.run(f"src/main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 --train_size {train} --eval_size {eval_size[0]} --test_size 0 --seed 42 --train_shuffle True --iteration_num 1 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo --optim_model gpt-4 --log_dir train_log/ --data_dir datasets/{dataset} --init_prompt '{description}'", shell=True)

# The subprocess.run command for 'penguins_in_a_table.json' remains as it is, unless it also needs to be looped over different parameters

#subprocess.run("python src/test.py --task_name bigbench --prompt_file '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgpwd
# ent/logs/20240303_162844-bigbench_penguins_in_a_table-algo_mcts-batch_5-train_10/20240303_162844-bigbench_penguins_in_a_table-algo_mcts-batch_5-train_10-train-000.log' --train_size 70 --eval_size 50 --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --data_dir 'datasets/penguins_in_a_table.json'", shell=True)




import os
import json
import subprocess

datasets = ["causal_judgement.json", "epistemic.json", "geometric_shapes.json","object_counting.json", "penguins_in_a_table.json","temporal_sequences.json"]
descriptions = ["Answer questions about causal attribution", "Determine whether one sentence entails the next", "Name geometric shapes from their SVG paths", "Questions that involve enumerating objects of different types and asking the model to count them", "Answer questions about penguins in a table", "Answer questions about temporal sequences"]

train_size = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
eval_size = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

#print(datasets)
#print(descriptions)
#print(train_size)

# Outer loop: iterate over datasets and descriptions
# ...
for i, dataset in enumerate(datasets):
    description = descriptions[i]

    for train in train_size:
        data_path = f"../datasets/{dataset}"
        print(f"Attempting to access dataset at: {data_path}")  # Debugging print statement

        # Check if the file exists before running the subprocess
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            continue

        subprocess.run(f"python main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 --train_size {train} --eval_size {eval_size[0]} --test_size 0 --seed 42 --train_shuffle True --iteration_num 1 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo --optim_model gpt-3.5-turbo --log_dir train_log/ --data_dir {data_path} --init_prompt '{description}'", shell=True)


# The subprocess.run command for 'penguins_in_a_table.json' remains as it is, unless it also needs to be looped over different parameters

#subprocess.run("python src/test.py --task_name bigbench --prompt_file '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent/logs/20240303_162844-bigbench_penguins_in_a_table-algo_mcts-batch_5-train_10/20240303_162844-bigbench_penguins_in_a_table-algo_mcts-batch_5-train_10-train-000.log' --train_size 70 --eval_size 50 --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --data_dir 'datasets/penguins_in_a_table.json'", shell=True)

subprocess.run("python src/test.py --task_name bigbench --prompt_file '/data.json/best_reward_path_selected_node' --train_size 70 --eval_size 50 --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --data_dir 'datasets/penguins_in_a_table.json'", shell=True)
"""
#new code------------------     ---------------------------------------------------------  
"""
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):

    rewards = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'path_reward' in line:
                rewards_str = line.split(': ')[-1].strip()
                rewards = [float(x) for x in rewards_str.strip('[]').split(',')]
                return rewards
    return []

def plot_convergence(all_data):
   
    plt.figure(figsize=(10, 6))
    for train_size, data in all_data.items():
        if data:
            iterations = np.arange(len(data))
            plt.plot(iterations, data, label=f'Train Size {train_size}')

    plt.xlabel('Number of Iterations')
    plt.ylabel('Reward')
    plt.title('Convergence Curves for Different Training Sizes')
    plt.legend()
    plt.grid()
    plt.show()

datasets = ["causal_judgement.json", "epistemic.json", "geometric_shapes.json",
            "object_counting.json", "penguins_in_a_table.json", "temporal_sequences.json"]
descriptions = ["Answer questions about causal attribution", 
                "Determine whether one sentence entails the next", 
                "Name geometric shapes from their SVG paths", 
                "Questions that involve enumerating objects of different types and asking the model to count them", 
                "Answer questions about penguins in a table", 
                "Answer questions about temporal sequences"]
train_size = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
eval_size = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

log_dir = "train_log/"
all_data = {size: [] for size in train_size}

# Running subprocesses and collecting data for both training and testing
for i, dataset in enumerate(datasets):
    description = descriptions[i]

    for train, eval in zip(train_size, eval_size):
        data_path = f"../datasets/{dataset}"
        train_log_file = os.path.join(log_dir, f"log_{dataset}_{train}.log")
        data_json_file = os.path.join(log_dir, f"data_{dataset}_{train}.json")  # JSON file with optimized prompt

        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            continue

        # Running main.py for training
        subprocess.run(
            f"python src/main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 "
            f"--train_size {train} --eval_size {eval} --test_size 0 --seed 42 --train_shuffle True "
            f"--iteration_num 10 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo "
            f"--optim_model gpt-4 --log_dir {log_dir} --data_dir {data_path} --init_prompt '{description}'", 
            shell=True
        )

        # Running test.py for testing
        subprocess.run(
            f"python src/test.py --task_name {dataset} --prompt_file '{data_json_file}' "
            f"--train_size {train} --eval_size {eval} --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' "
            f"--data_dir '{data_path}'", 
            shell=True
        )

        # Parse log file for training results
        train_rewards = parse_log_file(train_log_file)
        all_data[train].extend(train_rewards)

# Plotting the convergence curve for training results
plot_convergence(all_data)




#Newer code that worked but slept
"""
"""
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    
    #Parse a single log file to extract path rewards.

    rewards = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'path_reward' in line:
                rewards_str = line.split(': ')[-1].strip()
                rewards = [float(x) for x in rewards_str.strip('[]').split(',')]
                return rewards
    return []

def plot_convergence_for_size(train_size, rewards):
    
    #Plot the convergence curve for a given training size.
    
    plt.figure(figsize=(10, 6))
    iterations = np.arange(len(rewards))
    plt.plot(iterations, rewards, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Reward')
    plt.title(f'Convergence Curve for Training Size {train_size}')
    plt.grid()
    plt.show()

datasets = ["causal_judgement.json", "epistemic.json", "geometric_shapes.json",
            "object_counting.json", "penguins_in_a_table.json", "temporal_sequences.json"]
descriptions = ["Answer questions about causal attribution", 
                "Determine whether one sentence entails the next", 
                "Name geometric shapes from their SVG paths", 
                "Questions that involve enumerating objects of different types and asking the model to count them", 
                "Answer questions about penguins in a table", 
                "Answer questions about temporal sequences"]
train_size = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
eval_size = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

log_dir = "train_log/"

# Running subprocesses and collecting data for both training and testing
for i, dataset in enumerate(datasets):
    description = descriptions[i]

    for train, eval in zip(train_size, eval_size):
        data_path = f"../datasets/{dataset}"
        train_log_file = os.path.join(log_dir, f"log_{dataset}_{train}.log")
        test_log_file = os.path.join(log_dir, f"test_log_{dataset}_{train}.log")  # Placeholder for test log file

        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            continue

        # Running main.py for training
        subprocess.run(
            f"python main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 "
            f"--train_size {train} --eval_size {eval} --test_size 0 --seed 42 --train_shuffle True "
            f"--iteration_num 10 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo "
            f"--optim_model gpt-4 --log_dir {log_dir} --data_dir {data_path} --init_prompt '{description}' --api_key 'OPENAI_API_KEY'", 
            shell=True
        )

        # Running test.py for testing
        subprocess.run(
            f"python test.py --task_name bigbench --prompt_file '{test_log_file}' "
            f"--train_size {train} --eval_size {eval} --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --api_key 'OPENAI_API_KEY'"
            f"--data_dir '{data_path}'", 
            shell=True
        )

       # Parse log files for training and testing results
        train_rewards = parse_log_file(train_log_file)
        test_metric = parse_test_log_file(test_log_file)

        # Plotting the convergence curve and test results for this training size
        plot_convergence_for_size(train, train_rewards, test_metric)



"""
import os
import json
import subprocess

subprocess.run(

 f"python main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 "
            f"--train_size 70 --eval_size 50 --test_size 0 --seed 42 --train_shuffle True "
            f"--iteration_num 10 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo "
            f"--optim_model gpt-4 --log_dir logs/ --data_dir ../datasets/causal_judgement.json --init_prompt 'Answer questions about causal attribution' --api_key 'OPENAI_API_KEY'",
            shell=True
)