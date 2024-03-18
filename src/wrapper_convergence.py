
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    # Parse a single log file to extract path rewards.
    rewards = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'path_reward' in line:
                rewards_str = line.split(': ')[-1].strip()
                rewards = [float(x) for x in rewards_str.strip('[]').split(',')]
                return rewards
    return []

def parse_test_log_file(test_log_file_path):
    # Placeholder function to parse the test log file
    # Replace this with actual logic based on your test log format
    # Returning a random value for demonstration
    return np.random.uniform(0, 1)

def plot_convergence_for_size(train_size, train_rewards, test_metric):
    # Plot the convergence curve for training and bar graph for testing
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting training data
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Reward', color='tab:blue')
    ax1.plot(np.arange(len(train_rewards)), train_rewards, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plotting test data
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Test Metric', color='tab:orange')  # we already handled the x-label with ax1
    ax2.bar(train_size, test_metric, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title(f'Convergence Curve for Training Size {train_size}')
    fig.tight_layout()  # to ensure the right y-label is not clipped
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
        test_log_file = os.path.join(log_dir, f"test_log_{dataset}_{train}.log")

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
