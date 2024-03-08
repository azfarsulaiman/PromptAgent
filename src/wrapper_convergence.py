import os
import json
import subprocess

datasets = ["causal_judgement.json", "epistemic.json", "geometric_shapes.json","object_counting.json", "penguins_in_a_table.json","temporal_sequences.json"]
descriptions = ["Answer questions about causal attribution", "Determine whether one sentence entails the next", "Name geometric shapes from their SVG paths", "Questions that involve enumerating objects of different types and asking the model to count them", "Answer questions about penguins in a table", "Answer questions about temporal sequences"]

train_size = [100, 75, 50, 30, 25, 20, 15, 10, 5, 1]
eval_size = [50, 50, 50, 50, 50, 50, 50]

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

#subprocess.run("python src/test.py --task_name bigbench --prompt_file '/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgpwd
# ent/logs/20240303_162844-bigbench_penguins_in_a_table-algo_mcts-batch_5-train_10/20240303_162844-bigbench_penguins_in_a_table-algo_mcts-batch_5-train_10-train-000.log' --train_size 70 --eval_size 50 --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --data_dir 'datasets/penguins_in_a_table.json'", shell=True)
