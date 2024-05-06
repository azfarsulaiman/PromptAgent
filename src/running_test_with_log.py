import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

def get_optimized_prompt(train_log_file, prompts_path):
    """Retrieve the best prompt for a given training log file."""
    try:
        with open(prompts_path, 'r') as file:
            data = json.load(file)
            return data[train_log_file]['prompt']
    except KeyError:
        print(f"Warning: No entry found for '{train_log_file}' in JSON data.")
    except FileNotFoundError:
        print(f"Error: File not found {prompts_path}")
    except json.JSONDecodeError:
        print("Error: JSON Decode Error in the prompts file.")
    return None

def main():
    base_dir = Path('/Users/azfar/Documents/MIT/FutureTech/FutureTech/Thesis/PromptAgent')
    prompts_path = base_dir / 'optimized_prompts_dict.json'
    directory = base_dir / 'train_log'
    
    for file in os.listdir(directory):
        optimized_prompt = get_optimized_prompt(file, prompts_path)
        if optimized_prompt is None:
            continue

        file_elements = file.split('_')
        this_train_size, this_eval_size = file_elements[-2], file_elements[-1]
        dataset = '_'.join(file_elements[1:-2])
        
        if dataset == 'counting':
            dataset = 'object_counting'
            continue

        dataset_path = base_dir / 'datasets' / f'{dataset}.json'
        test_log_dir = base_dir / 'test_log' / f"{dataset}_{this_train_size}_{this_eval_size}"
        test_log_file = f"{dataset}_{this_train_size}_{this_eval_size}"

        #api_key = os.getenv('OPENAI_API_KEY', 'your-default-api-key')
        run_call = [
            "python", "test.py",
            "--task_name", "bigbench",
            "--eval_prompt", optimized_prompt,
            "--train_size", this_train_size,
            "--eval_size", this_eval_size,
            "--test_size", "1",
            "--seed", "42",
            "--pred_model", "gpt-3.5-turbo",
            "--api_key", "OPEN_AI_API_KEY",
            "--log_file", str(test_log_file),
            "--log_dir", str(test_log_dir),
            "--data_dir", str(dataset_path)
        ]
        print("Running subprocess:", ' '.join(run_call))
        subprocess.run(run_call, shell=False)

if __name__ == "__main__":
    main()
    