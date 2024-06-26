import argparse
from prompt_optim_agent import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description='Process prompt search agent arguments')
    # Remains the same
    parser.add_argument('--task_name', type=str, default='bigbench',  help='This is consistent to the task file names in tasks folder. The default is bigbench task.')  
    # Remains the same
    parser.add_argument('--search_algo', type=str, default='mcts', choices=['mcts', 'beam'], help='Prompt search algorithm. Choose from \'mcts\' and \'beam\'.')    
    # Remains the same
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size depending on the memory and model size')
    # Remains the same
    parser.add_argument('--depth_limit', type=int, default=5, help="The max depth of a single searching path.")
    # Train size - will change
    parser.add_argument('--train_size', type=int, default=None, help="The dataset that sample batches from.")

    parser.add_argument('--eval_size', type=int, default=50, help="Calculate reward on this set.")
    # Test set size - will not change
    parser.add_argument('--test_size', type=int, default=0, help="Test set size.")
    parser.add_argument('--seed', type=int, default=42, help="The seed to shuffle the dataset.")
    parser.add_argument('--train_shuffle', type=str2bool, default=True, help='Shuffle training set')
    
    #Search
    parser.add_argument('--init_prompt', type=str, default="Let's solve the problem.", help='Initial prompt written by human.')
    parser.add_argument('--iteration_num', type=int, default=12, help='MCTS iteration number.')
    parser.add_argument('--expand_width', type=int, default=3, help="The number of batches sampled in each expansion.")
    parser.add_argument('--num_new_prompts', type=int, default=1, help="The number of new prompts sampled in each batch.")
    parser.add_argument('--post_instruction', type=str2bool, default=False, help="True: the position of instruction is question+instruction; \nFalse: the position of instruction is instruction+question")
    
    # MCTS
    parser.add_argument('--min_depth', type=int, default=2, help="Early stop depth: early stop is only applied when depth is larger than min_depth.")
    parser.add_argument('--w_exp', type=float, default=2.5, help="Weight of MCTS.")

    # World Model
    parser.add_argument('--pred_model', type=str, default='gpt-3.5-turbo', help='The base model that makes predictions.')
    parser.add_argument('--optim_model', type=str, default='gpt-4', help='Prompt optimizer.') 
    parser.add_argument('--pred_temperature', type=float, default=0.0)
    parser.add_argument('--optim_temperature', type=float, default=1.0)
    
    # Others
    parser.add_argument('--log_dir', type=str, default='train_log/', help='Log directory.')
    parser.add_argument('--log_file', type=str, default='train.log', help='Log file.')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to the data file (if needed)')
    parser.add_argument('--api_key', type=str, default=os.getenv("OPENAI_API_KEY"), help='OpenAI api key or PaLM 2 api key')
    
    # BeamSearch
    parser.add_argument('--beam_width', type=int, default=3)
    args = parser.parse_args()

    args = vars(args)
    
    # Set up openai or PaLM keys
    api_key_config(args['api_key'])
    
    return args


def main(args):
    agent = BaseAgent(**args)
    agent.run(init_state=args['init_prompt'], iteration_num=args['iteration_num'])
    return

if __name__ == '__main__':
    args = config()
    main(args)