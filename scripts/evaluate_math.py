import sys 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.join( *os.path.split(sys.path[0])[:-1] ))

from custom_datasets.math import RussianMathEval, RussianPhysicsEval
from src.utils.data_utils import load_hf_model, load_local_model
from src.utils.math_checker import DoomSlayer
import argparse
import json 

def get_args():
    parser = argparse.ArgumentParser("math parser")
    parser.add_argument("--model_name", help="model for evaluate", type=str)
    parser.add_argument("--repeats", default=1, help="count of data repeats for more accurate score", type=int, required=False)
    parser.add_argument("--result_path", default="result.json", help="path to store results", type=str, required=False)
    parser.add_argument("--verbose", default=True, required=False, type=bool)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.verbose:
        print("MODEL LOAD...")
    if '/' in args.model_name:
        model = load_hf_model(args.model_name)
    else:
        model = load_local_model(args.model_name)
    if args.verbose:
        print("MODEL LOADED")
    
    checker = DoomSlayer()
    if args.verbose:
        print("READY TO EVAL")
    math_dataset = RussianMathEval(equality_checker=checker, 
                                 n_repeats=args.repeats,
                                 debug=args.verbose)
    math_score = math_dataset(model)
    if args.verbose:
        print("MATH OK")
        
    phys_dataset = RussianPhysicsEval(equality_checker=checker,
                                    n_repeats=args.repeats,
                                    debug=args.verbose)
    phys_score = phys_dataset(model)
    if args.verbose:
        print("PHYS OK")
        
    with open(args.result_path, mode='w') as f:
        json.dump({"Model": args.model_name,
                   "repeats": args.repeats,
                   "math_score": math_score,
                   "phys_score": phys_score}, f)
