import argparse

def main():
    argument_parser = argparse.ArgumentParser()

    argument_parser.
    argument_parser.add_argument('--algorithm', type=str, required=True, choices=['PPO', 'SAC', 'R-PPO', 'R-SAC'])


    args = argument_parser.parse_args()

    if args.algorithm == 'PPO':
