import argparse

parser = argparse.ArgumentParser(prog='TrackmanAI', description='Trackmania AI ImageNet Training')
parser.add_argument('--steps', type=int, default=10000, metavar='N', help='number of steps to train (default: 10000)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training ')
parser.add_argument('--rollout-steps', type=int, default=2048, metavar='N', help='rollout steps for training')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate (default: 0.0003)')
parser.add_argument('--scheduler', type=str, default='linear', metavar='SCHEDULER', help='scheduler (default: linear)')
parser.add_argument('--optimizer', type=str, default='adam', metavar='OPTIMIZER', help='optimizer (default: adam)',
                    choices=['adam', 'adamw', 'sgd', 'dadaptation-adam'])
parser.add_argument('--name', type=str, default='trackmania', metavar='NAME', help='name of the run')
parser.add_argument('--algorithm', type=str, default='ppo', metavar='ALGORITHM', help='algorithm (default: ppo)',
                    choices=['ppo', 'a2c', 'sac', 'td3', 'ddpg', 'r-ppo', 'ppg'])
parser.add_argument('--policy', type=str, default='MultiInputPolicy', metavar='POLICY',
                    help='policy (default: MultiInputPolicy)',
                    choices=['MultiInputPolicy', 'MlpPolicy', 'CnnPolicy', 'MultiInputPolicyWithActionMasking',
                             'MlpLstmPolicy'])

args = parser.parse_args()
