import argparse

from model.ppo_online import learn_ppo_online
from model.recurrent_ppo_online import learn_recurrent_ppo_online
from model.sac_online import learn_sac_online


def main():
    parser = argparse.ArgumentParser(prog='TrackmanAI', description='Trackmania AI ImageNet Training')
    parser.add_argument('--steps', type=int, default=10000, metavar='N',
                        help='number of steps to train (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training ')
    parser.add_argument('--rollout-steps', type=int, default=2048, metavar='N', help='rollout steps for training')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate (default: 0.0003)')
    parser.add_argument('--scheduler', type=str, default='linear', metavar='SCHEDULER',
                        help='scheduler (default: linear)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='OPTIMIZER', help='optimizer (default: adam)',
                        choices=['adam', 'adamw', 'sgd', 'dadaptation-adam'])
    parser.add_argument('--name', type=str, default='trackmania', metavar='NAME', help='name of the run')
    parser.add_argument('--algorithm', type=str, default='ppo', metavar='ALGORITHM', help='algorithm (default: ppo)',
                        choices=['ppo', 'a2c', 'sac', 'td3', 'ddpg', 'r-ppo', 'ppg'])
    parser.add_argument('--policy', type=str, default='MlpPolicy', metavar='POLICY',
                        help='policy (default: MlpPolicy)',
                        choices=['MultiInputPolicy', 'MlpPolicy', 'CnnPolicy', 'MultiInputPolicyWithActionMasking',
                                 'MlpLstmPolicy'])
    parser.add_argument('--frame-stacking', type=int, default=1, metavar='N', help='frame stacking (default: 1)')
    parser.add_argument('--model-name', type=str, metavar='NAME', help='name of the model to save/load')

    args = parser.parse_args()

    learn_func = None
    if args.algorithm == 'ppo':
        learn_func = learn_ppo_online
    elif args.algorithm == 'r-ppo':
        learn_func = learn_recurrent_ppo_online
    elif args.algorithm == 'sac':
        learn_func = learn_sac_online
    else:
        raise NotImplementedError('Algorithm not implemented yet')

    learn_func(env_name='Trackmania-v1', steps=args.steps, rollout_steps=args.rollout_steps, model_name=args.model_name,
               frame_stacking=args.frame_stacking)


if __name__ == '__main__':
    main()
