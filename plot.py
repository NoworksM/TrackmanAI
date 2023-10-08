import json
import os

import matplotlib.pyplot as plt
import config
import utils.movement as movement


def main():
    # Get all folders in the training data directory
    folders = os.listdir(config.data_base_path)

    for folder in folders:
        route_data_path = os.path.join(config.data_base_path, folder, 'route.json')
        plot_path = os.path.join(config.data_base_path, folder, 'data_plot.png')

        if not os.path.exists(route_data_path) or os.path.exists(plot_path):
            continue

        route = None
        with open(os.path.join(config.data_base_path, folder, 'route.json'), 'r') as f:
            route = json.load(f)

        if len(route) == 0:
            continue

        distance_covered = []
        steering = []
        rpm = []
        reward = []
        accelerate = []
        brake = []
        speed = []
        distance = []
        gear = []

        for i in range(len(route)):
            current_frame = route[i]

            steering.append(current_frame['steering_input'])
            rpm.append(current_frame['rpm'])
            reward.append(current_frame['reward'])
            accelerate.append(current_frame['accelerate'])
            brake.append(current_frame['brake'])
            speed.append(current_frame.get('speed', current_frame.get('unknown_0', 0)))
            distance.append(current_frame.get('distance', current_frame.get('unknown_1', 0)))
            gear.append(current_frame.get('gear', current_frame.get('unknown_5', 1)))
            distance_covered.append(distance)

        fig, axs = plt.subplots(8, 1, figsize=(40, 24))

        axs[0].plot(steering)
        axs[0].set_title('Steering')
        axs[0].set_ylabel('Steering Value')
        axs[0].grid(True)

        axs[1].plot(reward)
        axs[1].set_title('Reward')
        axs[1].set_ylabel('Reward Value')
        axs[1].grid(True)

        axs[2].plot(distance)
        axs[2].set_title('Distance')
        axs[2].set_ylabel('Distance Covered')
        axs[2].grid(True)

        axs[3].plot(rpm)
        axs[3].set_title('RPM')
        axs[3].set_ylabel('RPM Value')
        axs[3].grid(True)

        axs[4].plot(accelerate)
        axs[4].set_title('Accelerate')
        axs[4].set_ylabel('Accelerate Input')
        axs[4].grid(True)

        axs[5].plot(brake)
        axs[5].set_title('Brake')
        axs[5].set_ylabel('Brake Input')
        axs[5].grid(True)

        axs[6].plot(speed)
        axs[6].set_title('Speed')
        axs[6].set_ylabel('Speed (FR)')
        axs[6].grid(True)

        axs[7].plot(gear)
        axs[7].set_title('Gear')
        axs[7].set_ylabel('Gear')
        axs[7].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    main()
