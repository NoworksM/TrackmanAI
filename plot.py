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
        unknown_0 = []
        unknown_1 = []
        unknown_5 = []

        for i in range(1, len(route)):
            previous_frame = route[i - 1]
            current_frame = route[i]

            distance = movement.calculate_distance(previous_frame['x'], previous_frame['y'], previous_frame['z'],
                                                   current_frame['x'], current_frame['y'], current_frame['z'])

            steering.append(current_frame['steering_input'])
            rpm.append(current_frame['rpm'])
            reward.append(current_frame['reward'])
            accelerate.append(current_frame['accelerate'])
            brake.append(current_frame['brake'])
            unknown_0.append(current_frame['unknown_0'])
            unknown_1.append(current_frame['unknown_1'])
            unknown_5.append(current_frame['unknown_5'])
            distance_covered.append(distance)

        fig, axs = plt.subplots(9, 1, figsize=(40, 24))

        axs[0].plot(steering)
        axs[0].set_title('Steering')
        axs[0].set_ylabel('Steering Value')
        axs[0].grid(True)

        axs[1].plot(reward)
        axs[1].set_title('Reward')
        axs[1].set_ylabel('Reward Value')
        axs[1].grid(True)

        axs[2].plot(distance_covered)
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

        axs[6].plot(unknown_0)
        axs[6].set_title('Unknown 0')
        axs[6].set_ylabel('Unknown 0 Input')
        axs[6].grid(True)

        axs[7].plot(unknown_1)
        axs[7].set_title('Unknown 1')
        axs[7].set_ylabel('Unknown 1 Input')
        axs[7].grid(True)

        axs[8].plot(unknown_5)
        axs[8].set_title('Unknown 5')
        axs[8].set_ylabel('Unknown 5 Input')
        axs[8].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    main()
