import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def plot_tfevents(file_paths, tag, colors=None):
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    for idx, file_path in enumerate(file_paths):
        event_acc = EventAccumulator(file_path)
        event_acc.Reload()

        # Get all the event tags
        tags = event_acc.Tags()['scalars']

        if tag not in tags:
            print(f"Tag '{tag}' not found in the provided file.")
            return

        # Get the data
        data = event_acc.Scalars(tag)

        steps = [event.step for event in data]
        values = [event.value for event in data]

        # Plot
        plt.ylim(None, 0.05)
        if idx == 0:
            plt.plot(steps, values, label="oracle", color=colors[idx])
        elif idx == 1:
            plt.plot(steps, values, label="image_reconstruction", color=colors[idx])
        elif idx == 2:
            plt.plot(steps, values, label="image_reconstruction_finetune", color=colors[idx])

    plt.xlabel('Step')
    plt.ylabel(tag)
    plt.title(f'{tag} over Time')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_paths = [
        "saved_rl_runs/oracle_num_epochs=70env=SpritesState-v0_lr=0.001 ||23-04-2024_10-31-13/events.out.tfevents.1713893473.Matthews-MacBook-Pro.local.32705.0",
        "saved_rl_runs/reward_prediction_with_encoder_num_epochs=70env=Sprites-v0_lr=0.001 ||22-04-2024_17-03-30/events.out.tfevents.1713830610.Matthews-MacBook-Pro.local.34383.0",
        "saved_rl_runs/reward_finetune_1distractor_oracle_num_epochs=70env=SpritesState-v1_lr=0.001 ||23-04-2024_20-57-11/events.out.tfevents.1713931031.Matthews-MacBook-Pro.local.47382.0"
    ]

    plot_tfevents(file_paths, "Average Actor Loss")
