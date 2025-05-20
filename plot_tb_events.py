import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import seaborn # sets some style parameters automatically

# COLORS = [(57, 106, 177), (218, 124, 48)] 
COLORS = seaborn.color_palette('tab10')


def load_event_data(scalar_tag, events_dir, keys_mapping, first_epochs=184):
    # === List all TensorBoard event files ===
    event_files = []
    for dirpath, _, filenames in os.walk(events_dir):
        for filename in filenames:
            if filename.startswith("events.out.tfevents"):
                full_path = os.path.join(dirpath, filename)
                event_files.append(full_path)

    tags_to_plot = sorted(keys_mapping.keys())

    tagged_event_files = {}
    for event_file in event_files:
        for tag in tags_to_plot:
            if tag in event_file:
                if tag not in tagged_event_files:
                    tagged_event_files[tag] = []
                tagged_event_files[tag].append(event_file)


    if len(tagged_event_files) == 0:
        print("No event files found with the specified tags.")
        exit(1)

    kw_epochs = {}
    kw_eprewmean = {}

    for tag in tagged_event_files.keys():
        event_file_paths = tagged_event_files[tag]
        for event_file_path in event_file_paths:
            print(f"Loading event file: {event_file_path}")
            ea = event_accumulator.EventAccumulator(event_file_path)
            ea.Reload()

            scalar_tags = ea.Tags()['scalars']
            print(f"Available scalar tags: {scalar_tags}")

            if scalar_tag not in scalar_tags:
                print(f"Scalar tag '{scalar_tag}' not found in event file.")
                continue

            # Get the scalar values for the specified tag
            events = ea.Scalars(scalar_tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])

            if tag not in kw_epochs:
                kw_epochs[tag] = []
            kw_epochs[tag].append(steps[:first_epochs])
            if tag not in kw_eprewmean:
                kw_eprewmean[tag] = []
            kw_eprewmean[tag].append(values[:first_epochs])
    
    return kw_epochs, kw_eprewmean, tags_to_plot

def make_plots(ax, title, kw_epochs, kw_eprewmean, tags_to_plot, keys_mapping, env):
    color_idx = 0

    for key in tags_to_plot:
        steps = np.stack(kw_epochs[key], axis=0)
        values = np.stack(kw_eprewmean[key], axis=0)

        x = np.mean(steps, axis=0)
        y_mean = np.mean(values, axis=0)
        y_stderr = np.std(values, axis=0) / np.sqrt(len(values))
        print(key, 'last mean:', y_mean[-1], 'last stderr:', y_stderr[-1])

        ax.plot(x, y_mean, label=keys_mapping[key], color=COLORS[color_idx], linestyle='solid' if env == 'train' else 'dotted', linewidth=1.0, rasterized=True)
        ax.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=COLORS[color_idx], alpha=.25, linewidth=0.0, rasterized=True)
        # plt.xlabel("Epoch")
        # plt.ylabel("Episodic return")
        # plt.title(f"{title}")
        color_idx += 1

    # plt.grid(True)
    # ax.legend()

if __name__ == "__main__":
    # === Configuration ===
    dropout = 0.0
    title = f'ViT on CIFAR (tiny; ablation)'
    # title = f'ViT on CIFAR (tiny)'
    # title = f'Fine-Tuning on BERT'

    envs_list = ['train', 'test']
    scalar_tags = ['loss', 'acc']

    # events_dir = "vit_cifar/logs_small_vit/"
    events_dir = "vit_cifar/logs_tiny_vit/"
    # events_dir = "bert_finetuning/logs/"

    # envs_list = ['halfcheetah', 'hopper', 'swimmer', 'walker2d', 'humanoid']
    # envs_list = ['invertedpendulum', 'halfcheetah', 'hopper', 'walker2d']

    dimx, dimy = (1, len(envs_list))

    # keys_mapping = {'trust.mlp.dropout_0.0': 'Dropout_0.0', 
    #                 # 'trust.mlp.dropout_0.1': 'Dropout_0.1',
    #                 'trust.mlp.dropout_0.2': 'Dropout_0.2',
    #                 # 'trust.mlp.dropout_0.3': 'Dropout_0.3',
    #                 'trust.mlp.dropout_0.4': 'Dropout_0.4',
    #                 # 'trust.mlp.dropout_0.5': 'Dropout_0.5',
    #                 'trust.mlp.dropout_0.6': 'Dropout_0.6',
    #                 # 'trust.mlp.dropout_0.7': 'Dropout_0.7',
    #                 # 'trust.mlp.dropout_0.8': 'Dropout_0.8',
    #                 # 'trust.mlp.dropout_0.9': 'Dropout_0.9',
    #                 }
    # keys_mapping = {f'trust.mlp.dropout_{dropout}': f'Trust_{dropout}', 
    #                 f'rc.mlp.dropout_{dropout}': f'PPO_{dropout}', }
    keys_mapping = {'cifar.dgclip.0.2.1.0.0.01': 'delta-GClip-0.01', 
                    'cifar.dgclip.0.2.1.0.0.001': 'delta-GClip-0.001',
                    'cifar.dgclip.0.2.1.0.0.0001': 'delta-GClip-0.0001',
                    'cifar.dgclip.0.2.1.0.1e-05': 'delta-GClip-0.00001',
                    # f'cifar.adam.0.0005': f'Adam', 
                    }
    # keys_mapping = {f'bert_finetuning.dgclip': f'delta-GClip', 
    #                 f'bert_finetuning.adamw': f'AdamW'}

    fig, axarr = plt.subplots(dimx, dimy, figsize=(dimy * 3.8, dimx * 2.0), dpi=300)

    for i, tag in enumerate(scalar_tags):
        ax = axarr[i // dimy][i % dimy] if dimx > 1 else axarr[i % dimy]
        ax.set_title('Loss (ViT Tiny)' if tag == 'loss' else 'Accuracy (ViT Tiny)')
        ax.set_xlabel("Epoch")
        ax.set_ylabel('Loss' if tag == 'loss' else 'Accuracy')

        for j, env in enumerate(envs_list):
            # if (env == 'train' and tag == 'acc') or (env == 'test' and tag == 'loss'):
            #     continue
            new_keys_mapping = {k: f'{v}-{env}' for k, v in keys_mapping.items()}
            scalar_tag = f'{env}_{tag}'
            kw_epochs, kw_eprewmean, tags_to_plot = load_event_data(scalar_tag, events_dir, new_keys_mapping)
            make_plots(ax, env, kw_epochs, kw_eprewmean, tags_to_plot, new_keys_mapping, env)

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, loc='lower left', labels=labels[-4:], frameon=True, bbox_to_anchor=(0.1, -0.08), ncol=4)
    ax.legend(loc='upper left', frameon=True, bbox_to_anchor=(1.0, 1.0), ncol=1)

    # === Plot and save each scalar tag ===
    output_dir = "tensorboard_plots"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title}.pdf")

    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("\nAll plots saved to:", title)