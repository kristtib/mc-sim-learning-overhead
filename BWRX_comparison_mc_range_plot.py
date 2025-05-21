import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.lines import Line2D

# --- Load data ---
with open("noak_simulation_results_nuclear_smr.pkl", "rb") as f:
    noak_results = pickle.load(f)

years = list(noak_results.keys())


def custom_quartiles(data):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return None, None, None
    min_val = np.min(data)
    p25 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    max_val = np.max(data)
    q1 = (min_val + p25) / 2
    q2 = median
    q3 = (p75 + max_val) / 2
    return q1, q2, q3


def prepare_data_all_materials(results, reactor_prefix, learning_type):
    scenarios = ["premium", "no_premium", "market_price"]
    data = {scenario: {"bottoms": [], "medians": [], "tops": []} for scenario in scenarios}

    for year in years:
        for scenario in scenarios:
            key = f"{reactor_prefix}_noak_{learning_type}_{scenario}"
            if key in results[year]:
                q1, q2, q3 = custom_quartiles(results[year][key])
                data[scenario]["bottoms"].append(q1)
                data[scenario]["medians"].append(q2)
                data[scenario]["tops"].append(q3)
            else:
                data[scenario]["bottoms"].append(None)
                data[scenario]["medians"].append(None)
                data[scenario]["tops"].append(None)

    return data


def save_plot(fig, folder, filename):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_nuclear_all_materials_noak(nuclear_data, title, filename):
    scenarios = ["premium", "no_premium", "market_price"]
    colors = ["#4878A8", "#59935B", "#FFA500"]
    labels = [f"{year.replace('Deployment ', '')}" for year in years]
    fig, ax = plt.subplots(figsize=(20, 10))

    x_pos = np.arange(len(labels) * 3)
    bottoms, tops, medians = [], [], []

    for scenario in scenarios:
        bottoms += nuclear_data[scenario]["bottoms"]
        tops += nuclear_data[scenario]["tops"]
        medians += nuclear_data[scenario]["medians"]

    heights = [t - b for t, b in zip(tops, bottoms)]
    color_map = [colors[i // 5] for i in range(15)]
    ax.bar(x_pos, heights, bottom=bottoms, width=0.6, color=color_map, edgecolor='black', alpha=0.4)

    for i in range(15):
        ax.scatter(x_pos[i], medians[i], color='red', marker='o', s=80)
        ax.text(x_pos[i], medians[i] + 150, f"{medians[i]:.0f}", ha='center', va='bottom', fontsize=18, color='red')

    for group_start in range(0, 15, 5):
        base_q2 = medians[group_start]
        for i in range(1, 5):
            drop = 100 * (base_q2 - medians[group_start + i]) / base_q2
            ax.text(x_pos[group_start + i], tops[group_start + i] + 300,
                    f"{drop:.1f}% ↓", ha='center', fontsize=18, color='darkblue')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels * 3, rotation=45, ha='right')
    ax.set_ylabel("NOAK OCC+GCC (2024 USD/kW)", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(tops + medians) + 1000)

    custom_lines = [
        Line2D([0], [0], marker='o', color='#4878A8', label='Premium', linestyle=''),
        Line2D([0], [0], marker='o', color='#59935B', label='No Premium', linestyle=''),
        Line2D([0], [0], marker='o', color='#FFA500', label='Market Price', linestyle=''),
        Line2D([0], [0], marker='o', color='red', label='Q2 (Median)', linestyle='')
    ]
    ax.legend(handles=custom_lines,fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, folder, filename)


base_dir = "range_plots_smr_noak"
output_dirs = {
    "nuc": os.path.join(base_dir, "esbwr"),
    "smr": os.path.join(base_dir, "bwrx")
}

for prefix, subdir in output_dirs.items():
    for learning in ["fixed", "dynamic"]:
        data = prepare_data_all_materials(noak_results, prefix, learning)
        title = "Fixed" if learning == "fixed" else "Dynamic"
        filename = f"{prefix}_all_materials_{learning}.pdf"
        folder = subdir 
        plot_nuclear_all_materials_noak(data, title, filename)


with open("lcoe_simulation_results_nuclear_smr.pkl", "rb") as f:
    lcoe_results = pickle.load(f)

years = list(lcoe_results.keys())

def prepare_data_all_materials(results, reactor_prefix, learning_type):
    scenarios = ["premium", "no_premium", "market_price"]
    data = {scenario: {"bottoms": [], "medians": [], "tops": []} for scenario in scenarios}

    for year in years:
        for scenario in scenarios:
            key = f"{reactor_prefix}_lcoe_{learning_type}_{scenario}"
            if key in results[year]:
                q1, q2, q3 = custom_quartiles(results[year][key])
                data[scenario]["bottoms"].append(q1)
                data[scenario]["medians"].append(q2)
                data[scenario]["tops"].append(q3)
            else:
                data[scenario]["bottoms"].append(None)
                data[scenario]["medians"].append(None)
                data[scenario]["tops"].append(None)

    return data


def save_plot(fig, folder, filename):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_nuclear_all_materials_noak(nuclear_data, title, filename):
    scenarios = ["premium", "no_premium", "market_price"]
    colors = ["#4878A8", "#59935B", "#FFA500"]
    labels = [f"{year.replace('Deployment ', '')}" for year in years]
    fig, ax = plt.subplots(figsize=(20, 10))

    x_pos = np.arange(len(labels) * 3)
    bottoms, tops, medians = [], [], []

    for scenario in scenarios:
        bottoms += nuclear_data[scenario]["bottoms"]
        tops += nuclear_data[scenario]["tops"]
        medians += nuclear_data[scenario]["medians"]

    heights = [t - b for t, b in zip(tops, bottoms)]
    color_map = [colors[i // 5] for i in range(15)]
    ax.bar(x_pos, heights, bottom=bottoms, width=0.6, color=color_map, edgecolor='black', alpha=0.4)

    for i in range(15):
        ax.scatter(x_pos[i], medians[i], color='red', marker='o', s=80)
        ax.text(x_pos[i], medians[i] + 2, f"{medians[i]:.0f}", ha='center', va='bottom', fontsize=18, color='red')

    for group_start in range(0, 15, 5):
        base_q2 = medians[group_start]
        for i in range(1, 5):
            drop = 100 * (base_q2 - medians[group_start + i]) / base_q2
            ax.text(x_pos[group_start + i], tops[group_start + i] + 5,
                    f"{drop:.1f}% ↓", ha='center', fontsize=18, color='darkblue')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels * 3, rotation=45, ha='right')
    ax.set_ylabel("LCOE (2024 USD/MWh)",fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(tops + medians) + 10)

    custom_lines = [
        Line2D([0], [0], marker='o', color='#4878A8', label='Premium', linestyle=''),
        Line2D([0], [0], marker='o', color='#59935B', label='No Premium', linestyle=''),
        Line2D([0], [0], marker='o', color='#FFA500', label='Market Price', linestyle=''),
        Line2D([0], [0], marker='o', color='red', label='Q2 (Median)', linestyle='')
    ]
    ax.legend(handles=custom_lines,fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, folder, filename)


base_dir = "range_plots_smr_lcoe"
output_dirs = {
    "nuc": os.path.join(base_dir, "esbwr"),
    "smr": os.path.join(base_dir, "bwrx")
}

for prefix, subdir in output_dirs.items():
    for learning in ["fixed", "dynamic"]:
        data = prepare_data_all_materials(lcoe_results, prefix, learning)
        title = "Fixed" if learning == "fixed" else "Dynamic"
        filename = f"{prefix}_all_materials_{learning}.pdf"
        folder = subdir
        plot_nuclear_all_materials_noak(data, title, filename)