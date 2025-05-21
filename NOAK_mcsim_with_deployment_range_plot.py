import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.lines import Line2D

with open("noak_simulation_results_fixed_dynamic.pkl", "rb") as f:
    results = pickle.load(f)

years = list(results.keys())


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


def prepare_data_nuclear_wind(results, learning_type):
    nuclear_bottoms, nuclear_medians, nuclear_tops = [], [], []
    wind_bottoms, wind_medians, wind_tops = [], [], []

    for year in years:
        nuc_data = results[year][f"noak_{learning_type}_premium"]
        wind_data = results[year][f"noak_wind_{learning_type}"]

        q1, q2, q3 = custom_quartiles(nuc_data)
        nuclear_bottoms.append(q1)
        nuclear_medians.append(q2)
        nuclear_tops.append(q3)

        q1, q2, q3 = custom_quartiles(wind_data)
        wind_bottoms.append(q1)
        wind_medians.append(q2)
        wind_tops.append(q3)

    return nuclear_bottoms, nuclear_medians, nuclear_tops, wind_bottoms, wind_medians, wind_tops

def prepare_data_nuclear_all_materials(results, learning_type):
    scenarios = ["premium", "no_premium", "market_price"]
    data = {scenario: {"bottoms": [], "medians": [], "tops": []} for scenario in scenarios}

    for year in years:
        for scenario in scenarios:
            nuc_data = results[year][f"noak_{learning_type}_{scenario}"]
            q1, q2, q3 = custom_quartiles(nuc_data)
            data[scenario]["bottoms"].append(q1)
            data[scenario]["medians"].append(q2)
            data[scenario]["tops"].append(q3)

    return data

# --- Bar Plot Functions ---
def save_plot(fig, folder, filename):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_nuclear_wind(nuclear, wind, title, filename):
    nuc_bottoms, nuc_medians, nuc_tops = nuclear
    wind_bottoms, wind_medians, wind_tops = wind

    x_labels = [f"{year.replace('Deployment ', '')}\nNuclear" for year in years] + \
               [f"{year.replace('Deployment ', '')}\nWind" for year in years]

    x_pos = np.arange(len(x_labels))
    bottoms = nuc_bottoms + wind_bottoms
    tops = nuc_tops + wind_tops
    medians = nuc_medians + wind_medians
    heights = [top - bottom for top, bottom in zip(tops, bottoms)]

    fig, ax = plt.subplots(figsize=(18, 8))

    bar_colors = ['#4878A8'] * len(years) + ['#E75480'] * len(years)
    ax.bar(x_pos, heights, bottom=bottoms, width=0.6, color=bar_colors, edgecolor='black', alpha=0.4)

    ax.scatter(x_pos, bottoms, color='blue', marker='o')
    ax.scatter(x_pos, medians, color='red', marker='o')
    ax.scatter(x_pos, tops, color='green', marker='o')

    for i, q2 in enumerate(medians):
        ax.text(x_pos[i], q2 + 150, f"{q2:.0f}", ha='center', va='bottom', fontsize=18, color='red')

    for i in range(1, 5):
        drop = 100 * (nuc_medians[0] - nuc_medians[i]) / nuc_medians[0]
        ax.text(x_pos[i], tops[i] + 300, f"{drop:.1f}% ↓", ha='center', fontsize=18, color='darkblue')
        drop = 100 * (wind_medians[0] - wind_medians[i]) / wind_medians[0]
        ax.text(x_pos[i+5], tops[i+5] + 300, f"{drop:.1f}% ↓", ha='center', fontsize=18, color='darkgreen')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("NOAK OCC+GCC (2024 USD/kW)",fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)

    custom_lines = [
        Line2D([0], [0], marker='o', color='#4878A8', label='Nuclear (Premium)', linestyle=''),
        Line2D([0], [0], marker='o', color='#E75480', label='Wind', linestyle=''),
        Line2D([0], [0], marker='o', color='red', label='Q2 (Median)', linestyle='')
    ]
    ax.legend(handles=custom_lines, fontsize=18, loc='upper left')
    ax.set_ylim(0, max(tops + medians) + 1000)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    folder = f"bar_plots/{'fixed' if 'Fixed' in title else 'dynamic'}"
    save_plot(fig, folder, filename)

def plot_nuclear_all_materials(nuclear_data, title, filename):
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
    folder = f"bar_plots/{'fixed' if 'Fixed' in title else 'dynamic'}"
    save_plot(fig, folder, filename)


nuclear_fixed = prepare_data_nuclear_wind(results, "fixed")[:3]
wind_fixed = prepare_data_nuclear_wind(results, "fixed")[3:]
plot_nuclear_wind(nuclear_fixed, wind_fixed, "Fixed", "nuclear_wind_fixed.pdf")

nuclear_dynamic = prepare_data_nuclear_wind(results, "dynamic")[:3]
wind_dynamic = prepare_data_nuclear_wind(results, "dynamic")[3:]
plot_nuclear_wind(nuclear_dynamic, wind_dynamic, "dynamic", "nuclear_wind_dynamic.pdf")

nuclear_fixed_all = prepare_data_nuclear_all_materials(results, "fixed")
plot_nuclear_all_materials(nuclear_fixed_all, "Fixed", "nuclear_all_materials_fixed.pdf")

nuclear_dynamic_all = prepare_data_nuclear_all_materials(results, "dynamic")
plot_nuclear_all_materials(nuclear_dynamic_all, "dynamic", "nuclear_all_materials_dynamic.pdf")
