import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D


material_and_OCC_data = {
    "Project": ["PWR","ABWR","ESBWR","EPR","Flamanville 3", "Vogtle 3&4", "Hinkley Point C", "Barakah", "Olkiluoto 3", "Sizewell B", "Floating Offshore Wind"],
    "Capacity_per_unit_MW": [1000, 1380, 1500, 1600, 1650, 1100, 1600, 1400, 1600, 1345, 450],
    "Material_cost_with_nuclear_premium": [297.92, 399.42, 270.21, 381.31, 359.95, 402.56, 735.77, 523.62, 319.65, 589.66, None],
    "Material_cost_wo_nuclear_premium": [144.92, 199.61, 131.66, 189.93, 196.19, 214.29, 386.47, 283.79, 166.03, 317.87, None],
    "Material_cost_market_price": [74.33, 110.61, 67.87, 104.30, 133.33, 138.54, 242.50, 190.64, 101.50, 211.15, 1700],
}
df_manual = pd.DataFrame(material_and_OCC_data)

# --- Group definitions ---
group_1_projects = ["PWR", "ABWR", "ESBWR", "EPR"]
group_2_projects = ["Flamanville 3", "Vogtle 3&4", "Hinkley Point C", "Barakah", "Olkiluoto 3", "Sizewell B"]

group_1 = df_manual[df_manual["Project"].isin(group_1_projects)].sort_values(by="Capacity_per_unit_MW")
group_2 = df_manual[df_manual["Project"].isin(group_2_projects)].sort_values(by="Capacity_per_unit_MW")

# --- Colors ---
color_premium = '#4878A8' # Blue
color_nopremium = '#59935B' # Green
color_market = '#FFA500' # Orange

# --- Range Plot per group ---
def plot_group(df_group, group_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_group))

    for i, row in enumerate(df_group.itertuples()):
        ax.bar(i, row.Material_cost_with_nuclear_premium - row.Material_cost_market_price,
               bottom=row.Material_cost_market_price,
               width=0.5, color='#CCCCCC', alpha=0.4)

        # Q1 (Market)
        ax.scatter(i, row.Material_cost_market_price, color=color_market, s=40, zorder=3)
        ax.text(i + 0.1, row.Material_cost_market_price, f"{row.Material_cost_market_price:.0f}", ha='left', va='center', fontsize=12, color=color_market)

        # Q2 (No Premium)
        ax.scatter(i, row.Material_cost_wo_nuclear_premium, color=color_nopremium, s=50, zorder=3)
        ax.text(i + 0.1, row.Material_cost_wo_nuclear_premium, f"{row.Material_cost_wo_nuclear_premium:.0f}", ha='left', va='center', fontsize=12, color=color_nopremium)

        # Q3 (Premium)
        ax.scatter(i, row.Material_cost_with_nuclear_premium, color=color_premium, s=40, zorder=3)
        ax.text(i + 0.1, row.Material_cost_with_nuclear_premium, f"{row.Material_cost_with_nuclear_premium:.0f}", ha='left', va='center', fontsize=12, color=color_premium)

        # Project label with capacity
        ax.text(i, -40, f"{row.Project}\n{row.Capacity_per_unit_MW} MW", ha='center', va='top', fontsize=8)

    ax.set_xticks([])
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Material Input Cost [$/kW]")
    ax.grid(True, linestyle='--', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color=color_premium, label='MIC with Nuclear Premium', markersize=12, linestyle=''),
        Line2D([0], [0], marker='o', color=color_nopremium, label='MIC without Nuclear Premium', markersize=12, linestyle=''),
        Line2D([0], [0], marker='o', color=color_market, label='MIC based on Market Price', markersize=12, linestyle='')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True)

    os.makedirs("material_input_plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"material_input_plots/material_cost_range_{group_name.replace(' ', '_').lower()}.pdf", format="pdf")
    plt.close()

plot_group(group_1, "Group 1 - Standard Reactors")
plot_group(group_2, "Group 2 - Other Reactors")

# --- Print stats---
def print_median_and_std(label, series):
    clean_data = pd.to_numeric(series, errors='coerce').dropna()
    median = np.median(clean_data)
    std_dev = np.std(clean_data)
    print(f"{label}:")
    print(f"  Median: {median:.2f} $/kW")
    print(f"  Std Dev: {std_dev:.2f} $/kW\n")

print_median_and_std("With Nuclear Premium", df_manual["Material_cost_with_nuclear_premium"][:-1])
print_median_and_std("Without Nuclear Premium", df_manual["Material_cost_wo_nuclear_premium"][:-1])
print_median_and_std("Market Price", df_manual["Material_cost_market_price"][:-1])

# --- Median + Std bar plot ---
df_non_wind = df_manual[df_manual["Project"] != "Floating Offshore Wind"]

def compute_median_and_std(df):
    def clean(series): return pd.to_numeric(series, errors='coerce').dropna()
    stats = {}
    for label, series in {
        "With Nuclear Premium": df["Material_cost_with_nuclear_premium"],
        "No Nuclear Premium": df["Material_cost_wo_nuclear_premium"],
        "Market Price": df["Material_cost_market_price"]
    }.items():
        data = clean(series)
        median = np.median(data)
        std_dev = np.std(data)
        stats[label] = {"median": median, "std": std_dev}
    return stats

# Plotting
def plot_median_range(stats, output_dir="material_input_plots"):
    labels = list(stats.keys())
    medians = [v["median"] for v in stats.values()]
    errors = [v["std"] for v in stats.values()]
    
    colors = ["#4878A8", "#59935B", "#FFA500"] 

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))

    for i in range(len(x)):
        q1 = medians[i] - errors[i]
        q3 = medians[i] + errors[i]
        ax.bar(x[i], q3 - q1, bottom=q1, width=0.5, color=colors[i], alpha=0.3, edgecolor='none')

        ax.scatter(x[i], medians[i], color='black', s=50, zorder=3)
        ax.text(x[i] + 0.1, medians[i], f"{medians[i]:.0f}", ha='left', va='center', fontsize=12, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel("Material Input Cost [$/kW]")
    ax.set_ylim(0, 600)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Material Input Cost (Median ± Std Dev)")

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/material_cost_median_std_ranges_refined.pdf", format="pdf")
    plt.close()


material_stats = compute_median_and_std(df_non_wind)
plot_median_range(material_stats)
