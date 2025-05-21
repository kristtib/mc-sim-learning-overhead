import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from atb_data import values_from_ATB2024


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 18
})

with open("noak_simulation_results_fixed_dynamic.pkl", "rb") as f:
    results = pickle.load(f)


def clean_and_quartiles(data):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return data, None, None, None
    min_val = np.min(data)
    p25 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    max_val = np.max(data)
    q1 = (min_val + p25) / 2
    q2 = median
    q3 = (p75 + max_val) / 2
    return data, q1, q2, q3

nuclear_materials = ["premium", "no_premium", "market_price"]
colors = {
    "premium": "#4878A8",
    "no_premium": "#59935B",
    "market_price": "#FFA500",
    "wind": "#E75480"
}


os.makedirs("hist_plots/nuclear", exist_ok=True)
os.makedirs("hist_plots/wind", exist_ok=True)
os.makedirs("hist_plots/summary", exist_ok=True)

# --- Individual Nuclear KDE plots ---
def plot_learning_type(results, year_label, learning_type):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    materials = ["premium", "no_premium", "market_price"]
    titles = ["Premium", "Non Premium", "Market"]
    colors_local = ["#4878A8", "#59935B", "#FFA500"]

    for idx, (material, title, color) in enumerate(zip(materials, titles, colors_local)):
        ax = axes[idx]
        data, q1, q2, q3 = clean_and_quartiles(results[year_label][f"noak_{learning_type}_{material}"])
        sns.kdeplot(data, ax=ax, color=color, fill=True)
        ax.axvline(q2, color='red', linestyle='solid', linewidth=2)
        ax.text(q2 + 300, ax.get_ylim()[1] * 0.85, f"Q2:\n{q2:.0f} $/kW", color='red', fontsize=14, ha='left')
        ax.set_title(f"{title}")
        ax.set_xlabel("NOAK OCC+GCC [$/kW]")
        if idx == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(w_pad=0.5)
    plt.savefig(f"hist_plots/nuclear/nuclear_{year_label.replace(' ', '_').lower()}_{learning_type}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

# --- Individual Wind KDE plots ---
def plot_wind_learning_type(results, learning_type):
    years_sorted = sorted(results.keys(), key=lambda x: int(x.split()[-1]))
    fig, axes = plt.subplots(1, len(years_sorted), figsize=(12, 2.5), sharey=False)
    color = "#E75480"

    for idx, year in enumerate(years_sorted):
        ax = axes[idx]
        data, q1, q2, q3 = clean_and_quartiles(results[year][f"noak_wind_{learning_type}"])
        sns.kdeplot(data, ax=ax, color=color, fill=True)
        ax.axvline(q2, color='red', linestyle='solid', linewidth=2)
        ax.text(q2 + 300, ax.get_ylim()[1] * 0.75, f"Q2:\n{q2:.0f} $/kW", color='red', fontsize=14, ha='left')
        ax.set_title(f"{year.split()[-1]}")
        ax.set_xlabel("NOAK [$/kW]")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(w_pad=0.5)
    plt.savefig(f"hist_plots/wind/wind_{learning_type}_learning.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# --- OCC initial plot for 2030 ---
def plot_occ_initial_2030(results):
    material = "premium"
    color = "#4878A8"
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    data = np.array(results["Deployment 2030"][f"noak_fixed_{material}"])
    data = data[np.isfinite(data)]

    sns.kdeplot(data, ax=ax, fill=True, color=color)
    q2 = values_from_ATB2024["OCC + GCC"][1]
    ax.axvline(q2, color='red', linestyle='-', linewidth=1)
    ax.text(q2 + 800, ax.get_ylim()[1] * 0.7, f"Q2:\n{q2:.0f} $/kW", color='red', fontsize=14)

    xmin = int(np.floor(np.min(data) / 100.0)) * 100 - 300
    xmax = int(np.ceil(np.max(data) / 100.0)) * 100 + 300
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("NOAK OCC+GCC [$/kW]",fontsize=12)
    ax.set_ylabel("Density",fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("hist_plots/summary/initial_occ_2030_kde.pdf", format="pdf", bbox_inches="tight")
    plt.close()



for year in sorted(results.keys(), key=lambda x: int(x.split()[-1])):
    plot_learning_type(results, year, "fixed")
    plot_learning_type(results, year, "dynamic")

plot_wind_learning_type(results, "fixed")
plot_wind_learning_type(results, "dynamic")
plot_occ_initial_2030(results)

# --- Summary KDE Grid for Nuclear ---
def generate_noak_summary_nuclear():
    years = ["Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
    materials = nuclear_materials

    fig, axes = plt.subplots(len(years), 6, figsize=(18, 12), sharey=True)
    for i, year in enumerate(years):
        for j, (learning, col_offset) in enumerate(zip(["fixed", "dynamic"], [0, 3])):
            for k, mat in enumerate(materials):
                ax = axes[i, col_offset + k]
                data, _, q2, _ = clean_and_quartiles(results[year][f"noak_{learning}_{mat}"])
                sns.kdeplot(data, ax=ax, fill=True, color=colors[mat])
                ax.axvline(q2, color='red', linestyle='-', linewidth=2)
                ax.text(q2 + 300, ax.get_ylim()[1] * 0.8, f"Q2:\n{q2:.0f}", color='red', fontsize=18)
                ax.set_xlim(0, 9000)
                if i == len(years) - 1:
                    ax.set_xlabel("OCC+GCC [$/kW]")
                else:
                    ax.set_xticklabels([])
                if col_offset + k == 0:
                    ax.set_ylabel(year.replace("Deployment ", ""))
                else:
                    ax.set_yticklabels([])
                if i == 0:
                    header = "FLR" if learning == "fixed" else "OWLR"
                    ax.set_title(f"{mat.replace('_', ' ').title()}\n{header}")
                else:
                    ax.set_title("")

    plt.tight_layout()
    plt.savefig(f"hist_plots/summary/nuclear_kde_summary_grid.pdf", format="pdf")
    plt.close()

# --- Summary KDE Grid for Wind ---
def generate_noak_summary_wind():
    years = ["Deployment 2030", "Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
    learning_types = ["fixed", "dynamic"]

    fig, axes = plt.subplots(len(years), 2, figsize=(10, len(years)*2), sharex=True, sharey='row')
    for i, year in enumerate(years):
        for j, lt in enumerate(learning_types):
            ax = axes[i, j]
            data, _, q2, _ = clean_and_quartiles(results[year][f"noak_wind_{lt}"])
            sns.kdeplot(data, ax=ax, fill=True, color=colors["wind"])
            ax.axvline(q2, color='red', linestyle='-', linewidth=2)
            ax.text(q2 + 2000, ax.get_ylim()[1] * 0.5, f"Q2:\n{q2:.0f}", color='red', fontsize=18)
            ax.set_xlim(0, 15000)
            if i == len(years) - 1:
                ax.set_xlabel("NOAK OCC+GCC [$/kW]")
                ax.set_xticklabels([0, 5000, 10000, 15000],fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(year.replace("Deployment ", ""))
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            if i == 0:
                title = "FLR" if lt == "fixed" else "OWLR"
                ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"hist_plots/summary/wind_kde_summary_grid.pdf", format="pdf", bbox_inches="tight")
    plt.close()


generate_noak_summary_nuclear()
generate_noak_summary_wind()
