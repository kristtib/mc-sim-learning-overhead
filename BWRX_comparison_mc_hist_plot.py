import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Plotting config
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 18
})

# Load NOAK simulation results only
with open("noak_simulation_results_nuclear_smr.pkl", "rb") as f:
    noak_results = pickle.load(f)


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
    "market_price": "#FFA500"
}
reactors = {"nuc": "esbwr", "smr": "bwrx"}


for rdir in ["esbwr", "bwrx", "summary"]:
    os.makedirs(f"hist_plots_smr_noak/{rdir}", exist_ok=True)

# KDE plotting
def plot_kde_results(year_label, learning_type, reactor_prefix, output_dir, result_set, y_label, suffix):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for idx, mat in enumerate(nuclear_materials):
        ax = axes[idx]
        key = f"{reactor_prefix}_{suffix}_{learning_type}_{mat}"
        data, _, q2, _ = clean_and_quartiles(result_set[year_label][key])
        sns.kdeplot(data, ax=ax, color=colors[mat], fill=True)
        ax.axvline(q2, color='red', linestyle='-', linewidth=2)
        ax.text(q2 + 300, ax.get_ylim()[1] * 0.85, f"Q2:\n{q2:.0f}", color='red', fontsize=14)
        ax.set_title(mat.replace("_", " ").title())
        ax.set_xlabel(y_label)
        if idx == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(f"hist_plots_smr_noak/{output_dir}/{reactor_prefix}_{year_label.replace(' ', '_').lower()}_{learning_type}_{suffix}.pdf", format="pdf")
    plt.close()


for year in sorted(noak_results.keys(), key=lambda x: int(x.split()[-1])):
    for prefix, folder in reactors.items():
        plot_kde_results(year, "fixed", prefix, folder, noak_results, "NOAK OCC+GCC [$/kW]", "noak")
        plot_kde_results(year, "dynamic", prefix, folder, noak_results, "NOAK OCC+GCC [$/kW]", "noak")

# Summary grid
def generate_summary_grid(results, prefix, label, ylabel, suffix):
    years = ["Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
    fig, axes = plt.subplots(len(years), 6, figsize=(18, 12), sharey=True)
    for i, year in enumerate(years):
        for j, (learning, offset) in enumerate(zip(["fixed", "dynamic"], [0, 3])):
            for k, mat in enumerate(nuclear_materials):
                ax = axes[i, offset + k]
                key = f"{prefix}_{suffix}_{learning}_{mat}"
                data, _, q2, _ = clean_and_quartiles(results[year][key])
                sns.kdeplot(data, ax=ax, fill=True, color=colors[mat])
                ax.axvline(q2, color='red', linestyle='-', linewidth=2)
                ax.text(q2 + 300, ax.get_ylim()[1] * 0.8, f"Q2:\n{q2:.0f}", color='red', fontsize=18)
                ax.set_xlim(0, 9000)
                if i == len(years) - 1:
                    ax.set_xlabel(label)
                else:
                    ax.set_xticklabels([])
                if offset + k == 0:
                    ax.set_ylabel(year.replace("Deployment ", ""))
                else:
                    ax.set_yticklabels([])
                if i == 0:
                    title = f"{mat.replace('_', ' ').title()}\n{'FLR' if learning == 'fixed' else 'OWLR'}"
                    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"hist_plots_smr_noak/summary/{prefix}_{suffix}_summary_grid.pdf", format="pdf")
    plt.close()

generate_summary_grid(noak_results, "nuc", "OCC+GCC [$/kW]", "OCC+GCC [$/kW]", "noak")
generate_summary_grid(noak_results, "smr", "OCC+GCC [$/kW]", "OCC+GCC [$/kW]", "noak")

print("NOAK-only plots generated successfully.")

#LCOE plotting
for rdir in ["esbwr", "bwrx", "summary"]:
    os.makedirs(f"hist_plots_smr_lcoe/{rdir}", exist_ok=True)

# --- Load LCOE simulation results ---
with open("lcoe_simulation_results_nuclear_smr.pkl", "rb") as f:
    lcoe_results = pickle.load(f)
# KDE plot for each year and scenario
def plot_kde_results(year_label, learning_type, reactor_prefix, output_dir, result_set, y_label, suffix):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for idx, mat in enumerate(nuclear_materials):
        ax = axes[idx]
        key = f"{reactor_prefix}_{suffix}_{learning_type}_{mat}"
        data, _, q2, _ = clean_and_quartiles(result_set[year_label][key])
        sns.kdeplot(data, ax=ax, color=colors[mat], fill=True)
        ax.axvline(q2, color='red', linestyle='-', linewidth=2)
        ax.text(q2 + 5, ax.get_ylim()[1] * 0.85, f"Q2:\n{q2:.0f}", color='red', fontsize=14)
        ax.set_title(mat.replace("_", " ").title())
        ax.set_xlabel(y_label)
        if idx == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(f"hist_plots_smr_lcoe/{output_dir}/{reactor_prefix}_{year_label.replace(' ', '_').lower()}_{learning_type}_{suffix}.pdf", format="pdf")
    plt.close()

# Run individual plots
for year in sorted(lcoe_results.keys(), key=lambda x: int(x.split()[-1])):
    for prefix, folder in reactors.items():
        plot_kde_results(year, "fixed", prefix, folder, lcoe_results, "LCOE [$/MWh]", "lcoe")
        plot_kde_results(year, "dynamic", prefix, folder, lcoe_results, "LCOE [$/MWh]", "lcoe")

# Summary grid
def generate_summary_grid(results, prefix, label, ylabel, suffix):
    years = ["Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
    fig, axes = plt.subplots(len(years), 6, figsize=(18, 12), sharey=True)
    for i, year in enumerate(years):
        for j, (learning, offset) in enumerate(zip(["fixed", "dynamic"], [0, 3])):
            for k, mat in enumerate(nuclear_materials):
                ax = axes[i, offset + k]
                key = f"{prefix}_{suffix}_{learning}_{mat}"
                data, _, q2, _ = clean_and_quartiles(results[year][key])
                sns.kdeplot(data, ax=ax, fill=True, color=colors[mat])
                ax.axvline(q2, color='red', linestyle='-', linewidth=2)
                ax.text(q2 + 5, ax.get_ylim()[1] * 0.8, f"Q2:\n{q2:.0f}", color='red', fontsize=18)
                ax.set_xlim(0, 150)
                if i == len(years) - 1:
                    ax.set_xlabel(label)
                else:
                    ax.set_xticklabels([])
                if offset + k == 0:
                    ax.set_ylabel(year.replace("Deployment ", ""))
                else:
                    ax.set_yticklabels([])
                if i == 0:
                    title = f"{mat.replace('_', ' ').title()}\n{'FLR' if learning == 'fixed' else 'OWLR'}"
                    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"hist_plots_smr_lcoe/summary/{prefix}_{suffix}_summary_grid.pdf", format="pdf")
    plt.close()


generate_summary_grid(lcoe_results, "nuc", "LCOE [$/MWh]", "LCOE [$/MWh]", "lcoe")
generate_summary_grid(lcoe_results, "smr", "LCOE [$/MWh]", "LCOE [$/MWh]", "lcoe")

print("LCOE plots generated and saved in hist_plots_smr_lcoe/")