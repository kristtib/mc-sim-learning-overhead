import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 18
})

# Load LCOE results
with open("lcoe_simulation_results_fixed_dynamic.pkl", "rb") as f:
    lcoe_results = pickle.load(f)


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

output_dir = "hist_plots_lcoe"
os.makedirs(output_dir, exist_ok=True)

# --- Plotting KDE Histograms ---
for year_label in sorted(lcoe_results.keys(), key=lambda x: int(x.split()[-1])):
    for learning_type in ["fixed", "dynamic"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 6))

        for i, material in enumerate(nuclear_materials):
            ax = axes[0, i]
            data, q1, q2, q3 = clean_and_quartiles(lcoe_results[year_label][learning_type][material])
            sns.kdeplot(data, ax=ax, fill=True, color=colors[material])
            if q2:
                ax.axvline(q2, color='red', linewidth=2)
                ax.text(q2 + 0.3, ax.get_ylim()[1]*0.85, f"Q2:\n{q2:.1f} $/MWh", color='red', fontsize=12, ha='left')
            ax.set_title(material.replace('_', ' ').title())
            ax.set_xlabel("LCOE [$/MWh]")
            if i == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            ax.grid(True, linestyle='--', alpha=0.5)

        ax = axes[1, 1]
        data, q1, q2, q3 = clean_and_quartiles(lcoe_results[year_label][learning_type]["wind"])
        sns.kdeplot(data, ax=ax, fill=True, color=colors["wind"])
        if q2:
            ax.axvline(q2, color='red', linewidth=2)
            ax.text(q2 + 0.3, ax.get_ylim()[1]*0.85, f"Q2:\n{q2:.1f} $/MWh", color='red', fontsize=12, ha='left')
        ax.set_title("Floating Offshore Wind")
        ax.set_xlabel("LCOE [$/MWh]")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle='--', alpha=0.5)

        for j in range(3):
            if j != 1:
                axes[1, j].axis('off')

        plt.tight_layout()
        filename = f"{output_dir}/lcoe_{year_label.replace(' ', '_').lower()}_{learning_type}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()

# --- Summary KDE Grid for Nuclear ---
def generate_lcoe_summary_nuclear():
    years = sorted([y for y in lcoe_results.keys() if not y.endswith("2030")], key=lambda x: int(x.split()[-1]))
    materials = nuclear_materials

    all_values = []
    for year in years:
        for learning in ["fixed", "dynamic"]:
            for mat in materials:
                all_values.extend(lcoe_results[year][learning][mat])
    all_values = np.array(all_values)
    xmin = np.floor(np.min(all_values) / 5) * 5
    xmax = np.ceil(np.max(all_values) / 5) * 5

    fig, axes = plt.subplots(len(years), 6, figsize=(18, 12), sharey=True)
    for i, year in enumerate(years):
        for j, (learning, col_offset) in enumerate(zip(["fixed", "dynamic"], [0, 3])):
            for k, mat in enumerate(materials):
                ax = axes[i, col_offset + k]
                data, _, q2, _ = clean_and_quartiles(lcoe_results[year][learning][mat])
                sns.kdeplot(data, ax=ax, fill=True, color=colors[mat])
                ax.axvline(q2, color='red', linestyle='-', linewidth=2)
                ax.text(q2 + 10, ax.get_ylim()[1] * 0.8, f"Q2:\n{q2:.1f} ", color='red', fontsize=18)
                ax.set_xlim(xmin, xmax)

                if i == len(years) - 1:
                    ax.set_xlabel("LCOE [$/MWh]")
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
    os.makedirs("hist_plots_lcoe/summary", exist_ok=True)
    plt.savefig("hist_plots_lcoe/summary/nuclear_kde_summary_grid.pdf", format="pdf", bbox_inches="tight")
    plt.close()


# --- Summary KDE Grid for Wind ---
def generate_lcoe_summary_wind():
    years = sorted([y for y in lcoe_results.keys()], key=lambda x: int(x.split()[-1]))

    all_data = []
    for year in years:
        for learning_type in ["fixed", "dynamic"]:
            all_data.extend(lcoe_results[year][learning_type]["wind"])
    all_data = np.array(all_data)
    xmin = int(np.floor(np.min(all_data) / 5) * 5)
    xmax = int(np.ceil(np.max(all_data) / 5) * 5)

    fig, axes = plt.subplots(len(years), 2, figsize=(10, len(years)*2), sharey=True)

    for i, year in enumerate(years):
        for j, learning_type in enumerate(["fixed", "dynamic"]):
            ax = axes[i, j]
            data, _, q2, _ = clean_and_quartiles(lcoe_results[year][learning_type]["wind"])
            sns.kdeplot(data, ax=ax, fill=True, color=colors["wind"])
            ax.axvline(q2, color='red', linestyle='-', linewidth=2)
            ax.text(q2 + 20, ax.get_ylim()[1] * 0.6, f"Q2:\n{q2:.1f}", color='red', fontsize=18)

            ax.set_xlim(xmin, xmax)

            if i == len(years) - 1:
                ax.set_xlabel("LCOE [$/MWh]")
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(year.replace("Deployment ", ""))
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            if i == 0:
                header = "Fixed learning rate" if learning_type == "fixed" else "Overhead-weighted learning rate"
                ax.set_title(header)

    plt.tight_layout()
    os.makedirs("hist_plots_lcoe/summary", exist_ok=True)
    plt.savefig("hist_plots_lcoe/summary/wind_kde_summary_grid.pdf", format="pdf", bbox_inches="tight")
    plt.close()


# --- Generate Initial LCOE (2030) ---
def plot_lcoe_initial_2030():
    year = "Deployment 2030"
    data = lcoe_results[year]["fixed"]["premium"]
    data = np.array(data)
    data = data[np.isfinite(data)]
    q2 = np.percentile(data, 50)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(data, ax=ax, fill=True, color=colors["premium"])
    ax.axvline(q2, color='red', linestyle='-', linewidth=2)
    ax.text(q2 + 10, ax.get_ylim()[1] * 0.7, f"Q2:\n{q2:.1f} $/MWh", color='red', fontsize=12)

    ax.set_xlabel("LCOE [$/MWh]", fontsize = 12)
    ax.set_ylabel("Density",fontsize=12)
    ax.tick_params(axis='both', labelsize=12) 
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("hist_plots_lcoe/summary/initial_lcoe_2030_kde.pdf", format="pdf", bbox_inches="tight")
    plt.close()

generate_lcoe_summary_nuclear()
generate_lcoe_summary_wind()
plot_lcoe_initial_2030()
