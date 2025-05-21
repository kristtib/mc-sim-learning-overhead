import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- Load results ---
with open("lcoe_simulation_results_fixed_dynamic.pkl", "rb") as f:
    lcoe_results = pickle.load(f)

years = [2030, 2035, 2040, 2045, 2050]
year_labels = [f"Deployment {y}" for y in years]
colors = {"premium": "#4878A8", "no_premium": "#59935B", "market_price": "#FFA500", "wind": "#E75480"}


plt.rcParams.update({
    "font.size": 18
})


def custom_quartiles(data):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return None, None, None
    q1 = (np.min(data) + np.percentile(data, 25)) / 2
    q2 = np.percentile(data, 50)
    q3 = (np.percentile(data, 75) + np.max(data)) / 2
    return q1, q2, q3


def prepare_nuclear_wind(results, learning_type):
    nuc_b, nuc_m, nuc_t = [], [], []
    wind_b, wind_m, wind_t = [], [], []
    for year in year_labels:
        q1_n, q2_n, q3_n = custom_quartiles(results[year][learning_type]["premium"])
        q1_w, q2_w, q3_w = custom_quartiles(results[year][learning_type]["wind"])
        nuc_b.append(q1_n); nuc_m.append(q2_n); nuc_t.append(q3_n)
        wind_b.append(q1_w); wind_m.append(q2_w); wind_t.append(q3_w)
    return (nuc_b, nuc_m, nuc_t), (wind_b, wind_m, wind_t)

def prepare_nuclear_all_materials(results, learning_type):
    scenarios = ["premium", "no_premium", "market_price"]
    data = {s: {"bottoms": [], "medians": [], "tops": []} for s in scenarios}
    for year in year_labels:
        for s in scenarios:
            q1, q2, q3 = custom_quartiles(results[year][learning_type][s])
            data[s]["bottoms"].append(q1)
            data[s]["medians"].append(q2)
            data[s]["tops"].append(q3)
    return data

# --- Plot nuclear vs wind ---
def plot_nuclear_vs_wind(nuc, wind, learning_type):
    nuc_b, nuc_m, nuc_t = nuc
    wind_b, wind_m, wind_t = wind
    x_labels = [f"{y}\nNuclear" for y in years] + [f"{y}\nWind" for y in years]
    x_pos = np.arange(len(x_labels))

    bottoms = nuc_b + wind_b
    tops = nuc_t + wind_t
    medians = nuc_m + wind_m
    heights = [top - bottom for top, bottom in zip(tops, bottoms)]
    bar_colors = [colors["premium"]] * 5 + [colors["wind"]] * 5

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x_pos, heights, bottom=bottoms, color=bar_colors, edgecolor='black', alpha=0.4)
    ax.scatter(x_pos, medians, color='red', label='Q2 (Median)')

    for i, q2 in enumerate(medians):
        ax.text(x_pos[i], q2 + 5, f"{q2:.1f}", ha='center', fontsize=18, color='red')

    for i in range(1, 5):
        drop_n = 100 * (nuc_m[0] - nuc_m[i]) / nuc_m[0]
        drop_w = 100 * (wind_m[0] - wind_m[i]) / wind_m[0]
        ax.text(x_pos[i], tops[i] + 5, f"{drop_n:.1f}% ↓", ha='center', fontsize=18, color='darkblue')
        ax.text(x_pos[i + 5], tops[i + 5] + 5, f"{drop_w:.1f}% ↓", ha='center', fontsize=18, color='darkgreen')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("LCOE (USD/MWh)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(tops) + 10)

    plt.tight_layout()
    output_dir = f"range_plots_LCOE/bar_plots/{learning_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/nuclear_wind_{learning_type}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

# --- Plot all nuclear scenarios ---
def plot_nuclear_all(data, learning_type):
    scenarios = ["premium", "no_premium", "market_price"]
    color_map = [colors[s] for s in scenarios for _ in years]
    markers = ['o', 'o', 'o']
    x_pos = np.arange(len(scenarios) * len(years))

    bottoms, tops, medians = [], [], []
    for s in scenarios:
        bottoms.extend(data[s]["bottoms"])
        tops.extend(data[s]["tops"])
        medians.extend(data[s]["medians"])

    heights = [top - bottom for top, bottom in zip(tops, bottoms)]
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.bar(x_pos, heights, bottom=bottoms, color=color_map, edgecolor='black', alpha=0.4)

    for i in range(len(x_pos)):
        idx = i // 5
        ax.scatter(x_pos[i], medians[i], color=color_map[i], marker=markers[idx], s=80)
        ax.text(x_pos[i], medians[i] + 2, f"{medians[i]:.1f}", ha='center', fontsize=18, color='red')

    for group in range(0, 15, 5):
        base = medians[group]
        for i in range(1, 5):
            drop = 100 * (base - medians[group + i]) / base
            ax.text(x_pos[group + i], tops[group + i] + 5, f"{drop:.1f}% ↓", ha='center', fontsize=18, color='darkblue')

    xtick_labels = []
    for scenario in ["Premium", "No Premium", "Market Price"]:
        xtick_labels.extend([f"{year}" for year in years])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylabel("LCOE (USD/MWh)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(tops) + 10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors["premium"], lw=4, label='Premium'),
        Line2D([0], [0], color=colors["no_premium"], lw=4, label='No Premium'),
        Line2D([0], [0], color=colors["market_price"], lw=4, label='Market Price')
    ]
    ax.legend(handles=legend_elements, title="Material Scenario", fontsize=18)

    plt.tight_layout()
    output_dir = f"range_plots_LCOE/bar_plots/{learning_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/nuclear_all_materials_{learning_type}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

# --- Run Plots ---
nuc_fixed, wind_fixed = prepare_nuclear_wind(lcoe_results, "fixed")
plot_nuclear_vs_wind(nuc_fixed, wind_fixed, "fixed")

nuc_dyn, wind_dyn = prepare_nuclear_wind(lcoe_results, "dynamic")
plot_nuclear_vs_wind(nuc_dyn, wind_dyn, "dynamic")

nuc_fixed_all = prepare_nuclear_all_materials(lcoe_results, "fixed")
plot_nuclear_all(nuc_fixed_all, "fixed")

nuc_dyn_all = prepare_nuclear_all_materials(lcoe_results, "dynamic")
plot_nuclear_all(nuc_dyn_all, "dynamic")
