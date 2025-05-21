import numpy as np
import matplotlib.pyplot as plt
from atb_data import values_from_ATB2024
import pandas as pd
import pickle 
import os

os.makedirs("benchmark_comparison", exist_ok=True)

with open("noak_simulation_results_fixed_dynamic.pkl", "rb") as f:
    data = pickle.load(f)

with open("lcoe_simulation_results_fixed_dynamic.pkl", "rb") as f:
    data_lcoe = pickle.load(f)

with open("benchmark_simulation_results.pkl", "rb") as f:
    benchmark_data = pickle.load(f)

# ATB values
years = ["Deployment 2030", "Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
nuclear_idxs = [0, 1, 2]  # Adv, Mod, Cons
smr_idxs = [3, 4, 5]  # Adv, Mod, Cons
wind_idxs = [6, 7, 8]     # Adv, Mod, Cons
occ_gcc_foak = values_from_ATB2024["OCC + GCC"].values
learning_rates= values_from_ATB2024["learning_rate"].values

noak_nuc = {q: [] for q in ["q1", "q2", "q3"]}
noak_smr = {q: [] for q in ["q1","q2","q3"]}
noak_wind = {q: [] for q in ["q1", "q2", "q3"]}


def apply_learning(occ, deployment, learning_rate):
    noak = occ*deployment**(np.log(1-learning_rate)/np.log(2))
    return noak

for year in years:
    dep = values_from_ATB2024[year].values

    for i, q in enumerate(["q1", "q2", "q3"]):
        idx_nuc = nuclear_idxs[i]
        idx_smr = smr_idxs[i]
        idx_wind = wind_idxs[i]

        noak_nuc[q].append(apply_learning(occ_gcc_foak[idx_nuc], dep[idx_nuc], learning_rates[idx_nuc]))
        noak_smr[q].append(apply_learning(occ_gcc_foak[idx_smr],dep[idx_smr], learning_rates[idx_smr]))
        noak_wind[q].append(apply_learning(occ_gcc_foak[idx_wind], dep[idx_wind], learning_rates[idx_wind]))

deterministic_noak = {
    "noak_nuc": noak_nuc,
    "noak_smr": noak_smr,
    "noak_wind": noak_wind
}

with open("deterministic_noak.pkl","wb") as f:
    pickle.dump(deterministic_noak,f)

#Deterministic LCOE
# --- Fixed parameters from ATB ---
lcoe_deterministic = {
    "nuc": {"q1": [], "q2": [], "q3": []},
    "smr": {"q1": [], "q2": [], "q3": []},
    "wind": {"q1": [], "q2": [], "q3": []}
}

years_simple = ["2030", "2035", "2040", "2045", "2050"]

# Fixed values
WACC_nuc = values_from_ATB2024.loc[1, "WACC"]
WACC_smr = WACC_nuc
WACC_wind = values_from_ATB2024.loc[7, "WACC"]

CRP_nuc = values_from_ATB2024.loc[1, "CRP"]
CRP_smr = CRP_nuc
CRP_wind = values_from_ATB2024.loc[7, "CRP"]

PFF_nuc = values_from_ATB2024.loc[1, "PFF"]
PFF_smr = PFF_nuc
PFF_wind = values_from_ATB2024.loc[7, "PFF"]

def calc_crf(wacc, crp):
    return wacc / (1 - (1 / (1 + wacc) ** crp))

def calc_lcoe(crf, pff, cff, occ, fom, cf, vom=0, fuel=0):
    return ((crf * pff * cff * occ + fom) * 1000 / (cf * 8760)) + vom + fuel

# Compute for each quartile and year
for q_idx, q in enumerate(["q1", "q2", "q3"]):
    for y_idx, year in enumerate(years_simple):
        # Index mapping
        idx_nuc = nuclear_idxs[q_idx]
        idx_smr = smr_idxs[q_idx]
        idx_wind = wind_idxs[q_idx]

        # --- Nuclear ---
        occ_nuc = noak_nuc[q][y_idx]
        cff_nuc = values_from_ATB2024.loc[idx_nuc, "CFF"]
        fom_nuc = values_from_ATB2024.loc[idx_nuc, "FOM"]
        vom_nuc = values_from_ATB2024.loc[idx_nuc, "VOM"]
        fuel_nuc = values_from_ATB2024.loc[idx_nuc, "Fuel"]
        cf_nuc = values_from_ATB2024.loc[idx_nuc, "CF"]

        crf_nuc = calc_crf(WACC_nuc, CRP_nuc)
        lcoe_nuc = calc_lcoe(crf_nuc, PFF_nuc, cff_nuc, occ_nuc, fom_nuc, cf_nuc, vom_nuc, fuel_nuc)
        lcoe_deterministic["nuc"][q].append(lcoe_nuc)

        # --- SMR ---
        occ_smr = noak_smr[q][y_idx]
        cff_smr = values_from_ATB2024.loc[idx_smr, "CFF"]
        fom_smr = values_from_ATB2024.loc[idx_smr, "FOM"]
        vom_smr = values_from_ATB2024.loc[idx_smr, "VOM"]
        fuel_smr = values_from_ATB2024.loc[idx_smr, "Fuel"]
        cf_smr = values_from_ATB2024.loc[idx_smr, "CF"]

        crf_smr = calc_crf(WACC_smr, CRP_smr)
        lcoe_smr = calc_lcoe(crf_smr, PFF_smr, cff_smr, occ_smr, fom_smr, cf_smr, vom_smr, fuel_smr)
        lcoe_deterministic["smr"][q].append(lcoe_smr)

        # --- Wind ---
        occ_wind = noak_wind[q][y_idx]
        cff_wind = values_from_ATB2024.loc[idx_wind, "CFF"]
        fom_wind = values_from_ATB2024.loc[idx_wind, "FOM"]
        cf_wind = values_from_ATB2024.loc[idx_wind, "CF"]

        crf_wind = calc_crf(WACC_wind, CRP_wind)
        lcoe_wind = calc_lcoe(crf_wind, PFF_wind, cff_wind, occ_wind, fom_wind, cf_wind)
        lcoe_deterministic["wind"][q].append(lcoe_wind)


# --- Save deterministic LCOE results ---
with open("deterministic_lcoe.pkl", "wb") as f:
    pickle.dump(lcoe_deterministic, f)



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

# --- Compute q1, q2, q3 from benchmark simulation ---
simulated_benchmark_noak_nuclear = {q: [] for q in ["q1", "q2", "q3"]}
simulated_benchmark_noak_wind = {q: [] for q in ["q1", "q2", "q3"]}
sim_years = [y.split()[-1] for y in years] 

for year_label, year_key in zip(years, sim_years):
    sim_nuc = np.array(benchmark_data[year_key]["noak_nuclear"])
    sim_wind = np.array(benchmark_data[year_key]["noak_wind"])

    q1_nuc, q2_nuc, q3_nuc = custom_quartiles(sim_nuc)
    q1_wind, q2_wind, q3_wind = custom_quartiles(sim_wind)

    simulated_benchmark_noak_nuclear["q1"].append(q1_nuc)
    simulated_benchmark_noak_nuclear["q2"].append(q2_nuc)
    simulated_benchmark_noak_nuclear["q3"].append(q3_nuc)

    simulated_benchmark_noak_wind["q1"].append(q1_wind)
    simulated_benchmark_noak_wind["q2"].append(q2_wind)
    simulated_benchmark_noak_wind["q3"].append(q3_wind)

# --- Print results ---
print("\nSimulated Benchmark Nuclear NOAK ($/kW):")
for q in simulated_benchmark_noak_nuclear:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_noak_nuclear[q], 1)}")

print("\nSimulated Benchmark Wind NOAK ($/kW):")
for q in simulated_benchmark_noak_wind:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_noak_wind[q], 1)}")

with open("noak_simulation_results_fixed_dynamic.pkl", "rb") as f:
    noak_fd_data = pickle.load(f)

# --- Initialize storage ---
keys_of_interest = [
    "noak_fixed_premium", "noak_dynamic_premium",
    "noak_fixed_no_premium", "noak_dynamic_no_premium",
    "noak_fixed_market_price", "noak_dynamic_market_price",
    "noak_wind_fixed", "noak_wind_dynamic"
]

simulated_noak_fd = {key: {"q1": [], "q2": [], "q3": []} for key in keys_of_interest}

# --- Compute q1/q2/q3 for each key and year ---
for year in years:
    for key in keys_of_interest:
        values = np.array(noak_fd_data[year][key])
        q1, q2, q3 = custom_quartiles(values)
        simulated_noak_fd[key]["q1"].append(q1)
        simulated_noak_fd[key]["q2"].append(q2)
        simulated_noak_fd[key]["q3"].append(q3)

# --- Example printout ---
print("\nSimulated NOAK Fixed/Dynamic ($/kW):")
for key in keys_of_interest:
    print(f"\n{key.upper()}:")
    for q in ["q1", "q2", "q3"]:
        print(f"  {q.upper()}: {np.round(simulated_noak_fd[key][q], 1)}")


# --- Plotting ---
years_numeric = [int(y.split()[-1]) for y in years]

# --- Define grouped scenario data ---
scenario_groups = {
    "Fixed Learning Rate": {
        "Fixed Premium": simulated_noak_fd["noak_fixed_premium"],
        "Fixed No Premium": simulated_noak_fd["noak_fixed_no_premium"],
        "Fixed Market Price": simulated_noak_fd["noak_fixed_market_price"],
    },
    "Overhead-Weighted Learning Rate": {
        "OWLR Premium": simulated_noak_fd["noak_dynamic_premium"],
        "OWLR No Premium": simulated_noak_fd["noak_dynamic_no_premium"],
        "OWLR Market Price": simulated_noak_fd["noak_dynamic_market_price"],
    }
}


color_map = {
    "Benchmark": "black",
    "Fixed Premium": "tab:blue",
    "Fixed No Premium": "tab:green",
    "Fixed Market Price": "tab:orange",
    "OWLR Premium": "tab:blue",
    "OWLR No Premium": "tab:green",
    "OWLR Market Price": "tab:orange",
}

for learning_type, scenarios in scenario_groups.items():
    plt.figure(figsize=(12, 7))

    # Benchmark
    bench = simulated_benchmark_noak_nuclear
    color = color_map["Benchmark"]
    plt.plot(years_numeric, bench["q2"], label="Benchmark Q2", color=color, linewidth=2)
    plt.plot(years_numeric, bench["q1"], label="Benchmark Q1", color=color, linewidth=2)
    plt.plot(years_numeric, bench["q3"], label="Benchmark Q3", color=color, linewidth=2)
    x_annot = years_numeric[-1] + 0.5  # slightly to the right of last year

    plt.text(x_annot, bench["q1"][-1], "Benchmark Q1", color=color, fontsize=12, va='center')
    plt.text(x_annot, bench["q2"][-1], "Benchmark Q2", color=color, fontsize=12, va='center')
    plt.text(x_annot, bench["q3"][-1], "Benchmark Q3", color=color, fontsize=12, va='center')

    # --- Learning type scenarios ---
    for label, data in scenarios.items():
        color = color_map[label]
        plt.plot(years_numeric, data["q2"], label=f"{label} Q2", color=color, linewidth=1)
        plt.plot(years_numeric, data["q1"], label=f"{label} Q1", color=color, linewidth=1,linestyle="--")
        plt.plot(years_numeric, data["q3"], label=f"{label} Q3", color=color, linewidth=1, linestyle="-.")

    # Styling
    plt.xlabel("Year")
    plt.ylabel("NOAK Cost ($/kW)")
    plt.grid(True, linestyle=":")
    plt.xticks(years_numeric)
    plt.legend(loc="upper right", fontsize=12, ncol=2)
    plt.tight_layout()
    filename = f"benchmark_comparison/nuclear_noak_{learning_type.replace(' ', '_').lower()}.pdf"
    plt.savefig(filename, format="pdf")
    plt.close()


# --- Wind NOAK: Benchmark vs Fixed & OWLR ---
plt.figure(figsize=(12, 7))

# Data aliases with clear names
years_numeric = [int(y.split()[-1]) for y in years]
bench_wind = simulated_benchmark_noak_wind
fixed_wind = simulated_noak_fd["noak_wind_fixed"]
owlr_wind = simulated_noak_fd["noak_wind_dynamic"]


color_map_wind = {
    "Benchmark": "black",
    "Fixed Wind": "tab:blue",
    "OWLR Wind": "tab:orange"
}

color = color_map_wind["Benchmark"]
plt.plot(years_numeric, bench_wind["q2"], color=color, linewidth=2)
plt.plot(years_numeric, bench_wind["q1"], color=color, linewidth=2)
plt.plot(years_numeric, bench_wind["q3"], color=color, linewidth=2)

x_annot = years_numeric[-1] + 0.5
plt.text(x_annot, bench_wind["q1"][-1], "Benchmark Q1", color=color, fontsize=12, va='center')
plt.text(x_annot, bench_wind["q2"][-1], "Benchmark Q2", color=color, fontsize=12, va='center')
plt.text(x_annot, bench_wind["q3"][-1], "Benchmark Q3", color=color, fontsize=12, va='center')


color = color_map_wind["Fixed Wind"]
plt.plot(years_numeric, fixed_wind["q2"], label="Fixed Wind Q2", color=color, linewidth=1)
plt.plot(years_numeric, fixed_wind["q1"], label="Fixed Wind Q1", color=color, linewidth=1)
plt.plot(years_numeric, fixed_wind["q3"], label="Fixed Wind Q3", color=color, linewidth=1)


plt.xlabel("Year", fontsize=12)
plt.ylabel("NOAK Cost ($/kW)", fontsize=12)
plt.xticks(years_numeric, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle=":")
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()
plt.savefig("benchmark_comparison/wind_noak_fixed.pdf", format="pdf")
plt.close()


plt.figure(figsize=(12, 7))
years_numeric = [int(y.split()[-1]) for y in years]

bench = simulated_benchmark_noak_nuclear
det = noak_nuc

color_sim = "tab:blue"
color_det = "tab:orange"

# --- Plot Simulated Benchmark ---
plt.plot(years_numeric, bench["q1"], label="Simulated Q1", color=color_sim, linewidth=1)
plt.plot(years_numeric, bench["q2"], label="Simulated Q2", color=color_sim, linewidth=2)
plt.plot(years_numeric, bench["q3"], label="Simulated Q3", color=color_sim, linewidth=1)

# --- Plot Deterministic ---
plt.plot(years_numeric, det["q1"], label="Deterministic Q1", color=color_det, linewidth=1, linestyle="--")
plt.plot(years_numeric, det["q2"], label="Deterministic Q2", color=color_det, linewidth=2, linestyle="--")
plt.plot(years_numeric, det["q3"], label="Deterministic Q3", color=color_det, linewidth=1, linestyle="--")


plt.xlabel("Year", fontsize=12)
plt.ylabel("NOAK Cost ($/kW)", fontsize=12)
plt.xticks(years_numeric, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle=":")
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()


plt.savefig("benchmark_comparison/nuclear_sim_vs_deterministic.pdf", format="pdf")
plt.close()

plt.figure(figsize=(12, 7))
years_numeric = [int(y.split()[-1]) for y in years]


bench_wind = simulated_benchmark_noak_wind
det_wind = noak_wind

color_sim = "tab:blue"
color_det = "tab:orange"

# --- Plot Simulated Benchmark ---
plt.plot(years_numeric, bench_wind["q1"], label="Simulated Q1", color=color_sim, linewidth=1)
plt.plot(years_numeric, bench_wind["q2"], label="Simulated Q2", color=color_sim, linewidth=2)
plt.plot(years_numeric, bench_wind["q3"], label="Simulated Q3", color=color_sim, linewidth=1)

# --- Plot Deterministic ---
plt.plot(years_numeric, det_wind["q1"], label="Deterministic Q1", color=color_det, linewidth=1, linestyle="--")
plt.plot(years_numeric, det_wind["q2"], label="Deterministic Q2", color=color_det, linewidth=2, linestyle="--")
plt.plot(years_numeric, det_wind["q3"], label="Deterministic Q3", color=color_det, linewidth=1, linestyle="--")


plt.xlabel("Year", fontsize=12)
plt.ylabel("NOAK Cost ($/kW)", fontsize=12)
plt.xticks(years_numeric, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle=":")
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()

plt.savefig("benchmark_comparison/wind_sim_vs_deterministic.pdf", format="pdf")
plt.close()
