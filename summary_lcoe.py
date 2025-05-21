import pickle
import pandas as pd
import numpy as np

with open("lcoe_simulation_results_fixed_dynamic.pkl", "rb") as f:
    lcoe_results = pickle.load(f)
with open("benchmark_lcoe_simulation_results.pkl", "rb") as f:
    benchmark_data_lcoe = pickle.load(f)
with open("lcoe_simulation_results_nuclear_smr.pkl", "rb") as f:
    lcoe_results_smr = pickle.load(f)
with open("deterministic_lcoe.pkl", "rb") as f:
    deterministic_lcoe = pickle.load(f)

#printing deterministic noak
tech_labels = {
    "nuc": "ESBWR (Nuclear)",
    "smr": "BWRX-300 (SMR)",
    "wind": "Floating Wind"
}
# Years to label
years = ["2030", "2035", "2040", "2045", "2050"]
# Print values
for tech_key, tech_name in tech_labels.items():
    print(f"\n{tech_name} LCOE ($/kW):")
    for q in ["q1", "q2", "q3"]:
        values = deterministic_lcoe[tech_key][q]
        year_values = ", ".join([f"{year}: {val:.1f}" for year, val in zip(years, values)])
        print(f"  {q.upper()}: {year_values}")

# Define custom quartiles function
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


#Printing Benchmark
simulated_benchmark_lcoe_nuclear = {q: [] for q in ["q1", "q2", "q3"]}
simulated_benchmark_lcoe_smr = {q: [] for q in ["q1", "q2", "q3"]}
simulated_benchmark_lcoe_wind = {q: [] for q in ["q1", "q2", "q3"]}
years = [f"Deployment {y}" for y in ["2030", "2035", "2040", "2045", "2050"]]

for year_key in years:
    sim_nuc = np.array(benchmark_data_lcoe[year_key]["lcoe_nuc"])
    sim_smr =np.array(benchmark_data_lcoe[year_key]["lcoe_smr"])
    sim_wind = np.array(benchmark_data_lcoe[year_key]["lcoe_wind"])

    q1_nuc, q2_nuc, q3_nuc = custom_quartiles(sim_nuc)
    q1_smr,q2_smr,q3_smr = custom_quartiles(sim_smr)
    q1_wind, q2_wind, q3_wind = custom_quartiles(sim_wind)

    simulated_benchmark_lcoe_nuclear["q1"].append(q1_nuc)
    simulated_benchmark_lcoe_nuclear["q2"].append(q2_nuc)
    simulated_benchmark_lcoe_nuclear["q3"].append(q3_nuc)
    simulated_benchmark_lcoe_smr["q1"].append(q1_smr)
    simulated_benchmark_lcoe_smr["q2"].append(q2_smr)
    simulated_benchmark_lcoe_smr["q3"].append(q3_smr)
    simulated_benchmark_lcoe_wind["q1"].append(q1_wind)
    simulated_benchmark_lcoe_wind["q2"].append(q2_wind)
    simulated_benchmark_lcoe_wind["q3"].append(q3_wind)

# --- Print results ---
print("\nSimulated Benchmark Nuclear LCOE ($/MWh):")
for q in simulated_benchmark_lcoe_nuclear:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_lcoe_nuclear[q], 1)}")
print("\nSimulated Benchmark SMR LCOE ($/MWh):")
for q in simulated_benchmark_lcoe_smr:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_lcoe_smr[q], 1)}")
print("\nSimulated Benchmark Wind LCOE ($/MWh):")
for q in simulated_benchmark_lcoe_wind:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_lcoe_wind[q], 1)}")


#LCOE simulation Nuclear vs Wind
years = ["Deployment 2030", "Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
scenarios = ["premium", "no_premium", "market_price"]
methods = ["fixed", "dynamic"]
techs = scenarios + ["wind"]  # "wind" only appears under fixed/dynamic
lcoe_keys = [f"lcoe_{m}_{s}" for m in methods for s in techs if not (s in scenarios and m == "dynamic" and s == "wind")]

simulated_lcoe_fd = {key: {"q1": [], "q2": [], "q3": []} for key in lcoe_keys}
for year in years:
    for method in methods:
        for scenario in lcoe_results[year][method]:
            values = np.array(lcoe_results[year][method][scenario])
            q1, q2, q3 = custom_quartiles(values)
            key = f"lcoe_{method}_{scenario}"
            if key not in simulated_lcoe_fd:
                continue  # Skip if wind_dynamic doesn't exist
            simulated_lcoe_fd[key]["q1"].append(q1)
            simulated_lcoe_fd[key]["q2"].append(q2)
            simulated_lcoe_fd[key]["q3"].append(q3)

# --- Printout ---
print("\nSimulated LCOE Fixed/Dynamic ($/MWh):")
for key in lcoe_keys:
    print(f"\n{key.upper()}:")
    for q in ["q1", "q2", "q3"]:
        print(f"  {q.upper()}: {np.round(simulated_lcoe_fd[key][q], 1)}")

#LCOE simulation ESBWR vs BWRX-300
years = ["Deployment 2030", "Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
keys_of_interest_smr = [
    "nuc_lcoe_fixed_premium", "nuc_lcoe_dynamic_premium",
    "nuc_lcoe_fixed_no_premium", "nuc_lcoe_dynamic_no_premium",
    "nuc_lcoe_fixed_market_price", "nuc_lcoe_dynamic_market_price",
    "smr_lcoe_fixed_premium", "smr_lcoe_dynamic_premium",
    "smr_lcoe_fixed_no_premium", "smr_lcoe_dynamic_no_premium",
    "smr_lcoe_fixed_market_price", "smr_lcoe_dynamic_market_price"
]

sim_lcoe_smr = {key: {"q1": [], "q2": [], "q3": []} for key in keys_of_interest_smr}

for year in years:
    for key in keys_of_interest_smr:
        values = np.array(lcoe_results_smr[year][key])
        q1, q2, q3 = custom_quartiles(values)
        sim_lcoe_smr[key]["q1"].append(q1)
        sim_lcoe_smr[key]["q2"].append(q2)
        sim_lcoe_smr[key]["q3"].append(q3)

print("\nSimulated LCOE for ESBWR vs BWRX-300 ($/MWh):")
for key in keys_of_interest_smr:
    print(f"\n{key.upper()}:")
    for q in ["q1", "q2", "q3"]:
        print(f"  {q.upper()}: {np.round(sim_lcoe_smr[key][q], 1)}")

