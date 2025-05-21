import pickle
import pandas as pd
import numpy as np



with open("noak_simulation_results_fixed_dynamic.pkl", "rb") as f:
    noak_results = pickle.load(f)
with open("benchmark_simulation_results.pkl", "rb") as f:
    benchmark_data_noak = pickle.load(f)
with open("noak_simulation_results_nuclear_smr.pkl", "rb") as f:
    noak_results_smr = pickle.load(f)
with open("deterministic_noak.pkl", "rb") as f:
    deterministic_noak = pickle.load(f)

#printing deterministic noak
tech_labels = {
    "noak_nuc": "ESBWR (Nuclear)",
    "noak_smr": "BWRX-300 (SMR)",
    "noak_wind": "Floating Wind"
}
# Years to label
years = ["2030", "2035", "2040", "2045", "2050"]
# Print values
for tech_key, tech_name in tech_labels.items():
    print(f"\n{tech_name} NOAK OCC+GCC ($/kW):")
    for q in ["q1", "q2", "q3"]:
        values = deterministic_noak[tech_key][q]
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

# Setup
simulated_benchmark_noak_nuclear = {q: [] for q in ["q1", "q2", "q3"]}
simulated_benchmark_noak_smr = {q: [] for q in ["q1", "q2", "q3"]}
simulated_benchmark_noak_wind = {q: [] for q in ["q1", "q2", "q3"]}
sim_years = [y.split()[-1] for y in years] 

for year_label, year_key in zip(years, sim_years):
    sim_nuc = np.array(benchmark_data_noak[year_key]["noak_nuclear"])
    sim_smr =np.array(benchmark_data_noak[year_key]["noak_smr"])
    sim_wind = np.array(benchmark_data_noak[year_key]["noak_wind"])

    q1_nuc, q2_nuc, q3_nuc = custom_quartiles(sim_nuc)
    q1_smr,q2_smr,q3_smr = custom_quartiles(sim_smr)
    q1_wind, q2_wind, q3_wind = custom_quartiles(sim_wind)

    simulated_benchmark_noak_nuclear["q1"].append(q1_nuc)
    simulated_benchmark_noak_nuclear["q2"].append(q2_nuc)
    simulated_benchmark_noak_nuclear["q3"].append(q3_nuc)
    simulated_benchmark_noak_smr["q1"].append(q1_smr)
    simulated_benchmark_noak_smr["q2"].append(q2_smr)
    simulated_benchmark_noak_smr["q3"].append(q3_smr)
    simulated_benchmark_noak_wind["q1"].append(q1_wind)
    simulated_benchmark_noak_wind["q2"].append(q2_wind)
    simulated_benchmark_noak_wind["q3"].append(q3_wind)

# --- Print results ---
print("\nSimulated Benchmark Nuclear NOAK ($/kW):")
for q in simulated_benchmark_noak_nuclear:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_noak_nuclear[q], 1)}")
print("\nSimulated Benchmark SMR NOAK ($/kW):")
for q in simulated_benchmark_noak_smr:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_noak_smr[q], 1)}")
print("\nSimulated Benchmark Wind NOAK ($/kW):")
for q in simulated_benchmark_noak_wind:
    print(f"  {q.upper()}: {np.round(simulated_benchmark_noak_wind[q], 1)}")


#NOAK SIM nuclear vs WIND
keys_of_interest = [
    "noak_fixed_premium", "noak_dynamic_premium",
    "noak_fixed_no_premium", "noak_dynamic_no_premium",
    "noak_fixed_market_price", "noak_dynamic_market_price",
    "noak_wind_fixed", "noak_wind_dynamic"]
years = ["Deployment 2030", "Deployment 2035", "Deployment 2040", "Deployment 2045", "Deployment 2050"]
simulated_noak_fd = {key: {"q1": [], "q2": [], "q3": []} for key in keys_of_interest}
for year in years:
    for key in keys_of_interest:
        values = np.array(noak_results[year][key])
        q1, q2, q3 = custom_quartiles(values)
        simulated_noak_fd[key]["q1"].append(q1)
        simulated_noak_fd[key]["q2"].append(q2)
        simulated_noak_fd[key]["q3"].append(q3)

# --- Printout ---
print("\nSimulated NOAK Fixed/Dynamic ($/kW):")
for key in keys_of_interest:
    print(f"\n{key.upper()}:")
    for q in ["q1", "q2", "q3"]:
        print(f"  {q.upper()}: {np.round(simulated_noak_fd[key][q], 1)}")

#NOAK SIM ESBWR VS BWRX-300
keys_of_interest_smr=["nuc_noak_fixed_premium","nuc_noak_dynamic_premium", "nuc_noak_fixed_no_premium",
    "nuc_noak_dynamic_no_premium", "nuc_noak_fixed_market_price","nuc_noak_dynamic_market_price",
    "smr_noak_fixed_premium","smr_noak_dynamic_premium","smr_noak_fixed_no_premium","smr_noak_dynamic_no_premium",
    "smr_noak_fixed_market_price","smr_noak_dynamic_market_price"]
sim_noak_smr = {key: {"q1": [], "q2": [], "q3": []} for key in keys_of_interest_smr}
for year in years:
    for key in keys_of_interest_smr:
        values = np.array(noak_results_smr[year][key])
        q1, q2, q3 = custom_quartiles(values)
        sim_noak_smr[key]["q1"].append(q1)
        sim_noak_smr[key]["q2"].append(q2)
        sim_noak_smr[key]["q3"].append(q3)

# --- Example printout ---
print("\nSimulated NOAK Fixed/Dynamic ESBWR vs BWRX-300 ($/kW):")
for key in keys_of_interest_smr:
    print(f"\n{key.upper()}:")
    for q in ["q1", "q2", "q3"]:
        print(f"  {q.upper()}: {np.round(sim_noak_smr[key][q], 1)}")