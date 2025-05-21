import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.stats import beta
from lowest_NOAK_comparison_plot import df_manual
from atb_data import values_from_ATB2024


np.random.seed(42)
num_simulations = 10000
years = [2030, 2035, 2040, 2045, 2050]

nuclear_df = df_manual[:-1]
wind_df = df_manual[-1:]

# --- Nuclear Material Costs ---
median_mc_premium = np.median(nuclear_df["Material_cost_with_nuclear_premium"].dropna())
std_mc_premium = np.std(nuclear_df["Material_cost_with_nuclear_premium"].dropna())

median_mc_no_premium = np.median(nuclear_df["Material_cost_wo_nuclear_premium"])
std_mc_no_premium = np.std(nuclear_df["Material_cost_wo_nuclear_premium"])

median_mc_market_price = np.median(nuclear_df["Material_cost_market_price"].dropna())
std_mc_market_price = np.std(nuclear_df["Material_cost_market_price"].dropna())

# --- Wind Material Costs ---
median_mc_wind = wind_df["Material_cost_market_price"].values[0]
std_mc_wind = 0.35 * median_mc_wind

# --- OCC Ranges ---
nuclear_occ_range = list(values_from_ATB2024["OCC + GCC"][:3])
wind_occ_range = list(values_from_ATB2024["OCC + GCC"][6:])

def sample_lr_nuclear():
    return np.random.triangular(0.05, 0.10, 0.15)

def sample_lr_wind():
    return np.random.beta(5, 10) * 0.3

def compute_b(lr):
    return np.log(1 - lr) / np.log(2)

def sample_deployment_triangular(q3, q2, q1):
    if np.isclose(q3, q1):
        return q3
    else:
        return np.random.triangular(left=q3, mode=q2, right=q1)

results = {f"Deployment {year}": {
    "noak_fixed_premium": [],
    "noak_dynamic_premium": [],
    "noak_fixed_no_premium": [],
    "noak_dynamic_no_premium": [],
    "noak_fixed_market_price": [],
    "noak_dynamic_market_price": [],
    "noak_wind_fixed": [],
    "noak_wind_dynamic": []
} for year in years}

# --- Run Monte Carlo Simulations ---
for sim in tqdm(range(num_simulations), desc="Running simulations"):

    # --- Sample Initial OCC ---
    occ_initial_nuclear = np.random.triangular(min(nuclear_occ_range), nuclear_occ_range[1], max(nuclear_occ_range))
    occ_initial_wind = np.random.triangular(min(wind_occ_range), wind_occ_range[1], max(wind_occ_range))

    # --- Sample MICs ---
    mic_premium = np.random.normal(median_mc_premium, std_mc_premium)
    mic_no_premium = np.random.normal(median_mc_no_premium, std_mc_no_premium)
    mic_market = np.random.normal(median_mc_market_price, std_mc_market_price)
    mic_wind = np.random.normal(median_mc_wind, std_mc_wind)

    # --- Sample Learning Rates ---
    lr_nuclear = sample_lr_nuclear()
    lr_wind = sample_lr_wind()
    b_fixed_nuclear = compute_b(lr_nuclear)
    b_fixed_wind = compute_b(lr_wind)

    # --- Sample Deployments for this run ---
    deployments_nuclear = {}
    deployments_wind = {}
    prev_deployment_nuclear = 1
    prev_deployment_wind = 1

    for year in years:
        # Nuclear
        nuclear_adv = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Adv", f"Deployment {year}"].values[0]
        nuclear_mod = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Mod", f"Deployment {year}"].values[0]
        nuclear_cons = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Cons", f"Deployment {year}"].values[0]

        while True:
            sampled = sample_deployment_triangular(nuclear_cons, nuclear_mod, nuclear_adv)
            if sampled >= prev_deployment_nuclear:
                deployments_nuclear[year] = sampled
                prev_deployment_nuclear = sampled
                break

        # Wind
        wind_adv = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "FOW-Class11-Adv", f"Deployment {year}"].values[0]
        wind_mod = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "FOW_Class11-Mod", f"Deployment {year}"].values[0]
        wind_cons = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "FOW_Class11-Cons", f"Deployment {year}"].values[0]

        while True:
            sampled = sample_deployment_triangular(wind_cons, wind_mod, wind_adv)
            if sampled >= prev_deployment_wind:
                deployments_wind[year] = sampled
                prev_deployment_wind = sampled
                break

    # --- Simulate Nuclear ---
    for mic_name, mic_value in zip(
        ["premium", "no_premium", "market_price"],
        [mic_premium, mic_no_premium, mic_market]
    ):
        overhead_fixed = occ_initial_nuclear - mic_value
        overhead_dynamic = occ_initial_nuclear - mic_value
        occ_prev_fixed = occ_initial_nuclear
        occ_prev_dynamic = occ_initial_nuclear
        N_prev = 1

        for year in years:
            N_curr = deployments_nuclear[year]

            # Fixed Learning
            overhead_fixed_new = overhead_fixed * (N_curr / N_prev) ** b_fixed_nuclear
            noak_fixed = mic_value + overhead_fixed_new
            results[f"Deployment {year}"][f"noak_fixed_{mic_name}"].append(noak_fixed)

            # Dynamic Learning
            overhead_ratio = overhead_dynamic / occ_prev_dynamic
            lr_dynamic = min(0.99, lr_nuclear * (1 + overhead_ratio))
            b_dynamic = compute_b(lr_dynamic)
            overhead_dynamic_new = overhead_dynamic * (N_curr / N_prev) ** b_dynamic
            noak_dynamic = mic_value + overhead_dynamic_new
            results[f"Deployment {year}"][f"noak_dynamic_{mic_name}"].append(noak_dynamic)

            occ_prev_fixed = noak_fixed
            occ_prev_dynamic = noak_dynamic
            overhead_fixed = overhead_fixed_new
            overhead_dynamic = overhead_dynamic_new
            N_prev = N_curr

    # --- Simulate Wind ---
    overhead_fixed_wind = occ_initial_wind - mic_wind
    overhead_dynamic_wind = occ_initial_wind - mic_wind
    occ_prev_fixed_wind = occ_initial_wind
    occ_prev_dynamic_wind = occ_initial_wind
    N_prev = 1

    for year in years:
        N_curr = deployments_wind[year]

        # Fixed Learning
        overhead_fixed_new = overhead_fixed_wind * (N_curr / N_prev) ** b_fixed_wind
        noak_fixed = mic_wind + overhead_fixed_new
        results[f"Deployment {year}"]["noak_wind_fixed"].append(noak_fixed)

        # Dynamic Learning
        overhead_ratio_wind = overhead_dynamic_wind / occ_prev_dynamic_wind
        lr_dynamic_wind = min(0.99, lr_wind * (1 + overhead_ratio_wind))
        b_dynamic_wind = compute_b(lr_dynamic_wind)
        overhead_dynamic_new = overhead_dynamic_wind * (N_curr / N_prev) ** b_dynamic_wind
        noak_dynamic = mic_wind + overhead_dynamic_new
        results[f"Deployment {year}"]["noak_wind_dynamic"].append(noak_dynamic)

        occ_prev_fixed_wind = noak_fixed
        occ_prev_dynamic_wind = noak_dynamic
        overhead_fixed_wind = overhead_fixed_new
        overhead_dynamic_wind = overhead_dynamic_new
        N_prev = N_curr


with open("noak_simulation_results_fixed_dynamic.pkl", "wb") as f:
    pickle.dump(results, f)

print("Simulation completed and saved.")
