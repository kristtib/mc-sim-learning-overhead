import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

labels = [
    "Turbine (excl. foundation)",
    "HV Transformer (x2)",
    "Semi-submersible Substructure",
    "Secondary Steel",
    "Anchors and Moorings",
    "Cables and Accessories"
]
values = [211.622952, 0.6795, 600, 21.948, 787, 76.7]


colors = plt.get_cmap('tab20c').colors 
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title("Material Input Cost Breakdown for Floating Offshore Wind (Total: 1698 $/kW)", fontsize=11)

import os
os.makedirs("material_input_plots", exist_ok=True)
plt.tight_layout()
plt.savefig("material_input_plots/mic_fow_piechart.pdf", format="pdf")
plt.show()

base_df = pd.DataFrame({
    "Material": ["Steel, low alloy", "Steel, high alloy", "Cast Iron", "Aluminium", "Copper", "Zinc", "Lead", "Polymer", "Ceramic/Glass", "Concrete", "Synthetic rope"],
    "Turbine": [65.28, 9.89, 14.38, 0.90, 1.1, 0, 0, 4.6, 9.65, 0, 0],
    "Substructure": [0, 266.6, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    "Mooring": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Array cables": [1.6, 0, 0, 0.6, 0, 0, 1.24, 1, 0, 0, 0],
    "Export cables": [9.03, 0, 0, 0, 9.3, 0, 7.3, 7.95, 0, 0, 0],
    "Offshore substation": [10.32, 0.05, 0, 0.09, 0.14, 0.014, 0, 0.06, 0, 0, 0],
    "Onshore substation": [0.6, 0.01, 0, 0.12, 0.42, 0, 0, 0.06, 0, 1.2, 0],
    "Material cost": [500, 1000, 2200, 2600, 9000, 3000, 2000, 2000, 2000, 400, 39000]
})

material_cost = base_df["Material cost"].values
components = base_df.columns[1:-1]

depth_range = np.arange(30, 1310, 10) 
line_range = np.arange(3, 10)
X, Y = np.meshgrid(depth_range, line_range)
Z_mooring = np.zeros_like(X, dtype=float)
Z_total = np.zeros_like(X, dtype=float)

for i, n_mooring_lines in enumerate(line_range):
    for j, water_depth in enumerate(depth_range):
        length_of_mooring_line = 6 * water_depth

        # Mooring material calculations
        steel_mass_kg = 0.8 * length_of_mooring_line * 500 * n_mooring_lines
        synt_mass_kg = 0.2 * length_of_mooring_line * 50 * n_mooring_lines

        anchors_kg = 50000 * n_mooring_lines # 50 tonnes per anchor
        mooring_connectors_kg = 5000 * n_mooring_lines
        shackles_kg = 5000 * n_mooring_lines
        tri_plate_kg = 2000 * 2 * n_mooring_lines
        clumps_kg = 5000 * 2 * n_mooring_lines

        steel_total_t = (steel_mass_kg + anchors_kg + mooring_connectors_kg + shackles_kg + tri_plate_kg) / 1000 / 15
        synt_total_t = synt_mass_kg / 1000 / 15
        cast_iron_t = clumps_kg / 1000 / 15

        df = base_df.copy()
        df.loc[df["Material"] == "Steel, high alloy", "Mooring"] = steel_total_t
        df.loc[df["Material"] == "Synthetic rope", "Mooring"] = synt_total_t
        df.loc[df["Material"] == "Cast Iron", "Mooring"] = cast_iron_t

        mic_dict = {}
        for component in components:
            material_mass = df[component].values
            mic_total = np.sum(material_mass * material_cost) / 1000 
            mic_dict[component] = mic_total

        Z_mooring[i, j] = mic_dict["Mooring"]
        Z_total[i, j] = sum(mic_dict.values())


fig = plt.figure(figsize=(16, 7))

# Mooring MIC
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_mooring, cmap='viridis', edgecolor='none')
ax1.set_title("Mooring MIC ($/kW)")
ax1.set_xlabel("Water Depth (m)")
ax1.set_ylabel("Mooring Lines")
ax1.set_zlabel("Mooring MIC ($/kW)")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Total MIC
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_total, cmap='plasma', edgecolor='none')
ax2.set_title("Total MIC ($/kW)")
ax2.set_xlabel("Water Depth (m)")
ax2.set_ylabel("Mooring Lines")
ax2.set_zlabel("Total MIC ($/kW)")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.suptitle("MIC vs Water Depth and Number of Mooring Lines", fontsize=14)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))

# Mooring MIC plot
plt.subplot(1, 2, 1)
for i, n_mooring_lines in enumerate(line_range):
    plt.plot(depth_range, Z_mooring[i], label=f"{n_mooring_lines} mooring lines")
plt.title("Mooring MIC vs Water Depth")
plt.xlabel("Water Depth (m)")
plt.ylabel("Mooring MIC ($/kW)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Number of Mooring Lines")

# Total MIC plot
plt.subplot(1, 2, 2)
for i, n_mooring_lines in enumerate(line_range):
    plt.plot(depth_range, Z_total[i], label=f"{n_mooring_lines} mooring lines")
plt.title("Total MIC vs Water Depth")
plt.xlabel("Water Depth (m)")
plt.ylabel("Total MIC ($/kW)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Number of Mooring Lines")

plt.tight_layout()
plt.show()