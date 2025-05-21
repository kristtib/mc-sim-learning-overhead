# mc-sim-learning-overhead
# Simulation Framework for Capital Cost Learning in Clean Energy Technologies

This repository contains the Python simulation framework developed as part of my Master's thesis, titled:

**"Overhead-Focused Learning in Cost Projections for SMRs, Nuclear, and Floating Offshore Wind"**

## 📄 Overview

This project models future cost trajectories (OCC and LCOE)  for three clean energy technologies:

- ⚛️ Large Nuclear Reactors
- ⚛️ Small Modular Reactors (SMRs)
- 🌊 Floating Offshore Wind (FOW)

The framework is built around the idea that **only overhead (non-material) costs are subject to learning effects**, while material costs are treated as fixed baselines derived from material intensity and pricing data.

---

## 🧠 Key Concepts

- **Cost Disaggregation**: Overnight capital cost is split into:
  - Material Input Costs (MIC)
  - Overhead costs (labor, management, engineering, etc.)

- **Learning Models**:
  - **Fixed Learning Rate (FLR)**: Learning applies only to overhead, at a fixed rate.
  - **Overhead-Weighted Learning Rate (OWLR)**: Learning rate adapts to the overhead share in the cost structure.

- **Monte Carlo Simulations**: Used to incorporate uncertainty in:
  - Learning rates
  - Material prices
  - Deployment trajectories
  - Economic variables

---
