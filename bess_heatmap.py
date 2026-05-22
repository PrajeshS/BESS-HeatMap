import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title='BESS Constraint Explorer', layout='wide')

# Sidebar Parameters
st.sidebar.header("Sim Parameters")
eff = st.sidebar.slider('One-Way Efficiency', 0.85, 1.0, 0.96, 0.01)
init_soc = st.sidebar.slider('Initial Year SOC (%)', 0, 100, 50) / 100.0
soc_min = st.sidebar.slider('Min SOC (%)', 0, 50, 10) / 100.0
soc_max = st.sidebar.slider('Max SOC (%)', 50, 100, 90) / 100.0

@st.cache_data
def load_pv():
    if os.path.exists('power time series 1 min (1)(in).csv'):
        return pd.read_csv('power time series 1 min (1)(in).csv')['Solar_MW'].values
    return np.zeros(525600)

# Optimized Vectorized Simulation Engine
def run_sim_vectorized(pv_data, p_cap, e_cap, eff_val, s_min, s_max, start_soc):
    n = len(pv_data)
    grid_export = np.zeros(n)
    soc_history = np.zeros(n)
    pv_capped = np.minimum(pv_data, 100.0)
    e_min, e_max = s_min * e_cap, s_max * e_cap
    curr_energy = start_soc * e_cap

    for t in range(n):
        prev_exp = grid_export[t-1] if t > 0 else pv_capped[0]
        raw_ramp = pv_capped[t] - prev_exp
        target = (raw_ramp - 3.0) if raw_ramp > 3.0 else ((raw_ramp + 3.0) if raw_ramp < -3.0 else 0)

        actual_bess = 0
        if target > 0:
            actual_bess = min(target, p_cap, (e_max - curr_energy) * 60 / eff_val)
            curr_energy += (actual_bess * eff_val) / 60
        elif target < 0:
            actual_bess = -min(abs(target), p_cap, (curr_energy - e_min) * 60 * eff_val)
            curr_energy += (actual_bess / eff_val) / 60

        grid_export[t] = pv_capped[t] - actual_bess
        soc_history[t] = (curr_energy / e_cap) * 100

    at_limit = np.sum((soc_history >= (s_max*100 - 0.1)) | (soc_history <= (s_min*100 + 0.1)))
    return (at_limit / n) * 100

pv_data = load_pv()

st.title("Live BESS Sizing Sensitivity Matrix (Vectorized Engine)")
st.markdown(f"**Operational Window:** {soc_min*100:.0f}% - {soc_max*100:.0f}% SOC | **Efficiency:** {eff*100:.0f}%")

powers = np.arange(5, 55, 5)
energies = np.arange(5, 55, 5)

with st.spinner('Calculating sensitivity matrix...'):
    matrix = np.zeros((len(powers), len(energies)))
    for i, p in enumerate(powers):
        for j, e in enumerate(energies):
            matrix[i, j] = run_sim_vectorized(pv_data, p, e, eff, soc_min, soc_max, init_soc)

    df = pd.DataFrame(matrix, index=powers, columns=energies)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': '% Time at Limits'})
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Energy (MWh)')
    ax.invert_yaxis()
    st.pyplot(fig)

st.info("The optimized engine allows for faster real-time exploration of sizing tradeoffs.")
