import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from numba import njit

st.set_page_config(page_title='BESS Constraint Sensitivity Simulator', layout='wide')

# --- Professional Engineering CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #58a6ff; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">BESS Constraint Sensitivity Simulator: % Time at Min or Max SOC</div>', unsafe_allow_html=True)

# Sidebar Parameters
st.sidebar.header("⚙️ Simulation Parameters")
eff = st.sidebar.slider('One-Way Efficiency', 0.85, 1.0, 0.96, 0.01)
init_soc = st.sidebar.slider('Initial Year SOC (%)', 0, 100, 50) / 100.0
soc_min_val = st.sidebar.slider('Min SOC (%)', 0, 50, 10) / 100.0
soc_max_val = st.sidebar.slider('Max SOC (%)', 50, 100, 90) / 100.0

@st.cache_data
def load_pv():
    path = 'power time series 1 min (1)(in).csv'
    if os.path.exists(path):
        return pd.read_csv(path)['Solar_MW'].values
    return np.zeros(525600)

# Numba JIT Optimized Engine (Compiled to Machine Code)
@njit
def run_sim_fast(pv_data, p_cap, e_cap, eff_val, s_min, s_max, start_soc):
    n = len(pv_data)
    # Pre-allocate for speed, though we only strictly need current values for this metric
    # we maintain the logic structure
    curr_energy = start_soc * e_cap
    e_min, e_max = s_min * e_cap, s_max * e_cap
    at_limit_count = 0
    
    prev_export = pv_data[0]
    
    for t in range(n):
        pv = pv_data[t]
        if pv > 100.0: pv = 100.0
        
        raw_ramp = pv - prev_export
        target = 0.0
        if raw_ramp > 3.0: target = raw_ramp - 3.0
        elif raw_ramp < -3.0: target = raw_ramp + 3.0

        actual_bess = 0.0
        if target > 0:
            space_pwr = (e_max - curr_energy) * 60.0 / eff_val
            actual_bess = min(target, p_cap, space_pwr)
            curr_energy += (actual_bess * eff_val) / 60.0
        elif target < 0:
            avail_pwr = (curr_energy - e_min) * 60.0 * eff_val
            actual_bess = -min(abs(target), p_cap, avail_pwr)
            curr_energy += (actual_bess / eff_val) / 60.0

        prev_export = pv - actual_bess
        
        # Check limits with epsilon
        soc_val = (curr_energy / e_cap) * 100.0
        if soc_val >= (s_max*100.0 - 0.1) or soc_val <= (s_min*100.0 + 0.1):
            at_limit_count += 1

    return (at_limit_count / n) * 100.0

pv_data = load_pv()

powers = np.arange(5, 55, 5)
energies = np.arange(5, 55, 5)

with st.spinner('Calculating Sensitivity Matrix...'):
    matrix = np.zeros((len(powers), len(energies)))
    for i in range(len(powers)):
        for j in range(len(energies)):
            matrix[i, j] = run_sim_fast(pv_data, float(powers[i]), float(energies[j]), eff, soc_min_val, soc_max_val, init_soc)

    df = pd.DataFrame(matrix, index=powers, columns=energies)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': '% Time at Limits'})
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Energy (MWh)')
    ax.invert_yaxis()
    st.pyplot(fig)
