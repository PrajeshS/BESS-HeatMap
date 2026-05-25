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
    .stSlider > label { font-weight: bold; color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">BESS Sizing Constraint Sensitivity Simulator</div>', unsafe_allow_html=True)

# Sidebar Parameters
st.sidebar.header("⚙️ Simulation Parameters")
eff = st.sidebar.number_input('One-Way Efficiency', 0.80, 1.0, 0.96, 0.01)
init_soc = st.sidebar.number_input('Initial Year SOC (%)', 0, 100, 50) / 100.0
soc_min_val = st.sidebar.number_input('Min Operating SOC (%)', 0, 50, 10) / 100.0
soc_max_val = st.sidebar.number_input('Max Operating SOC (%)', 50, 100, 90) / 100.0

@st.cache_data
def load_pv():
    path = 'power time series 1 min (1)(in).csv'
    if os.path.exists(path):
        return pd.read_csv(path)['Solar_MW'].values
    return np.zeros(525600)

@njit
def run_sim_fast(pv_data, p_cap, e_cap, eff_val, s_min, s_max, start_soc):
    n = len(pv_data)
    curr_energy = start_soc * e_cap
    e_min, e_max = s_min * e_cap, s_max * e_cap
    at_limit_count = 0
    prev_export = pv_data[0] if pv_data[0] < 100.0 else 100.0

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
        soc_val = (curr_energy / e_cap) * 100.0
        if soc_val >= (s_max*100.0 - 0.1) or soc_val <= (s_min*100.0 + 0.1):
            at_limit_count += 1
    return (at_limit_count / n) * 100.0

pv_data = load_pv()

st.sidebar.markdown("--- ")

# Industry Standard Durations
durations = [0.25, 0.5, 1.0, 2.0, 4.0]
# Fixed Power Range as requested
powers = np.arange(5, 45, 5) # 5, 10, 15, 20, 25

with st.spinner('Generating Industry-Standard Sensitivity Matrix...'):
    matrix = np.zeros((len(powers), len(durations)))
    for i in range(len(powers)):
        for j in range(len(durations)):
            p_val = float(powers[i])
            e_val = p_val * durations[j]
            matrix[i, j] = run_sim_fast(pv_data, p_val, e_val, eff, soc_min_val, soc_max_val, init_soc)

    cols = [f"{d}h ({1/d if d!=0 else 0:.2f}C)" for d in durations]
    df = pd.DataFrame(matrix, index=powers, columns=cols)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': '% Time at SOC Limits'})
    ax.set_ylabel('BESS Power Rating (MW)')
    ax.set_xlabel('BESS Discharge Duration (Hours / C-Rate)')
    ax.invert_yaxis()
    plt.title("BESS Constraint Sensitivity Matrix")
    st.pyplot(fig)


