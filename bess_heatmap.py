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

def run_sim(pv_data, p_cap, e_cap, eff_val, s_min, s_max, start_soc):
    n = len(pv_data)
    grid_export = np.zeros(n)
    soc_history = np.zeros(n)
    curr_energy = start_soc * e_cap
    e_min, e_max = s_min * e_cap, s_max * e_cap

    for t in range(n):
        pv = np.minimum(pv_data[t], 100.0)
        prev_export = grid_export[t-1] if t > 0 else pv
        raw_ramp = pv - prev_export
        target = (raw_ramp - 3.0) if raw_ramp > 3.0 else ((raw_ramp + 3.0) if raw_ramp < -3.0 else 0)
        
        actual_bess = 0
        if target > 0:
            space = (e_max - curr_energy) * 60 / eff_val
            actual_bess = min(target, p_cap, space)
            curr_energy += (actual_bess * eff_val) / 60
        elif target < 0:
            avail = (curr_energy - e_min) * 60 * eff_val
            actual_bess = -min(abs(target), p_cap, avail)
            curr_energy += (actual_bess / eff_val) / 60

        grid_export[t] = pv - actual_bess
        soc_history[t] = (curr_energy / e_cap) * 100
    
    at_limit = np.sum((soc_history >= (s_max*100 - 0.1)) | (soc_history <= (s_min*100 + 0.1)))
    return (at_limit / n) * 100

pv_data = load_pv()

st.title("BESS Sizing Sensitivity Matrix")
st.markdown(f"Exploring % time at limits ({soc_min*100}%-{soc_max*100}% SOC) with {eff*100}% Efficiency.")

if st.button('GENERATE MATRIX'):
    powers = [10, 20, 30, 40, 50]
    energies = [10, 20, 30, 40, 50]
    matrix = np.zeros((len(powers), len(energies)))
    
    progress_bar = st.progress(0)
    total_steps = len(powers) * len(energies)
    
    for i, p in enumerate(powers):
        for j, e in enumerate(energies):
            matrix[i, j] = run_sim(pv_data, p, e, eff, soc_min, soc_max, init_soc)
            progress_bar.progress((i * len(energies) + j + 1) / total_steps)
            
    df = pd.DataFrame(matrix, index=powers, columns=energies)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Energy (MWh)')
    st.pyplot(fig)
