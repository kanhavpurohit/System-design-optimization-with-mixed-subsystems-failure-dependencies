import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy

# Import backend modules
import backend.core_math as core_math
from backend.core_math import evaluate_solution
from backend.algorithms import optimize_de, optimize_mrfo, optimize_sfla, optimize_shade
from backend.transient import system_availability_at, mission_penalised_objective
from backend.mo_algorithms import optimize_mode 

st.set_page_config(
    page_title="System design optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .main {background-color: #FAFAFA;}
    h1, h2, h3 {color: #2C3E50;}
    .stButton>button {width: 100%; background-color: #2C3E50; color: white;}
    .stButton>button:hover {background-color: #1A252F; color: white;}
    </style>
""", unsafe_allow_html=True)

#Sidebar Inputs
st.sidebar.header("Simulation Parameters")

system_choice = st.sidebar.selectbox(
    "Select System Configuration",
    ["Complex Bridge Network", "Series-Parallel System"]
)

opt_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Single-Objective (Cost minimization)", "Multi-Objective (Cost vs. Availability)"]
)

st.sidebar.markdown("---")

pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=500, value=100)
max_gen = st.sidebar.number_input("Max Generations", min_value=10, max_value=1000, value=200)

if opt_mode == "Single-Objective (Cost minimization)":
    # Number input parsing and strict validation for Availability Targets
    as_input_str = st.sidebar.text_input(
        "Minimum System Availability ($A_{s,min}$)",
        value="0.90, 0.95, 0.99",
        help="Enter values between 0.0 and 0.9999 separated by commas. E.g., 0.90, 0.95"
    )

    avail_targets = []
    input_error = False

    try:
        raw_vals = [float(x.strip()) for x in as_input_str.split(",") if x.strip()]
        for val in raw_vals:
            if 0.0 <= val <= 0.9999:
                avail_targets.append(val)
            else:
                st.sidebar.error(f"Value {val} is invalid. $A_{{s,min}}$ must be between 0.0 and 0.9999.")
                input_error = True
    except ValueError:
        st.sidebar.error("Invalid input. Please enter valid numbers separated by commas.")
        input_error = True

    avail_targets = sorted(list(set(avail_targets)))

    num_runs = st.sidebar.slider("Number of Independent Runs", min_value=1, max_value=20, value=5)

    st.sidebar.markdown("### Algorithms to Run")
    run_de   = st.sidebar.checkbox("Differential Evolution (DE)", value=True)
    run_shade = st.sidebar.checkbox("SHADE (Self-Adaptive DE) ★", value=True)
    run_mrfo = st.sidebar.checkbox("Manta Ray Foraging (MRFO)", value=True)
    run_sfla = st.sidebar.checkbox("Shuffled Frog Leaping (SFLA)", value=True)

else:
    st.sidebar.markdown("### Multi-Objective Algorithm")
    run_mode = st.sidebar.checkbox("Multi-Objective DE (MODE)", value=True)

# --- Plotting Functions ---
def plot_box_diagrams(data, system_name, targets):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
    fig.tight_layout(pad=5.0)
    axes = np.atleast_1d(axes)
    labels = [f'({chr(97+i)})' for i in range(n)]

    for ax, As_min, label in zip(axes, targets, labels):
        algos = list(data[As_min].keys())
        box_data = [data[As_min][alg]['costs'] for alg in algos]
        ax.boxplot(box_data, labels=algos, patch_artist=True,
                   boxprops=dict(facecolor='#E8F0FE', color='#2C3E50'),
                   medianprops=dict(color='#E74C3C', linewidth=2))
        ax.set_title(f"{system_name} ($A_{{s,min}} = {As_min}$)")
        ax.set_ylabel("System cost")
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.text(0.5, -0.2, label, transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold')
    return fig

def plot_convergence_curves(data, system_name, targets):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
    fig.tight_layout(pad=5.0)
    axes = np.atleast_1d(axes)
    labels = [f'({chr(97+i)})' for i in range(n)]
    colors = {'DE': '#1f77b4', 'SHADE': '#d62728', 'MRFO': '#ff7f0e', 'SFLA': '#2ca02c'}

    for ax, As_min, label in zip(axes, targets, labels):
        algos = list(data[As_min].keys())
        for alg in algos:
            history = data[As_min][alg]['history']
            if history:
                ax.plot(range(len(history)), history, label=alg, color=colors.get(alg, '#333'), linewidth=1.5)
        ax.set_title(f"{system_name} ($A_{{s,min}} = {As_min}$)")
        ax.set_xlabel("Generation number")
        ax.set_ylabel("System cost")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.text(0.5, -0.25, label, transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold')
    return fig

def plot_bar_charts(data, system_name, targets):
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, 4 * n))
    fig.tight_layout(pad=5.0)
    axes = np.atleast_1d(axes)
    labels = [f'({chr(97+i)})' for i in range(n)]

    for ax, As_min, label in zip(axes, targets, labels):
        algos = list(data[As_min].keys())
        best_costs = [data[As_min][alg]['best_cost'] for alg in algos]
        ax.bar(algos, best_costs, color='#0052A5', edgecolor='black', alpha=0.8)
        ax.set_title(f"{system_name} ($A_{{s,min}} = {As_min}$)")
        ax.set_ylabel("System cost")
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.text(0.5, -0.2, label, transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold')
    return fig

def plot_pareto_front(pareto_front, system_name):
    costs = [sol['cost'] for sol in pareto_front]
    avails = [sol['avail'] for sol in pareto_front]
    
    # Make the figure wider and shorter for a better vertical stack
    fig, ax = plt.subplots(figsize=(12, 5)) 
    
    # Plot the line and points
    ax.plot(costs, avails, marker='o', linestyle='-', color='#0052A5', markersize=8, linewidth=2.5, label="Pareto Front", zorder=3)
    
    # Add a subtle shading under the curve for a modern UI look
    ax.fill_between(costs, avails, 0.85, color='#E8F0FE', alpha=0.5, zorder=2)
    
    ax.set_ylim(bottom=0.85, top=0.995) 
    
    ax.set_title(f"Cost vs Availability Trade-off ({system_name})", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("System Cost (Minimization)", fontsize=12, fontweight='bold')
    ax.set_ylabel("System Availability (Maximization)", fontsize=12, fontweight='bold')
    
    # Clean up the grid and spines
    ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig

#Application
st.title("System Design Optimization with Mixed Failure Dependencies")

default_subsystem_data = {
    "Subsystem ID": list(range(1, 11)),
    "Dependency Type (0=L, 1=W, 2=S)": [0, 0, 1, 0, 1, 2, 0, 2, 0, 1],
    "Failure Rate (λ)": [0.07, 0.04, 0.02, 0.03, 0.08, 0.05, 0.01, 0.06, 0.09, 0.05],
    "Repair Rate (μ)": [0.25, 0.12, 0.27, 0.10, 0.15, 0.26, 0.18, 0.30, 0.14, 0.28],
    "Component Cost (C_comp)": [30.0, 55.0, 40.0, 75.0, 60.0, 80.0, 70.0, 45.0, 50.0, 25.0],
    "Repair Cost (C_rep)": [20.0, 35.0, 25.0, 45.0, 30.0, 60.0, 30.0, 25.0, 30.0, 20.0]
}
df_system_default = pd.DataFrame(default_subsystem_data).set_index("Subsystem ID")

with st.expander("Modify Subsystem Parameters (λ, μ, Costs, Dependencies)", expanded=False):
    edited_df = st.data_editor(df_system_default, use_container_width=True)

st.write("---")

#Execution Block
if st.sidebar.button("Run Optimization Benchmark"):
    custom_system_data = {}
    for idx, row in edited_df.iterrows():
        custom_system_data[idx] = [
            int(row["Dependency Type (0=L, 1=W, 2=S)"]), 
            float(row["Failure Rate (λ)"]), 
            float(row["Repair Rate (μ)"]), 
            float(row["Component Cost (C_comp)"]), 
            float(row["Repair Cost (C_rep)"])
        ]
    core_math.SYSTEM_DATA = custom_system_data

    m_subsys = 5 if "Bridge" in system_choice else 10
    sys_type = 'bridge' if "Bridge" in system_choice else 'series_parallel'

    if opt_mode == "Single-Objective (Cost minimization)":
        if input_error:
            st.error("Please fix the validation errors in the Availability Targets before running.")
        elif len(avail_targets) == 0:
            st.error("Please provide at least one valid Availability Target ($A_{s,min}$).")
        elif not (run_de or run_shade or run_mrfo or run_sfla):
            st.error("Please select at least one algorithm to run.")
        else:
            algorithms = {}
            if run_de:    algorithms['DE']    = optimize_de
            if run_shade: algorithms['SHADE'] = optimize_shade
            if run_mrfo:  algorithms['MRFO']  = optimize_mrfo
            if run_sfla:  algorithms['SFLA']  = optimize_sfla

            plot_data = {target: {} for target in avail_targets}
            results_summary = []

            total_tasks = len(avail_targets) * len(algorithms) * num_runs
            completed_tasks = 0

            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for As_min in avail_targets:
                for alg_name, opt_fn in algorithms.items():
                    run_costs = []
                    best_cost = np.inf
                    best_n = None
                    best_r = None
                    best_avail = 0.0
                    best_history = []
                    
                    for run in range(num_runs):
                        status_text.text(f"Processing target {As_min} | Running {alg_name} (Run {run + 1}/{num_runs})...")
                        
                        if alg_name == 'SFLA':
                            n_v, r_v, history = opt_fn(m_subsys, sys_type, As_min, pop_size=pop_size, num_memeplexes=5, local_iters=10, max_gen=max_gen, track_history=True)
                        else:
                            n_v, r_v, history = opt_fn(m_subsys, sys_type, As_min, pop_size=pop_size, max_gen=max_gen, track_history=True)
                        
                        x_flat = np.zeros(m_subsys * 2)
                        x_flat[0::2] = n_v
                        x_flat[1::2] = r_v
                        cost, avail = evaluate_solution(x_flat, m_subsys, sys_type)
                        
                        run_costs.append(cost)
                        if avail >= As_min and cost < best_cost:
                            best_cost = cost
                            best_n = n_v.tolist()
                            best_r = r_v.tolist()
                            best_avail = avail
                            best_history = copy.deepcopy(history)
                        
                        completed_tasks += 1
                        progress_bar.progress(completed_tasks / total_tasks)

                    mean_c = np.mean(run_costs)
                    std_c = np.std(run_costs)
                    
                    results_summary.append({
                        "Target $A_{s,min}$": As_min,
                        "Algorithm": alg_name,
                        "Best n-vector": str(best_n),
                        "Best r-vector": str(best_r),
                        "Achieved $A_s$": round(best_avail, 4),
                        "Best Cost ($C_s$)": round(best_cost, 2) if best_cost < np.inf else "Infeasible",
                        "Mean Cost": round(mean_c, 2),
                        "Std Dev ($\sigma$)": round(std_c, 2)
                    })
                    
                    plot_data[As_min][alg_name] = {
                        'costs': run_costs,
                        'best_cost': best_cost,
                        'history': best_history
                    }

            status_text.success(f"Benchmark completed in {time.time() - start_time:.2f} seconds!")
            progress_bar.empty()

            st.header("Tabular Results")
            df_results = pd.DataFrame(results_summary)
            st.dataframe(df_results.set_index(["Target $A_{s,min}$", "Algorithm"]), use_container_width=True)

            st.write("---")
            st.header("Visualizations")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<h4 style='text-align: center;'>Convergence Curves</h4>", unsafe_allow_html=True)
                fig_conv = plot_convergence_curves(plot_data, system_choice, avail_targets)
                st.pyplot(fig_conv)
                
            with col2:
                st.markdown("<h4 style='text-align: center;'>Box Diagrams</h4>", unsafe_allow_html=True)
                fig_box = plot_box_diagrams(plot_data, system_choice, avail_targets)
                st.pyplot(fig_box)
                
            with col3:
                st.markdown("<h4 style='text-align: center;'>Best System Cost</h4>", unsafe_allow_html=True)
                fig_bar = plot_bar_charts(plot_data, system_choice, avail_targets)
                st.pyplot(fig_bar)

    # --- MISSION-TIME ANALYSIS (always shown after single-obj results) ---
    if opt_mode == "Single-Objective (Cost minimization)" and not input_error and len(avail_targets) > 0:
        st.write("---")
        st.header("🕐 Mission-Time Analysis (Extension)")
        st.markdown(
            """
            The paper optimises **steady-state availability** (mission length → ∞).
            This extension uses a validated **continuous-time Markov chain** to find the
            cheapest design meeting the availability target at a *finite mission time T*.

            > **Finding:** short missions tolerate less redundancy and cost up to ~22% less.
            > As T → ∞ the optimum converges to the paper's steady-state design.
            """
        )

        As_mission = avail_targets[-1] if avail_targets else 0.99
        col_t, col_run = st.columns([3, 1])
        with col_t:
            mission_T = st.slider(
                f"Mission time T (hours)  —  availability target: {As_mission}",
                min_value=5, max_value=500, value=100, step=5
            )
        with col_run:
            st.write("")
            run_mission = st.button("Compute mission-time optimal design")

        if run_mission:
            with st.spinner("Optimising for finite mission time…"):
                obj = lambda x: mission_penalised_objective(x, m_subsys, sys_type, As_mission, mission_T)
                n_v, r_v, _ = optimize_shade(
                    m_subsys, sys_type, As_mission,
                    pop_size=int(pop_size), max_gen=int(max_gen), objective=obj
                )
                x = np.zeros(m_subsys * 2); x[0::2] = n_v; x[1::2] = r_v
                from backend.core_math import evaluate_solution as ev
                cost_T, A_ss = ev(x, m_subsys, sys_type)
                A_T = system_availability_at(n_v, r_v, m_subsys, sys_type, mission_T)

                # Steady-state optimum for comparison
                n_ss, r_ss, _ = optimize_shade(m_subsys, sys_type, As_mission,
                                               pop_size=int(pop_size), max_gen=int(max_gen))
                x_ss = np.zeros(m_subsys * 2); x_ss[0::2] = n_ss; x_ss[1::2] = r_ss
                cost_ss, _ = ev(x_ss, m_subsys, sys_type)

            saving = (cost_ss - cost_T) / max(cost_ss, 1) * 100
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Mission-time optimal cost", f"{cost_T:.0f}",
                       delta=f"{-saving:.1f}% vs steady-state" if saving > 0 else None)
            mc2.metric(f"A(T={mission_T})", f"{A_T:.4f}",
                       delta="✓ meets target" if A_T >= As_mission else "✗ below target")
            mc3.metric("Steady-state cost (paper)", f"{cost_ss:.0f}")

            comp_df = pd.DataFrame({
                "Model": [f"Mission T={mission_T}", "Steady-state (paper)"],
                "Cost": [round(cost_T, 0), round(cost_ss, 0)],
                f"A(T={mission_T})": [round(A_T, 4), "—"],
                "Steady-state A": [round(A_ss, 4), round(_, 4)],
                "n-vector": [str(list(map(int, n_v))), str(list(map(int, n_ss)))],
                "r-vector": [str(list(map(int, r_v))), str(list(map(int, r_ss)))],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Cost vs mission-time curve
            st.subheader("Cost vs Mission Time")
            T_vals = list(range(5, min(mission_T * 2, 501), max(5, mission_T // 10)))
            costs_curve = []
            with st.spinner("Computing cost curve…"):
                for Tv in T_vals:
                    o2 = lambda x, Tv=Tv: mission_penalised_objective(x, m_subsys, sys_type, As_mission, Tv)
                    nv2, rv2, _ = optimize_shade(m_subsys, sys_type, As_mission,
                                                pop_size=40, max_gen=80, objective=o2)
                    xv2 = np.zeros(m_subsys * 2); xv2[0::2] = nv2; xv2[1::2] = rv2
                    cv2, _ = ev(xv2, m_subsys, sys_type)
                    costs_curve.append(cv2)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(T_vals, costs_curve, marker='o', color='#0052A5', linewidth=2)
            ax2.axhline(cost_ss, color='#E74C3C', linestyle='--', label=f"Steady-state ({cost_ss:.0f})")
            ax2.set_xlabel("Mission time T"); ax2.set_ylabel("Optimal cost")
            ax2.set_title(f"Cost vs Mission Time (A ≥ {As_mission})")
            ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig2)

    # --- PATHWAY B: MULTI OBJECTIVE ---
    else:
        if not run_mode:
            st.error("Please select the Multi-Objective Algorithm to run.")
        else:
            status_text = st.empty()
            status_text.info("Running Multi-Objective Differential Evolution (MODE). Generating Pareto Front...")
            start_time = time.time()
            
            # Execute MODE
            pareto_front = optimize_mode(m_subsys, sys_type, pop_size=pop_size, max_gen=max_gen)
            
            status_text.success(f"Pareto Front Generated in {time.time() - start_time:.2f} seconds! Found {len(pareto_front)} optimal trade-off solutions.")
            st.write("---")
            
            # 1. GRAPH SECTION (Centered)
            st.markdown("<h3 style='text-align: center; color: #2C3E50;'>Trade-off Curve (Pareto Front)</h3>", unsafe_allow_html=True)
            
            # Use columns to center the plot and give it breathing room
            c1, c2, c3 = st.columns([1, 10, 1])
            with c2:
                fig_pareto = plot_pareto_front(pareto_front, system_choice)
                st.pyplot(fig_pareto)
                
            st.write("<br>", unsafe_allow_html=True) # Spacer
            
            # 2. TABLE SECTION (Centered underneath)
            st.markdown("<h3 style='text-align: center; color: #2C3E50;'>Detailed Optimal Solutions</h3>", unsafe_allow_html=True)
            
            df_pareto = pd.DataFrame(pareto_front)
            
            # Convert lists to strings so they display perfectly in Streamlit's table
            df_pareto['n'] = df_pareto['n'].apply(lambda x: str(x))
            df_pareto['r'] = df_pareto['r'].apply(lambda x: str(x))
            
            # Rename columns for presentation
            df_pareto.rename(columns={
                'n': 'Redundancy Vector (n)', 
                'r': 'Repair Teams Vector (r)', 
                'cost': 'System Cost', 
                'avail': 'Availability'
            }, inplace=True)
            
            # Use columns to slightly indent the table from the full width of the screen
            t1, t2, t3 = st.columns([1, 10, 1])
            with t2:
                # hide_index=True removes the default 0,1,2,3 row numbers for a cleaner look
                st.dataframe(df_pareto, use_container_width=True, hide_index=True)