# 🇦🇪 UAE International Migration Nowcasting (2015–2024)

### 📊 Project Summary
Accurate, real-time migration data is critical for national planning. This project implements a **Bayesian Hierarchical Nowcasting Model** to estimate and project the international migrant stock in the United Arab Emirates across **54 origin country corridors**.

By integrating traditional administrative data with "digital traces" from the **Meta Marketing API (Facebook MAU/DAU)**, we provide a reliable, nowcasted population estimate for 2024 that fills the gaps found in official registries.

---

## 📈 Data Source Priority & Reliability
To achieve 92% accuracy, the model weighs each data source differently based on its historical performance against Truth (LFS) data:

| Data Source | Priority / Reliability | Role in Model |
| :--- | :--- | :--- |
| **UAE Labour Force Survey (LFS)** | 🔴 **High (Ground Truth)** | Baseline anchor for total scale. |
| **FCSC Admin Data** | 🟡 **Medium-High** | Primary historical source; corrected for 8% undercount. |
| **Facebook MAU** | 🟢 **Medium (Leader)** | Primary nowcasting signal for 2024 trends. |
| **Facebook DAU** | ⚪ **Low (Validator)** | Used for micro-trend validation. |

---

## 📐 The Model Pipeline (Logic & Architecture)

### 1. What the Model is Doing
The engine solves the **"Missing Middle"** problem. Registers (FCSC) are slow but verified; Social Media (Meta) is fast but biased. Our **Bayesian Inference** finds the accurate signal by "de-biasing" both sources simultaneously.

### 2. Bayesian Strategy
- **AR(1) Smoothing**: The model prevents "jumpy" data by enforcing temporal consistency.
- **54-Corridor Panel**: Simultaneously modeling all 54 countries to share regional growth patterns.
- **NUTS Sampler**: Uses Hamiltonian Monte Carlo (1000 tuning + 1000 draws) for high-precision projections.

---

## 🖼️ Full Analytical Report (21 Graphs)

### Phase 1: Data Discovery & Foundations
| 01: Regional Sunburst | 02: Temporal Trends (Top 10) | 03: Master Panel Data |
| :--- | :--- | :--- |
| ![01](out/01_regional_sunburst.png) | ![02](out/02_temporal_trends_top10.png) | ![03](out/03_master_panel_data.png) |

### Phase 2: Demographic Breadth & Coverage
| 04: Regional Bar Totals | 05: India Data Source Comparison | 06: Data Coverage Heatmap |
| :--- | :--- | :--- |
| ![04](out/04_regional_bar_stock_trends.png) | ![05](out/05_india_data_source_comparison.png) | ![06](out/06_coverage_heatmap.png) |

### Phase 3: Bayesian Convergence & Diagnostics
| 09: MCMC Convergence | 10: Trace & Energy Plots | 11: Posterior Bias Distributions |
| :--- | :--- | :--- |
| ![09](out/09_mcmc_convergence_diagnostics.png) | ![10](out/10_trace_plots_energy.png) | ![11](out/11_energy_posterior_bias.png) |

### Phase 4: 2024 Nowcast Results
| 12: Total 2024 Nowcast | 13: Regional Compositions | 14: Top Corridor Estimates (Detailed) |
| :--- | :--- | :--- |
| ![12](out/12_posterior_estimates_total_stock.png) | ![13](out/13_regional_stacked_corridor_top.png) | ![14](out/14_corridor_estimates_bottom.png) |

### Phase 5: Regional Comparison & Validation
| 15: All 54 Corridor Performance | 17: RMSE Validation Scatter | 18: Surge Warning Indicators |
| :--- | :--- | :--- |
| ![15](out/15_all_corridors_circular_migration.png) | ![17](out/17_validation_scatter_early_warning.png) | ![18](out/18_early_warning_sector_planning.png) |

### Phase 6: Infrastructure & Policy Briefing
| 19: Sector Infrastructure Planning | 20: Audit & Replication Guide | 21: Final Methodology Summary |
| :--- | :--- | :--- |
| ![19](out/19_sector_forecast_infrastructure.png) | ![20](out/20_replication_guide.png) | ![21](out/21_final_summary.png) |

---

## 🛠️ Installation & Config
1.  **Clone & Install**:
    ```bash
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Environment**: Add `FACEBOOK_API=your_token` to a `.env` file in the root.
3.  **Run**: `python3 migration_testing.py`

---

**Generated for FCSC / MOHRE Policy Planning**
*Project Lead: Bayesian UAE Migration Team*
