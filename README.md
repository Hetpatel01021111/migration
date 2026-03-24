# 🇦🇪 UAE International Migration Nowcasting (2015–2024)

### 📊 Project Summary
Accurate, real-time migration data is critical for national planning. This project implements a **Bayesian Hierarchical Nowcasting Model** to estimate and project the international migrant stock in the United Arab Emirates across **54 origin country corridors**.

By integrating traditional administrative data with "digital traces" from the **Meta Marketing API (Facebook MAU/DAU)**, we provide a reliable, nowcasted population estimate for 2024 that fills the gaps found in official registries.

---

## 🔍 What the Model is Doing & Why
The core objective is to solve the **"Missing Middle"** problem in migration statistics. Traditional administrative registers (FCSC) are often delayed by 1–2 years, while real-time indicators (Social Media) are noisy and biased.

**Our model performs three critical functions:**
1.  **Nowcasting**: Projects the migration stock into the current year (2024) before official census data is released.
2.  **Bias Adjustment**: Identifies if a source systematically undercounts (like Admin registers for informal workers) or overcounts (like 'Recently In' metrics which include tourists) and adjusts accordingly.
3.  **Uncertainty Quantification**: Instead of a single number, it provides a **80% Credible Interval**, allowing policy makers to see the "best-case" and "worst-case" scenarios.

---

## 📐 How the Model Works (The Logic)
The engine behind this project is a **Bayesian AR(1) Latent Process Model**:

- **$\mu_t = \mu_{t-1} + \delta$**: The model assumes that migration stock follows a smooth temporal trend (Auto-Regressive) where last year's stock strongly predicts this year's.
- **Hierarchical Regional Priors**: Countries are grouped into **8 regions** (e.g., South Asia, Arab World). This allows the model to "learn" from regional neighbors when a specific country has missing data.
- **Bayesian Hamiltonian Monte Carlo (NUTS)**: We use the **No-U-Turn Sampler** in PyMC to explore millions of possible population scenarios, settling on the most mathematically probable one.
- **Data Fusion**: The model treats both Admin data and FB data as "noisy observers" of a single, hidden **True Stock**.

---

## 📂 Data Sources & Origins
We integrated 54 unique country corridors by fusing these specific datasets:

| Data Type | Source Provider | Description |
| :--- | :--- | :--- |
| **Priors / Baselines** | *UAE Migration Data Compendium* | 54-country baseline stocks and growth rates (2015-2024). |
| **Administrative** | FCSC / MOHRE | Official registries of resident visas and labor contracts. |
| **Digital Traces** | Meta Marketing API | Real-time Monthly (MAU) and Daily (DAU) Active Users in the UAE. |
| **Labor Surveys** | UAE LFS | Representative survey data used to anchor the model's total scale. |

---

## 🛠️ Detailed System Architecture & Pipelines

### 1. Data Acquisition Pipeline
- **API Engine**: `facebook_api.py` securely authenticates using an environment-stored token. It queries Meta's `adTargetingCategory` to find specific "Lived in [Country]" segments.
- **Panel Construction**: Automatically merges the 54 countries and 10 years into a single **Master Panel Dataframe** used for the Bayesian likelihood.

### 2. Analytical Visualization Pipeline (The 21-Plot Sequence)
The script produces a standardized report in the `out/` folder, saved as 21 individual PNGs for policy briefs:
- **01–03 (Foundation)**: Sunbursts and Master Panel previews.
- **04–06 (Context)**: Regional stock trends and data coverage heatmaps.
- **09–11 (Diagnostics)**: MCMC trace plots and Bayesian energy distributions.
- **12–15 (Projections)**: Total UAE projection and 54-country individual nowcasts.
- **17–21 (Policy)**: RMSE validation, "Surge" early-warning indicators, and sector planning.

---

## 🛠️ Installation & Execution
1.  **Requirements**: Python 3.9+, `pymc`, `arviz`, `pandas`, `seaborn`, `python-dotenv`.
2.  **Setup Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Config**: Add `FACEBOOK_API=your_token` to a `.env` file.
4.  **Run**:
    ```bash
    python3 migration_testing.py
    ```

---

**Generated for FCSC / MOHRE Policy Planning**
*Project Lead: Bayesian UAE Migration Team*
