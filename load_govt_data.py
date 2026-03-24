# ── load_govt_data.py ────────────────────────────────────────────────────────
# Module for loading official government migration statistics.
#
# Supports two modes:
#   1. load_complete_dataset() — loads the full "UAE Migration Complete Dataset.xlsx"
#      which contains a pre-formatted Annex2 panel (2,124 rows) ready for the model.
#   2. load_govt_data() — auto-detects CSV/XLSX files in a folder (generic loader)
#
# Usage:
#   from load_govt_data import load_complete_dataset
#   panel, fb_data, census_data, admin_data, lfs_data, priors = load_complete_dataset("Govt/")
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import glob


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMARY LOADER — UAE Migration Complete Dataset.xlsx
# ══════════════════════════════════════════════════════════════════════════════

def load_complete_dataset(govt_folder="Govt/", verbose=True):
    """
    Load the 'UAE Migration Complete Dataset.xlsx' file and extract all
    data sources into standardised DataFrames.

    Returns
    -------
    dict with keys:
      'panel'    : Full Annex2 panel (origin × year × source × stock)
      'fb_mau'   : Facebook MAU/DAU data by origin
      'census'   : Census 2015 anchor data
      'admin'    : FCSC administrative registry (aggregate)
      'lfs'      : UAE Labour Force Survey
      'un_desa'  : UN DESA migrant stock by origin
      'priors'   : Bayesian prior distributions
      'fb_pen'   : Facebook penetration rates by region
    """
    govt_folder = os.path.abspath(govt_folder)
    target = os.path.join(govt_folder, "UAE Migration Complete Dataset.xlsx")

    if not os.path.exists(target):
        # Try case-insensitive search
        for f in os.listdir(govt_folder):
            if "complete dataset" in f.lower() and f.endswith(".xlsx"):
                target = os.path.join(govt_folder, f)
                break
        else:
            raise FileNotFoundError(
                f"'UAE Migration Complete Dataset.xlsx' not found in {govt_folder}"
            )

    if verbose:
        print(f"📂 Loading: {os.path.basename(target)}")
        print(f"   Size: {os.path.getsize(target) / 1024:.0f} KB")

    result = {}

    # ── 1. Annex2 Panel (MAIN MODEL INPUT) ────────────────────────────────
    if verbose:
        print("\n── Sheet: 8_Annex2_Panel (main model input) ──")
    df = pd.read_excel(target, sheet_name="8_Annex2_Panel", header=None)

    # Find header row (row containing "origin")
    header_idx = None
    for i, row in df.iterrows():
        if any(str(v).strip().lower() == "origin" for v in row.values if pd.notna(v)):
            header_idx = i
            break

    if header_idx is not None:
        df.columns = [str(v).strip() for v in df.iloc[header_idx].values]
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
    else:
        raise ValueError("Could not find header row in 8_Annex2_Panel")

    # Clean column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "origin":
            col_map[c] = "origin"
        elif cl == "destination":
            col_map[c] = "destination"
        elif cl == "region":
            col_map[c] = "region"
        elif cl == "year":
            col_map[c] = "year"
        elif cl == "source":
            col_map[c] = "source"
        elif cl == "stock":
            col_map[c] = "stock"
        elif cl == "log_stock":
            col_map[c] = "log_stock"
        elif cl == "data_available":
            col_map[c] = "data_available"
        elif cl == "bias_prior_mean":
            col_map[c] = "bias_prior_mean"
        elif cl == "bias_prior_precision":
            col_map[c] = "bias_prior_precision"
        elif cl == "source_url":
            col_map[c] = "source_url"
        elif cl == "notes":
            col_map[c] = "notes"
        elif cl == "corridor_id":
            col_map[c] = "corridor_id"
        elif cl == "year_idx":
            col_map[c] = "year_idx"

    df = df.rename(columns=col_map)

    # Convert types
    for col in ["year", "stock", "corridor_id", "year_idx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["bias_prior_mean", "bias_prior_precision", "log_stock"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN origin
    df = df.dropna(subset=["origin"]).copy()

    # Convert stock to thousands (if in raw counts)
    if df["stock"].median() > 10000:
        df["stock_thousands"] = df["stock"] / 1000.0
    else:
        df["stock_thousands"] = df["stock"]

    result["panel"] = df

    if verbose:
        print(f"   Rows     : {len(df):,}")
        print(f"   Countries: {df['origin'].nunique()}")
        print(f"   Sources  : {sorted(df['source'].unique())}")
        print(f"   Years    : {sorted(df['year'].dropna().unique().astype(int))}")
        print()

    # ── 2. Facebook MAU/DAU ───────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 1_Facebook_MAU_DAU ──")
        df_fb = pd.read_excel(target, sheet_name="1_Facebook_MAU_DAU", header=None)
        # Find header row
        for i, row in df_fb.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if "origin" in vals:
                df_fb.columns = [str(v).strip() for v in df_fb.iloc[i].values]
                df_fb = df_fb.iloc[i + 1:].reset_index(drop=True)
                break

        # Standardise column names
        fb_col_map = {}
        for c in df_fb.columns:
            cl = c.lower().strip() if pd.notna(c) else ""
            if cl == "origin": fb_col_map[c] = "origin"
            elif cl == "region": fb_col_map[c] = "region"
            elif cl == "gender": fb_col_map[c] = "gender"
            elif "age min" in cl: fb_col_map[c] = "age_min"
            elif "age max" in cl: fb_col_map[c] = "age_max"
            elif "mau" in cl and "estimate" in cl: fb_col_map[c] = "mau_estimate"
            elif "dau" in cl and "estimate" in cl: fb_col_map[c] = "dau_estimate"
            elif "penetration" in cl: fb_col_map[c] = "penetration_group"
            elif "bias" in cl and "mean" in cl: fb_col_map[c] = "bias_prior_mean"
            elif "precision" in cl: fb_col_map[c] = "precision"
            elif "notes" in cl: fb_col_map[c] = "notes"
        df_fb = df_fb.rename(columns=fb_col_map)
        df_fb = df_fb.dropna(subset=["origin"])

        for col in ["mau_estimate", "dau_estimate", "age_min", "age_max",
                     "bias_prior_mean", "precision"]:
            if col in df_fb.columns:
                df_fb[col] = pd.to_numeric(df_fb[col], errors="coerce")

        result["fb_mau"] = df_fb
        if verbose:
            print(f"   Rows: {len(df_fb)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["fb_mau"] = pd.DataFrame()

    # ── 3. Census ─────────────────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 2_UAE_Census ──")
        df_c = pd.read_excel(target, sheet_name="2_UAE_Census", header=None)
        for i, row in df_c.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if any("origin" in v for v in vals):
                df_c.columns = [str(v).strip() for v in df_c.iloc[i].values]
                df_c = df_c.iloc[i + 1:].reset_index(drop=True)
                break
        df_c = df_c.dropna(subset=[df_c.columns[0]])
        result["census"] = df_c
        if verbose:
            print(f"   Rows: {len(df_c)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["census"] = pd.DataFrame()

    # ── 4. FCSC Admin ─────────────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 3_FCSC_Admin ──")
        df_a = pd.read_excel(target, sheet_name="3_FCSC_Admin", header=None)
        for i, row in df_a.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if "year" in vals:
                df_a.columns = [str(v).strip() for v in df_a.iloc[i].values]
                df_a = df_a.iloc[i + 1:].reset_index(drop=True)
                break
        df_a = df_a.dropna(subset=[df_a.columns[0]])
        result["admin"] = df_a
        if verbose:
            print(f"   Rows: {len(df_a)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["admin"] = pd.DataFrame()

    # ── 5. UN DESA ────────────────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 5_UN_DESA_Stock ──")
        df_un = pd.read_excel(target, sheet_name="5_UN_DESA_Stock", header=None)
        for i, row in df_un.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if "region" in vals or "origin" in vals:
                df_un.columns = [str(v).strip() for v in df_un.iloc[i].values]
                df_un = df_un.iloc[i + 1:].reset_index(drop=True)
                break
        df_un = df_un.dropna(subset=[df_un.columns[1]])
        result["un_desa"] = df_un
        if verbose:
            print(f"   Rows: {len(df_un)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["un_desa"] = pd.DataFrame()

    # ── 6. LFS ────────────────────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 4_UAE_LFS ──")
        df_lfs = pd.read_excel(target, sheet_name="4_UAE_LFS", header=None)
        for i, row in df_lfs.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if "year" in vals:
                df_lfs.columns = [str(v).strip() for v in df_lfs.iloc[i].values]
                df_lfs = df_lfs.iloc[i + 1:].reset_index(drop=True)
                break
        df_lfs = df_lfs.dropna(subset=[df_lfs.columns[0]])
        result["lfs"] = df_lfs
        if verbose:
            print(f"   Rows: {len(df_lfs)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["lfs"] = pd.DataFrame()

    # ── 7. Bias Priors ────────────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 7_Bias_Priors ──")
        df_bp = pd.read_excel(target, sheet_name="7_Bias_Priors", header=None)
        for i, row in df_bp.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if "data source" in " ".join(vals):
                df_bp.columns = [str(v).strip() for v in df_bp.iloc[i].values]
                df_bp = df_bp.iloc[i + 1:].reset_index(drop=True)
                break
        df_bp = df_bp.dropna(subset=[df_bp.columns[0]])
        result["priors"] = df_bp
        if verbose:
            print(f"   Rows: {len(df_bp)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["priors"] = pd.DataFrame()

    # ── 8. FB Penetration ─────────────────────────────────────────────────
    try:
        if verbose:
            print("── Sheet: 6_FB_Penetration ──")
        df_pen = pd.read_excel(target, sheet_name="6_FB_Penetration", header=None)
        for i, row in df_pen.iterrows():
            vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
            if "region" in vals:
                df_pen.columns = [str(v).strip() for v in df_pen.iloc[i].values]
                df_pen = df_pen.iloc[i + 1:].reset_index(drop=True)
                break
        df_pen = df_pen.dropna(subset=[df_pen.columns[0]])
        result["fb_pen"] = df_pen
        if verbose:
            print(f"   Rows: {len(df_pen)}")
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not load: {e}")
        result["fb_pen"] = pd.DataFrame()

    if verbose:
        print(f"\n✅ Complete Dataset loaded successfully!")
        print(f"   Panel: {len(result['panel'])} rows × {len(result['panel'].columns)} cols")
        sources = result['panel']['source'].value_counts()
        print(f"   Breakdown by source:")
        for src, cnt in sources.items():
            print(f"      {src:20s}: {cnt:>4d} rows")

    return result


def panel_to_model_inputs(panel_df, years=None, countries=None):
    """
    Convert the Annex2 panel DataFrame into the separate DataFrames
    expected by the migration model (df_admin, df_fb, df_fb_recent, df_lfs).

    Parameters
    ----------
    panel_df : pd.DataFrame — the 'panel' key from load_complete_dataset()
    years : list — filter to these years (default: all)
    countries : list — filter to these countries (default: all)

    Returns
    -------
    df_admin, df_fb, df_lfs : pd.DataFrame
    """
    df = panel_df.copy()

    if years is not None:
        df = df[df["year"].isin(years)]
    if countries is not None:
        df = df[df["origin"].isin(countries)]

    # ── Administrative data (Census + FCSC Admin) ─────────────────────────
    admin_sources = ["Census_2015", "FCSC_Admin"]
    df_admin_raw = df[df["source"].isin(admin_sources)].copy()

    # Pivot: for each origin × year, take the first available admin stock
    # Priority: Census > FCSC_Admin
    admin_rows = []
    for (origin, year), grp in df_admin_raw.groupby(["origin", "year"]):
        # Prefer Census, then FCSC_Admin
        best = grp.sort_values("source").iloc[0]
        admin_rows.append({
            "origin": origin,
            "region": best.get("region", ""),
            "year": int(year),
            "admin_stock": best["stock_thousands"],
            "true_stock_approx": best["stock_thousands"],
            "data_available": True,
            "bias_prior_mean": best.get("bias_prior_mean", 0.88),
            "bias_prior_precision": best.get("bias_prior_precision", 100),
        })

    df_admin = pd.DataFrame(admin_rows)

    # ── Facebook MAU data ─────────────────────────────────────────────────
    fb_sources = ["Facebook_MAU"]
    df_fb_raw = df[df["source"].isin(fb_sources)].copy()

    fb_rows = []
    for _, row in df_fb_raw.iterrows():
        fb_rows.append({
            "origin": row["origin"],
            "region": row.get("region", ""),
            "year": int(row["year"]),
            "fb_mau_lived_in": row["stock_thousands"],
        })
    df_fb = pd.DataFrame(fb_rows)

    # ── Facebook DAU data (extra signal) ──────────────────────────────────
    # Not directly used in model but available
    dau_sources = ["Facebook_DAU"]
    df_dau_raw = df[df["source"].isin(dau_sources)].copy()

    # ── LFS data ──────────────────────────────────────────────────────────
    lfs_sources = ["UAE_LFS"]
    df_lfs_raw = df[df["source"].isin(lfs_sources)].copy()

    lfs_rows = []
    for _, row in df_lfs_raw.iterrows():
        lfs_rows.append({
            "origin": row["origin"],
            "region": row.get("region", ""),
            "year": int(row["year"]),
            "lfs_stock": row["stock_thousands"],
        })
    df_lfs = pd.DataFrame(lfs_rows)

    # ── UN DESA anchor data ───────────────────────────────────────────────
    desa_sources = ["UN_DESA_2015", "UN_DESA_2020"]
    df_desa_raw = df[df["source"].isin(desa_sources)].copy()

    return df_admin, df_fb, df_lfs




# ── Column name mapping ───────────────────────────────────────────────────────
# Maps common column names from different data sources to our standard format.
# Case-insensitive matching is applied.

COLUMN_ALIASES = {
    # Origin country columns
    "origin": ["origin", "country", "country_of_birth", "nationality",
               "country_name", "source_country", "origin_country",
               "country of birth", "country of origin"],

    # Year columns
    "year": ["year", "yr", "date", "period", "reference_year",
             "ref_year", "census_year"],

    # Official stock columns (in thousands or raw)
    "official_stock": ["official_stock", "stock", "population", "count",
                       "migrant_stock", "total", "pop_by_nationality",
                       "expat_population", "foreign_population",
                       "admin_stock", "govt_stock", "official_count",
                       "migrant_count", "number"],
}

# Country name standardisation
COUNTRY_NAME_MAP = {
    # Common aliases → standard names used in our model
    "uae": "United Arab Emirates",
    "uk": "United Kingdom",
    "usa": "United States",
    "us": "United States",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "korea": "South Korea",
    "china, people's republic of": "China",
    "peoples republic of china": "China",
    "hong kong sar": "Hong Kong",
    "hong kong, china": "Hong Kong",
    "myanmar (burma)": "Myanmar",
    "russian federation": "Russia",
    "viet nam": "Vietnam",
    "congo, democratic republic of the": "Congo",
    "united republic of tanzania": "Tanzania",
    "iran, islamic republic of": "Iran",
    "state of palestine": "Palestine",
    "syrian arab republic": "Syria",
    "lao people's democratic republic": "Laos",
    "the philippines": "Philippines",
    "brunei darussalam": "Brunei",
    "ivory coast": "Côte d'Ivoire",
    "new zealand": "New Zealand",
    "unitedkingdom": "United Kingdom",
}


def _find_column(df, target_key):
    """
    Find the column in df that matches one of the aliases for target_key.
    Returns the matched column name or None.
    """
    aliases = COLUMN_ALIASES.get(target_key, [])
    df_cols_lower = {col.lower().strip(): col for col in df.columns}

    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    return None


def _standardise_country_name(name):
    """Standardise country name to match our model's naming convention."""
    if pd.isna(name):
        return name
    name_clean = str(name).strip()
    name_lower = name_clean.lower()

    # Check direct mapping
    if name_lower in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[name_lower]

    # Return as-is (with title case normalisation)
    return name_clean


def load_single_csv(filepath, verbose=True):
    """
    Load a single CSV file and auto-detect column mappings.

    Returns a standardised DataFrame with columns:
        origin, year, official_stock, source_file
    """
    if verbose:
        print(f"   📂 Loading: {os.path.basename(filepath)}")

    # Try reading with different encodings
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        print(f"      ⚠️ Could not read {filepath} with any encoding — skipping")
        return pd.DataFrame()

    if df.empty:
        print(f"      ⚠️ Empty file — skipping")
        return pd.DataFrame()

    if verbose:
        print(f"      Shape: {df.shape}, Columns: {list(df.columns)}")

    # ── Auto-detect columns ───────────────────────────────────────────────
    origin_col = _find_column(df, "origin")
    year_col = _find_column(df, "year")
    stock_col = _find_column(df, "official_stock")

    if origin_col is None:
        print(f"      ⚠️ No origin/country column found. "
              f"Available: {list(df.columns)}")
        return pd.DataFrame()

    if stock_col is None:
        # Try to find any numeric column as stock
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stock_col = numeric_cols[-1]  # take last numeric column
            if verbose:
                print(f"      ℹ️ Using '{stock_col}' as stock column (auto-detected)")
        else:
            print(f"      ⚠️ No stock/population column found — skipping")
            return pd.DataFrame()

    # ── Build standardised output ─────────────────────────────────────────
    result = pd.DataFrame()
    result["origin"] = df[origin_col].apply(_standardise_country_name)

    if year_col is not None:
        result["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    else:
        # If no year column, try to infer from filename
        basename = os.path.basename(filepath)
        for yr in range(2010, 2030):
            if str(yr) in basename:
                result["year"] = yr
                if verbose:
                    print(f"      ℹ️ Year inferred from filename: {yr}")
                break
        else:
            print(f"      ⚠️ No year column or year in filename — skipping")
            return pd.DataFrame()

    result["official_stock"] = pd.to_numeric(df[stock_col], errors="coerce")
    result["source_file"] = os.path.basename(filepath)

    # Drop rows with missing essential data
    result = result.dropna(subset=["origin", "year", "official_stock"])

    if verbose:
        print(f"      ✅ {len(result)} valid rows loaded")

    return result


def detect_stock_unit(df):
    """
    Auto-detect if stock values are in raw counts or thousands.
    Returns 'raw' or 'thousands'.
    """
    if df["official_stock"].empty:
        return "raw"

    median = df["official_stock"].median()
    if median > 10_000:
        return "raw"      # Likely raw counts (e.g., 2,900,000)
    elif median > 10:
        return "thousands" # Already in thousands (e.g., 2,900)
    else:
        return "raw"


def load_govt_data(govt_folder="Govt/", convert_to_thousands=True, verbose=True):
    """
    Load all government migration data CSV files from a folder.

    Parameters
    ----------
    govt_folder : str — path to folder containing CSV files
    convert_to_thousands : bool — auto-detect and convert raw counts to thousands
    verbose : bool — print progress

    Returns
    -------
    pd.DataFrame with columns:
        origin, year, official_stock (in thousands), source_file
    """
    govt_folder = os.path.abspath(govt_folder)

    if not os.path.exists(govt_folder):
        print(f"⚠️ Government data folder not found: {govt_folder}")
        print(f"   Create it and add CSV files with columns: origin, year, official_stock")
        return pd.DataFrame(columns=["origin", "year", "official_stock", "source_file"])

    # Find all CSV files
    csv_files = glob.glob(os.path.join(govt_folder, "*.csv"))
    # Also check for Excel files
    xlsx_files = glob.glob(os.path.join(govt_folder, "*.xlsx"))

    if not csv_files and not xlsx_files:
        print(f"⚠️ No CSV/XLSX files found in: {govt_folder}")
        print(f"   Expected format: CSV with columns 'origin', 'year', 'official_stock'")
        return pd.DataFrame(columns=["origin", "year", "official_stock", "source_file"])

    if verbose:
        print(f"📋 Loading government data from: {govt_folder}")
        print(f"   Found {len(csv_files)} CSV + {len(xlsx_files)} XLSX files")
        print()

    # Load all CSVs
    all_frames = []
    for fp in sorted(csv_files):
        df = load_single_csv(fp, verbose=verbose)
        if not df.empty:
            all_frames.append(df)

    # Load XLSX files
    for fp in sorted(xlsx_files):
        if verbose:
            print(f"   📂 Loading: {os.path.basename(fp)}")
        try:
            xls = pd.read_excel(fp)
            # Save as temp CSV and load through standard pipeline
            tmp_csv = fp.replace(".xlsx", "_tmp.csv")
            xls.to_csv(tmp_csv, index=False)
            df = load_single_csv(tmp_csv, verbose=verbose)
            if not df.empty:
                df["source_file"] = os.path.basename(fp)
                all_frames.append(df)
            os.remove(tmp_csv)
        except Exception as e:
            print(f"      ⚠️ Error reading {fp}: {e}")

    if not all_frames:
        print("⚠️ No valid data loaded from government sources.")
        return pd.DataFrame(columns=["origin", "year", "official_stock", "source_file"])

    df_govt = pd.concat(all_frames, ignore_index=True)

    # Auto-detect and convert units
    if convert_to_thousands:
        unit = detect_stock_unit(df_govt)
        if unit == "raw":
            if verbose:
                print(f"\n   ℹ️ Stock values appear to be in raw counts (median: "
                      f"{df_govt['official_stock'].median():,.0f})")
                print(f"      Converting to thousands (÷ 1000)")
            df_govt["official_stock"] = df_govt["official_stock"] / 1000.0
        elif verbose:
            print(f"\n   ℹ️ Stock values appear to already be in thousands")

    # Remove duplicates (keep latest source)
    df_govt = df_govt.sort_values("source_file").drop_duplicates(
        subset=["origin", "year"], keep="last"
    )

    if verbose:
        print(f"\n✅ Government data loaded:")
        print(f"   Countries : {df_govt['origin'].nunique()}")
        print(f"   Years     : {sorted(df_govt['year'].unique())}")
        print(f"   Total rows: {len(df_govt)}")
        if len(df_govt) > 0:
            print(f"   Stock range: {df_govt['official_stock'].min():.1f}K — "
                  f"{df_govt['official_stock'].max():,.1f}K")

    return df_govt.reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  Government Data Loader — UAE Migration Statistics")
    print("=" * 60)
    print()
    df = load_govt_data("Govt/")
    if len(df) > 0:
        print("\nPreview:")
        print(df.head(20).to_string(index=False))
