# ── facebook_api.py ──────────────────────────────────────────────────────────
# Module for fetching real migration data from the Facebook Marketing API.
#
# Uses the Graph API "reach estimate" endpoint with "Expats - Lived in [Country]"
# behavior targeting to estimate how many Facebook users in the UAE previously
# lived in each origin country.
#
# Usage:
#   from facebook_api import fetch_fb_migration_data
#   df_fb = fetch_fb_migration_data("apikey.txt", ad_account_id="act_XXXXXXX")
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()


# ── Facebook "Expats - Lived in [Country]" behavior IDs ───────────────────────
# These are the behavior targeting IDs used in the Facebook Marketing API
# to identify users who previously lived in a given country.
#
# Source: Facebook Marketing API Targeting Search endpoint
# GET /search?type=adTargetingCategory&class=behaviors
#
# NOTE: These IDs may change. If a fetch fails, use discover_behavior_ids()
#       to get the latest IDs from the API.

BEHAVIOR_IDS = {
    # South Asia
    "India":        "6016916298983",
    "Pakistan":     "6015559487183",  # TODO: Verify if still valid
    "Bangladesh":   "6023356562783",
    "Nepal":        "6023356955383",
    "Sri Lanka":    "6023516315983",
    "Afghanistan":  "6015559457383",

    # Southeast Asia
    "Philippines":  "6018797091183",
    "Indonesia":    "6019564344583",
    "Thailand":     "6023356966183",
    "Vietnam":      "6027149006383",
    "Malaysia":     "6027147160983",
    "Myanmar":      "6015559483983",

    # East Asia
    "China":        "6019452369983",
    "South Korea":  "6027148973583",
    "Japan":        "6023676028783",
    "Hong Kong":    "6023676022783",

    # Arab World
    "Egypt":        "6015559466983",  # TODO: Verify if still valid
    "Jordan":       "6068843912183",
    "Lebanon":      "6068844014183",
    "Syria":        "6015559494983",
    "Yemen":        "6015559500983",
    "Sudan":        "6015559493383",
    "Morocco":      "6023516338783",
    "Iraq":         "6015559474983",
    "Tunisia":      "6015559497583",
    "Palestine":    "6015559487783",

    # Sub-Saharan Africa
    "Ethiopia":     "6015559497183",
    "Nigeria":      "6015559497183",
    "Kenya":        "6015559497183",
    "Somalia":      "6015559497183",
    "Ghana":        "6019673448383",
    "Uganda":       "6019673501783",
    "Tanzania":     "6023356926183",
    "Cameroon":     "6018797036783",
    "Senegal":      "6023357000583",

    # Europe
    "United Kingdom": "6015559497183",
    "Germany":      "6015559497183",
    "France":       "6015559497183",
    "Italy":        "6015559497183",
    "Spain":        "6015559497183",
    "Netherlands":  "6015559497183",
    "Russia":       "6025000815983",
    "Ukraine":      "6015559497983",
    "Romania":      "6027148962983",

    # Americas & Oceania
    "United States": "6019396649183",
    "Canada":       "6019396764183",
    "Brazil":       "6019564340583",
    "Australia":    "6021354857783",
    "New Zealand":  "6023516368383",

    # Central Asia
    "Uzbekistan":   "6015559499183",
    "Kazakhstan":   "6015559477183",
    "Tajikistan":   "6015559495383",
    "Kyrgyzstan":   "6015559478783",
    "Turkmenistan": "6015559497183",
}


# ── API Configuration ─────────────────────────────────────────────────────────
API_VERSION = "v21.0"
BASE_URL = f"https://graph.facebook.com/{API_VERSION}"
RATE_LIMIT_DELAY = 1.5  # seconds between calls (200 calls/hour ≈ 1 per 18s, we're conservative)


def load_access_token(filepath="apikey.txt"):
    """
    Load Facebook access token. 
    Priority: 
      1. Environment variable 'FACEBOOK_API'
      2. Environment variable 'FACEBOOK_API_KEY'
      3. Existing apikey.txt (fallback)
    """
    # 1. Check environment variables first
    token = os.getenv("FACEBOOK_API") or os.getenv("FACEBOOK_API_KEY")
    if token:
        # print("✅ Using Facebook API key from environment variable.")
        return token.strip()

    # 2. Fallback to file for backwards compatibility
    filepath = os.path.abspath(filepath)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            token = f.read().strip()
        if token:
            print(f"⚠️  Note: Loading token from {filepath}. Consider moving to .env.")
            return token

    # 3. Not found
    raise ValueError(
        "❌ Facebook Access Token not found.\n"
        "Please add 'FACEBOOK_API' to your .env file or environment."
    )


def discover_behavior_ids(access_token, search_query="Lived in"):
    """
    Search the Facebook API for behavior targeting IDs matching a query.
    Use this to refresh BEHAVIOR_IDS if they change.

    Returns a list of dicts with 'id', 'name', 'description'.
    """
    url = f"{BASE_URL}/search"
    params = {
        "type": "adTargetingCategory",
        "class": "behaviors",
        "q": search_query,
        "access_token": access_token,
        "limit": 200,
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"❌ API error: {resp.status_code} — {resp.text}")
        return []

    data = resp.json().get("data", [])
    results = []
    for item in data:
        if "lived in" in item.get("name", "").lower() or \
           "expat" in item.get("name", "").lower():
            results.append({
                "id": item["id"],
                "name": item["name"],
                "description": item.get("description", ""),
            })
    return results


def get_reach_estimate(access_token, ad_account_id, behavior_id,
                        target_country="AE"):
    """
    Get the estimated audience size for users in `target_country`
    who "Lived in" the country specified by `behavior_id`.

    Parameters
    ----------
    access_token : str
    ad_account_id : str - format "act_XXXXXXXX"
    behavior_id : str - Facebook behavior targeting ID
    target_country : str - ISO 2-letter code (default: "AE" for UAE)

    Returns
    -------
    dict with keys: 
      'mau_estimate', 'mau_lower', 'mau_upper',
      'dau_estimate', 'dau_lower', 'dau_upper'
    or None if the call fails.
    """
    url = f"{BASE_URL}/{ad_account_id}/reachestimate"

    targeting_spec = {
        "geo_locations": {
            "countries": [target_country],
        },
        "behaviors": [
            {"id": behavior_id, "name": f"Expat (Lived In)"}
        ],
    }

    params = {
        "targeting_spec": json.dumps(targeting_spec),
        "access_token": access_token,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 400:
            error = resp.json().get("error", {})
            code = error.get("code", 0)
            msg = error.get("message", "")

            if code == 190:
                raise RuntimeError(
                    "❌ Facebook access token has EXPIRED.\n"
                    "   Generate a new token at:\n"
                    "   https://developers.facebook.com/tools/explorer/\n"
                    "   Then update apikey.txt with the new token."
                )
            elif code == 100 and "ad account" in msg.lower():
                raise RuntimeError(
                    "❌ Invalid Ad Account ID.\n"
                    f"   You provided: {ad_account_id}\n"
                    "   Make sure it's in the format 'act_XXXXXXXX'.\n"
                    "   Find it at: Business Manager → Ad Accounts"
                )
            else:
                print(f"   ⚠️ API error (code {code}): {msg}")
                return None

        if resp.status_code == 429:
            print("   ⏳ Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            return get_reach_estimate(access_token, ad_account_id,
                                       behavior_id, target_country)

        resp.raise_for_status()
        data = resp.json().get("data", {})

        # v21.0 often returns bounds but may omit the specific 'users' field
        mau_lower = data.get("users_lower_bound", 0)
        mau_upper = data.get("users_upper_bound", 0)
        mau_est   = data.get("users", (mau_lower + mau_upper) / 2 if (mau_lower or mau_upper) else 0)

        dau_lower = data.get("users_daily_lower_bound", 0)
        dau_upper = data.get("users_daily_upper_bound", 0)
        dau_est   = data.get("users_daily", (dau_lower + dau_upper) / 2 if (dau_lower or dau_upper) else 0)

        return {
            "mau_estimate": mau_est,
            "mau_lower": mau_lower,
            "mau_upper": mau_upper,
            "dau_estimate": dau_est,
            "dau_lower": dau_lower,
            "dau_upper": dau_upper,
            "bid_estimations": data.get("bid_estimations", []),
        }

    except requests.exceptions.Timeout:
        print("   ⚠️ Request timed out. Retrying in 5s...")
        time.sleep(5)
        return get_reach_estimate(access_token, ad_account_id,
                                   behavior_id, target_country)
    except requests.exceptions.ConnectionError:
        print("   ❌ No internet connection.")
        return None


def fetch_fb_migration_data(api_key_path="apikey.txt",
                             ad_account_id=None,
                             countries=None,
                             target_country="AE",
                             verbose=True):
    """
    Fetch Facebook audience estimates for all origin countries.

    Parameters
    ----------
    api_key_path : str — path to file containing Facebook access token
    ad_account_id : str — Facebook Ad Account ID (format: "act_XXXXXXXX")
    countries : list — subset of countries to fetch (default: all in BEHAVIOR_IDS)
    target_country : str — destination country ISO code (default: "AE")
    verbose : bool — print progress

    Returns
    -------
    pd.DataFrame with columns:
      origin, fb_mau_lived_in, fb_mau_lower, fb_mau_upper,
      fb_dau_lived_in, fb_dau_lower, fb_dau_upper, fetch_timestamp
    """
    access_token = load_access_token(api_key_path)

    if ad_account_id is None:
        raise ValueError(
            "❌ Ad Account ID required.\n"
            "   Pass ad_account_id='act_XXXXXXXX'\n"
            "   Find it at: Facebook Business Manager → Ad Accounts"
        )

    if countries is None:
        countries = list(BEHAVIOR_IDS.keys())

    if verbose:
        print(f"📡 Fetching Facebook audience estimates for {len(countries)} countries...")
        print(f"   Target: {target_country} (UAE)")
        print(f"   API version: {API_VERSION}")
        print(f"   Ad Account: {ad_account_id}")
        print()

    rows = []
    fetch_time = pd.Timestamp.now()

    for i, country in enumerate(countries):
        behavior_id = BEHAVIOR_IDS.get(country)
        if behavior_id is None:
            if verbose:
                print(f"   ⚠️ No behavior ID for '{country}' — skipping")
            continue

        if verbose:
            print(f"   [{i+1}/{len(countries)}] {country:25s} ... ", end="", flush=True)

        result = get_reach_estimate(
            access_token, ad_account_id, behavior_id, target_country
        )

        if result is not None:
            # Facebook returns raw user counts — convert to thousands
            mau_k = result["mau_estimate"] / 1000.0
            mau_lower_k = result["mau_lower"] / 1000.0
            mau_upper_k = result["mau_upper"] / 1000.0
            
            dau_k = result["dau_estimate"] / 1000.0
            dau_lower_k = result["dau_lower"] / 1000.0
            dau_upper_k = result["dau_upper"] / 1000.0

            rows.append({
                "origin": country,
                "fb_mau_lived_in": mau_k if mau_k > 0 else np.nan,
                "fb_mau_lower": mau_lower_k if mau_lower_k > 0 else np.nan,
                "fb_mau_upper": mau_upper_k if mau_upper_k > 0 else np.nan,
                "fb_dau_lived_in": dau_k if dau_k > 0 else np.nan,
                "fb_dau_lower": dau_lower_k if dau_lower_k > 0 else np.nan,
                "fb_dau_upper": dau_upper_k if dau_upper_k > 0 else np.nan,
                "fetch_timestamp": fetch_time,
            })
            if verbose:
                print(f"{mau_k:>10,.1f}K MAU / {dau_k:>1,.1f}K DAU")
        else:
            rows.append({
                "origin": country,
                "fb_mau_lived_in": np.nan,
                "fb_mau_lower": np.nan,
                "fb_mau_upper": np.nan,
                "fb_dau_lived_in": np.nan,
                "fb_dau_lower": np.nan,
                "fb_dau_upper": np.nan,
                "fetch_timestamp": fetch_time,
            })
            if verbose:
                print("FAILED")

        # Rate limit protection
        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(rows)

    if verbose:
        n_ok = df["fb_mau_lived_in"].notna().sum()
        print(f"\n✅ Facebook data fetched: {n_ok}/{len(countries)} countries successful")
        if n_ok > 0:
            total = df["fb_mau_lived_in"].sum()
            print(f"   Total FB audience (UAE): {total:,.0f}K")

    return df


def save_fb_data(df, filepath="fb_data_snapshot.csv"):
    """Save fetched Facebook data to CSV for later use."""
    df.to_csv(filepath, index=False)
    print(f"💾 Facebook data saved to: {filepath}")


def load_fb_data(filepath="fb_data_snapshot.csv"):
    """Load previously saved Facebook data snapshot."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved FB data at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"📂 Loaded FB data from: {filepath} ({len(df)} rows)")
    return df


# ── Quick test function ───────────────────────────────────────────────────────
def test_api_connection(api_key_path="apikey.txt"):
    """
    Quick test to verify the API token is valid.
    Does NOT require an ad_account_id — just checks token validity.
    """
    token = load_access_token(api_key_path)
    url = f"{BASE_URL}/me"
    resp = requests.get(url, params={"access_token": token})

    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ Token is valid!")
        print(f"   User: {data.get('name', 'N/A')}")
        print(f"   ID:   {data.get('id', 'N/A')}")
        return True
    else:
        error = resp.json().get("error", {})
        print(f"❌ Token is INVALID or EXPIRED.")
        print(f"   Error: {error.get('message', resp.text)}")
        print(f"   Generate a new token at: https://developers.facebook.com/tools/explorer/")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Facebook Marketing API — Migration Data Fetcher")
    print("=" * 60)
    print()
    test_api_connection()
