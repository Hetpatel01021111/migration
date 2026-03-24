import pandas as pd
from facebook_api import fetch_fb_migration_data, save_fb_data
import os

# Ad account from user
AD_ACCOUNT_ID = 'act_1640054656564082'
OUTPUT_PATH = 'fb_data_snapshot.csv'

# Full list from transition zones
ORIGIN_REGIONS = {
    "South Asia": ["India", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka", "Afghanistan"],
    "Southeast Asia": ["Philippines", "Indonesia", "Thailand", "Vietnam", "Malaysia", "Myanmar"],
    "East Asia": ["China", "South Korea", "Japan", "Hong Kong"],
    "Arab World": ["Egypt", "Jordan", "Lebanon", "Syria", "Yemen", "Sudan", "Morocco", "Iraq", "Tunisia", "Palestine"],
    "Sub-Saharan Africa": ["Ethiopia", "Nigeria", "Kenya", "Somalia", "Ghana", "Uganda", "Tanzania", "Cameroon", "Senegal"],
    "Europe": ["United Kingdom", "Germany", "France", "Italy", "Spain", "Netherlands", "Russia", "Ukraine", "Romania"],
    "Americas & Oceania": ["United States", "Canada", "Brazil", "Australia", "New Zealand"],
    "Central Asia": ["Uzbekistan", "Kazakhstan", "Tajikistan", "Kyrgyzstan", "Turkmenistan"]
}
ALL_ORIGINS = [c for countries in ORIGIN_REGIONS.values() for c in countries]

def fetch_all():
    print(f"🚀 Starting full fetch for {len(ALL_ORIGINS)} countries...")
    # fetch_fb_migration_data(api_key_path="apikey.txt", ad_account_id=None, countries=None, ...)
    df = fetch_fb_migration_data(
        api_key_path="apikey.txt",
        ad_account_id=AD_ACCOUNT_ID,
        countries=ALL_ORIGINS
    )
    
    print(f"✅ Fetch complete. {len(df)} rows retrieved.")
    save_fb_data(df, OUTPUT_PATH)
    print(f"📁 Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_all()
