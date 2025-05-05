"""
Download Chicago Energy Benchmarking data and save to data/raw/.
"""
import os
import sys
import urllib.request
import zipfile
import pandas as pd

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create data directories if they don't exist
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# URLs for Chicago Energy Benchmarking data
# Note: Update these URLs if they change
data_urls = {
    'Chicago_Energy_Benchmarking_20250403.csv': 'https://data.cityofchicago.org/api/views/xq83-jr8c/rows.csv?accessType=DOWNLOAD',
    'Chicago_Energy_Benchmarking_-_Covered_Buildings_20250403.csv': 'https://data.cityofchicago.org/api/views/g5i5-yz37/rows.csv?accessType=DOWNLOAD'
}

# Download files
for filename, url in data_urls.items():
    output_path = os.path.join(raw_dir, filename)
    
    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
    else:
        print(f"{filename} already exists, skipping download.")

print("Data download complete.")