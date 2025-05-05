import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    """
    Class for loading, cleaning, and preprocessing Chicago Energy Benchmarking data.
    """
    
    def __init__(self):
        """Initialize the DataProcessor class."""
        self.energy_df = None
        self.buildings_df = None
        self.merged_df = None
    
    def load_data(self, energy_file_path, buildings_file_path):
        """
        Load data from energy benchmarking and covered buildings CSV files.
        
        Parameters:
        -----------
        energy_file_path : str
            Path to the energy benchmarking CSV file
        buildings_file_path : str
            Path to the covered buildings CSV file
        """
        print("Loading data...")
        
        # Load the Chicago Energy Benchmarking dataset
        self.energy_df = pd.read_csv(energy_file_path)
        print(f"Energy Benchmarking Dataset Shape: {self.energy_df.shape}")
        
        # Load the Covered Buildings dataset
        self.buildings_df = pd.read_csv(buildings_file_path)
        print(f"Covered Buildings Dataset Shape: {self.buildings_df.shape}")
        
        print("Data loaded successfully.")
        
        return self.energy_df, self.buildings_df
    
    def check_missing_values(self):
        """
        Check for missing values in both datasets and print summaries.
        """
        print("Missing values in Energy Benchmarking Dataset:")
        print(self.energy_df.isnull().sum()[self.energy_df.isnull().sum() > 0])

        print("\nMissing values in Covered Buildings Dataset:")
        print(self.buildings_df.isnull().sum()[self.buildings_df.isnull().sum() > 0])
    
    def get_latest_year_data(self):
        """
        Extract data for the latest available year from the energy dataset.
        
        Returns:
        --------
        pandas.DataFrame
            Energy data for the latest year
        """
        latest_year = self.energy_df['Data Year'].max()
        energy_latest = self.energy_df[self.energy_df['Data Year'] == latest_year].copy()
        
        print(f"Latest year in dataset: {latest_year}")
        print(f"Number of records for {latest_year}: {len(energy_latest)}")
        
        return energy_latest, latest_year
    
    def merge_datasets(self, energy_latest):
        """
        Merge the energy benchmarking data with the buildings data.
        
        Parameters:
        -----------
        energy_latest : pandas.DataFrame
            Energy data for the latest year
            
        Returns:
        --------
        pandas.DataFrame
            Merged dataset
        """
        # Rename ID column in energy dataset to match buildings dataset for merging
        energy_latest.rename(columns={'ID': 'Building ID'}, inplace=True)
        
        # Merge datasets
        self.merged_df = pd.merge(energy_latest, self.buildings_df, on='Building ID', 
                                  how='left', suffixes=('', '_buildings'))
        
        print(f"Merged dataset shape: {self.merged_df.shape}")
        print(f"Number of buildings with matching IDs: {self.merged_df['Building ID'].notna().sum()}")
        
        return self.merged_df
    
    def handle_missing_values(self):
        """
        Handle missing values in the merged dataset using strategic imputation.
        
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataset with imputed values
        """
        # Make sure we're working with a copy to avoid warnings
        self.merged_df = self.merged_df.copy()
        
        # Define key numerical columns
        numerical_cols = [
            'Gross Floor Area - Buildings (sq ft)',
            'Year Built',
            'Site EUI (kBtu/sq ft)',
            'Source EUI (kBtu/sq ft)',
            'ENERGY STAR Score',
            'GHG Intensity (kg CO2e/sq ft)',
            'Total GHG Emissions (Metric Tons CO2e)'
        ]
        
        # For numerical columns, use median imputation by property type
        for col in numerical_cols:
            if col in self.merged_df.columns:
                # Calculate median by property type for more accurate imputation
                medians = self.merged_df.groupby('Primary Property Type')[col].median()
                
                # For each property type, fill missing values with the median for that type
                for prop_type in medians.index:
                    mask = (self.merged_df['Primary Property Type'] == prop_type) & self.merged_df[col].isna()
                    # Use loc instead of chained assignment
                    self.merged_df.loc[mask, col] = medians[prop_type]
                
                # For any remaining NAs (property types with all NA values), use overall median
                # Use loc instead of inplace=True
                self.merged_df.loc[self.merged_df[col].isna(), col] = self.merged_df[col].median()
        
        # For categorical columns, use mode or leave as is if appropriate
        categorical_cols = ['Primary Property Type', 'Community Area', 'Community Area Name']
        for col in categorical_cols:
            if col in self.merged_df.columns and self.merged_df[col].isna().sum() > 0:
                # Use loc instead of inplace=True
                self.merged_df.loc[self.merged_df[col].isna(), col] = 'Unknown'
        
        print("Missing values handled with strategic imputation.")
        
        return self.merged_df
        
    def process_data(self, energy_file_path, buildings_file_path):
        """
        Full data processing pipeline: load, merge, and clean the datasets.
        
        Parameters:
        -----------
        energy_file_path : str
            Path to the energy benchmarking CSV file
        buildings_file_path : str
            Path to the covered buildings CSV file
            
        Returns:
        --------
        tuple
            Tuple containing (energy_df, buildings_df, merged_df, latest_year)
        """
        # Load data
        self.load_data(energy_file_path, buildings_file_path)
        
        # Check missing values
        self.check_missing_values()
        
        # Get latest year data
        energy_latest, latest_year = self.get_latest_year_data()
        
        # Merge datasets
        self.merged_df = self.merge_datasets(energy_latest)
        
        # Handle missing values
        self.merged_df = self.handle_missing_values()
        
        return self.energy_df, self.buildings_df, self.merged_df, latest_year