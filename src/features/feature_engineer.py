import pandas as pd
import numpy as np
from itertools import combinations

class FeatureEngineer:
    """
    Class for feature engineering on the Chicago Energy Benchmarking data.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the FeatureEngineer class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to engineer features for
        """
        self.merged_df = merged_df.copy()
    
    def create_building_age_features(self):
        """
        Create building age and age category features.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new age features
        """
        current_year = 2025
        self.merged_df['Building Age'] = current_year - self.merged_df['Year Built']
        
        # Create age categories
        self.merged_df['Age Category'] = pd.cut(
            self.merged_df['Building Age'],
            bins=[0, 25, 50, 75, 100, float('inf')],
            labels=['< 25 years', '25-49 years', '50-74 years', '75-99 years', '100+ years']
        )
        
        return self.merged_df
    
    def create_size_features(self):
        """
        Create building size category features.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new size features
        """
        # Create size categories
        self.merged_df['Size Category'] = pd.cut(
            self.merged_df['Gross Floor Area - Buildings (sq ft)'],
            bins=[0, 50000, 100000, 250000, 500000, float('inf')],
            labels=['< 50K sq ft', '50K-100K sq ft', '100K-250K sq ft', '250K-500K sq ft', '> 500K sq ft']
        )
        
        return self.merged_df
    
    def create_energy_mix_features(self):
        """
        Calculate energy mix percentages (electricity vs. natural gas vs. other).
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new energy mix features
        """
        energy_cols = [
            'Electricity Use (kBtu)',
            'Natural Gas Use (kBtu)',
            'District Steam Use (kBtu)',
            'District Chilled Water Use (kBtu)',
            'All Other Fuel Use (kBtu)'
        ]
        
        # Filter to energy columns that exist in the dataframe
        energy_cols = [col for col in energy_cols if col in self.merged_df.columns]
        
        # Replace NaN with 0 for energy columns
        for col in energy_cols:
            self.merged_df[col] = self.merged_df[col].fillna(0)
        
        # Calculate total energy and percentages
        self.merged_df['Total Energy (kBtu)'] = self.merged_df[energy_cols].sum(axis=1)
        for col in energy_cols:
            new_col = col.replace(' Use (kBtu)', ' Percentage')
            self.merged_df[new_col] = self.merged_df[col] / self.merged_df['Total Energy (kBtu)'] * 100
        
        # Handle division by zero
        for col in [c.replace(' Use (kBtu)', ' Percentage') for c in energy_cols]:
            self.merged_df[col] = self.merged_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Group electricity usage into categories
        if 'Electricity Percentage' in self.merged_df.columns:
            self.merged_df['Electricity Usage Category'] = pd.cut(
                self.merged_df['Electricity Percentage'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            )
        
        return self.merged_df
    
    def create_performance_metrics(self):
        """
        Create relative performance metrics compared to peers.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new performance metrics
        """
        # Reset the index if it contains groupby keys
        if any(key in self.merged_df.index.names for key in ['Primary Property Type', 'Size Category']):
            self.merged_df = self.merged_df.reset_index()
        
        # Define a function to calculate relative performance
        def calculate_relative_performance(group):
            if len(group) >= 5:  # Only for groups with enough buildings for comparison
                group['Avg Peer EUI'] = group['Site EUI (kBtu/sq ft)'].mean()
                group['EUI Performance vs Peers (%)'] = (group['Site EUI (kBtu/sq ft)'] - group['Avg Peer EUI']) / group['Avg Peer EUI'] * 100
            return group
        
        # Group by property type and size category
        # Use transform instead of apply to avoid changing the index
        avg_eui = self.merged_df.groupby(['Primary Property Type', 'Size Category'])['Site EUI (kBtu/sq ft)'].transform('mean')
        self.merged_df['Avg Peer EUI'] = avg_eui
        
        # Calculate performance vs peers only where avg_eui is not zero or null
        self.merged_df['EUI Performance vs Peers (%)'] = np.where(
            (self.merged_df['Avg Peer EUI'].notna()) & (self.merged_df['Avg Peer EUI'] != 0),
            (self.merged_df['Site EUI (kBtu/sq ft)'] - self.merged_df['Avg Peer EUI']) / self.merged_df['Avg Peer EUI'] * 100,
            0
        )
        
        # Create performance categories
        self.merged_df['Performance Category'] = pd.cut(
            self.merged_df['EUI Performance vs Peers (%)'].fillna(0),
            bins=[-float('inf'), -25, -10, 10, 25, float('inf')],
            labels=['Excellent (>25% better)', 'Good (10-25% better)', 'Average (±10%)', 'Poor (10-25% worse)', 'Very Poor (>25% worse)']
        )
        
        return self.merged_df

    def create_advanced_features(self):
        """
        Create advanced features for more sophisticated analysis.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new advanced features
        """
        # Make sure we're working with a reset index
        if 'Primary Property Type' in self.merged_df.index.names:
            self.merged_df = self.merged_df.reset_index()
        
        # Energy Efficiency Ratio - lower is better
        if all(col in self.merged_df.columns for col in ['Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)']):
            self.merged_df['Site to Source Ratio'] = self.merged_df['Site EUI (kBtu/sq ft)'] / self.merged_df['Source EUI (kBtu/sq ft)']
        
        # Emissions per energy unit - measure of energy source cleanliness
        if all(col in self.merged_df.columns for col in ['GHG Intensity (kg CO2e/sq ft)', 'Site EUI (kBtu/sq ft)']):
            self.merged_df['Emissions per Energy Unit'] = self.merged_df['GHG Intensity (kg CO2e/sq ft)'] / self.merged_df['Site EUI (kBtu/sq ft)']
        
        # Energy intensity relative to age
        if all(col in self.merged_df.columns for col in ['Site EUI (kBtu/sq ft)', 'Building Age']):
            # First handle zeros to avoid division by zero
            building_age_non_zero = self.merged_df['Building Age'].copy()
            building_age_non_zero[building_age_non_zero <= 0] = 1
            self.merged_df['Energy Intensity per Year of Age'] = self.merged_df['Site EUI (kBtu/sq ft)'] / building_age_non_zero
        
        # Benchmark percentile rank within property type
        if 'Site EUI (kBtu/sq ft)' in self.merged_df.columns:
            # Reset index first to ensure no ambiguity
            self.merged_df['EUI Percentile Rank'] = self.merged_df.reset_index(drop=True).groupby('Primary Property Type')['Site EUI (kBtu/sq ft)'].transform(lambda x: x.rank(pct=True) * 100)
        
        # Energy Star Score squared (to capture non-linear effects in modeling)
        if 'ENERGY STAR Score' in self.merged_df.columns:
            self.merged_df['ENERGY STAR Score Squared'] = self.merged_df['ENERGY STAR Score'] ** 2
        
        # Log transform of floor area (to handle skewed distribution)
        if 'Gross Floor Area - Buildings (sq ft)' in self.merged_df.columns:
            self.merged_df['Log Floor Area'] = np.log1p(self.merged_df['Gross Floor Area - Buildings (sq ft)'])
        
        return self.merged_df

    def create_system_efficiency_indicators(self):
        """
        Create indicators for different building system efficiencies.
        These will be used for targeted recommendations.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with system efficiency indicators
        """
        # Normalize key metrics for easier comparisons
        self.merged_df['EUI Type Percentile'] = self.merged_df.groupby('Primary Property Type')['Site EUI (kBtu/sq ft)'].transform(
            lambda x: pd.Series(x).rank(pct=True)
        )
        
        if 'ENERGY STAR Score' in self.merged_df.columns:
            self.merged_df['ENERGY STAR Type Percentile'] = self.merged_df.groupby('Primary Property Type')['ENERGY STAR Score'].transform(
                lambda x: pd.Series(x).rank(pct=True)
            )
        
        # 1. Lighting efficiency indicator (0=Poor, 1=Average, 2=Good)
        if 'Electricity Percentage' in self.merged_df.columns and 'ENERGY STAR Score' in self.merged_df.columns:
            self.merged_df['Lighting Efficiency'] = 1  # Default to average
            
            # Modified lighting logic: consider overall performance more strongly
            poor_lighting_mask = (
                ((self.merged_df['Electricity Percentage'] > 60) & (self.merged_df['ENERGY STAR Score'] < 50)) |
                (self.merged_df['ENERGY STAR Score'] < 25)  # Very poor overall performance suggests all systems need work
            )
            self.merged_df.loc[poor_lighting_mask, 'Lighting Efficiency'] = 0  # Poor
            
            good_lighting_mask = (
                (self.merged_df['ENERGY STAR Score'] > 75) &
                (self.merged_df['Electricity Percentage'] > 50)
            )
            self.merged_df.loc[good_lighting_mask, 'Lighting Efficiency'] = 2  # Good
        
        # 2. HVAC efficiency indicator - considering ENERGY STAR score more heavily
        if 'EUI Type Percentile' in self.merged_df.columns:
            self.merged_df['HVAC Efficiency'] = 1  # Default to average
            
            poor_hvac_mask = (
                ((self.merged_df['EUI Type Percentile'] < 0.25) &
                ((self.merged_df['Natural Gas Percentage'] > 60) | (self.merged_df['Electricity Percentage'] > 60))) |
                (self.merged_df['ENERGY STAR Score'] < 25)  # Very poor overall performance
            )
            self.merged_df.loc[poor_hvac_mask, 'HVAC Efficiency'] = 0  # Poor
            
            # Make sure buildings with excellent ENERGY STAR scores don't get poor HVAC ratings
            # unless there's strong evidence of HVAC issues
            if 'ENERGY STAR Score' in self.merged_df.columns:
                self.merged_df.loc[(self.merged_df['ENERGY STAR Score'] > 85) &
                                  (self.merged_df['HVAC Efficiency'] == 0), 'HVAC Efficiency'] = 1
            
            good_hvac_mask = (
                (self.merged_df['EUI Type Percentile'] > 0.75) &
                (self.merged_df['ENERGY STAR Score'] > 60)
            )
            self.merged_df.loc[good_hvac_mask, 'HVAC Efficiency'] = 2  # Good
        
        # 3. Envelope efficiency indicator
        if 'Building Age' in self.merged_df.columns and 'Natural Gas Percentage' in self.merged_df.columns:
            self.merged_df['Envelope Efficiency'] = 1  # Default to average
            
            # Envelope: Older buildings with high heating demand likely have envelope issues
            poor_envelope_mask = (
                (self.merged_df['Building Age'] > 50) &
                (self.merged_df['Natural Gas Percentage'] > 60) &
                (self.merged_df['EUI Type Percentile'] < 0.25)
            )
            self.merged_df.loc[poor_envelope_mask, 'Envelope Efficiency'] = 0  # Poor
            
            good_envelope_mask = (
                (self.merged_df['Building Age'] > 40) &
                (self.merged_df['ENERGY STAR Score'] > 70)
            )
            self.merged_df.loc[good_envelope_mask, 'Envelope Efficiency'] = 2  # Good
        
        # 4. Controls efficiency indicator
        if 'Gross Floor Area - Buildings (sq ft)' in self.merged_df.columns:
            self.merged_df['Controls Efficiency'] = 1  # Default to average
            
            # Controls: Large buildings often benefit most from advanced controls
            gross_floor_area_med = self.merged_df['Gross Floor Area - Buildings (sq ft)'].median()
            
            poor_controls_mask = (
                (self.merged_df['Gross Floor Area - Buildings (sq ft)'] > gross_floor_area_med) &
                (self.merged_df['ENERGY STAR Score'] < 50)
            )
            self.merged_df.loc[poor_controls_mask, 'Controls Efficiency'] = 0  # Poor
            
            good_controls_mask = (
                (self.merged_df['Gross Floor Area - Buildings (sq ft)'] > gross_floor_area_med) &
                (self.merged_df['ENERGY STAR Score'] > 70)
            )
            self.merged_df.loc[good_controls_mask, 'Controls Efficiency'] = 2  # Good
        
        return self.merged_df
    
    def engineer_all_features(self):
        """
        Run all feature engineering methods and return the enhanced dataset.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all engineered features
        """
        print("Starting feature engineering...")
        
        # Create building age features
        self.create_building_age_features()
        print("Building age features created")
        
        # Create size features
        self.create_size_features()
        print("Building size features created")
        
        # Create energy mix features
        self.create_energy_mix_features()
        print("Energy mix features created")
        
        # Create performance metrics
        self.create_performance_metrics()
        print("Performance metrics created")
        
        # Create advanced features
        self.create_advanced_features()
        print("Advanced features created")
        
        # Create system efficiency indicators
        self.create_system_efficiency_indicators()
        print("System efficiency indicators created")
        
        print("Feature engineering completed")
        
        return self.merged_df