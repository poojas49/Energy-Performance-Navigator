import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ExploratoryAnalysis:
    """
    Class for conducting exploratory data analysis on the Chicago Energy Benchmarking data.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the ExploratoryAnalysis class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to analyze
        """
        self.merged_df = merged_df
    
    def generate_summary_statistics(self, columns=None):
        """
        Generate summary statistics for specified energy metrics.
        
        Parameters:
        -----------
        columns : list, optional
            List of column names to include in summary. If None, uses default energy metrics.
            
        Returns:
        --------
        pandas.DataFrame
            Summary statistics
        """
        if columns is None:
            columns = ['Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)', 
                      'ENERGY STAR Score', 'GHG Intensity (kg CO2e/sq ft)']
        
        # Calculate summary statistics
        summary_stats = self.merged_df[columns].describe()
        print("Summary Statistics for Key Energy Metrics:")
        print(summary_stats)
        
        return summary_stats
    
    def analyze_building_types(self, n=10):
        """
        Analyze the distribution of building types.
        
        Parameters:
        -----------
        n : int, optional
            Number of top building types to display
            
        Returns:
        --------
        pandas.Series
            Counts of top building types
        """
        building_type_counts = self.merged_df['Primary Property Type'].value_counts().head(n)
        
        # Create bar plot of building type distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(x=building_type_counts.values, y=building_type_counts.index)
        plt.title(f'Top {n} Building Types')
        plt.xlabel('Count')
        plt.ylabel('Building Type')
        plt.tight_layout()
        plt.savefig('building_types_distribution.png', dpi=300)
        plt.close()
        
        return building_type_counts
    
    def analyze_energy_ratings(self):
        """
        Analyze the distribution of Chicago Energy Ratings.
        
        Returns:
        --------
        pandas.Series
            Counts of energy ratings
        """
        rating_counts = self.merged_df['Chicago Energy Rating'].value_counts().sort_index()
        
        # Create bar plot of energy ratings
        plt.figure(figsize=(10, 6))
        sns.barplot(x=rating_counts.index, y=rating_counts.values)
        plt.title('Distribution of Chicago Energy Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.xticks(range(len(rating_counts.index)), rating_counts.index)
        plt.tight_layout()
        plt.savefig('energy_ratings_distribution.png', dpi=300)
        plt.close()
        
        return rating_counts
    
    def correlation_analysis(self, columns=None):
        """
        Analyze correlations between key energy metrics.
        
        Parameters:
        -----------
        columns : list, optional
            List of column names to include in correlation analysis. 
            If None, uses default set of metrics.
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        if columns is None:
            columns = [
                'Site EUI (kBtu/sq ft)',
                'Source EUI (kBtu/sq ft)',
                'ENERGY STAR Score',
                'GHG Intensity (kg CO2e/sq ft)',
                'Chicago Energy Rating',
                'Building Age',
                'Gross Floor Area - Buildings (sq ft)',
                'Electricity Percentage',
                'Natural Gas Percentage'
            ]
            
            # Make sure all columns exist in the dataframe
            columns = [col for col in columns if col in self.merged_df.columns]
        
        # Calculate correlation matrix
        corr_matrix = self.merged_df[columns].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        plt.title('Correlation Between Key Metrics')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300)
        plt.close()
        
        return corr_matrix
    
    def analyze_building_age_vs_performance(self):
        """
        Analyze how building age relates to energy performance.
        
        Returns:
        --------
        pandas.DataFrame
            Performance metrics grouped by building age
        """
        # Ensure Building Age column exists
        if 'Building Age' not in self.merged_df.columns:
            print("Building Age column not found. Please run feature engineering first.")
            return None
            
        # Create or use age categories if they exist
        if 'Age Category' not in self.merged_df.columns:
            current_year = 2025
            self.merged_df['Building Age'] = current_year - self.merged_df['Year Built']
            self.merged_df['Age Category'] = pd.cut(
                self.merged_df['Building Age'],
                bins=[0, 25, 50, 75, 100, float('inf')],
                labels=['< 25 years', '25-49 years', '50-74 years', '75-99 years', '100+ years']
            )
        
        # Group by age category and calculate mean metrics
        age_performance = self.merged_df.groupby('Age Category').agg({
            'Site EUI (kBtu/sq ft)': 'mean',
            'ENERGY STAR Score': 'mean',
            'GHG Intensity (kg CO2e/sq ft)': 'mean',
            'Building ID': 'count'
        }).rename(columns={'Building ID': 'Count'}).reset_index()
        
        # Order categories properly for plotting
        correct_order = ['< 25 years', '25-49 years', '50-74 years', '75-99 years', '100+ years']
        age_performance['Age Category'] = pd.Categorical(
            age_performance['Age Category'],
            categories=correct_order,
            ordered=True
        )
        age_performance = age_performance.sort_values('Age Category')
        
        # Return the processed data
        return age_performance
    
    def run_full_eda(self):
        """
        Run all EDA analyses and return compiled results.
        
        Returns:
        --------
        dict
            Dictionary containing all EDA results
        """
        results = {}
        
        # Generate summary statistics
        results['summary_statistics'] = self.generate_summary_statistics()
        
        # Analyze building types
        results['building_type_counts'] = self.analyze_building_types()
        
        # Analyze energy ratings
        results['energy_rating_counts'] = self.analyze_energy_ratings()
        
        # Correlation analysis
        results['correlation_matrix'] = self.correlation_analysis()
        
        # Building age vs performance
        results['age_performance'] = self.analyze_building_age_vs_performance()
        
        return results