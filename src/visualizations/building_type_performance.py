import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BuildingTypePerformanceVisualizer:
    """
    Class for creating visualizations of energy performance by building type.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the BuildingTypePerformanceVisualizer class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to visualize
        """
        self.merged_df = merged_df.reset_index(drop=True)
    
    def visualize_performance_by_building_type(self, top_n=15):
        """
        Create visualization comparing energy performance metrics across top building types.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top building types to include
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the visualization
        """
        # Get the top building types by count
        top_property_types = self.merged_df['Primary Property Type'].value_counts().head(top_n).index.tolist()
        
        # Prepare data for visualization - filter to top types
        building_performance = self.merged_df[self.merged_df['Primary Property Type'].isin(top_property_types)].groupby('Primary Property Type').agg({
            'Site EUI (kBtu/sq ft)': 'mean',
            'Source EUI (kBtu/sq ft)': 'mean',
            'GHG Intensity (kg CO2e/sq ft)': 'mean',
            'Building ID': 'count'
        }).rename(columns={'Building ID': 'Count'}).reset_index()
        
        # Create a figure with one subplot per metric
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
        plt.subplots_adjust(hspace=0.3)
        
        # Define colors for each metric
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        metrics = ['Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)', 'GHG Intensity (kg CO2e/sq ft)']
        titles = ['Site Energy Use Intensity (Lower is Better)',
                  'Source Energy Use Intensity (Lower is Better)',
                  'Greenhouse Gas Intensity (Lower is Better)']
        
        # Create horizontal bar plots for each metric, sorting each by its own metric
        for i, (metric, title, ax, color) in enumerate(zip(metrics, titles, axes, colors)):
            # Sort data for this specific metric from highest to lowest (worst to best)
            metric_sorted_data = building_performance.sort_values(metric, ascending=False)
            
            # Create horizontal bar chart
            ax.barh(metric_sorted_data['Primary Property Type'],
                    metric_sorted_data[metric],
                    color=color,
                    alpha=0.7)
            
            # Add count information next to each bar
            for j, (value, count) in enumerate(zip(metric_sorted_data[metric], metric_sorted_data['Count'])):
                ax.text(value + (max(metric_sorted_data[metric]) * 0.02),
                        j,
                        f'n={count}',
                        va='center')
            
            # Add a vertical line for the average
            avg_value = metric_sorted_data[metric].mean()
            ax.axvline(x=avg_value, color='red', linestyle='--', alpha=0.7)
            ax.text(avg_value, -0.6, f'Avg: {avg_value:.1f}', color='red', ha='center')
            
            # Improve aesthetics
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(metric, fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Add a short explanation about the metric
            if i == 0:
                explanation = "Site EUI measures actual energy consumed onsite"
            elif i == 1:
                explanation = "Source EUI includes energy used in production & delivery"
            else:
                explanation = "GHG Intensity reflects carbon impact"
            
            ax.text(0.98, 0.95, explanation,
                    transform=ax.transAxes,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),
                    style='italic', fontsize=10)
        
        # Set a main title
        plt.suptitle('Energy Performance Metrics by Building Type', fontsize=16, y=0.98)
        
        # Add a simple explanatory note
        fig.text(0.5, 0.01,
                 'Building types are sorted independently for each metric (worst to best). Lower values indicate better efficiency.',
                 ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        
        # Save the figure
        plt.savefig('building_type_performance.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_implications(self):
        """
        Generate implications of the building type performance visualization.
        
        Returns:
        --------
        dict
            Dictionary containing implications for building owners and city authorities
        """
        implications = {
            'building_owners': """
            # Implications for Building Owners
            
            This visualization helps building owners understand how their property type typically performs and what energy metrics they should expect. For example, if you own an office building and your Site EUI is significantly higher than the average shown here, it suggests substantial room for improvement. The fact that different building types show distinct energy profiles confirms our hypothesis that building characteristics significantly predict energy performance.
            
            Building owners can use this information to:
            1. Benchmark their building against the average for their property type
            2. Identify reasonable efficiency targets based on peer performance
            3. Prioritize energy improvements that address the specific challenges of their building type
            4. Understand which energy metrics are most relevant for their property type
            """,
            
            'city_authorities': """
            # Implications for City Authorities
            
            For city planners, this visualization identifies which building categories should be prioritized in energy efficiency programs. Building types with both high average energy intensity and large building counts (like multifamily housing) represent the greatest opportunity for citywide emissions reduction. Targeted policies and incentives for these specific building types would yield the highest impact.
            
            City authorities can use this information to:
            1. Design targeted incentive programs for building types with the highest energy intensity
            2. Set building-type-specific benchmarks in energy efficiency policies
            3. Allocate resources to programs targeting the building types with the greatest potential impact
            4. Better understand the distribution of energy usage across the building stock
            """
        }
        
        return implications