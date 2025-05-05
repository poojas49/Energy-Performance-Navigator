import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EnergySourceVisualizer:
    """
    Class for creating visualizations of energy source distribution by building type.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the EnergySourceVisualizer class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to visualize
        """
        self.merged_df = merged_df.reset_index(drop=True)
    
    def visualize_energy_source_distribution(self, top_n=15):
        """
        Create visualization showing energy source distribution across top building types.
        
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
        
        # Prepare data for visualization
        energy_source_df = self.merged_df[self.merged_df['Primary Property Type'].isin(top_property_types)].copy()
        
        # Calculate average percentages for each building type
        energy_source_summary = energy_source_df.groupby('Primary Property Type').agg({
            'Electricity Percentage': 'mean',
            'Natural Gas Percentage': 'mean',
            'District Steam Percentage': 'mean',
            'District Chilled Water Percentage': 'mean',
            'All Other Fuel Percentage': 'mean',
            'Building ID': 'count'
        }).rename(columns={'Building ID': 'Count'}).reset_index()
        
        # Sort by count
        energy_source_summary = energy_source_summary.sort_values('Count', ascending=False)
        
        # Remove categories with all-zero energy percentages
        energy_source_summary = energy_source_summary[
            energy_source_summary[['Electricity Percentage', 'Natural Gas Percentage',
                                  'District Steam Percentage', 'District Chilled Water Percentage',
                                  'All Other Fuel Percentage']].sum(axis=1) > 0
        ]
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 8))
        
        # Define energy sources with vibrant colors
        energy_sources = [
            ('Electricity Percentage', '#00b4d8', 'Electricity'),
            ('Natural Gas Percentage', '#ff9500', 'Natural Gas'),
            ('District Steam Percentage', '#2ecc71', 'District Steam'),
            ('District Chilled Water Percentage', '#e74c3c', 'District Chilled Water'),
            ('All Other Fuel Percentage', '#9b59b6', 'Other Fuel')
        ]
        
        # Initialize bottom values
        bottom_values = np.zeros(len(energy_source_summary))
        x = np.arange(len(energy_source_summary))
        
        # Plot stacked bars
        for source, color, label in energy_sources:
            values = energy_source_summary[source].fillna(0)
            plt.bar(
                x,
                values,
                bottom=bottom_values,
                label=label,
                color=color,
                width=0.65,
                edgecolor='black',
                linewidth=0.8,
                alpha=0.95
            )
            # Add percentage labels if > 5%
            for i, (val, bottom) in enumerate(zip(values, bottom_values)):
                if val > 5:
                    plt.text(
                        x[i], bottom + val / 2, f'{val:.1f}%',
                        ha='center', va='center', color='white', fontsize=11, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1)
                    )
            bottom_values += values
        
        # Customize axes and labels
        plt.xlabel('Building Type', fontsize=13, weight='bold')
        plt.ylabel('Energy Mix (%)', fontsize=13, weight='bold')
        plt.title('Energy Source Distribution by Building Type', fontsize=18, weight='bold', pad=15,
                  color='#333333')
        plt.xticks(x, energy_source_summary['Primary Property Type'], rotation=25, ha='right', fontsize=11)
        plt.yticks(np.arange(0, 101, 10), fontsize=11)
        
        # Add count annotations
        for i, count in enumerate(energy_source_summary['Count']):
            total_height = sum(energy_source_summary[source].iloc[i] for source, _, _ in energy_sources)
            plt.text(
                x[i], total_height + 2, f'n={count}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
            )
        
        # Enhance legend
        plt.legend(
            title='Energy Source', title_fontsize=13, fontsize=11,
            bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, edgecolor='black',
            facecolor='white', framealpha=1
        )
        
        # Add grid and adjust layout
        plt.grid(axis='y', linestyle='--', alpha=0.2, color='gray')
        plt.ylim(0, 110)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('energy_source_distribution.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_implications(self):
        """
        Generate implications of the energy source distribution visualization.
        
        Returns:
        --------
        dict
            Dictionary containing implications for building owners and city authorities
        """
        implications = {
            'building_owners': """
            # Implications for Building Owners
            
            The energy source distribution reveals important patterns that building owners can use to optimize their energy strategy. Buildings with higher electricity usage often have different efficiency challenges than those relying primarily on natural gas. Owners can compare their building's energy mix to similar properties and identify opportunities to shift toward a more optimal distribution. For instance, if your office building uses significantly more natural gas than the average, exploring electrification options might be beneficial.
            
            Building owners can use this information to:
            1. Compare their energy mix to peers of the same building type
            2. Identify potential imbalances in their energy sources
            3. Target efficiency improvements toward their dominant energy source
            4. Evaluate opportunities for fuel switching or electrification
            """,
            
            'city_authorities': """
            # Implications for City Authorities
            
            This visualization helps authorities understand the energy infrastructure needs across different building sectors. It can inform policies related to grid modernization, electrification incentives, or natural gas efficiency programs based on the predominant energy sources used by different building types. For example, targeted electrification programs for building types that currently rely heavily on natural gas could yield significant emissions reductions.
            
            City authorities can use this information to:
            1. Design sector-specific energy transition programs
            2. Prioritize grid infrastructure upgrades in areas with high electricity demand
            3. Target electrification incentives to building types that currently rely heavily on fossil fuels
            4. Better understand the implications of energy policies across different building sectors
            """
        }
        
        return implications