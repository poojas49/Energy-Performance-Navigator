import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BuildingAgePerformanceVisualizer:
    """
    Class for creating visualizations of energy performance by building age.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the BuildingAgePerformanceVisualizer class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to visualize
        """
        self.merged_df = merged_df.reset_index(drop=True)
    
    def visualize_performance_by_age(self):
        """
        Create visualization showing energy performance metrics across building age categories.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the visualization
        """
        # Ensure we have age categories
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
        
        # Set color scheme
        site_eui_color = '#0066cc'  # Blue
        ghg_color = '#00aa55'       # Green
        energy_star_color = '#ff6600'  # Orange
        
        # Create a figure with more space at the bottom for legend and annotations
        plt.figure(figsize=(12, 9))  # Increased height to make room for annotations
        
        # Create positions for the bars
        x = np.arange(len(age_performance['Age Category']))
        width = 0.25  # Width of the bars
        
        # Create bars for each metric
        plt.bar(x - width, age_performance['Site EUI (kBtu/sq ft)'], width, label='Site EUI (kBtu/sq ft)', color=site_eui_color)
        plt.bar(x, age_performance['GHG Intensity (kg CO2e/sq ft)'], width, label='GHG Intensity (kg CO2e/sq ft)', color=ghg_color)
        plt.bar(x + width, age_performance['ENERGY STAR Score'], width, label='ENERGY STAR Score', color=energy_star_color)
        
        # Add labels and title
        plt.xlabel('Building Age', fontsize=14, fontweight='bold')
        plt.ylabel('Metric Value', fontsize=14, fontweight='bold')
        plt.title('Energy Performance Metrics by Building Age', fontsize=16, fontweight='bold', pad=20)
        
        # Add custom x-tick labels with counts
        plt.xticks(x, [f"{age}\nn={count}" for age, count in zip(age_performance['Age Category'], age_performance['Count'])])
        
        # Add legend in a better position - move to upper right to avoid overlap
        plt.legend(loc='upper right', fontsize=12)
        
        # Add annotation explaining metrics - moved to the top left instead of bottom
        # Also using a text box to make it more readable
        plt.annotate(
            "Site EUI & GHG Intensity: Lower values indicate better performance\nENERGY STAR Score: Higher values indicate better performance",
            xy=(0.01, 0.97),
            xycoords='figure fraction',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=0.5')
        )
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('building_age_performance.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_implications(self):
        """
        Generate implications of the building age performance visualization.
        
        Returns:
        --------
        dict
            Dictionary containing implications for building owners and city authorities
        """
        implications = {
            'building_owners': """
            # Implications for Building Owners
            
            This visualization shows how a building's age typically affects its energy performance. Owners of older buildings can use this information to contextualize their building's performance and set realistic improvement targets. Interestingly, the data shows that very old buildings (100+ years) sometimes perform better than those from certain mid-century periods, possibly due to retrofits or architectural features. This suggests that even owners of very old buildings can achieve significant efficiency improvements with the right approaches.
            
            Building owners can use this information to:
            1. Understand how their building's age may impact its energy performance
            2. Set appropriate benchmarks based on age-typical performance
            3. Identify age-specific improvement strategies
            4. Recognize that older buildings can still achieve good performance with appropriate upgrades
            """,
            
            'city_authorities': """
            # Implications for City Authorities
            
            For policymakers, this visualization helps identify which building age groups should be targeted for retrofitting programs. Buildings from eras with consistently poor performance represent prime opportunities for efficiency improvements. This could inform age-specific building codes, retrofit requirements, or incentive programs tailored to the unique challenges of buildings from different time periods.
            
            City authorities can use this information to:
            1. Design age-targeted retrofit programs
            2. Set appropriate performance expectations for buildings of different ages
            3. Identify historical construction eras that may require special attention
            4. Create tailored incentives for improving specific age cohorts
            """
        }
        
        return implications