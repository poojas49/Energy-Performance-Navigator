import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PerformanceOutlierVisualizer:
    """
    Class for creating visualizations that identify and analyze buildings that significantly 
    outperform or underperform their peers.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the PerformanceOutlierVisualizer class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to visualize
        """
        self.merged_df = merged_df.reset_index(drop=True)
    
    def visualize_performance_outliers(self):
        """
        Create a scatter plot of ENERGY STAR Score vs Site EUI, highlighting outliers.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the visualization
        """
        # Create a scatter plot of ENERGY STAR Score vs Site EUI, colored by performance category
        # First, filter to buildings with both metrics
        performance_df = self.merged_df.dropna(subset=['ENERGY STAR Score', 'Site EUI (kBtu/sq ft)', 'Performance Category'])
        
        # Create a bigger figure to allow more space for insights box
        plt.figure(figsize=(16, 10))
        
        # Get the top 5 property types by count
        top_property_types = performance_df['Primary Property Type'].value_counts().head(5).index.tolist()
        
        # Filter to these property types
        filtered_df = performance_df[performance_df['Primary Property Type'].isin(top_property_types)]
        
        # Create a new highly contrasting color palette for maximum distinction
        property_colors = {
            top_property_types[0]: '#0066cc',  # Deep blue
            top_property_types[1]: '#ff9900',  # Orange
            top_property_types[2]: '#00cc66',  # Bright green
            top_property_types[3]: '#cc0000',  # Bright red
            top_property_types[4]: '#9933cc'   # Deep purple
        }
        
        # Create a scatter plot with property types
        for prop_type, color in property_colors.items():
            subset = filtered_df[filtered_df['Primary Property Type'] == prop_type]
            plt.scatter(
                subset['ENERGY STAR Score'],
                subset['Site EUI (kBtu/sq ft)'],
                c=color,
                label=prop_type,
                alpha=0.8,  # Increased from 0.6 for better visibility
                s=60,
                edgecolors='none'
            )
        
        # Calculate overall medians for all filtered properties
        median_score = filtered_df['ENERGY STAR Score'].median()
        median_eui = filtered_df['Site EUI (kBtu/sq ft)'].median()
        
        # Add quadrant lines
        plt.axvline(x=median_score, color='black', linestyle='--', alpha=0.5, label=f'Median Score: {median_score:.0f}')
        plt.axhline(y=median_eui, color='black', linestyle='--', alpha=0.5, label=f'Median EUI: {median_eui:.0f}')
        
        # Find extreme outliers - top 3 best and worst performers across filtered property types
        # Best: High ENERGY STAR, Low EUI
        best_performers = filtered_df.nlargest(3, 'ENERGY STAR Score')
        # Worst: Low ENERGY STAR, High EUI
        worst_performers = filtered_df.nsmallest(3, 'ENERGY STAR Score')
        
        # Highlight these extreme outliers with thicker borders for better visibility
        plt.scatter(
            best_performers['ENERGY STAR Score'],
            best_performers['Site EUI (kBtu/sq ft)'],
            s=200,
            facecolors='none',
            edgecolors='#00aa00',  # Darker green for better contrast
            linewidths=4,         # Thicker line
            label='Best Performers',
            zorder=10
        )
        
        plt.scatter(
            worst_performers['ENERGY STAR Score'],
            worst_performers['Site EUI (kBtu/sq ft)'],
            s=200,
            facecolors='none',
            edgecolors='#cc0000',  # Bright red
            linewidths=4,          # Thicker line
            label='Worst Performers',
            zorder=10
        )
        
        # Add annotations with better positioning for best performers
        for i, (idx, row) in enumerate(best_performers.iterrows()):
            prop_name = str(row.get('Property Name', f'Building {idx}'))
            prop_type = str(row.get('Primary Property Type', 'Unknown'))
            
            # Calculate different offset positions for each point to avoid overlap
            if i == 0:
                x_offset = 15
                y_offset = -25
                connection_style = 'arc3,rad=0.2'
            elif i == 1:
                x_offset = 15
                y_offset = -55  # More vertical separation
                connection_style = 'arc3,rad=0.1'
            else:
                x_offset = 15
                y_offset = -85  # Even more vertical separation
                connection_style = 'arc3,rad=0.3'
            
            plt.annotate(
                f"{prop_name[:10]}... ({prop_type[:10]})",
                xy=(row['ENERGY STAR Score'], row['Site EUI (kBtu/sq ft)']),
                xytext=(x_offset, y_offset),  # Different offset for each annotation
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle=connection_style,
                    color='#00aa00',
                    lw=2
                ),
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    fc='white',
                    ec='#00aa00',
                    alpha=0.9
                )
            )
        
        # Add annotations with better positioning for worst performers
        for i, (idx, row) in enumerate(worst_performers.iterrows()):
            prop_name = str(row.get('Property Name', f'Building {idx}'))
            prop_type = str(row.get('Primary Property Type', 'Unknown'))
            
            # Calculate different offset positions for each point
            if i == 0:
                x_offset = -20  # Position more to the left
                y_offset = 20
                connection_style = 'arc3,rad=-0.2'  # Curve the arrow differently
            elif i == 1:
                x_offset = -20  # Even more to the left
                y_offset = 30
                connection_style = 'arc3,rad=-0.3'
            else:
                x_offset = -100
                y_offset = 80  # Significant vertical separation
                connection_style = 'arc3,rad=-0.2'
            
            plt.annotate(
                f"{prop_name[:10]}... ({prop_type[:10]})",
                xy=(row['ENERGY STAR Score'], row['Site EUI (kBtu/sq ft)']),
                xytext=(x_offset, y_offset),  # Different offset for each annotation
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle=connection_style,  # Different curve for each arrow
                    color='#cc0000',
                    lw=2
                ),
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    fc='white',
                    ec='#cc0000',
                    alpha=0.9
                )
            )
        
        # Add quadrant labels
        x_max = filtered_df['ENERGY STAR Score'].max()
        x_min = filtered_df['ENERGY STAR Score'].min()
        y_max = filtered_df['Site EUI (kBtu/sq ft)'].max()
        y_min = filtered_df['Site EUI (kBtu/sq ft)'].min()
        
        # Position the HIGH PERFORMERS label with more space to avoid overlap with annotations
        plt.text(x_max * 0.75, y_min * 1.15,
                'HIGH PERFORMERS\n(High Score, Low EUI)',
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
        
        # Position the LOW PERFORMERS label to avoid overlap with annotations
        plt.text(x_min * 1.3, y_max * 0.85,
                'LOW PERFORMERS\n(Low Score, High EUI)',
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
        
        # Add title and labels
        plt.title('Building Performance Outliers by Property Type', fontsize=18, fontweight='bold')
        plt.xlabel('ENERGY STAR Score (Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Site EUI (kBtu/sq ft) (Lower is Better)', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve legend with larger marker size and a border
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, framealpha=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.5)
        
        # Create insights text
        insights_text = f'''Key Insights:
        • Most effective buildings use {best_performers["Site EUI (kBtu/sq ft)"].mean():.1f} kBtu/sq ft (avg)
        • Least effective buildings use {worst_performers["Site EUI (kBtu/sq ft)"].mean():.1f} kBtu/sq ft (avg)
        • Performance ratio: {worst_performers["Site EUI (kBtu/sq ft)"].mean() / best_performers["Site EUI (kBtu/sq ft)"].mean():.1f}x difference
        
        Best performers typically have:
        • Higher ENERGY STAR Scores (avg: {best_performers["ENERGY STAR Score"].mean():.1f})
        • Lower Site EUI values
        • More recent renovations or efficiency upgrades'''
        
        # BETTER POSITIONING: Create a separate axes for the insights box
        # This prevents overlapping with the main visualization
        ins_ax = plt.axes([0.05, 0.03, 0.3, 0.2], frameon=True)  # [left, bottom, width, height]
        ins_ax.text(0.05, 0.95, insights_text, fontsize=12,
                   verticalalignment='top', horizontalalignment='left',
                   transform=ins_ax.transAxes)
        ins_ax.axis('off')  # Hide the axes
        
        # Add a thin border around the insights box for better definition
        ins_ax.patch.set_edgecolor('black')
        ins_ax.patch.set_linewidth(0.5)
        ins_ax.patch.set_facecolor('white')
        ins_ax.patch.set_alpha(0.9)
        
        # Save the figure
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.25)
        plt.savefig('performance_outliers.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_implications(self):
        """
        Generate implications of the performance outlier visualization.
        
        Returns:
        --------
        dict
            Dictionary containing implications for building owners and city authorities
        """
        implications = {
            'building_owners': """
            # Implications for Building Owners
            
            This outlier analysis provides building owners with a clear picture of how their property performs relative to peers with similar characteristics. It helps owners set realistic targets based on what has actually been achieved by comparable buildings. For owners of underperforming buildings (shown in orange and red), it demonstrates that significant improvements are possible, since similar buildings are performing much better. For those with high-performing buildings, it validates their efficiency efforts and may identify opportunities for recognition or certification.
            
            Building owners can use this information to:
            1. Benchmark their building against top performers of the same type
            2. Set realistic improvement targets based on proven peer performance
            3. Identify performance gaps that represent opportunity for improvement
            4. Study best practices from similar buildings that outperform theirs
            """,
            
            'city_authorities': """
            # Implications for City Authorities
            
            City planners can use this outlier analysis to identify success stories and problem cases across the building stock. The exceptional performers (green) can serve as case studies for effective efficiency strategies, while the worst performers (red) might need targeted intervention or enforcement. The wide performance spread among similar buildings indicates that there's significant untapped potential for energy savings without requiring major technological breakthroughs – simply bringing underperforming buildings up to the standards of their peers would yield substantial energy and emissions reductions.
            
            City authorities can use this information to:
            1. Identify high-impact targets for efficiency improvement programs
            2. Document and publicize case studies of exceptional performers
            3. Develop realistic performance targets for building regulations
            4. Quantify the citywide potential for energy savings by bringing underperformers to median levels
            5. Design recognition programs to highlight and reward top-performing buildings
            """
        }
        
        return implications