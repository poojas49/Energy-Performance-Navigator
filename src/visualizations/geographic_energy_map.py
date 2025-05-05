import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import folium
from folium.plugins import MarkerCluster
import os

class GeographicEnergyMapVisualizer:
    """
    Class for creating geographic visualizations of energy efficiency in Chicago.
    """
    
    def __init__(self, merged_df):
        """
        Initialize the GeographicEnergyMapVisualizer class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame
            The merged dataset to visualize
        """
        self.merged_df = merged_df.reset_index(drop=True)
    
    def create_geographic_map(self, save_html=True):
        """
        Create an interactive folium map showing energy efficiency patterns across Chicago.
        
        Parameters:
        -----------
        save_html : bool, optional
            Whether to save the map as an HTML file
            
        Returns:
        --------
        folium.Map
            Interactive map object
        """
        import folium
        from folium.plugins import MarkerCluster
        
        # Prepare data for mapping
        # Calculate average metrics by community area
        if 'Community Area' in self.merged_df.columns:
            community_col = 'Community Area'
        elif 'Community Area Name' in self.merged_df.columns:
            community_col = 'Community Area Name'
        else:
            print("No Community Area column found in dataset")
            return None
        
        community_metrics = self.merged_df.groupby(community_col).agg({
            'Site EUI (kBtu/sq ft)': 'mean',
            'GHG Intensity (kg CO2e/sq ft)': 'mean',
            'ENERGY STAR Score': 'mean',
            'Latitude': 'median',
            'Longitude': 'median',
            'Building ID': 'count'
        }).rename(columns={'Building ID': 'Count'}).reset_index()
        
        # Remove rows with missing coordinates
        community_metrics = community_metrics.dropna(subset=['Latitude', 'Longitude'])
        
        # Create a map centered on Chicago
        chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
        
        # Define a function to determine circle color based on Site EUI
        def get_color(eui):
            if pd.isna(eui):
                return 'grey'
            elif eui < 50:
                return 'green'
            elif eui < 75:
                return 'lightgreen'
            elif eui < 100:
                return 'orange'
            else:
                return 'red'
        
        # Create a circle marker for each community area
        for idx, row in community_metrics.iterrows():
            if pd.isna(row['Site EUI (kBtu/sq ft)']) or pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
                continue
            
            # Format popup content
            popup_content = f"""
            <b>Community Area:</b> {row[community_col]}<br>
            <b>Avg Site EUI:</b> {row['Site EUI (kBtu/sq ft)']:.1f} kBtu/sq ft<br>
            <b>Avg GHG Intensity:</b> {row['GHG Intensity (kg CO2e/sq ft)']:.1f} kg CO2e/sq ft<br>
            <b>Avg ENERGY STAR Score:</b> {row['ENERGY STAR Score']:.1f}<br>
            <b>Number of Buildings:</b> {row['Count']}
            """
            
            # Determine circle size based on building count (with minimum size for visibility)
            radius = max(5, min(15, row['Count'] / 10))
            
            # Add circle marker
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                color=get_color(row['Site EUI (kBtu/sq ft)']),
                fill=True,
                fill_color=get_color(row['Site EUI (kBtu/sq ft)']),
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(chicago_map)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 180px; height: 120px;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px;
                    border-radius: 5px;">
            <div style="margin-bottom: 5px;"><b>Site EUI (kBtu/sq ft)</b></div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: green; margin-right: 10px;"></div>
                < 50 (Excellent)
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: lightgreen; margin-right: 10px;"></div>
                50 - 75 (Good)
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: orange; margin-right: 10px;"></div>
                75 - 100 (Fair)
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: red; margin-right: 10px;"></div>
                > 100 (Poor)
            </div>
        </div>
        '''
        chicago_map.get_root().html.add_child(folium.Element(legend_html))
        
        # Save the map to an HTML file
        if save_html:
            chicago_map.save('chicago_energy_map.html')
        
        # Create a static alternative for the report using matplotlib
        self.create_static_map(community_metrics, community_col)
        
        return chicago_map
    
    def create_static_map(self, community_metrics, community_col):
        """
        Create a static matplotlib version of the geographic map for reports.
        
        Parameters:
        -----------
        community_metrics : pandas.DataFrame
            DataFrame containing community area metrics
        community_col : str
            Column name for community area
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the static map
        """
        plt.figure(figsize=(14, 10))
        
        # Create a scatter plot with better readability
        scatter = plt.scatter(
            community_metrics['Longitude'],
            community_metrics['Latitude'],
            c=community_metrics['Site EUI (kBtu/sq ft)'],
            cmap='RdYlGn_r',
            s=community_metrics['Count'] * 3,  # Slightly larger circles
            alpha=1,  # Add transparency for overlap
            edgecolors='black',
            linewidths=0.5
        )
        
        # Zoom in to reduce empty space and enhance focus
        plt.xlim(-87.94, -87.53)
        plt.ylim(41.65, 42.02)
        
        offsets = [(80, 40), (-90, -30), (100, -50), (-70, 60), (90, 70)]
        
        for i, (idx, row) in enumerate(community_metrics.sort_values('Count', ascending=False).head(5).iterrows()):
            dx, dy = offsets[i % len(offsets)]
            plt.annotate(
                row[community_col],
                xy=(row['Longitude'], row['Latitude']),           # Anchor point
                xytext=(dx, dy),                                  # Distant offset
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=1),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', alpha=0.9)
            )
        
        # Add a colorbar and labels
        cbar = plt.colorbar(scatter)
        cbar.set_label('Site EUI (kBtu/sq ft) - Lower is Better', fontsize=12)
        
        plt.title('Geographic Distribution of Building Energy Efficiency in Chicago', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Custom legend (same as folium)
        legend_patches = [
            mpatches.Patch(color='green', label='Excellent (< 50 kBtu/sq ft)'),
            mpatches.Patch(color='lightgreen', label='Good (50–75 kBtu/sq ft)'),
            mpatches.Patch(color='orange', label='Fair (75–100 kBtu/sq ft)'),
            mpatches.Patch(color='red', label='Poor (> 100 kBtu/sq ft)')
        ]
        plt.legend(handles=legend_patches, title='Site EUI Categories', loc='lower left', frameon=True)
        
        plt.tight_layout()
        plt.savefig('chicago_energy_map_static.png', dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_implications(self):
        """
        Generate implications of the geographic energy map visualization.
        
        Returns:
        --------
        dict
            Dictionary containing implications for building owners and city authorities
        """
        implications = {
            'building_owners': """
            # Implications for Building Owners
            
            This geographic visualization helps building owners understand how their location might influence energy performance. Owners can see if their building is in a high-performing or low-performing area, which might indicate neighborhood-specific factors affecting efficiency (such as building age clusters, local infrastructure, or microclimates). This information can help owners contextualize their building's performance and identify location-specific strategies for improvement.
            
            Building owners can use this information to:
            1. Understand neighborhood-specific energy performance patterns
            2. Identify if local factors may be influencing their building's performance
            3. Connect with other building owners in their area to share best practices
            4. Consider location-specific strategies for energy improvements
            """,
            
            'city_authorities': """
            # Implications for City Authorities
            
            For city planners, this map reveals neighborhood-level patterns that can guide targeted interventions. Areas with consistently poor energy performance (shown in red) could benefit from focused outreach programs, community-based efficiency initiatives, or neighborhood-specific incentives. The map also helps identify if certain areas have been left behind in energy improvements, enabling more equitable distribution of resources across the city. Additionally, this spatial analysis can inform infrastructure planning, such as district heating/cooling systems for neighborhoods with clusters of inefficient buildings.
            
            City authorities can use this information to:
            1. Target energy efficiency programs to neighborhoods with the greatest need
            2. Design community-based outreach and education initiatives
            3. Identify potential locations for district energy systems
            4. Develop neighborhood-specific energy policies and incentives
            5. Ensure equitable distribution of energy efficiency resources across the city
            """
        }
        
        return implications