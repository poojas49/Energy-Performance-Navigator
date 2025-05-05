import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os

class BuildingClusteringModel:
    """
    Class for clustering buildings based on energy performance characteristics.
    """
    
    def __init__(self, merged_df=None):
        """
        Initialize the BuildingClusteringModel class.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame, optional
            The merged dataset to use for clustering
        """
        self.merged_df = merged_df
        self.cluster_data = None
        self.cluster_centers_df = None
        self.cluster_analysis = None
        self.optimal_k = None
        self.kmeans = None
        self.scaler = None
    
    def prepare_data(self, merged_df=None):
        """
        Prepare data for clustering analysis.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame, optional
            The merged dataset to use for clustering.
            If None, uses the dataset provided at initialization.
            
        Returns:
        --------
        pandas.DataFrame
            Data prepared for clustering
        """
        if merged_df is not None:
            self.merged_df = merged_df
        
        # Select features for clustering
        cluster_features = [
            'Site EUI (kBtu/sq ft)',
            'GHG Intensity (kg CO2e/sq ft)',
            'Electricity Percentage',
            'Natural Gas Percentage',
            'Building Age'
        ]
        
        # Filter to features that exist in the dataframe
        cluster_features = [f for f in cluster_features if f in self.merged_df.columns]
        
        # Ensure all features have values
        self.cluster_data = self.merged_df.dropna(subset=cluster_features).copy()
        
        print(f"Number of buildings for clustering: {len(self.cluster_data)}")
        
        return self.cluster_data
    
    def find_optimal_clusters(self, max_k=10):
        """
        Find the optimal number of clusters using silhouette score.
        
        Parameters:
        -----------
        max_k : int, optional
            Maximum number of clusters to consider
            
        Returns:
        --------
        int
            Optimal number of clusters
        """
        # Select features for clustering
        cluster_features = [
            'Site EUI (kBtu/sq ft)',
            'GHG Intensity (kg CO2e/sq ft)',
            'Electricity Percentage',
            'Natural Gas Percentage',
            'Building Age'
        ]
        
        # Filter to features that exist in the dataframe
        cluster_features = [f for f in cluster_features if f in self.cluster_data.columns]
        
        # Scale the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.cluster_data[cluster_features])
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        k_values = range(2, max_k+1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {k}, silhouette score is {silhouette_avg:.3f}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, 'o-')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Score for Different Numbers of Clusters', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(k_values)
        plt.tight_layout()
        plt.savefig('silhouette_scores.png', dpi=300)
        plt.close()
        
        # Select optimal number of clusters based on silhouette scores
        self.optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        print(f"Optimal number of clusters: {self.optimal_k}")
        
        return self.optimal_k
    
    def perform_clustering(self, k=None):
        """
        Perform K-means clustering on the prepared data.
        
        Parameters:
        -----------
        k : int, optional
            Number of clusters. If None, uses the optimal k from find_optimal_clusters.
            
        Returns:
        --------
        pandas.DataFrame
            Cluster analysis results
        """
        if k is None:
            if self.optimal_k is None:
                self.find_optimal_clusters()
            k = self.optimal_k
        
        # Select features for clustering
        cluster_features = [
            'Site EUI (kBtu/sq ft)',
            'GHG Intensity (kg CO2e/sq ft)',
            'Electricity Percentage',
            'Natural Gas Percentage',
            'Building Age'
        ]
        
        # Filter to features that exist in the dataframe
        cluster_features = [f for f in cluster_features if f in self.cluster_data.columns]
        
        # Scale the data if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.cluster_data[cluster_features])
        else:
            scaled_data = self.scaler.transform(self.cluster_data[cluster_features])
        
        # Apply K-means clustering
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.cluster_data['Cluster'] = self.kmeans.fit_predict(scaled_data)
        
        # Get cluster centers
        cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.cluster_centers_df = pd.DataFrame(
            cluster_centers,
            columns=cluster_features
        )
        
        # Analyze clusters
        self.cluster_analysis = self.cluster_data.groupby('Cluster').agg({
            'Site EUI (kBtu/sq ft)': 'mean',
            'Source EUI (kBtu/sq ft)': 'mean',
            'GHG Intensity (kg CO2e/sq ft)': 'mean',
            'ENERGY STAR Score': 'mean',
            'Electricity Percentage': 'mean',
            'Natural Gas Percentage': 'mean',
            'Building Age': 'mean',
            'Primary Property Type': lambda x: x.value_counts().index[0],  # Most common type
            'Chicago Energy Rating': 'mean',
            'Building ID': 'count'
        }).rename(columns={'Building ID': 'Count'})
        
        print("Cluster Analysis:")
        print(self.cluster_analysis)
        
        return self.cluster_analysis
    
    def visualize_clusters_2d(self):
        """
        Visualize clusters in 2D using the two most important features.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the 2D cluster visualization
        """
        if self.cluster_data is None or 'Cluster' not in self.cluster_data.columns:
            print("No cluster data available. Run perform_clustering() first.")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with Site EUI vs GHG Intensity
        scatter = plt.scatter(
            self.cluster_data['Site EUI (kBtu/sq ft)'],
            self.cluster_data['GHG Intensity (kg CO2e/sq ft)'],
            c=self.cluster_data['Cluster'],
            cmap='viridis',
            alpha=0.7,
            s=70,
            edgecolors='w',
            linewidths=0.5
        )
        
        # Plot cluster centers
        plt.scatter(
            self.cluster_centers_df['Site EUI (kBtu/sq ft)'],
            self.cluster_centers_df['GHG Intensity (kg CO2e/sq ft)'],
            marker='X',
            s=200,
            c='red',
            edgecolors='k',
            linewidths=1.5,
            label='Cluster Centers'
        )
        
        # Add labels and legend
        plt.xlabel('Site EUI (kBtu/sq ft)', fontsize=12)
        plt.ylabel('GHG Intensity (kg CO2e/sq ft)', fontsize=12)
        plt.title('Building Energy Efficiency Clusters', fontsize=16)
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate cluster centers with cluster numbers
        for i, (x, y) in enumerate(zip(self.cluster_centers_df['Site EUI (kBtu/sq ft)'],
                                      self.cluster_centers_df['GHG Intensity (kg CO2e/sq ft)'])):
            plt.annotate(
                f"Cluster {i}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig('cluster_visualization_2d.png', dpi=300)
        
        return plt.gcf()
    
    def visualize_clusters_parallel(self):
        """
        Create a parallel coordinates plot to visualize clusters across all dimensions.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the parallel coordinates visualization
        """
        if self.cluster_data is None or 'Cluster' not in self.cluster_data.columns:
            print("No cluster data available. Run perform_clustering() first.")
            return None
        
        # Select features for clustering
        cluster_features = [
            'Site EUI (kBtu/sq ft)',
            'GHG Intensity (kg CO2e/sq ft)',
            'Electricity Percentage',
            'Natural Gas Percentage',
            'Building Age'
        ]
        
        # Filter to features that exist in the dataframe
        cluster_features = [f for f in cluster_features if f in self.cluster_data.columns]
        
        # Prepare data for parallel coordinates
        parallel_coords_data = self.cluster_data[cluster_features + ['Cluster']].copy()
        
        # Scale all features to 0-1 range for better visualization
        for feature in cluster_features:
            min_val = parallel_coords_data[feature].min()
            max_val = parallel_coords_data[feature].max()
            range_val = max_val - min_val
            parallel_coords_data[feature] = (parallel_coords_data[feature] - min_val) / range_val
        
        # Create the parallel coordinates plot
        plt.figure(figsize=(14, 8))
        
        # Get a colormap with the same number of colors as clusters
        cmap = plt.cm.get_cmap('viridis', self.optimal_k)
        
        # Plot each cluster
        for i in range(self.optimal_k):
            # Get data for this cluster
            cluster_i_data = parallel_coords_data[parallel_coords_data['Cluster'] == i]
            
            # Plot each building in this cluster with partial transparency
            for _, row in cluster_i_data.sample(min(50, len(cluster_i_data))).iterrows():
                plt.plot(cluster_features, row[cluster_features], color=cmap(i), alpha=0.1)
            
            # Calculate and plot the cluster mean
            cluster_mean = cluster_i_data[cluster_features].mean()
            plt.plot(cluster_features, cluster_mean, color=cmap(i), linewidth=3, label=f'Cluster {i}')
        
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Normalized Value', fontsize=12)
        plt.title('Parallel Coordinates Plot of Building Clusters', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('cluster_visualization_parallel.png', dpi=300)
        
        return plt.gcf()
    
    def run_full_clustering_pipeline(self, merged_df=None):
        """
        Run the complete clustering pipeline from data preparation to visualization.
        
        Parameters:
        -----------
        merged_df : pandas.DataFrame, optional
            The merged dataset to use for clustering.
            If None, uses the dataset provided at initialization.
            
        Returns:
        --------
        tuple
            Tuple of (cluster_data, cluster_centers_df, cluster_analysis)
        """
        print("Starting building energy efficiency clustering...")
        
        # Prepare data
        self.prepare_data(merged_df)
        
        # Find optimal number of clusters
        self.find_optimal_clusters()
        
        # Perform clustering
        self.perform_clustering()
        
        # Visualize clusters in 2D
        self.visualize_clusters_2d()
        
        # Visualize clusters in parallel coordinates
        self.visualize_clusters_parallel()
        
        return self.cluster_data, self.cluster_centers_df, self.cluster_analysis
    
    def generate_cluster_interpretation(self):
        """
        Generate interpretation of the clustering results.
        
        Returns:
        --------
        str
            Detailed interpretation of the clustering results
        """
        if self.cluster_analysis is None:
            print("No cluster analysis available. Run perform_clustering() first.")
            return None
        
        interpretation = f"""
        # Cluster Interpretation and Implications

        The K-means clustering algorithm has identified {self.optimal_k} distinct groups of buildings with similar energy performance characteristics. Based on the silhouette score analysis, the optimal number of clusters is {self.optimal_k}, suggesting {self.optimal_k} natural groupings within Chicago's building stock.

        ## Cluster Characteristics:
        """
        
        # Add characteristics for each cluster
        for i in range(self.optimal_k):
            cluster_data = self.cluster_analysis.loc[i]
            
            # Determine key characteristics
            high_eui = cluster_data['Site EUI (kBtu/sq ft)'] > self.cluster_analysis['Site EUI (kBtu/sq ft)'].mean()
            high_ghg = cluster_data['GHG Intensity (kg CO2e/sq ft)'] > self.cluster_analysis['GHG Intensity (kg CO2e/sq ft)'].mean()
            high_energy_star = cluster_data['ENERGY STAR Score'] > self.cluster_analysis['ENERGY STAR Score'].mean()
            high_electricity = cluster_data['Electricity Percentage'] > self.cluster_analysis['Electricity Percentage'].mean()
            high_gas = cluster_data['Natural Gas Percentage'] > self.cluster_analysis['Natural Gas Percentage'].mean()
            older_building = cluster_data['Building Age'] > self.cluster_analysis['Building Age'].mean()
            
            # Create cluster name
            if high_energy_star and not high_eui:
                cluster_name = "High Efficiency"
                if high_electricity:
                    cluster_name += " Electric"
                elif high_gas:
                    cluster_name += " Gas-Heated"
                
                if not older_building:
                    cluster_name += " Modern Buildings"
                else:
                    cluster_name += " Older Buildings"
            elif high_eui and high_electricity and not high_gas:
                cluster_name = "Electricity-Dominant Inefficient Buildings"
            elif high_eui and high_gas and not high_electricity:
                cluster_name = "Natural Gas-Dependent"
                if older_building:
                    cluster_name += " Older Buildings"
                else:
                    cluster_name += " Buildings"
            else:
                cluster_name = "Mixed-Source"
                if high_energy_star:
                    cluster_name += " Moderate Performers"
                else:
                    cluster_name += " Low Performers"
            
            # Add cluster description
            interpretation += f"""
            {i+1}. **Cluster {i} - {cluster_name}**: These buildings have {
                'low' if not high_eui else 'high'} Site EUI, {
                'low' if not high_ghg else 'high'} GHG intensity, and {
                'high' if high_energy_star else 'low'} ENERGY STAR scores. They tend to be {
                'older' if older_building else 'newer'} and have {
                'higher electricity usage' if high_electricity else 'higher natural gas usage' if high_gas else 'a balanced energy mix'
            }. The most common building type is {cluster_data['Primary Property Type']}. This cluster contains {int(cluster_data['Count'])} buildings.
            """
        
        # Add practical applications
        interpretation += """
        ## Practical Applications:

        - For Building Owners: The clustering provides a more nuanced benchmarking approach than simple building type or size categories. Owners can identify which cluster their building belongs to and understand its performance relative to that specific peer group. The characteristics of higher-performing clusters in the same building category can provide a roadmap for improvement.

        - For City Authorities: The clustering reveals distinct building archetypes that may require different policy approaches. For example, buildings in natural gas-dependent clusters might benefit most from heating system upgrade incentives, while those in electricity-dominant but inefficient clusters might need targeted programs for electrical efficiency or demand management.

        This unsupervised approach has validated our hypothesis that buildings naturally group into distinct performance categories that cut across traditional classifications, offering new ways to target efficiency improvements.
        """
        
        return interpretation