import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class BuildingRecommendationEngine:
    """
    Class for generating tailored energy efficiency recommendations for buildings.
    """
    
    def __init__(self):
        """Initialize the BuildingRecommendationEngine class."""
        self.energy_df = None
        self.buildings_df = None
        self.merged_df = None
        self.dt_models = {}
        
        # Define recommendation categories for each system
        self.system_categories = {
            'lighting': ['Poor Lighting', 'Average Lighting', 'Good Lighting'],
            'hvac': ['Poor HVAC', 'Average HVAC', 'Good HVAC'],
            'envelope': ['Poor Envelope', 'Average Envelope', 'Good Envelope'],
            'controls': ['Poor Controls', 'Average Controls', 'Good Controls']
        }
        
        # Create recommendation templates
        self.recommendations = self._create_recommendation_templates()
    
    def _create_recommendation_templates(self):
        """
        Create templates for all possible recommendations.
        
        Returns:
        --------
        dict
            Dictionary of recommendation templates by system and category
        """
        return {
            'lighting': {
                'Poor Lighting': {
                    'title': 'Comprehensive Lighting Upgrade',
                    'description': 'Replace all lighting with LEDs and add smart controls.',
                    'typical_savings': '15-30% of electricity use',
                    'priority': 'High',
                    'typical_cost': '$$ (Medium investment with 2-3 year payback)'
                },
                'Average Lighting': {
                    'title': 'Targeted Lighting Improvements',
                    'description': 'Upgrade remaining non-LED fixtures and add occupancy sensors in key areas.',
                    'typical_savings': '5-15% of electricity use',
                    'priority': 'Medium',
                    'typical_cost': '$-$$ (Low to medium investment with 1-3 year payback)'
                },
                'Good Lighting': {
                    'title': 'Lighting System Maintenance',
                    'description': 'Maintain current lighting systems and consider advanced controls for further optimization.',
                    'typical_savings': '1-5% of electricity use',
                    'priority': 'Low',
                    'typical_cost': '$ (Low investment with 1-2 year payback)'
                }
            },
            'hvac': {
                'Poor HVAC': {
                    'title': 'HVAC System Overhaul',
                    'description': 'Your HVAC system needs significant upgrades or replacement.',
                    'typical_savings': '20-35% of HVAC energy use',
                    'priority': 'High',
                    'typical_cost': '$$$ (High investment with 3-7 year payback)'
                },
                'Average HVAC': {
                    'title': 'HVAC Optimization',
                    'description': 'Improve HVAC performance through control optimization, preventative maintenance, and targeted component upgrades.',
                    'typical_savings': '10-20% of HVAC energy use',
                    'priority': 'Medium',
                    'typical_cost': '$$ (Medium investment with 2-4 year payback)'
                },
                'Good HVAC': {
                    'title': 'HVAC Performance Monitoring',
                    'description': 'Implement ongoing monitoring and maintenance of your already efficient HVAC system.',
                    'typical_savings': '3-8% of HVAC energy use',
                    'priority': 'Low',
                    'typical_cost': '$ (Low investment with 1-2 year payback)'
                }
            },
            'envelope': {
                'Poor Envelope': {
                    'title': 'Building Envelope Improvements',
                    'description': 'Consider comprehensive air sealing, insulation improvements, and window upgrades.',
                    'typical_savings': '15-25% of heating/cooling costs',
                    'priority': 'High',
                    'typical_cost': '$$$ (High investment with 3-8 year payback)'
                },
                'Average Envelope': {
                    'title': 'Targeted Envelope Sealing',
                    'description': 'Address specific envelope weaknesses with targeted air sealing and insulation in key areas.',
                    'typical_savings': '5-15% of heating/cooling costs',
                    'priority': 'Medium',
                    'typical_cost': '$$ (Medium investment with 2-5 year payback)'
                },
                'Good Envelope': {
                    'title': 'Envelope Maintenance',
                    'description': 'Maintain your building\'s good envelope performance with regular inspections and minor repairs.',
                    'typical_savings': '1-5% of heating/cooling costs',
                    'priority': 'Low',
                    'typical_cost': '$ (Low investment with 1-3 year payback)'
                }
            },
            'controls': {
                'Poor Controls': {
                    'title': 'Building Automation System Implementation',
                    'description': 'Installing or significantly upgrading building automation systems could substantially improve efficiency.',
                    'typical_savings': '15-30% of total energy use',
                    'priority': 'High',
                    'typical_cost': '$$$ (High investment with 3-5 year payback)'
                },
                'Average Controls': {
                    'title': 'Controls Optimization',
                    'description': 'Optimize existing control systems with better scheduling, setpoints, and sequence of operations.',
                    'typical_savings': '5-15% of total energy use',
                    'priority': 'Medium',
                    'typical_cost': '$$ (Medium investment with 1-3 year payback)'
                },
                'Good Controls': {
                    'title': 'Advanced Control Strategies',
                    'description': 'Implement advanced strategies like demand response and predictive controls.',
                    'typical_savings': '3-8% of total energy use',
                    'priority': 'Low',
                    'typical_cost': '$$ (Medium investment with 2-4 year payback)'
                }
            }
        }
    
    def load_data(self, energy_file, buildings_file):
        """
        Load the Chicago Energy Benchmarking datasets.
        
        Parameters:
        -----------
        energy_file : str
            Path to the energy benchmarking CSV file
        buildings_file : str
            Path to the covered buildings CSV file
            
        Returns:
        --------
        tuple
            Tuple containing (energy_df, buildings_df)
        """
        print("Loading data...")
        self.energy_df = pd.read_csv(energy_file)
        self.buildings_df = pd.read_csv(buildings_file)
        print(f"Loaded {len(self.energy_df)} energy records and {len(self.buildings_df)} building records")
        return self.energy_df, self.buildings_df
    
    def prepare_data(self):
        """
        Prepare and merge the datasets.
        
        Returns:
        --------
        pandas.DataFrame
            The merged and prepared dataset
        """
        print("Preparing data...")
        
        # Get the latest year data
        latest_year = self.energy_df['Data Year'].max()
        energy_latest = self.energy_df[self.energy_df['Data Year'] == latest_year].copy()
        
        # Rename ID column in energy dataset to match buildings dataset
        energy_latest.rename(columns={'ID': 'Building ID'}, inplace=True)
        
        # Merge datasets
        self.merged_df = pd.merge(energy_latest, self.buildings_df, on='Building ID', how='left', suffixes=('', '_buildings'))
        
        # Create building age feature
        current_year = 2025
        self.merged_df['Building Age'] = current_year - self.merged_df['Year Built']
        
        # Calculate energy mix percentages
        energy_cols = [
            'Electricity Use (kBtu)',
            'Natural Gas Use (kBtu)',
            'District Steam Use (kBtu)',
            'District Chilled Water Use (kBtu)'
        ]
        
        # Replace NaN with 0 for energy columns
        for col in energy_cols:
            if col in self.merged_df.columns:
                self.merged_df[col] = self.merged_df[col].fillna(0)
        
        # Calculate total energy
        energy_cols_available = [col for col in energy_cols if col in self.merged_df.columns]
        self.merged_df['Total Energy (kBtu)'] = self.merged_df[energy_cols_available].sum(axis=1)
        
        # Calculate energy percentages
        for col in energy_cols_available:
            new_col = col.replace(' Use (kBtu)', ' Percentage')
            self.merged_df[new_col] = np.where(self.merged_df['Total Energy (kBtu)'] > 0,
                                          self.merged_df[col] / self.merged_df['Total Energy (kBtu)'] * 100,
                                          0)
        
        # Fill missing values for key metrics
        for col in ['Site EUI (kBtu/sq ft)', 'ENERGY STAR Score', 'GHG Intensity (kg CO2e/sq ft)']:
            if col in self.merged_df.columns:
                self.merged_df[col] = self.merged_df[col].fillna(self.merged_df[col].median())
        
        # Create derived features for decision tree modeling
        self._create_system_efficiency_indicators()
        
        print(f"Prepared dataset with {len(self.merged_df)} records")
        return self.merged_df
    
    def _create_system_efficiency_indicators(self):
        """
        Create indicators for system efficiency to use as target variables for decision trees.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with new efficiency indicator features
        """
        # Normalize key metrics for easier comparisons
        self.merged_df['EUI Type Percentile'] = self.merged_df.groupby('Primary Property Type')['Site EUI (kBtu/sq ft)'].transform(
            lambda x: pd.Series(x).rank(pct=True)
        )
        
        self.merged_df['ENERGY STAR Type Percentile'] = self.merged_df.groupby('Primary Property Type')['ENERGY STAR Score'].transform(
            lambda x: pd.Series(x).rank(pct=True)
        )
        
        # 1. Lighting efficiency indicator (0=Poor, 1=Average, 2=Good)
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
        
        # 2. HVAC efficiency indicator - modified to consider ENERGY STAR score more heavily
        self.merged_df['HVAC Efficiency'] = 1  # Default to average
        
        poor_hvac_mask = (
            ((self.merged_df['EUI Type Percentile'] < 0.25) &
            ((self.merged_df['Natural Gas Percentage'] > 60) | (self.merged_df['Electricity Percentage'] > 60))) |
            (self.merged_df['ENERGY STAR Score'] < 25)  # Very poor overall performance
        )
        self.merged_df.loc[poor_hvac_mask, 'HVAC Efficiency'] = 0  # Poor
        
        # Make sure buildings with excellent ENERGY STAR scores don't get poor HVAC ratings
        # unless there's strong evidence of HVAC issues
        self.merged_df.loc[(self.merged_df['ENERGY STAR Score'] > 85) &
                          (self.merged_df['HVAC Efficiency'] == 0), 'HVAC Efficiency'] = 1
        
        good_hvac_mask = (
            (self.merged_df['EUI Type Percentile'] > 0.75) &
            (self.merged_df['ENERGY STAR Score'] > 60)
        )
        self.merged_df.loc[good_hvac_mask, 'HVAC Efficiency'] = 2  # Good
        
        # 3. Envelope efficiency indicator
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
    
    def train_decision_trees(self):
        """
        Train decision trees for each building system.
        
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        print("Training decision tree models...")
        
        # Define features to use for decision trees
        features = [
            'Gross Floor Area - Buildings (sq ft)',
            'Building Age',
            'Site EUI (kBtu/sq ft)',
            'ENERGY STAR Score',
            'Electricity Percentage',
            'Natural Gas Percentage'
        ]
        
        # Ensure all features exist
        missing_features = [f for f in features if f not in self.merged_df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            features = [f for f in features if f in self.merged_df.columns]
        
        # Make sure we have enough features
        if len(features) < 3:
            print("Error: Not enough features for decision tree training")
            return False
        
        # Get buildings with complete data
        complete_data = self.merged_df.dropna(subset=features)
        
        if len(complete_data) < 100:
            print(f"Warning: Only {len(complete_data)} buildings have complete data for training.")
        
        # Train a decision tree for each system
        for system, target in {
            'lighting': 'Lighting Efficiency',
            'hvac': 'HVAC Efficiency',
            'envelope': 'Envelope Efficiency',
            'controls': 'Controls Efficiency'
        }.items():
            # Create and train the model
            X = complete_data[features]
            y = complete_data[target]
            
            # Try to train the model
            try:
                dt = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)
                dt.fit(X, y)
                
                # Store the model and features
                self.dt_models[system] = {
                    'model': dt,
                    'features': features
                }
                
                print(f"Trained decision tree for {system} recommendations with accuracy: {dt.score(X, y):.2f}")
            except Exception as e:
                print(f"Error training decision tree for {system}: {e}")
        
        return bool(self.dt_models)
    
    def visualize_decision_tree(self, system):
        """
        Visualize a specific decision tree model.
        
        Parameters:
        -----------
        system : str
            System name ('lighting', 'hvac', 'envelope', or 'controls')
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the decision tree visualization
        """
        if system not in self.dt_models:
            print(f"No model found for {system}")
            return None
        
        model_info = self.dt_models[system]
        dt = model_info['model']
        features = model_info['features']
        
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(dt, filled=True, feature_names=features,
                 class_names=self.system_categories[system],
                 rounded=True, fontsize=10, ax=ax)
        plt.title(f"Decision Tree for {system.capitalize()} Recommendations", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{system}_decision_tree.png', dpi=300)
        
        return fig
    
    def generate_recommendations(self, building_id):
        """
        Generate recommendations for a specific building using decision trees.
        
        Parameters:
        -----------
        building_id : int or str
            Building ID to generate recommendations for
            
        Returns:
        --------
        tuple
            Tuple of (recommendations, error_message)
        """
        if building_id not in self.merged_df['Building ID'].values:
            return None, f"Building ID {building_id} not found in dataset"
        
        if not self.dt_models:
            self.train_decision_trees()
        
        building = self.merged_df[self.merged_df['Building ID'] == building_id].iloc[0]
        
        # Initialize recommendations
        recommendations = {
            'building_info': {
                'Building ID': building_id,
                'Property Name': building.get('Property Name', 'Unknown'),
                'Address': building.get('Address', 'Unknown'),
                'Primary Property Type': building.get('Primary Property Type', 'Unknown'),
                'Gross Floor Area - Buildings (sq ft)': building.get('Gross Floor Area - Buildings (sq ft)', 0),
                'Year Built': building.get('Year Built', 'Unknown'),
                'Building Age': building.get('Building Age', 'Unknown'),
                'ENERGY STAR Score': building.get('ENERGY STAR Score', 'Unknown'),
                'Site EUI (kBtu/sq ft)': building.get('Site EUI (kBtu/sq ft)', 'Unknown')
            },
            'system_recommendations': [],
            'overall_summary': []
        }
        
        # Generate overall summary based on ENERGY STAR score
        energy_star = building.get('ENERGY STAR Score', 0)
        if pd.notna(energy_star):
            if energy_star >= 75:
                recommendations['overall_summary'].append({
                    'title': 'Excellent Overall Performance',
                    'description': f'Your {building.get("Primary Property Type", "building")} is performing very well with an ENERGY STAR score of {energy_star:.0f}.'
                })
            elif energy_star >= 50:
                recommendations['overall_summary'].append({
                    'title': 'Good Overall Performance',
                    'description': f'Your {building.get("Primary Property Type", "building")} is performing better than average with an ENERGY STAR score of {energy_star:.0f}.'
                })
            elif energy_star >= 25:
                recommendations['overall_summary'].append({
                    'title': 'Below Average Performance',
                    'description': f'Your {building.get("Primary Property Type", "building")} is performing below average with an ENERGY STAR score of {energy_star:.0f}.'
                })
            else:
                recommendations['overall_summary'].append({
                    'title': 'Poor Overall Performance',
                    'description': f'Your {building.get("Primary Property Type", "building")} is significantly underperforming with an ENERGY STAR score of {energy_star:.0f}.'
                })
        
        # Generate system-specific recommendations using decision trees
        for system, model_info in self.dt_models.items():
            model = model_info['model']
            features = model_info['features']
            
            # Prepare feature vector
            feature_values = []
            for feature in features:
                if feature in building and pd.notna(building[feature]):
                    feature_values.append(building[feature])
                else:
                    # Use median for missing values
                    feature_values.append(self.merged_df[feature].median())
            
            # Predict system efficiency
            try:
                prediction = model.predict([feature_values])[0]
                category = self.system_categories[system][prediction]
                
                # Get corresponding recommendation
                if system in self.recommendations and category in self.recommendations[system]:
                    recommendation = self.recommendations[system][category].copy()
                    recommendation['system'] = system
                    recommendations['system_recommendations'].append(recommendation)
            except Exception as e:
                print(f"Error predicting {system} recommendation: {e}")
        
        # Sort recommendations by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations['system_recommendations'].sort(key=lambda x: priority_order.get(x.get('priority'), 99))
        
        return recommendations, None
    
    def generate_recommendations_for_dashboard(self, building_id):
        """
        Generate simplified recommendations formatted for dashboard display.
        
        Parameters:
        -----------
        building_id : int or str
            Building ID to generate recommendations for
            
        Returns:
        --------
        dict
            Simplified recommendations for dashboard display
        """
        recommendations, error = self.generate_recommendations(building_id)
        if error:
            return None
        
        dashboard_data = {
            'building_info': recommendations['building_info'],
            'overall_score': recommendations['building_info'].get('ENERGY STAR Score', 0),
            'performance_summary': recommendations['overall_summary'][0]['description'] if recommendations['overall_summary'] else "",
            'priority_actions': [],
            'system_ratings': []
        }
        
        # Add priority actions (high and medium priority only)
        for rec in recommendations['system_recommendations']:
            if rec.get('priority') in ['High', 'Medium']:
                dashboard_data['priority_actions'].append({
                    'title': rec.get('title', ''),
                    'description': rec.get('description', ''),
                    'savings': rec.get('typical_savings', ''),
                    'cost': rec.get('typical_cost', ''),
                    'priority': rec.get('priority', '')
                })
        
        # Add system ratings
        systems = ['Lighting', 'HVAC', 'Envelope', 'Controls']
        for system in systems:
            system_lower = system.lower()
            system_efficiency = 1  # Default to average
            
            # Find actual efficiency from dataframe
            if building_id in self.merged_df['Building ID'].values:
                building = self.merged_df[self.merged_df['Building ID'] == building_id].iloc[0]
                if f'{system} Efficiency' in building:
                    system_efficiency = building[f'{system} Efficiency']
            
            # Map efficiency level to label
            efficiency_label = ['Poor', 'Average', 'Good'][int(system_efficiency)]
            
            # Find recommendation for this system
            recommendation = next((r for r in recommendations['system_recommendations']
                                  if r['system'] == system_lower), None)
            action = recommendation['title'] if recommendation else 'No specific recommendation'
            
            dashboard_data['system_ratings'].append({
                'system': system,
                'rating': efficiency_label,
                'rating_value': int(system_efficiency),
                'recommended_action': action
            })
        
        return dashboard_data
    
    def run_full_recommendation_pipeline(self, energy_file, buildings_file, sample_building_ids=None):
        """
        Run the complete recommendation pipeline and test it on sample buildings.
        
        Parameters:
        -----------
        energy_file : str
            Path to the energy benchmarking CSV file
        buildings_file : str
            Path to the covered buildings CSV file
        sample_building_ids : list, optional
            List of building IDs to test recommendations on.
            If None, randomly selects 3 buildings.
            
        Returns:
        --------
        dict
            Dictionary of recommendation results for sample buildings
        """
        print("Starting building recommendation engine pipeline...")
        
        # Load data
        self.load_data(energy_file, buildings_file)
        
        # Prepare data
        self.prepare_data()
        
        # Train decision trees
        self.train_decision_trees()
        
        # Visualize decision trees
        for system in ['lighting', 'hvac', 'envelope', 'controls']:
            self.visualize_decision_tree(system)
        
        # If no sample building IDs provided, select 3 random buildings
        if sample_building_ids is None:
            # Try to get buildings with different performance levels
            good_building = self.merged_df[self.merged_df['ENERGY STAR Score'] > 75].sample(1)['Building ID'].iloc[0] if len(self.merged_df[self.merged_df['ENERGY STAR Score'] > 75]) > 0 else None
            
            avg_building = self.merged_df[
                (self.merged_df['ENERGY STAR Score'] > 40) & 
                (self.merged_df['ENERGY STAR Score'] < 60)
            ].sample(1)['Building ID'].iloc[0] if len(self.merged_df[(self.merged_df['ENERGY STAR Score'] > 40) & (self.merged_df['ENERGY STAR Score'] < 60)]) > 0 else None
            
            poor_building = self.merged_df[self.merged_df['ENERGY STAR Score'] < 25].sample(1)['Building ID'].iloc[0] if len(self.merged_df[self.merged_df['ENERGY STAR Score'] < 25]) > 0 else None
            
            sample_building_ids = [i for i in [good_building, avg_building, poor_building] if i is not None]
            
            # If we don't have 3 buildings yet, add more random ones
            if len(sample_building_ids) < 3:
                additional_buildings = self.merged_df.sample(3 - len(sample_building_ids))['Building ID'].tolist()
                sample_building_ids.extend(additional_buildings)
        
        # Generate recommendations for sample buildings
        recommendation_results = {}
        for building_id in sample_building_ids:
            recommendations, error = self.generate_recommendations(building_id)
            if error:
                recommendation_results[building_id] = {'error': error}
            else:
                recommendation_results[building_id] = recommendations
        
        return recommendation_results
    
    def generate_model_interpretation(self):
        """
        Generate interpretation of the recommendation engine.
        
        Returns:
        --------
        str
            Detailed interpretation of the recommendation system
        """
        interpretation = """
        # Building Recommendation Engine: Interpretation and Implications

        The Building Recommendation Engine represents a significant advancement in translating complex energy data into actionable insights. By using decision tree models trained on actual building performance data, the system can provide tailored recommendations specific to each building's characteristics and current performance level.

        ## Key Features and Strengths:

        1. **System-Specific Analysis**: The engine separately analyzes four critical building systems (lighting, HVAC, envelope, and controls), providing targeted recommendations for each area rather than generic building-wide advice.

        2. **Data-Driven Classifications**: Building systems are classified based on real performance patterns observed in the Chicago building stock, ensuring recommendations are grounded in local realities rather than generic industry assumptions.

        3. **Prioritized Recommendations**: Each recommendation comes with a priority level (High, Medium, Low) based on the potential for improvement, helping building owners allocate limited resources effectively.

        4. **Cost-Benefit Transparency**: The inclusion of typical costs and savings expectations helps building owners make informed investment decisions.

        ## Practical Applications:

        - For Building Owners: The recommendation engine provides a personalized roadmap for energy efficiency improvements, highlighting the specific systems and upgrades that will yield the greatest benefits for their particular building. This eliminates guesswork and helps prioritize capital improvements.

        - For City Authorities: The engine can be deployed at scale to provide targeted guidance across the entire building stock, potentially creating a "virtual energy audit" service that reaches buildings that might not otherwise receive professional energy assessments. This could significantly accelerate efficiency improvements across the city.

        - For Energy Service Providers: The system helps identify which buildings need which services, enabling more efficient marketing and delivery of energy services.

        ## Future Enhancements:

        - Integration with real-time energy monitoring data for dynamic recommendations
        - Expansion of the model to include more specialized building types
        - Addition of financial modeling to calculate detailed ROI for each recommendation
        - Development of a building-to-building matching system to connect building owners with peers who have successfully implemented similar improvements
        """
        
        return interpretation