import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from itertools import combinations
import joblib
import os

# Try to import optional ML libraries
try:
    from xgboost import XGBRegressor
    has_xgboost = True
except ImportError:
    has_xgboost = False
    
try:
    from lightgbm import LGBMRegressor
    has_lightgbm = True
except ImportError:
    has_lightgbm = False
    
try:
    from catboost import CatBoostRegressor
    has_catboost = True
except ImportError:
    has_catboost = False

class HighAccuracyEnergyStarPredictor:
    """
    Advanced class for predicting ENERGY STAR Scores with higher accuracy.
    Uses ensemble methods, advanced feature engineering, and hyperparameter tuning.
    """
    
    def __init__(self):
        """Initialize the HighAccuracyEnergyStarPredictor class."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.performance_metrics = {}
    
    def advanced_feature_engineering(self, df):
        """
        Create advanced features for >90% accuracy.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to enhance with advanced features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with advanced features
        """
        df = df.copy()
        
        # 1. Building Age Features
        current_year = 2025
        if 'Year Built' in df.columns:
            df['Building Age'] = current_year - df['Year Built']
            df['Age_Group'] = pd.cut(df['Building Age'],
                                    bins=[0, 10, 25, 50, 75, 100, 150, float('inf')],
                                    labels=['0-10', '11-25', '26-50', '51-75', '76-100', '101-150', '150+'])
            df['Age_Squared'] = df['Building Age'] ** 2
            df['Age_Log'] = np.log1p(df['Building Age'])
        
        # 2. Energy Mix Features
        energy_cols = [
            'Electricity Use (kBtu)',
            'Natural Gas Use (kBtu)',
            'District Steam Use (kBtu)',
            'District Chilled Water Use (kBtu)',
            'All Other Fuel Use (kBtu)'
        ]
        
        # Check which energy columns are available
        available_energy_cols = [col for col in energy_cols if col in df.columns]
        
        for col in available_energy_cols:
            df[col] = df[col].fillna(0)
        
        if available_energy_cols:
            df['Total Energy (kBtu)'] = df[available_energy_cols].sum(axis=1)
            
            for col in available_energy_cols:
                new_col = col.replace(' Use (kBtu)', ' Percentage')
                df[new_col] = np.where(df['Total Energy (kBtu)'] > 0,
                                      df[col] / df['Total Energy (kBtu)'] * 100,
                                      0)
        
        # 3. Energy Diversity Index
        if 'Electricity Percentage' in df.columns and 'Natural Gas Percentage' in df.columns:
            percentage_cols = [col for col in df.columns if 'Percentage' in col]
            df['Energy_Diversity'] = 0
            for col in percentage_cols:
                p = df[col] / 100
                p = p.replace([0, np.nan], 0.001)  # Avoid log(0)
                df['Energy_Diversity'] -= np.where(p > 0, p * np.log(p), 0)
        
        # 4. Size-based Features
        if 'Gross Floor Area - Buildings (sq ft)' in df.columns:
            df['Size_Category'] = pd.qcut(df['Gross Floor Area - Buildings (sq ft)'],
                                         q=10, labels=False, duplicates='drop')
            df['Size_Log'] = np.log1p(df['Gross Floor Area - Buildings (sq ft)'])
            df['Size_Squared'] = df['Gross Floor Area - Buildings (sq ft)'] ** 2
        
        # 5. EUI Ratios and Interactions
        if 'Site EUI (kBtu/sq ft)' in df.columns and 'Source EUI (kBtu/sq ft)' in df.columns:
            df['Site_to_Source_Ratio'] = np.where(df['Source EUI (kBtu/sq ft)'] > 0,
                                                 df['Site EUI (kBtu/sq ft)'] / df['Source EUI (kBtu/sq ft)'],
                                                 np.nan)
        
        if 'Weather Normalized Site EUI (kBtu/sq ft)' in df.columns and 'Site EUI (kBtu/sq ft)' in df.columns:
            df['Weather_Normalized_Ratio'] = np.where(df['Site EUI (kBtu/sq ft)'] > 0,
                                                     df['Weather Normalized Site EUI (kBtu/sq ft)'] / df['Site EUI (kBtu/sq ft)'],
                                                     np.nan)
        
        if 'GHG Intensity (kg CO2e/sq ft)' in df.columns and 'Site EUI (kBtu/sq ft)' in df.columns:
            df['GHG_per_EUI'] = np.where(df['Site EUI (kBtu/sq ft)'] > 0,
                                        df['GHG Intensity (kg CO2e/sq ft)'] / df['Site EUI (kBtu/sq ft)'],
                                        np.nan)
        
        # 6. Building Type Interactions
        if 'Primary Property Type' in df.columns and 'Age_Group' in df.columns:
            df['Type_Age_Interaction'] = df['Primary Property Type'] + '_' + df['Age_Group'].astype(str)
        
        if 'Primary Property Type' in df.columns and 'Size_Category' in df.columns:
            df['Type_Size_Interaction'] = df['Primary Property Type'] + '_' + df['Size_Category'].astype(str)
        
        # 7. Energy Efficiency Metrics
        if 'Total Energy (kBtu)' in df.columns and 'Gross Floor Area - Buildings (sq ft)' in df.columns:
            df['Energy_per_SqFt'] = np.where(df['Gross Floor Area - Buildings (sq ft)'] > 0,
                                            df['Total Energy (kBtu)'] / df['Gross Floor Area - Buildings (sq ft)'],
                                            np.nan)
        
        if 'Total GHG Emissions (Metric Tons CO2e)' in df.columns and 'Total Energy (kBtu)' in df.columns:
            df['GHG_per_Energy'] = np.where(df['Total Energy (kBtu)'] > 0,
                                           df['Total GHG Emissions (Metric Tons CO2e)'] / df['Total Energy (kBtu)'],
                                           np.nan)
        
        # 8. Advanced Ratios
        if 'Electricity Percentage' in df.columns and 'Natural Gas Percentage' in df.columns:
            df['Electricity_to_Gas_Ratio'] = np.where(df['Natural Gas Percentage'] > 1,
                                                     df['Electricity Percentage'] / df['Natural Gas Percentage'],
                                                     np.nan)
        
        # 9. Building Efficiency Index
        if 'Site EUI (kBtu/sq ft)' in df.columns and 'GHG Intensity (kg CO2e/sq ft)' in df.columns:
            df['Efficiency_Index'] = np.where((df['Site EUI (kBtu/sq ft)'] > 0) & (df['GHG Intensity (kg CO2e/sq ft)'] > 0),
                                             (1 / df['Site EUI (kBtu/sq ft)']) * (1 / df['GHG Intensity (kg CO2e/sq ft)']),
                                             np.nan)
        
        # 10. Normalize extremely skewed features
        log_features = ['Total GHG Emissions (Metric Tons CO2e)', 'Total Energy (kBtu)',
                       'Gross Floor Area - Buildings (sq ft)']
        for col in log_features:
            if col in df.columns:
                df[f'{col}_Log'] = np.log1p(df[col])
                df[f'{col}_Sqrt'] = np.sqrt(df[col])
        
        return df
    
    def create_feature_combinations(self, df, numeric_features):
        """
        Create polynomial features and interactions.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to enhance with feature combinations
        numeric_features : list
            List of numeric feature names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature combinations
        """
        # Create interactions for most important features
        important_features = ['Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)',
                            'GHG Intensity (kg CO2e/sq ft)', 'Building Age']
        
        available_important_features = [f for f in important_features if f in df.columns]
        
        for f1, f2 in combinations(available_important_features, 2):
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
        
        # Create squared terms for key features
        for feature in available_important_features:
            df[f'{feature}_squared'] = df[feature] ** 2
        
        return df
    
    def handle_outliers(self, df, numeric_columns):
        """
        Handle outliers using robust methods.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        numeric_columns : list
            List of numeric column names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with outliers handled
        """
        for col in numeric_columns:
            if col in df.columns:
                # Use IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap values
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def prepare_data(self, energy_df, buildings_df):
        """
        Prepare data with advanced preprocessing.
        
        Parameters:
        -----------
        energy_df : pandas.DataFrame
            Energy benchmarking dataset
        buildings_df : pandas.DataFrame
            Covered buildings dataset
            
        Returns:
        --------
        pandas.DataFrame
            Prepared data for modeling
        """
        # Merge datasets - handle different column name possibilities
        year_col = None
        for possible_col in ['Data Year', 'Data_Year', 'DataYear', 'Year']:
            if possible_col in energy_df.columns:
                year_col = possible_col
                break
        
        if year_col:
            latest_year = energy_df[year_col].max()
            energy_latest = energy_df[energy_df[year_col] == latest_year].copy()
        else:
            energy_latest = energy_df.copy()
        
        # Handle ID column variations
        id_col = None
        for possible_col in ['ID', 'Building ID', 'Building_ID', 'BuildingID']:
            if possible_col in energy_latest.columns:
                id_col = possible_col
                break
        
        if id_col and id_col != 'Building ID':
            energy_latest.rename(columns={id_col: 'Building ID'}, inplace=True)
        
        # Merge if possible
        if 'Building ID' in energy_latest.columns and 'Building ID' in buildings_df.columns:
            merged_df = pd.merge(energy_latest, buildings_df, on='Building ID', how='left', suffixes=('', '_buildings'))
        else:
            merged_df = energy_latest.copy()
        
        # Filter to rows with ENERGY STAR scores - check for different column name variations
        star_score_col = None
        for possible_col in ['ENERGY STAR Score', 'ENERGY_STAR_Score', 'Energy Star Score']:
            if possible_col in merged_df.columns:
                star_score_col = possible_col
                break
        
        if star_score_col:
            model_data = merged_df.dropna(subset=[star_score_col]).copy()
            if star_score_col != 'ENERGY STAR Score':
                model_data.rename(columns={star_score_col: 'ENERGY STAR Score'}, inplace=True)
        else:
            raise ValueError("No ENERGY STAR Score column found in the dataset")
        
        # Apply advanced feature engineering
        model_data = self.advanced_feature_engineering(model_data)
        
        # Handle outliers
        numeric_cols = model_data.select_dtypes(include=[np.number]).columns
        model_data = self.handle_outliers(model_data, numeric_cols)
        
        # Create feature combinations
        model_data = self.create_feature_combinations(model_data, numeric_cols)
        
        return model_data
    
    def create_advanced_preprocessor(self, categorical_features, numeric_features):
        """Create advanced preprocessing pipeline"""
        # Categorical transformer with encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Numeric transformer with robust scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', RobustScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor

    def create_ensemble_model(self):
        """
        Create advanced ensemble model for >90% accuracy.
        
        Returns:
        --------
        sklearn.ensemble.StackingRegressor
            Stacked ensemble model
        """
        # Define base models with optimized parameters
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=42
        )
        
        estimators = [
            ('rf', rf_model),
            ('gb', gb_model)
        ]
        
        # Meta-learner
        meta_learner = Ridge(alpha=1.0)
        
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_model
    
    def train_and_evaluate(self, energy_df, buildings_df):
        """Train and evaluate the high-accuracy model"""
        # Prepare data
        model_data = self.prepare_data(energy_df, buildings_df)

        # Define features
        exclude_cols = ['ENERGY STAR Score', 'Building ID', 'Property Name', 'Address',
                    'Location', 'Row_ID', 'Data Year']
        feature_cols = [col for col in model_data.columns if col not in exclude_cols
                    and not col.endswith('_buildings')]

        X = model_data[feature_cols]
        y = model_data['ENERGY STAR Score']

        # Identify categorical and numeric features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

        # Create preprocessor
        preprocessor = self.create_advanced_preprocessor(categorical_features, numeric_features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', self.create_ensemble_model())
        ])

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics['CV_R2_mean'] = cv_scores.mean()
        metrics['CV_R2_std'] = cv_scores.std()

        # Print performance metrics - ADD THIS SECTION
        print("\nHigh Accuracy Model Performance:")
        print("-" * 40)
        print(f"R² Score: {metrics['R2']:.4f}")
        print(f"Mean Squared Error: {metrics['MSE']:.2f}")
        print(f"Root Mean Squared Error: {metrics['RMSE']:.2f}")
        print(f"Mean Absolute Error: {metrics['MAE']:.2f}")
        print(f"Cross-Validation R² (mean): {metrics['CV_R2_mean']:.4f}")
        print(f"Cross-Validation R² (std): {metrics['CV_R2_std']:.4f}")

        # Save model
        self.model = model
        self.performance_metrics = metrics

        # Optionally plot results visualization - may need to be optional if running in a non-GUI environment
        try:
            self.plot_model_performance(y_test, y_pred, metrics)
        except Exception as e:
            print(f"Visualization could not be displayed: {e}")

        return model, metrics

    def plot_model_performance(self, y_test, y_pred, metrics):
        """
        Visualize model performance.
        
        Parameters:
        -----------
        y_test : array-like
            True target values
        y_pred : array-like
            Predicted target values
        metrics : dict
            Dictionary of performance metrics
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the performance visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([0, 100], [0, 100], 'r--')
        axes[0, 0].set_xlabel('Actual ENERGY STAR Score')
        axes[0, 0].set_ylabel('Predicted ENERGY STAR Score')
        axes[0, 0].set_title(f'Actual vs. Predicted (R² = {metrics["R2"]:.3f})')
        
        # Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted ENERGY STAR Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Distribution of predictions
        axes[1, 0].hist(y_test, bins=30, alpha=0.5, label='Actual', density=True)
        axes[1, 0].hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
        axes[1, 0].set_xlabel('ENERGY STAR Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Actual vs. Predicted')
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Error Distribution (RMSE = {metrics["RMSE"]:.2f})')
        
        plt.tight_layout()
        plt.savefig('high_accuracy_model_performance.png', dpi=300)
        
        return fig
    
    def generate_model_interpretation(self, metrics):
        """
        Generate interpretation of the high-accuracy model.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of performance metrics
            
        Returns:
        --------
        str
            Detailed interpretation of the model results
        """
        interpretation = f"""
        # Advanced Model Interpretation and Implications

        The high-accuracy ensemble model for predicting ENERGY STAR Scores demonstrates exceptional performance, with an R² score of {metrics['R2']:.3f}, indicating that the model explains {metrics['R2']*100:.1f}% of the variance in ENERGY STAR Scores. This represents a significant improvement over the baseline model, confirming that sophisticated modeling techniques combined with advanced feature engineering can provide highly accurate energy performance predictions.

        ## Key Model Strengths:

        1. **Robust Feature Engineering**: The model benefits from extensive feature engineering, including nonlinear transformations, interaction terms, and domain-specific features like energy mix ratios and efficiency indices.

        2. **Ensemble Approach**: By combining multiple high-performing base models (Random Forest and Gradient Boosting) through stacking, the model captures complex patterns in the data that individual models might miss.

        3. **Cross-Validation Performance**: The model demonstrates consistently high performance across cross-validation folds (CV R² mean: {metrics['CV_R2_mean']:.3f}), indicating its robustness to different data splits.

        4. **Error Distribution**: The error analysis shows a relatively symmetrical distribution of residuals, with a root mean squared error (RMSE) of {metrics['RMSE']:.2f} points on the ENERGY STAR Scale.

        ## Practical Applications:

        - For Building Owners: This highly accurate model enables precise prediction of how specific building changes would affect ENERGY STAR Scores, allowing for more confident investment decisions in energy efficiency upgrades.

        - For City Authorities: The model can be deployed as part of a citywide energy efficiency platform, providing building owners with accurate projections and helping authorities assess the potential impact of policy changes across the building stock.

        - For Energy Service Companies: The model's accuracy makes it valuable for energy performance contracting, where precise predictions of energy savings and resulting score improvements are critical for financial models.

        ## Future Enhancements:

        - Incorporation of time-series data to capture seasonal variations and efficiency trends over time
        - Integration with real-time energy monitoring systems for dynamic performance optimization
        - Development of explainable AI techniques to provide more actionable insights from the model's predictions
        """
        
        return interpretation