# Chicago Energy Performance Navigator

A data-driven tool that transforms raw building energy benchmarking data into actionable insights for both building owners and city planners in Chicago.

## Team Members

- Pooja Shinde – [pshin8@uic.edu](mailto:pshin8@uic.edu) – [GitHub: poojas49](https://github.com/poojas49)
- Riya Mehta – [rmeht43@uic.edu](mailto:rmeht43@uic.edu) – [GitHub: riyagmehta](https://github.com/riyagmehta)
- Saakshi Patel – [spate808@uic.edu](mailto:spate808@uic.edu) – [GitHub: saakshipatel](https://github.com/saakshipatel)
- Heniben Prajapati – [hpraj6@uic.edu](mailto:hpraj6@uic.edu) – [GitHub: heni-29](https://github.com/heni-29)
- Het Nagda – [hnagd@uic.edu](mailto:hnagd@uic.edu) – [GitHub: hetnagda20](https://github.com/hetnagda20)

## Project Overview

The Chicago Energy Performance Navigator project aims to bridge the gap between data collection and meaningful action in building energy efficiency. Chicago's Energy Benchmarking Ordinance collects valuable building performance data, but currently fails to translate this into actionable insights. Our analysis reveals that similar buildings can show energy consumption variations of up to 3-5 times, indicating significant untapped potential for efficiency improvements.

We investigate how building characteristics, location, age, and energy sources correlate with energy efficiency metrics, and develop tools that provide customized recommendations. By translating complex energy data into clear, actionable pathways, we help building owners improve efficiency while enabling city planners to effectively target resources and policies to reduce Chicago's carbon footprint.

## Key Features

- **Building Performance Benchmarking**: Compare buildings against relevant peer groups
- **Energy Efficiency Analysis**: Identify factors that influence building energy performance
- **Recommendation Engine**: Generate tailored suggestions for efficiency improvements
- **Geographic Analysis**: Visualize energy performance patterns across Chicago neighborhoods
- **Predictive Modeling**: Forecast ENERGY STAR scores and identify efficiency opportunities

## Data Sources

This project uses two primary datasets:
1. **Chicago Energy Benchmarking Dataset**: Contains detailed energy performance metrics for buildings covered by Chicago's benchmarking ordinance
2. **Chicago Covered Buildings Dataset**: Provides additional information about buildings subject to the ordinance

## Technical Implementation

### Data Analysis & Visualization
- **Data Preparation**: Cleaning, merging, and feature engineering
- **Exploratory Analysis**: Statistical analysis and visualization of energy performance patterns
- **Geographic Mapping**: Spatial analysis of building efficiency across Chicago neighborhoods
- **Performance Outlier Detection**: Identification of exceptionally efficient and inefficient buildings

### Machine Learning Models
- **Energy Star Score Prediction**: Random Forest regression model to predict energy ratings based on building characteristics
- **Building Energy Efficiency Clustering**: K-means clustering to identify natural groupings of buildings with similar energy profiles

### Interactive Dashboard (Under Development)
- **User-Friendly Interface**: Intuitive visualization and exploration tools
- **Custom Recommendations**: Tailored efficiency improvement suggestions based on building characteristics
- **Impact Estimation**: Quantification of potential energy, cost, and emissions savings

## Key Findings

Our analysis reveals several important insights:

1. **Building Type Impact**: Different building types show distinct energy performance patterns, with some categories consistently outperforming others.

2. **Energy Source Variations**: The distribution of energy sources (electricity vs. natural gas) significantly impacts overall efficiency and emissions.

3. **Age-Performance Relationship**: Building age influences energy efficiency, but the relationship is complex, with some older buildings outperforming newer ones.

4. **Geographic Patterns**: Energy efficiency shows distinct spatial patterns across Chicago neighborhoods, suggesting localized factors affecting performance.

5. **Performance Outliers**: Similar buildings show energy consumption variations of up to 3-5 times, highlighting significant improvement potential.

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, folium, streamlit

### Installation

```bash
# Clone this repository
git clone https://github.com/poojas49/Energy-Performance-Navigator.git

# Navigate to the project directory
cd Energy-Performance-Navigator

# Install dependencies
pip install -r requirements.txt

# Run the dashboard (when available)
cd dashboard
streamlit run app.py
```

## Future Work

- Refine the ENERGY STAR prediction model with hyperparameter tuning
- Develop a classification model for Chicago Energy Rating prediction
- Integrate additional neighborhood-level data for enhanced spatial analysis
- Create a comprehensive recommendation system with cost-benefit analysis
- Implement a fully interactive dashboard for public use

## Acknowledgments

- City of Chicago for providing the Energy Benchmarking data
- UIC Department of Computer Science for project support
