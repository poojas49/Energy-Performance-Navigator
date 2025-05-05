# Chicago Energy Performance Navigator

## Project Overview

The Chicago Energy Performance Navigator is a data science project that analyzes building energy benchmarking data to provide actionable insights for improving energy efficiency in Chicago buildings. The project leverages Chicago's Energy Benchmarking Ordinance data to bridge the gap between data collection and meaningful action.

Our analysis reveals that similar buildings can show energy consumption variations of up to 3-5 times, indicating significant untapped potential for efficiency improvements. By transforming complex energy data into clear, actionable pathways, this tool helps building owners improve efficiency while enabling city planners to effectively target resources and policies to reduce Chicago's carbon footprint.

![Dashboard Overview](visualization/dashboard_overview.png)

## Features

- **Interactive Dashboard**: Explore energy performance metrics across Chicago's building stock
- **Building Explorer**: Analyze individual buildings and benchmark against peers
- **Advanced Visualizations**: Examine energy usage patterns by building type, age, and location
- **Machine Learning Models**: Predict ENERGY STAR scores and identify optimal improvements
- **Recommendation Engine**: Generate tailored energy efficiency recommendations
- **Geographic Analysis**: View energy performance patterns across Chicago neighborhoods

## Team Members

- Pooja Shinde – pshin8@uic.edu – [GitHub: poojas49]
- Riya Mehta – rmeht43@uic.edu – [GitHub: riyagmehta]
- Saakshi Patel – spate808@uic.edu – [GitHub: saakshipatel]
- Heniben Prajapati – hpraj6@uic.edu – [GitHub: heni-29]
- Het Nagda – hnagd@uic.edu – [GitHub: hetnagda20]

## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/poojas49/Energy-Performance-Navigator.git
cd Energy-Performance-Navigator
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
# Create the virtual environment
```bash
python -m venv venv
```

# Activate the virtual environment
```bash
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the Data
```bash
python scripts/download_data.py
```

### Step 5: Train the Models (Optional)
```bash
python scripts/train_models.py
```
This script will train all machine learning models and save them to the models/ directory, as well as generate visualizations in the visualization/ directory. This step is optional as pre-trained models are included in the repository.

### Usage
### Running the Dashboard
```bash
streamlit run scripts/run_dashboard.py
```
This will start the Streamlit dashboard. Open your web browser and navigate to http://localhost:8501 to access it.

### Using the Dashboard
The dashboard includes several sections:

-**Dashboard Overview**: Summary statistics and key insights from the dataset
-**Building Explorer**: Detailed analysis of individual buildings with performance comparisons
-**Visualizations**: Explore energy performance patterns across building types, ages, and locations
-**Machine Learning Models**: Interact with predictive models and view their findings
-**Recommendations Demo**: Generate custom energy efficiency recommendations for any building
-**Building Map**: Geographic visualization of energy performance across Chicago
-**About the Project**: Information about the project and team members

### Project Structure
```bash
chicago-energy-navigator/
│
├── data/                        # Data folder
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
│
├── models/                      # Saved model files
│
├── visualization/               # Saved visualization files
│
├── src/                         # Source code
│   ├── data/                    # Data processing scripts
│   ├── eda/                     # Exploratory data analysis
│   ├── features/                # Feature engineering
│   ├── models/                  # Machine learning models
│   ├── visualizations/          # Visualization components
│   └── dashboard/               # Dashboard components
│
├── scripts/                     # Utility scripts
│
├── requirements.txt             # Required Python packages
├── setup.py                     # Package setup file
└── README.md                    # This file
```

### Key Findings

Building characteristics (type, age, size) significantly predict energy performance
Energy source distribution (electricity vs. natural gas) plays a critical role in efficiency
Geographic patterns show neighborhood-level variations in building performance
Machine learning models can accurately predict ENERGY STAR scores
Clustering analysis reveals distinct energy efficiency archetypes among the building stock

### Models
The project includes five machine learning models:

-**ENERGY STAR Score Predictor**: Predicts building energy performance ratings
-**Building Clustering Model**: Identifies natural groupings of buildings with similar characteristics
-**High Accuracy ENERGY STAR Predictor**: Advanced model with enhanced prediction accuracy
-**Building Recommendation Engine**: Provides tailored energy efficiency recommendations
-**Energy Efficiency Classifier**: Classifies buildings into energy efficiency categories

### Visualizations
The project features five key visualizations:

-**Building Type Performance**: Compares energy performance across different building types
-**Energy Source Distribution**: Shows how different building types utilize various energy sources
-**Performance by Building Age**: Examines how building age relates to energy performance
-**Geographic Energy Map**: Maps energy efficiency patterns across Chicago neighborhoods
-**Performance Outliers**: Identifies buildings that significantly outperform or underperform peers

### Acknowledgments

-City of Chicago for providing the Energy Benchmarking data
-University of Illinois Chicago for supporting this project
