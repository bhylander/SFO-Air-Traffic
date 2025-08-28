# ✈️ Air Traffic Passenger Prediction with Machine Learning

## 📋 Project Overview

This project develops a machine learning pipeline to predict passenger counts using historical air traffic data from San Francisco International Airport. The solution achieves **91.7% accuracy (R²)** with a tuned XGBoost model, enabling airports and airlines to optimize operations, staffing, and resource allocation through accurate passenger forecasting.


## 🎯 Business Value

Accurate passenger count predictions enable:

- **Optimized Staffing**: Reduce labor costs while maintaining service quality
- **Resource Allocation**: Better gate assignments and security checkpoint planning  
- **Revenue Forecasting**: Predict parking, retail, and concession revenues
- **Capacity Planning**: Data-driven terminal expansion decisions

Without accurate predictions, airports face over-staffing (unnecessary costs), under-staffing (poor customer experience), inefficient resource usage, and missed revenue opportunities.

## 📊 Model Performance

### Test Set Metrics
| Metric | Score | Interpretation |
|--------|-------|----------------|
| **R²** | 0.9167 | Model explains 91.7% of variance |
| **RMSE** | 4,096.30 | Average error of ±4,096 passengers |
| **MAE** | 2,228.75 | Typical prediction off by ~2,229 passengers |
| **MAPE** | 106.43%* | Error relative to small passenger counts |

*Note: High MAPE is due to very small passenger counts (<100). For practical passenger volumes (>1,000), accuracy is strong.*

### Algorithm Comparison
Evaluated 5 algorithms, with **XGBoost performing best**:

| Algorithm | R² Score |
|-----------|----------|
| ✅ **XGBoost** | **0.917** |
| LightGBM | 0.915 |
| Random Forest | 0.910 |
| Gradient Boosting | 0.898 |
| Extra Trees | 0.887 |

## 🔍 Key Insights

- **Seasonal Patterns**: Clear monthly passenger traffic trends with seasonal variations
- **Strong Correlation**: Excellent alignment between predicted and actual passenger counts (R² = 0.917)
- **Feature Importance**: Terminal efficiency and airline size are the strongest predictors

## 🛠️ Technologies

- **Python 3.10+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - ML pipeline and preprocessing
- **XGBoost** - Primary prediction model
- **LightGBM** - Alternative gradient boosting
- **Matplotlib & Seaborn** - Data visualization
- **Google Colab** - Development environment

## 📁 Project Structure
```bash
air-traffic-passenger-prediction/
│
├── notebooks/
│ └── air_traffic_ml_pipeline.ipynb # Complete analysis and modeling
│
├── data/
│ └── README.md # Data source information
│
├── model_artifacts/
│ ├── air_traffic_model.joblib # Trained XGBoost model
│ ├── preprocessor.joblib # Feature preprocessing pipeline
│ └── model_metadata.json # Model parameters and metrics
│
├── images/
│ ├── passenger_trend.png # Visualizations for README
│ ├── predicted_vs_actual.png
│ └── feature_importance.png
│
├── requirements.txt # Python dependencies
└── README.md # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
pip install -r requirements.txt
```
Run the Analysis
Clone this repository

Download the dataset (see Data section below)

Open notebooks/air_traffic_ml_pipeline.ipynb in Jupyter/Colab

Run all cells to reproduce the analysis

Use the Trained Model
```bash
import joblib
import pandas as pd

# Load the model
model = joblib.load('model_artifacts/air_traffic_model.joblib')

# Prepare your data (example)
new_data = pd.DataFrame({
    'Year': [2024],
    'Month': [7],
    'Operating Airline': ['United Airlines'],
    'GEO Region': ['Asia'],
    # ... other required features
})

# Make predictions
predictions = model.predict(new_data)
print(f"Predicted passenger count: {predictions[0]:,.0f}")
```
📊 Dataset
* Source: San Francisco International Airport - Air Traffic Passenger Statistics

* Records: 38,370 observations

* Features: 15 variables

* Time Period: 2005-2023

* Target Variable: Passenger Count

Key features include:

* Operating Airline

* GEO Region (geographical region)

* Terminal

* Activity Type (Deplaned/Enplaned)

* Price Category

* Aircraft Type

⚠️ Important Usage Notes
Seasonal Nature: Airport data is highly seasonal. Comparative analyses should be done:

✅ Correct: Year-over-year (January 2024 vs January 2023)

❌ Incorrect: Month-to-month (January 2024 vs February 2024)

Aggregation: Passenger counts are additive across categories. For example, United Airlines passengers appear in multiple attribute fields and can be aggregated as needed.

🔮 Future Improvements
* Incorporate external features (weather, holidays, events)

* Implement time series forecasting for future predictions

* Deploy as REST API for real-time predictions

* Add automated retraining pipeline

* Create interactive dashboard for stakeholders

👥 Author
Benjamin Hylander

LinkedIn: https://www.linkedin.com/in/ben-hylander-0665b3b5/

GitHub: https://github.com/bhylander

Email: benhylander92@gmail.com

📄 License
This project is open source and available under the MIT License.

🙏 Acknowledgments
San Francisco International Airport for providing the public dataset

XGBoost and Scikit-learn communities for excellent documentation
