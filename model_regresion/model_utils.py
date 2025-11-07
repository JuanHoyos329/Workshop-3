import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================
# ETL PROCESS: Extract -> Transform
# =========================================================================

# Column mappings per year - normalizes heterogeneous column names
COLUMN_MAPPINGS = {
    2015: {
        'Country': 'Country',
        'Happiness Score': 'Score',
        'Economy (GDP per Capita)': 'GDP per capita',
        'Family': 'Social support',
        'Health (Life Expectancy)': 'Healthy life expectancy',
        'Freedom': 'Freedom to make life choices',
        'Trust (Government Corruption)': 'Perceptions of corruption',
        'Generosity': 'Generosity'
    },
    2016: {
        'Country': 'Country',
        'Happiness Score': 'Score',
        'Economy (GDP per Capita)': 'GDP per capita',
        'Family': 'Social support',
        'Health (Life Expectancy)': 'Healthy life expectancy',
        'Freedom': 'Freedom to make life choices',
        'Trust (Government Corruption)': 'Perceptions of corruption',
        'Generosity': 'Generosity'
    },
    2017: {
        'Country': 'Country',
        'Happiness.Score': 'Score',
        'Economy..GDP.per.Capita.': 'GDP per capita',
        'Family': 'Social support',
        'Health..Life.Expectancy.': 'Healthy life expectancy',
        'Freedom': 'Freedom to make life choices',
        'Trust..Government.Corruption.': 'Perceptions of corruption',
        'Generosity': 'Generosity'
    },
    2018: {
        'Country or region': 'Country',
        'Score': 'Score',
        'GDP per capita': 'GDP per capita',
        'Social support': 'Social support',
        'Healthy life expectancy': 'Healthy life expectancy',
        'Freedom to make life choices': 'Freedom to make life choices',
        'Perceptions of corruption': 'Perceptions of corruption',
        'Generosity': 'Generosity'
    },
    2019: {
        'Country or region': 'Country',
        'Score': 'Score',
        'GDP per capita': 'GDP per capita',
        'Social support': 'Social support',
        'Healthy life expectancy': 'Healthy life expectancy',
        'Freedom to make life choices': 'Freedom to make life choices',
        'Perceptions of corruption': 'Perceptions of corruption',
        'Generosity': 'Generosity'
    }
}

# Numeric features
FEATURE_COLUMNS = [
    'GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
]

# Categorical variable
CATEGORICAL_COLUMNS = ['Country']


def extract_csv_files():
    """Extract and normalize data from CSV files (2015-2019)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(os.path.dirname(script_dir), 'csv')
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    csv_files = [f for f in csv_files if any(year in os.path.basename(f) 
                 for year in ['2015', '2016', '2017', '2018', '2019'])]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    dfs = []
    for file in sorted(csv_files):
        year = int(os.path.basename(file).split('.')[0])
        df = pd.read_csv(file)
        
        if year in COLUMN_MAPPINGS:
            mapping = COLUMN_MAPPINGS[year]
            cols_to_select = {old: new for old, new in mapping.items() if old in df.columns}
            df = df[list(cols_to_select.keys())].rename(columns=cols_to_select)
        
        df['Year'] = year
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def transform_data(df):
    """Clean and select features for model training."""
    # Remove nulls in critical columns
    critical_columns = FEATURE_COLUMNS + ['Score']
    df_clean = df.dropna(subset=critical_columns)
    
    # Select required columns
    required_columns = CATEGORICAL_COLUMNS + ['Year', 'Score'] + FEATURE_COLUMNS
    available_cols = [col for col in required_columns if col in df_clean.columns]
    
    return df_clean[available_cols].copy()


def run_etl_pipeline():
    """Execute ETL pipeline and return processed DataFrame."""
    df = extract_csv_files()
    df = transform_data(df)
    logger.info(f"ETL completed: {len(df)} records from 2015-2019")
    return df


# =========================================================================
# MODEL TRAINING
# =========================================================================

def train_and_save_model(model_path: str = None):
    """Train Linear Regression model with One-Hot Encoding."""
    # Execute ETL
    df = run_etl_pipeline()
    
    # Prepare data
    df_clean = df.dropna(subset=FEATURE_COLUMNS + CATEGORICAL_COLUMNS + ['Score'])
    X = df_clean[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y = df_clean['Score']
    
    # Split data 70-30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create preprocessor with OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', FEATURE_COLUMNS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                                 handle_unknown='ignore'), CATEGORICAL_COLUMNS)
        ])
    
    # Apply transformations
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Train model
    modelo = LinearRegression()
    modelo.fit(X_train_transformed, y_train)
    
    # Evaluate
    y_pred = modelo.predict(X_test_transformed)
    metrics = {
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    logger.info(f"Model trained - R²: {metrics['R²']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, 'modelo_regresion_lineal.pkl')
    
    os.makedirs(script_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'modelo': modelo, 'preprocessor': preprocessor}, f)
    
    logger.info(f"Model saved: {model_path}")
    return modelo, preprocessor


def load_model(model_path: str = None):
    """Load model and preprocessor from .pkl file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, 'modelo_regresion_lineal.pkl')

    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        return model_package['modelo'], model_package['preprocessor']
    except FileNotFoundError:
        logger.error(f"Model not found: {model_path}")
        return None, None


def predict_happiness(country: str, gdp: float, social_support: float, 
                      life_expectancy: float, freedom: float, 
                      generosity: float, corruption: float,
                      model_path: str = None):
    """Make happiness prediction for given country and features."""
    modelo, preprocessor = load_model(model_path)
    
    if modelo is None or preprocessor is None:
        return None
    
    input_df = pd.DataFrame({
        'GDP per capita': [gdp],
        'Social support': [social_support],
        'Healthy life expectancy': [life_expectancy],
        'Freedom to make life choices': [freedom],
        'Generosity': [generosity],
        'Perceptions of corruption': [corruption],
        'Country': [country]
    })
    
    input_transformed = preprocessor.transform(input_df)
    prediction = modelo.predict(input_transformed)[0]
    
    return prediction


if __name__ == "__main__":
    print("="*80)
    print("UTILITIES - World Happiness Report ML System")
    print("="*80)
    
    # Train model
    modelo, preprocessor = train_and_save_model()
    
    # Test prediction with Finland 2019
    test_prediction = predict_happiness(
        'Finland', 1.340, 1.587, 0.986, 0.596, 0.153, 0.393
    )
    print(f"\nTest - Finland prediction: {test_prediction:.4f} (actual: 7.769)")
    print("="*80)

