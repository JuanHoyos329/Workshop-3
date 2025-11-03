import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save_model(model_path: str = 'modelo_regresion_lineal.pkl'):

    logger.info("🚀 Entrenando modelo de Regresión Lineal con datos 2015-2019...")
    
    # Cargar datos combinados del ETL
    import os
    combined_path = '../data/combined_data.csv'
    
    if not os.path.exists(combined_path):
        logger.error(f"❌ No se encontró {combined_path}")
        logger.error("⚠️  Por favor ejecuta primero: python etl.py")
        raise FileNotFoundError(f"No se encontró {combined_path}. Ejecuta 'python etl.py' primero.")
    
    # Cargar el CSV combinado directamente
    df = pd.read_csv(combined_path)
    logger.info(f"✅ Datos combinados cargados desde: {combined_path}")
    logger.info(f"   Total: {df.shape[0]} registros")
    
    # Mostrar registros por año
    year_counts = df['Year'].value_counts().sort_index()
    logger.info(f"✅ Registros por año:")
    for year, count in year_counts.items():
        logger.info(f"   {year}: {count} registros")
    
    logger.info(f"✅ Datos combinados: {df.shape[0]} registros totales")
    
    # Características (6 features)
    feature_columns = [
        'GDP per capita', 'Social support', 'Healthy life expectancy',
        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]
    
    # Limpiar datos nulos
    df_clean = df.dropna(subset=feature_columns + ['Score'])
    logger.info(f"✅ Registros limpios: {df_clean.shape[0]} (eliminados: {len(df) - len(df_clean)})")
    
    X = df_clean[feature_columns]
    y = df_clean['Score']
    
    # Dividir datos 70-30 (igual que en el notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    logger.info(f"✅ División 70-30:")
    logger.info(f"   Entrenamiento: {X_train.shape[0]} registros")
    logger.info(f"   Prueba: {X_test.shape[0]} registros")
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Evaluar con conjunto de prueba
    y_pred_test = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MÉTRICAS DEL MODELO (Conjunto de Prueba)")
    logger.info(f"{'='*80}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"MAE:      {mae:.4f}")
    logger.info(f"RMSE:     {rmse:.4f}")
    logger.info(f"MAPE:     {mape:.2f}%")
    logger.info(f"{'='*80}\n")
    
    # Guardar modelo
    with open(model_path, 'wb') as f:
        pickle.dump(modelo, f)
    
    logger.info(f"💾 Modelo guardado en: {model_path}")
    
    return modelo


def load_model(model_path: str = 'modelo_regresion_lineal.pkl'):
    """
    Carga el modelo desde archivo .pkl
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    try:
        with open(model_path, 'rb') as f:
            modelo = pickle.load(f)
        logger.info(f"✅ Modelo cargado desde: {model_path}")
        return modelo
    except FileNotFoundError:
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        logger.info("💡 Ejecuta train_and_save_model() primero")
        return None


def test_model_prediction(model_path: str = 'modelo_regresion_lineal.pkl'):
    """
    Prueba el modelo con datos de ejemplo
    
    Args:
        model_path: Ruta al modelo
    """
    modelo = load_model(model_path)
    
    if modelo is None:
        return
    
    # Datos de ejemplo (Finlandia 2019)
    ejemplo = np.array([[
        1.340,  # GDP per capita
        1.587,  # Social support
        0.986,  # Healthy life expectancy
        0.596,  # Freedom to make life choices
        0.153,  # Generosity
        0.393   # Perceptions of corruption
    ]])
    
    prediccion = modelo.predict(ejemplo)[0]
    
    logger.info(f"Predicción: {prediccion:.4f}")
    logger.info(f"Error: {abs(7.769 - prediccion):.4f}")


if __name__ == "__main__":
    print("="*80)
    print("🔧 UTILIDADES - World Happiness Report ML System")
    print("="*80)
    
    # 1. Entrenar y guardar modelo
    modelo = train_and_save_model()
    
    # 2. Probar modelo
    test_model_prediction()
