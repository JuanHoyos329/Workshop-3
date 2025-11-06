import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================
# ETL PROCESS: Extract → Transform → Load
# =========================================================================

def extract_csv_files():
    """
    [ETL - EXTRACT] Extrae datos desde archivos CSV individuales (2015-2019).
    
    Returns:
        DataFrame combinado con todos los años
    """
    import os
    import glob
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_dir = os.path.join(project_root, 'csv')
    
    logger.info("📥 [EXTRACT] Cargando archivos CSV individuales...")
    
    # Buscar todos los CSV de años
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    csv_files = [f for f in csv_files if any(year in os.path.basename(f) for year in ['2015', '2016', '2017', '2018', '2019'])]
    
    if not csv_files:
        raise FileNotFoundError(f"❌ No se encontraron archivos CSV en {csv_dir}")
    
    # Mapeo de columnas por año
    column_mappings = {
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
    
    dfs = []
    for file in sorted(csv_files):
        year = int(os.path.basename(file).split('.')[0])
        df = pd.read_csv(file)
        
        # Aplicar mapeo de columnas específico del año
        if year in column_mappings:
            mapping = column_mappings[year]
            # Seleccionar y renombrar solo las columnas que existen
            cols_to_select = {old: new for old, new in mapping.items() if old in df.columns}
            df = df[list(cols_to_select.keys())].rename(columns=cols_to_select)
        
        df['Year'] = year
        dfs.append(df)
        logger.info(f"   ✅ {os.path.basename(file)}: {len(df)} registros - {len(df.columns)} columnas")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"📊 [EXTRACT] Total extraído: {len(df_combined)} registros (2015-2019)")
    
    return df_combined


def transform_clean_nulls(df):
    """
    [ETL - TRANSFORM] Elimina registros con valores nulos en columnas críticas.
    
    Args:
        df: DataFrame combinado
        
    Returns:
        DataFrame limpio
    """
    logger.info("🔄 [TRANSFORM] Limpiando valores nulos...")
    
    initial_count = len(df)
    
    # Columnas críticas que no deben tener nulos
    critical_columns = [
        'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]
    
    df_clean = df.dropna(subset=critical_columns)
    removed_count = initial_count - len(df_clean)
    
    logger.info(f"   ✅ Registros limpios: {len(df_clean)}")
    logger.info(f"   ❌ Registros eliminados: {removed_count}")
    
    return df_clean


def transform_select_features(df):
    """
    [ETL - TRANSFORM] Selecciona las columnas necesarias para el modelo.
    
    Args:
        df: DataFrame con columnas normalizadas
        
    Returns:
        DataFrame con columnas seleccionadas
    """
    logger.info("🔄 [TRANSFORM] Seleccionando features para el modelo...")
    logger.info(f"   Columnas disponibles: {df.columns.tolist()}")
    
    required_columns = [
        'Country', 'Year', 'Score',
        'GDP per capita', 'Social support', 'Healthy life expectancy',
        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
    ]
    
    # Verificar que existan las columnas
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"⚠️  Columnas faltantes: {missing_cols}")
        logger.info(f"   💡 Columnas actuales en el DataFrame:")
        for col in df.columns:
            logger.info(f"      - {col}")
    
    # Seleccionar solo columnas que existen
    available_cols = [col for col in required_columns if col in df.columns]
    
    if len(available_cols) < len(required_columns):
        logger.error(f"❌ Solo {len(available_cols)}/{len(required_columns)} columnas disponibles")
    
    df_selected = df[available_cols].copy()
    
    logger.info(f"   ✅ {len(available_cols)} columnas seleccionadas")
    
    return df_selected





def run_etl_pipeline():
    """
    [ETL PIPELINE] Ejecuta el pipeline Extract → Transform (sin Load a CSV).
    Los datos procesados solo se usan para entrenar el modelo.
    
    Returns:
        DataFrame procesado listo para entrenamiento
    """
    logger.info("="*80)
    logger.info("🔄 INICIANDO PIPELINE ETL - MODEL TRAINING")
    logger.info("="*80)
    
    # EXTRACT (ya normaliza columnas por año)
    df = extract_csv_files()
    
    # TRANSFORM
    df = transform_clean_nulls(df)
    df = transform_select_features(df)
    
    logger.info("="*80)
    logger.info("✅ PIPELINE ETL COMPLETADO")
    logger.info("="*80)
    
    return df


# =========================================================================
# MODEL TRAINING
# =========================================================================

def train_and_save_model(model_path: str = None):
    """
    Entrena el modelo de Regresión Lineal ejecutando ETL desde CSV originales.
    
    Args:
        model_path: Ruta donde guardar el modelo .pkl
        
    Returns:
        Modelo entrenado
    """
    logger.info("🚀 Entrenando modelo de Regresión Lineal con datos 2015-2019...")

    # Ejecutar ETL desde archivos CSV originales
    df = run_etl_pipeline()

    logger.info(f"✅ Datos procesados: {df.shape[0]} registros")
    
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
    
    # Determinar ruta por defecto dentro de la carpeta model_regresion junto al script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, 'modelo_regresion_lineal.pkl')

    # Guardar modelo
    os.makedirs(script_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(modelo, f)

    logger.info(f"💾 Modelo guardado en: {model_path}")
    
    return modelo


def load_model(model_path: str = None):
    """
    Carga el modelo desde archivo .pkl
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, 'modelo_regresion_lineal.pkl')

    try:
        with open(model_path, 'rb') as f:
            modelo = pickle.load(f)
        logger.info(f"✅ Modelo cargado desde: {model_path}")
        return modelo
    except FileNotFoundError:
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        logger.info("💡 Ejecuta train_and_save_model() primero")
        return None


def test_model_prediction(model_path: str = None):
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
    print("Este script incluye su propio proceso ETL")
    print("="*80)
    
    # 1. Entrenar y guardar modelo (con ETL automático si es necesario)
    modelo = train_and_save_model()
    
    # 2. Probar modelo
    test_model_prediction()
