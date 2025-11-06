import json
import pickle
import time
import os
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
import mysql.connector
from mysql.connector import Error
import logging
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HappinessKafkaConsumer:
    """
    Consumidor de Kafka para predicción de Happiness Score en tiempo real.
    
    Proceso ETL:
    1. EXTRACT: Lee mensajes desde Kafka topic
    2. TRANSFORM: Extrae features, aplica modelo ML, calcula métricas
    3. LOAD: Almacena predicciones y resultados en MySQL
    """
    
    def __init__(self,
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'happiness-data',
                 group_id: str = 'happiness-prediction-group',
                 model_path: str = 'modelo_regresion_lineal.pkl',
                 mysql_config: Dict[str, Any] = None):
        """
        Inicializa el consumidor de Kafka.
        
        Args:
            bootstrap_servers: Dirección del servidor Kafka
            topic: Nombre del topic de Kafka
            group_id: ID del grupo de consumidores
            model_path: Ruta al modelo .pkl guardado
            mysql_config: Configuración de conexión MySQL
        """
        self.topic = topic
        self.model = None
        self.mysql_config = mysql_config or self._default_mysql_config()
        self.model_reload_attempts = 0  # Contador de intentos de recarga
        
        # Cargar modelo
        self._load_model(model_path)
        
        # Configurar consumidor de Kafka
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',  # Leer desde el inicio
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=10
            )
            logger.info(f"✅ Consumidor Kafka inicializado: {bootstrap_servers}")
            logger.info(f"📥 Suscrito al topic: {topic}")
        except Exception as e:
            logger.error(f"❌ Error al inicializar consumidor Kafka: {e}")
            raise
        
        # Configurar conexión MySQL
        self._setup_mysql_connection()
        self._create_predictions_table()
    
    def _default_mysql_config(self) -> Dict[str, Any]:
        """Retorna configuración por defecto de MySQL"""
        return {
            'host': 'localhost',
            'port': 3306,
            'database': 'happiness_db',
            'user': 'root',
            'password': 'root'
        }
    
    def _load_model(self, model_path: str) -> None:
        """
        Carga el modelo ML desde un archivo .pkl (contiene modelo + preprocessor).
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Verificar que el modelo se cargó correctamente
            if self.model is None:
                logger.error(f"❌ El modelo cargado es None")
                raise ValueError("Modelo cargado es None")
            
            # Verificar que es un diccionario con 'modelo' y 'preprocessor'
            if not isinstance(self.model, dict):
                logger.error(f"❌ El modelo debe ser un diccionario con 'modelo' y 'preprocessor'")
                raise ValueError("Modelo inválido: formato incorrecto")
            
            if 'modelo' not in self.model or 'preprocessor' not in self.model:
                logger.error(f"❌ El modelo debe contener keys 'modelo' y 'preprocessor'")
                raise ValueError("Modelo inválido: faltan componentes")
            
            # Verificar que el modelo tiene el método predict
            if not hasattr(self.model['modelo'], 'predict'):
                logger.error(f"❌ El modelo no tiene método 'predict'")
                raise ValueError("Modelo inválido: no tiene método 'predict'")
            
            logger.info(f"✅ Modelo cargado exitosamente desde {model_path}")
            logger.info(f"   Tipo de modelo: {type(self.model['modelo']).__name__}")
            logger.info(f"   Preprocessor: {type(self.model['preprocessor']).__name__}")
            
        except FileNotFoundError:
            logger.error(f"❌ Archivo de modelo no encontrado: {model_path}")
            logger.info("💡 Por favor, ejecuta primero: python model_regresion/model_utils.py")
            self.model = None
            raise
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {e}")
            self.model = None
            raise
    
    def _save_current_model(self, model_path: str) -> None:
        """Guarda el modelo actual (placeholder para generar .pkl si no existe)"""
        logger.warning("⚠️ Por favor, guarda tu modelo entrenado como .pkl")
        logger.info("💡 Ejemplo: pickle.dump(modelo_lr, open('modelo_regresion_lineal.pkl', 'wb'))")
    
    def _setup_mysql_connection(self) -> None:
        """Configura la conexión a MySQL y crea la base de datos si no existe"""
        try:
            # Primero conectar sin especificar base de datos para crearla
            config_without_db = self.mysql_config.copy()
            database_name = config_without_db.pop('database')
            
            # Conectar a MySQL sin BD
            temp_conn = mysql.connector.connect(**config_without_db)
            cursor = temp_conn.cursor()
            
            # Crear base de datos si no existe
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
            logger.info(f"✅ Base de datos '{database_name}' verificada/creada")
            
            cursor.close()
            temp_conn.close()
            
            # Ahora conectar a la base de datos específica
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            if self.mysql_conn.is_connected():
                logger.info(f"✅ Conectado a MySQL: {self.mysql_config['database']}")
        except Error as e:
            logger.error(f"❌ Error al conectar a MySQL: {e}")
            logger.info("💡 Asegúrate de que MySQL está corriendo y las credenciales son correctas")
            raise
    
    def _create_predictions_table(self) -> None:
        """Crea la tabla de predicciones si no existe"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            record_id INT AUTO_INCREMENT PRIMARY KEY,
            country VARCHAR(100),
            year INT,
            
            -- Características (Features)
            gdp_per_capita FLOAT,
            social_support FLOAT,
            healthy_life_expectancy FLOAT,
            freedom_to_make_life_choices FLOAT,
            generosity FLOAT,
            perceptions_of_corruption FLOAT,
            
            -- Scores
            actual_score FLOAT,
            predicted_score FLOAT,
            prediction_error FLOAT,
            
            -- Metadata
            type_model VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_country (country),
            INDEX idx_year (year),
            INDEX idx_type_model (type_model),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        
        try:
            cursor = self.mysql_conn.cursor()
            cursor.execute(create_table_query)
            self.mysql_conn.commit()
            cursor.close()
            logger.info("✅ Tabla 'predictions' verificada/creada")
        except Error as e:
            logger.error(f"❌ Error al crear tabla: {e}")
            raise
    
    # =========================================================================
    # EXTRACT: Extracción de datos desde Kafka
    # =========================================================================
    
    def extract_from_kafka_message(self, message) -> Dict[str, Any]:
        """
        [ETL - EXTRACT] Extrae el registro completo del mensaje de Kafka.
        
        Args:
            message: Mensaje de Kafka
            
        Returns:
            Diccionario con los datos del registro
        """
        record = message.value
        logger.debug(f"📥 [EXTRACT] Mensaje recibido: Record ID {record['record_id']}")
        return record
    
    # =========================================================================
    # TRANSFORM: Transformación y procesamiento de datos
    # =========================================================================
    
    def transform_extract_features(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        [ETL - TRANSFORM] Extrae y ordena las características para el modelo.
        
        Args:
            record: Registro del mensaje de Kafka
            
        Returns:
            DataFrame con las 6 características numéricas + Country (para One-Hot Encoding)
        """
        features = record['features']
        
        # Crear DataFrame con el orden correcto (6 numéricas + Country)
        # IMPORTANTE: El preprocessor del modelo aplicará One-Hot Encoding a Country
        feature_df = pd.DataFrame({
            'GDP per capita': [features['GDP_per_capita']],
            'Social support': [features['Social_support']],
            'Healthy life expectancy': [features['Healthy_life_expectancy']],
            'Freedom to make life choices': [features['Freedom_to_make_life_choices']],
            'Generosity': [features['Generosity']],
            'Perceptions of corruption': [features['Perceptions_of_corruption']],
            'Country': [features['Country']]  # ✅ AÑADIDO: Variable categórica
        })
        
        return feature_df
    
    def transform_predict_score(self, features: pd.DataFrame) -> float:
        """
        [ETL - TRANSFORM] Aplica el modelo ML para predecir el Happiness Score.
        
        Args:
            features: DataFrame con características (6 numéricas + Country)
            
        Returns:
            Score predicho por el modelo
        """
        try:
            # Verificar que el modelo existe y tiene preprocessor
            if self.model is None:
                if self.model_reload_attempts < 3:
                    self.model_reload_attempts += 1
                    logger.warning(f"⚠️ Modelo no está cargado. Intento de recarga #{self.model_reload_attempts}...")
                    try:
                        self._load_model('modelo_regresion_lineal.pkl')
                        if self.model is not None:
                            self.model_reload_attempts = 0  # Resetear contador si éxito
                    except:
                        pass
                
                if self.model is None:
                    logger.error(f"❌ No se pudo recargar el modelo después de {self.model_reload_attempts} intentos")
                    return 0.0
            
            # El modelo es un diccionario con 'modelo' y 'preprocessor'
            modelo = self.model['modelo']
            preprocessor = self.model['preprocessor']
            
            # Aplicar preprocessor (One-Hot Encoding a Country)
            features_transformed = preprocessor.transform(features)
            
            # Realizar predicción
            prediction = modelo.predict(features_transformed)[0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            logger.error(f"   Tipo de modelo: {type(self.model)}")
            logger.error(f"   Features type: {type(features)}")
            if isinstance(features, pd.DataFrame):
                logger.error(f"   Features columns: {features.columns.tolist()}")
            return 0.0
    
    def transform_calculate_metrics(self, actual_score: float, predicted_score: float) -> Dict[str, float]:
        """
        [ETL - TRANSFORM] Calcula métricas de error de la predicción.
        
        Args:
            actual_score: Score real
            predicted_score: Score predicho
            
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {
            'prediction_error': abs(actual_score - predicted_score),
            'squared_error': (actual_score - predicted_score) ** 2,
            'percentage_error': abs((actual_score - predicted_score) / actual_score) * 100 if actual_score != 0 else 0
        }
        logger.debug(f"📊 [TRANSFORM] Métricas calculadas: Error={metrics['prediction_error']:.4f}")
        return metrics
    
    # =========================================================================
    # LOAD: Carga de datos a MySQL y CSV
    # =========================================================================
    
    def load_to_mysql(self, record: Dict[str, Any], 
                      predicted_score: float,
                      type_model: str) -> None:
        """
        [ETL - LOAD] Persiste el registro, predicción y métricas en MySQL.
        Normaliza nombres de países antes de insertar.
        
        Args:
            record: Registro original desde Kafka
            predicted_score: Score predicho por el modelo
            type_model: Tipo de conjunto de datos ('train' o 'test')
        """
        insert_query = """
        INSERT INTO predictions (
            country, year,
            gdp_per_capita, social_support, healthy_life_expectancy,
            freedom_to_make_life_choices, generosity, perceptions_of_corruption,
            actual_score, predicted_score, prediction_error, type_model
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        features = record['features']
        actual_score = record['actual_score']
        prediction_error = abs(actual_score - predicted_score)
        
        # Obtener país del mensaje de Kafka y normalizar
        country = record['country']
        # Unificar Somaliland region -> Somalia
        if 'Somaliland' in country:
            country = 'Somalia'
        
        values = (
            country,
            record['year'],
            features['GDP_per_capita'],
            features['Social_support'],
            features['Healthy_life_expectancy'],
            features['Freedom_to_make_life_choices'],
            features['Generosity'],
            features['Perceptions_of_corruption'],
            actual_score,
            predicted_score,
            prediction_error,
            type_model
        )
        
        try:
            cursor = self.mysql_conn.cursor()
            cursor.execute(insert_query, values)
            self.mysql_conn.commit()
            cursor.close()
            logger.debug(f"💾 Registro guardado en MySQL")
        except Error as e:
            logger.error(f"❌ Error al guardar en MySQL: {e}")
            self.mysql_conn.rollback()
    
    def load_to_csv(self, csv_filename: str = 'predictions_streaming.csv') -> None:
        """
        [ETL - LOAD] Exporta todos los datos de MySQL a un archivo CSV en la carpeta data.
        
        Args:
            csv_filename: Nombre del archivo CSV a generar
        """
        import pandas as pd
        
        # Determinar ruta del archivo CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data')
        
        # Crear directorio si no existe
        os.makedirs(data_dir, exist_ok=True)
        
        csv_path = os.path.join(data_dir, csv_filename)
        
        try:
            logger.info("📤 [LOAD] Exportando datos a CSV...")
            
            # Consulta para obtener todos los datos
            query = """
            SELECT 
                country, region, year,
                gdp_per_capita, social_support, healthy_life_expectancy,
                freedom_to_make_life_choices, generosity, perceptions_of_corruption,
                actual_score, predicted_score, prediction_error,
                type_model, created_at
            FROM predictions
            ORDER BY created_at
            """
            
            # Cargar datos desde MySQL
            df = pd.read_sql(query, self.mysql_conn)
            
            # Guardar a CSV
            df.to_csv(csv_path, index=False)
            
            logger.info(f"✅ [LOAD] Datos exportados exitosamente")
            logger.info(f"   📁 Archivo: {csv_path}")
            logger.info(f"   📊 Registros: {len(df)}")
            
            # Mostrar estadísticas por type_model
            if 'type_model' in df.columns:
                split_counts = df['type_model'].value_counts()
                logger.info(f"   📈 Distribución:")
                for split, count in split_counts.items():
                    logger.info(f"      - {split}: {count} registros")
            
        except Exception as e:
            logger.error(f"❌ [LOAD] Error al exportar CSV: {e}")
    
    # =========================================================================
    # PROCESO ETL COMPLETO
    # =========================================================================
    
    def run_etl_on_message(self, message) -> None:
        """
        [ETL PIPELINE] Ejecuta el pipeline completo Extract → Transform → Load por mensaje.
        
        Proceso:
        1. EXTRACT: Lee mensaje desde Kafka
        2. TRANSFORM: Extrae features, predice score, calcula métricas
        3. LOAD: Guarda predicción en MySQL
        
        Args:
            message: Mensaje de Kafka
        """
        try:
            # ==================== EXTRACT ====================
            record = self.extract_from_kafka_message(message)
            
            # ==================== TRANSFORM ====================
            # Transformación 1: Extraer features
            features = self.transform_extract_features(record)
            
            # Transformación 2: Predecir score
            predicted_score = self.transform_predict_score(features)
            
            # Transformación 3: Calcular métricas
            metrics = self.transform_calculate_metrics(
                record['actual_score'], 
                predicted_score
            )
            
            # Obtener type_model del registro (viene del producer)
            type_model = record.get('type_model', 'unknown')
            
            # ==================== LOAD ====================
            self.load_to_mysql(record, predicted_score, type_model)
            
            # Log de resultado
            logger.info(
                f"✅ [ETL] Record #{record['record_id']}: "
                f"{record['country']} ({record['year']}) | "
                f"Type: {type_model} | "
                f"Real: {record['actual_score']:.2f} | "
                f"Predicho: {predicted_score:.2f} | "
                f"Error: {metrics['prediction_error']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"❌ [ETL] Error al procesar mensaje: {e}")
    
    def start_etl_streaming(self, timeout_ms: int = 1000) -> None:
        """
        Inicia el procesamiento ETL en streaming de mensajes desde Kafka.
        
        Args:
            timeout_ms: Timeout para poll de mensajes
        """
        logger.info("="*80)
        logger.info("� INICIANDO PIPELINE ETL - CONSUMER")
        logger.info("="*80)
        logger.info("🚀 Consumiendo mensajes desde Kafka...")
        logger.info("⏸️  Presiona Ctrl+C para detener")
        logger.info("="*80)
        
        try:
            messages_processed = 0
            train_processed = 0
            test_processed = 0
            total_error = 0.0
            
            for message in self.consumer:
                # Ejecutar ETL pipeline por mensaje
                self.run_etl_on_message(message)
                
                # Actualizar estadísticas
                messages_processed += 1
                record = message.value
                type_model = record.get('type_model', 'unknown')
                
                if type_model == 'train':
                    train_processed += 1
                elif type_model == 'test':
                    test_processed += 1
                
                # Log cada 10 mensajes
                if messages_processed % 10 == 0:
                    logger.info(
                        f"\n📊 [ESTADÍSTICAS] Total procesados: {messages_processed} | "
                        f"Train: {train_processed} | Test: {test_processed}\n"
                    )
                
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Consumo interrumpido por usuario")
        except Exception as e:
            logger.error(f"❌ Error en consumo: {e}")
            raise
        finally:
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE ETL FINALIZADO")
            logger.info("="*80)
            logger.info(f"📊 Total mensajes procesados: {messages_processed}")
            logger.info(f"   - Train: {train_processed}")
            logger.info(f"   - Test: {test_processed}")
            logger.info("="*80)
            
            # Exportar datos a CSV antes de cerrar
            if messages_processed > 0:
                logger.info("\n📥 Exportando datos procesados a CSV...")
                self.load_to_csv()
            
            self.close()
    
    def close(self):
        """Cierra las conexiones de Kafka y MySQL"""
        try:
            self.consumer.close()
            logger.info("🔒 Consumidor Kafka cerrado")
        except Exception as e:
            logger.error(f"❌ Error al cerrar consumidor: {e}")
        
        try:
            if self.mysql_conn.is_connected():
                self.mysql_conn.close()
                logger.info("🔒 Conexión MySQL cerrada")
        except Exception as e:
            logger.error(f"❌ Error al cerrar MySQL: {e}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para ejecutar el consumidor"""
    
    # Configuración Kafka
    KAFKA_SERVER = 'localhost:9092'
    TOPIC = 'happiness-data'
    GROUP_ID = 'happiness-prediction-group'
    
    # Ruta absoluta al modelo
    script_dir = os.path.dirname(os.path.abspath(__file__))  # kafka/
    project_root = os.path.dirname(script_dir)  # Workshop 3/
    MODEL_PATH = os.path.join(project_root, 'model_regresion', 'modelo_regresion_lineal.pkl')
    
    # Configuración MySQL
    MYSQL_CONFIG = {
        'host': 'localhost',
        'port': 3306,
        'database': 'happiness_db',
        'user': 'root',
        'password': 'root'  # ⚠️ Cambia esto por tu contraseña de MySQL
    }
    
    # Crear consumidor
    consumer = HappinessKafkaConsumer(
        bootstrap_servers=KAFKA_SERVER,
        topic=TOPIC,
        group_id=GROUP_ID,
        model_path=MODEL_PATH,
        mysql_config=MYSQL_CONFIG
    )
    
    # Iniciar pipeline ETL streaming
    consumer.start_etl_streaming()


if __name__ == "__main__":
    print("="*80)
    print("🚀 KAFKA CONSUMER - World Happiness Report ML System")
    print("="*80)
    main()
