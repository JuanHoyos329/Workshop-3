import json
import pickle
import time
import numpy as np
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
    Consume mensajes, realiza predicciones y almacena en MySQL.
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
        Carga el modelo de ML desde archivo .pkl
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Modelo cargado desde {model_path}")
        except FileNotFoundError:
            logger.error(f"❌ Archivo de modelo no encontrado: {model_path}")
            logger.info("💡 Guardando modelo actual...")
            self._save_current_model(model_path)
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {e}")
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
            id INT AUTO_INCREMENT PRIMARY KEY,
            record_id INT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            processing_time_ms FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_country (country),
            INDEX idx_year (year),
            INDEX idx_timestamp (timestamp)
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
    
    def extract_features(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Extrae las características del registro para predicción.
        
        Args:
            record: Registro del mensaje de Kafka
            
        Returns:
            Array de numpy con las características ordenadas
        """
        features = record['features']
        
        # Orden debe coincidir con el orden de entrenamiento del modelo
        feature_vector = np.array([
            features['GDP_per_capita'],
            features['Social_support'],
            features['Healthy_life_expectancy'],
            features['Freedom_to_make_life_choices'],
            features['Generosity'],
            features['Perceptions_of_corruption']
        ]).reshape(1, -1)
        
        return feature_vector
    
    def predict(self, features: np.ndarray) -> float:
        """
        Realiza predicción usando el modelo cargado.
        
        Args:
            features: Array de características
            
        Returns:
            Score predicho
        """
        try:
            prediction = self.model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return 0.0
    
    def save_to_mysql(self, record: Dict[str, Any], 
                      predicted_score: float,
                      processing_time: float) -> None:
        """
        Guarda el registro y la predicción en MySQL.
        
        Args:
            record: Registro original
            predicted_score: Score predicho
            processing_time: Tiempo de procesamiento en ms
        """
        insert_query = """
        INSERT INTO predictions (
            record_id, country, year,
            gdp_per_capita, social_support, healthy_life_expectancy,
            freedom_to_make_life_choices, generosity, perceptions_of_corruption,
            actual_score, predicted_score, prediction_error, processing_time_ms
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        features = record['features']
        actual_score = record['actual_score']
        prediction_error = abs(actual_score - predicted_score)
        
        values = (
            record['record_id'],
            record['country'],
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
            processing_time
        )
        
        try:
            cursor = self.mysql_conn.cursor()
            cursor.execute(insert_query, values)
            self.mysql_conn.commit()
            cursor.close()
            logger.debug(f"💾 Registro {record['record_id']} guardado en MySQL")
        except Error as e:
            logger.error(f"❌ Error al guardar en MySQL: {e}")
            self.mysql_conn.rollback()
    
    def process_message(self, message) -> None:
        """
        Procesa un mensaje de Kafka: predice y guarda.
        
        Args:
            message: Mensaje de Kafka
        """
        start_time = time.time()
        
        try:
            # Extraer datos del mensaje
            record = message.value
            
            # Extraer características
            features = self.extract_features(record)
            
            # Realizar predicción
            predicted_score = self.predict(features)
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Guardar en MySQL
            self.save_to_mysql(record, predicted_score, processing_time)
            
            # Log de resultado
            logger.info(
                f"✅ Procesado #{record['record_id']}: "
                f"{record['country']} ({record['year']}) | "
                f"Real: {record['actual_score']:.2f} | "
                f"Predicho: {predicted_score:.2f} | "
                f"Error: {abs(record['actual_score'] - predicted_score):.2f} | "
                f"⏱️ {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"❌ Error al procesar mensaje: {e}")
    
    def consume_and_predict(self, timeout_ms: int = 1000) -> None:
        """
        Inicia el consumo de mensajes y procesamiento en tiempo real.
        
        Args:
            timeout_ms: Timeout para poll de mensajes
        """
        logger.info("🚀 Iniciando consumo de mensajes...")
        logger.info("⏸️  Presiona Ctrl+C para detener")
        
        try:
            messages_processed = 0
            
            for message in self.consumer:
                # Procesar mensaje
                self.process_message(message)
                messages_processed += 1
                
                # Log cada 10 mensajes
                if messages_processed % 10 == 0:
                    logger.info(f"📊 Total procesados: {messages_processed}")
                
        except KeyboardInterrupt:
            logger.warning("⚠️ Consumo interrumpido por usuario")
        except Exception as e:
            logger.error(f"❌ Error en consumo: {e}")
            raise
        finally:
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
    
    # Configuración
    KAFKA_SERVER = 'localhost:9092'
    TOPIC = 'happiness-data'
    GROUP_ID = 'happiness-prediction-group'
    MODEL_PATH = 'modelo_regresion_lineal.pkl'
    
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
    
    # Iniciar consumo
    consumer.consume_and_predict()


if __name__ == "__main__":
    print("="*80)
    print("🚀 KAFKA CONSUMER - World Happiness Report ML System")
    print("="*80)
    main()
