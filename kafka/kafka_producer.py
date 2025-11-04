import json
import time
import pandas as pd
import numpy as np
from kafka import KafkaProducer
import logging
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HappinessKafkaProducer:
    """
    Productor de Kafka para datos de World Happiness Report.
    Envía registros de características de felicidad a un topic de Kafka.
    """
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 topic: str = 'happiness-data',
                 batch_size: int = 10):
        """
        Inicializa el productor de Kafka.
        
        Args:
            bootstrap_servers: Dirección del servidor Kafka
            topic: Nombre del topic de Kafka
            batch_size: Cantidad de registros a enviar antes de hacer flush
        """
        self.topic = topic
        self.batch_size = batch_size
        
        try:
            # Configurar el productor de Kafka
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Esperar confirmación de todos los replicas
                retries=3,   # Reintentos en caso de fallo
                compression_type='gzip',  # Comprimir datos
                max_in_flight_requests_per_connection=1  # Garantizar orden
            )
            logger.info(f"✅ Productor Kafka inicializado: {bootstrap_servers}")
            logger.info(f"📤 Topic: {topic}")
        except Exception as e:
            logger.error(f"❌ Error al inicializar productor Kafka: {e}")
            raise
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            csv_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Datos cargados desde {csv_path}: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Error al cargar datos: {e}")
            raise
    
    def prepare_record(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """
        Prepara un registro para enviar a Kafka.
        
        Args:
            row: Fila del DataFrame
            index: Índice del registro
            
        Returns:
            Diccionario con el registro formateado
        """
        # Características para el modelo (features)
        features = {
            'GDP_per_capita': float(row.get('GDP per capita', 0)),
            'Social_support': float(row.get('Social support', 0)),
            'Healthy_life_expectancy': float(row.get('Healthy life expectancy', 0)),
            'Freedom_to_make_life_choices': float(row.get('Freedom to make life choices', 0)),
            'Generosity': float(row.get('Generosity', 0)),
            'Perceptions_of_corruption': float(row.get('Perceptions of corruption', 0))
        }
        
        # Registro completo con metadata
        record = {
            'record_id': index,
            'timestamp': time.time(),
            'country': str(row.get('Country', 'Unknown')),
            'year': int(row.get('Year', 0)),
            'actual_score': float(row.get('Score', 0)),
            'features': features
        }
        
        return record
    
    def send_record(self, record: Dict[str, Any]) -> None:
        """
        Envía un registro a Kafka.
        
        Args:
            record: Diccionario con el registro a enviar
        """
        try:
            # Usar el record_id como key para particionar
            key = str(record['record_id'])
            
            # Enviar mensaje a Kafka
            future = self.producer.send(
                self.topic,
                key=key,
                value=record
            )
            
            # Callback para éxito
            future.add_callback(self._on_send_success)
            # Callback para error
            future.add_errback(self._on_send_error)
            
        except Exception as e:
            logger.error(f"❌ Error al enviar registro {record['record_id']}: {e}")
    
    def _on_send_success(self, record_metadata):
        """Callback cuando el mensaje se envía exitosamente"""
        logger.debug(
            f"✅ Mensaje enviado a {record_metadata.topic} "
            f"[Partition: {record_metadata.partition}, "
            f"Offset: {record_metadata.offset}]"
        )
    
    def _on_send_error(self, excp):
        """Callback cuando hay error al enviar mensaje"""
        logger.error(f"❌ Error al enviar mensaje: {excp}")
    
    def stream_data(self, 
                    csv_path: str, 
                    delay: float = 0.5,
                    max_records: int = None) -> None:
        """
        Transmite datos desde CSV a Kafka en modo streaming.
        
        Args:
            csv_path: Ruta al archivo CSV
            delay: Tiempo de espera entre registros (segundos)
            max_records: Máximo número de registros a enviar (None = todos)
        """
        try:
            # Cargar datos
            df = self.load_data(csv_path)
            
            # Limpiar datos nulos
            feature_columns = [
                'GDP per capita', 'Social support', 'Healthy life expectancy',
                'Freedom to make life choices', 'Generosity', 
                'Perceptions of corruption', 'Score'
            ]
            df_clean = df.dropna(subset=feature_columns)
            
            logger.info(f"📊 Registros válidos: {len(df_clean)}")
            
            # Limitar registros si se especifica
            if max_records:
                df_clean = df_clean.head(max_records)
                logger.info(f"🔢 Limitando a {max_records} registros")
            
            # Transmitir datos
            logger.info("🚀 Iniciando transmisión de datos...")
            
            records_sent = 0
            for idx, row in df_clean.iterrows():
                # Preparar registro
                record = self.prepare_record(row, idx)
                
                # Enviar a Kafka
                self.send_record(record)
                records_sent += 1
                
                # Log cada batch
                if records_sent % self.batch_size == 0:
                    self.producer.flush()  # Forzar envío
                    logger.info(f"📤 Enviados {records_sent}/{len(df_clean)} registros")
                
                # Esperar antes del siguiente registro (simular streaming)
                time.sleep(delay)
            
            # Flush final
            self.producer.flush()
            logger.info(f"✅ Transmisión completada: {records_sent} registros enviados")
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Transmisión interrumpida por usuario")
        except Exception as e:
            logger.error(f"❌ Error en transmisión: {e}")
            raise
        finally:
            self.close()
    
    def close(self):
        """Cierra el productor de Kafka"""
        try:
            self.producer.close()
            logger.info("🔒 Productor Kafka cerrado")
        except Exception as e:
            logger.error(f"❌ Error al cerrar productor: {e}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def verificar_combined_data():
    """
    Verifica si existe combined_data.csv generado por el ETL.
    Si no existe, ejecuta el ETL automáticamente.
    
    Returns:
        Ruta del archivo combined_data.csv
    """
    import os
    import subprocess
    
    # Ruta absoluta al archivo de datos
    script_dir = os.path.dirname(os.path.abspath(__file__))  # kafka/
    project_root = os.path.dirname(script_dir)  # Workshop 3/
    combined_path = os.path.join(project_root, 'data', 'combined_data.csv')
    
    # Verificar si existe el archivo
    if os.path.exists(combined_path):
        logger.info(f"✅ Archivo encontrado: {combined_path}")
        return combined_path
    else:
        logger.warning(f"⚠️  No se encontró {combined_path}")
        logger.info("🔄 Ejecutando ETL para generar combined_data.csv...")
        
        try:
            # Ejecutar el script ETL
            result = subprocess.run(['python', 'etl.py'], 
                                  capture_output=True, 
                                  text=True,
                                  check=True)
            logger.info("✅ ETL ejecutado exitosamente")
            return combined_path
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error al ejecutar ETL: {e}")
            logger.error(f"Salida: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("❌ No se encontró etl.py. Por favor ejecuta 'python etl.py' primero.")
            raise


def main():
    """Función principal para ejecutar el productor"""
    
    # Configuración
    KAFKA_SERVER = 'localhost:9092'
    TOPIC = 'happiness-data'
    DELAY = 0.1  # Segundos entre registros 
    MAX_RECORDS = None  
    
    # Verificar y cargar datos combinados del ETL
    logger.info("📊 Verificando datos combinados (2015-2019)...")
    CSV_FILE = verificar_combined_data()
    logger.info(f"📁 Usando archivo: {CSV_FILE}")
    
    # Crear productor
    producer = HappinessKafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        topic=TOPIC,
        batch_size=10
    )
    
    # Transmitir datos
    producer.stream_data(
        csv_path=CSV_FILE,
        delay=DELAY,
        max_records=MAX_RECORDS
    )


if __name__ == "__main__":
    print("="*80)
    print("🚀 KAFKA PRODUCER - World Happiness Report ML System")
    print("="*80)
    main()
