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
    
    Proceso ETL:
    1. EXTRACT: Carga datos desde CSV
    2. TRANSFORM: Limpia nulos, divide train/test (70-30), prepara registros
    3. LOAD: Envía registros a Kafka topic para streaming
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
    
    # =========================================================================
    # EXTRACT: Extracción de datos desde fuente (CSV)
    # =========================================================================
    
    def extract_csv_files(self) -> pd.DataFrame:
        """
        [ETL - EXTRACT] Extrae datos desde archivos CSV individuales (2015-2019).
        Aplica normalización de columnas específica por año.
        
        Returns:
            DataFrame combinado con todos los años
        """
        import os
        import glob
        
        # Obtener rutas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_dir = os.path.join(project_root, 'csv')
        
        logger.info(f"📥 [EXTRACT] Cargando archivos CSV desde {csv_dir}...")
        
        # Buscar todos los CSV de años
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        csv_files = [f for f in csv_files if any(year in os.path.basename(f) for year in ['2015', '2016', '2017', '2018', '2019'])]
        
        if not csv_files:
            raise FileNotFoundError(f"❌ No se encontraron archivos CSV en {csv_dir}")
        
        # Mapeo de columnas por año (mismo que model_utils.py)
        # NOTA: Region se excluye intencionalmente ya que no está en todos los años
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
            # NOTA: No incluimos Region porque solo existe en 2015-2016 y no se usa en el modelo
            if year in column_mappings:
                mapping = column_mappings[year]
                cols_to_select = {old: new for old, new in mapping.items() if old in df.columns}
                df = df[list(cols_to_select.keys())].rename(columns=cols_to_select)
            
            df['Year'] = year
            dfs.append(df)
            logger.info(f"   ✅ {os.path.basename(file)}: {len(df)} registros")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"📊 [EXTRACT] Total extraído: {len(df_combined)} registros (2015-2019)")
        
        return df_combined
    
    # =========================================================================
    # TRANSFORM: Transformación y preparación de datos
    # =========================================================================
    
    def transform_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [ETL - TRANSFORM] Limpia datos eliminando valores nulos en columnas críticas.
        También normaliza nombres de países (ej: Somaliland region -> Somalia).
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame limpio sin valores nulos (con índices reseteados)
        """
        feature_columns = [
            'GDP per capita', 'Social support', 'Healthy life expectancy',
            'Freedom to make life choices', 'Generosity', 
            'Perceptions of corruption', 'Score'
        ]
        
        initial_count = len(df)
        
        # Normalizar nombres de países (unificar Somaliland region -> Somalia)
        # Somaliland es una región autónoma de facto de Somalia
        if 'Country' in df.columns:
            somaliland_count = df['Country'].str.contains('Somaliland', case=False, na=False).sum()
            if somaliland_count > 0:
                df['Country'] = df['Country'].str.replace('Somaliland region', 'Somalia', case=False, regex=False)
                df['Country'] = df['Country'].str.replace('Somaliland Region', 'Somalia', case=False, regex=False)
                logger.info(f"🔄 [TRANSFORM] Unificados {somaliland_count} registros: Somaliland region -> Somalia")
        
        df_clean = df.dropna(subset=feature_columns)
        removed_count = initial_count - len(df_clean)
        
        # CRÍTICO: Resetear índices para que sean consecutivos (0, 1, 2, ...)
        # Esto asegura que train_test_split funcione correctamente
        df_clean = df_clean.reset_index(drop=True)
        
        logger.info(f"🧹 [TRANSFORM] Limpieza completada:")
        logger.info(f"   - Registros válidos: {len(df_clean)}")
        logger.info(f"   - Registros eliminados: {removed_count}")
        logger.info(f"   - Índices reseteados: 0 a {len(df_clean)-1}")
        
        return df_clean
    
    def transform_split_train_test(self, df: pd.DataFrame) -> tuple:
        """
        [ETL - TRANSFORM] Divide datos en conjuntos de entrenamiento y prueba.
        Usa el mismo random_state=42 que el modelo para consistencia.
        
        Args:
            df: DataFrame limpio
            
        Returns:
            Tupla (train_indices, test_indices) como sets
        """
        from sklearn.model_selection import train_test_split
        
        # Crear índices para el split
        indices = df.index.tolist()
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=0.3, 
            random_state=42
        )
        
        train_indices = set(train_idx)
        test_indices = set(test_idx)
        
        logger.info(f"📊 [TRANSFORM] División 70-30 completada:")
        logger.info(f"   - Entrenamiento: {len(train_indices)} registros (70%)")
        logger.info(f"   - Prueba: {len(test_indices)} registros (30%)")
        
        return train_indices, test_indices
    
    def transform_prepare_record(self, row: pd.Series, index: int, type_model: str = 'unknown') -> Dict[str, Any]:
        """
        [ETL - TRANSFORM] Transforma una fila del DataFrame en un registro estructurado.
        
        Args:
            row: Fila del DataFrame
            index: Índice del registro
            type_model: Tipo de conjunto ('train' o 'test')
            
        Returns:
            Diccionario con el registro formateado para Kafka
        """
        # Características para el modelo (features numéricas + categórica Country)
        features = {
            'Country': str(row.get('Country', 'Unknown')),  # ✅ AÑADIDO: Variable categórica
            'GDP_per_capita': float(row.get('GDP per capita', 0)),
            'Social_support': float(row.get('Social support', 0)),
            'Healthy_life_expectancy': float(row.get('Healthy life expectancy', 0)),
            'Freedom_to_make_life_choices': float(row.get('Freedom to make life choices', 0)),
            'Generosity': float(row.get('Generosity', 0)),
            'Perceptions_of_corruption': float(row.get('Perceptions of corruption', 0))
        }
        
        # Registro completo con metadata
        # NOTA: Region se excluye porque no está disponible en todos los años (solo 2015-2016)
        record = {
            'record_id': index,
            'timestamp': time.time(),
            'country': str(row.get('Country', 'Unknown')),
            'year': int(row.get('Year', 0)),
            'actual_score': float(row.get('Score', 0)),
            'type_model': type_model,
            'features': features
        }
        
        return record
    
    # =========================================================================
    # LOAD: Carga de datos a Kafka
    # =========================================================================
    
    def load_to_kafka(self, record: Dict[str, Any]) -> None:
        """
        [ETL - LOAD] Carga un registro transformado al topic de Kafka.
        
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
    
    # =========================================================================
    # PROCESO ETL COMPLETO
    # =========================================================================
    
    def run_etl_pipeline(self, 
                         delay: float = 0.5,
                         max_records: int = None) -> None:
        """
        [ETL PIPELINE] Ejecuta el pipeline completo Extract → Transform → Load.
        
        Proceso:
        1. EXTRACT: Lee datos desde CSV individuales (2015-2019)
        2. TRANSFORM: Limpia datos, divide train/test, prepara registros
        3. LOAD: Envía registros a Kafka en modo streaming
        
        Args:
            delay: Tiempo de espera entre registros (segundos)
            max_records: Máximo número de registros a enviar (None = todos)
        """
        try:
            logger.info("="*80)
            logger.info("🔄 INICIANDO PIPELINE ETL - PRODUCER")
            logger.info("="*80)
            
            # ==================== EXTRACT ====================
            logger.info("\n📥 FASE 1: EXTRACT")
            df = self.extract_csv_files()
            
            # ==================== TRANSFORM ====================
            logger.info("\n🔄 FASE 2: TRANSFORM")
            
            # Transformación 1: Limpiar datos
            df_clean = self.transform_clean_data(df)
            
            # Limitar registros si se especifica
            if max_records:
                df_clean = df_clean.head(max_records)
                logger.info(f"🔢 [TRANSFORM] Limitando a {max_records} registros")
            
            # Transformación 2: Dividir train/test
            train_indices, test_indices = self.transform_split_train_test(df_clean)
            
            # ==================== LOAD ====================
            logger.info("\n📤 FASE 3: LOAD")
            logger.info("🚀 Iniciando carga a Kafka...")
            
            records_sent = 0
            train_sent = 0
            test_sent = 0
            
            for idx, row in df_clean.iterrows():
                # Determinar si es train o test
                if idx in train_indices:
                    type_model = 'train'
                    train_sent += 1
                elif idx in test_indices:
                    type_model = 'test'
                    test_sent += 1
                else:
                    type_model = 'unknown'
                    logger.warning(f"⚠️ Índice {idx} no encontrado en train ni test")
                
                # Transformación 3: Preparar registro
                record = self.transform_prepare_record(row, idx, type_model)
                
                # Load: Enviar a Kafka
                self.load_to_kafka(record)
                records_sent += 1
                
                # Log cada batch
                if records_sent % self.batch_size == 0:
                    self.producer.flush()  # Forzar envío
                    logger.info(
                        f"📤 [LOAD] Enviados {records_sent}/{len(df_clean)} | "
                        f"Train: {train_sent} | Test: {test_sent}"
                    )
                
                # Esperar antes del siguiente registro (simular streaming)
                time.sleep(delay)
            
            # Flush final
            self.producer.flush()
            
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE ETL COMPLETADO")
            logger.info("="*80)
            logger.info(f"📊 Total registros enviados: {records_sent}")
            logger.info(f"   - Train: {train_sent} ({train_sent/records_sent*100:.1f}%)")
            logger.info(f"   - Test: {test_sent} ({test_sent/records_sent*100:.1f}%)")
            logger.info("="*80)
            
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

def main():
    """Función principal para ejecutar el productor"""
    
    # Configuración
    KAFKA_SERVER = 'localhost:9092'
    TOPIC = 'happiness-data'
    DELAY = 0.1  # Segundos entre registros 
    MAX_RECORDS = None  # None = enviar todos los registros
    
    logger.info("📊 El productor ejecutará su propio ETL desde CSV (2015-2019)")
    
    # Crear productor
    producer = HappinessKafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        topic=TOPIC,
        batch_size=10
    )
    
    # Ejecutar pipeline ETL (extrae desde CSV originales)
    producer.run_etl_pipeline(
        delay=DELAY,
        max_records=MAX_RECORDS
    )


if __name__ == "__main__":
    print("="*80)
    print("🚀 KAFKA PRODUCER - World Happiness Report ML System")
    print("="*80)
    main()
