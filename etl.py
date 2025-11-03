"""
ETL Script - World Happiness Report
====================================
Combina los datos de 5 años (2015-2019) en un solo archivo CSV consolidado.

Funcionalidades:
- Carga y estandariza columnas de 3 formatos diferentes
- Limpia valores nulos y duplicados
- Genera estadísticas del proceso ETL
- Guarda archivo combined_data.csv en carpeta data/

Autor: Juan A.
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HappinessETL:
    """
    Clase ETL para procesar y combinar datos del World Happiness Report
    """
    
    def __init__(self, csv_dir='csv', output_dir='data'):
        """
        Inicializa el ETL
        
        Args:
            csv_dir: Directorio con los CSVs originales
            output_dir: Directorio de salida para el archivo combinado
        """
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.dataframes = {}
        self.combined_df = None
        
        # Columnas objetivo (formato estandarizado)
        self.target_columns = [
            'Country',
            'Score',
            'GDP per capita',
            'Social support',
            'Healthy life expectancy',
            'Freedom to make life choices',
            'Generosity',
            'Perceptions of corruption',
            'Year'
        ]
        
        # Estadísticas del proceso
        self.stats = {
            'registros_originales': 0,
            'registros_finales': 0,
            'registros_eliminados': 0,
            'nulos_eliminados': 0,
            'duplicados_eliminados': 0
        }
    
    def estandarizar_columnas(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Estandariza los nombres de columnas según el año
        
        Args:
            df: DataFrame a estandarizar
            year: Año del dataset
            
        Returns:
            DataFrame con columnas estandarizadas
        """
        df = df.copy()
        df['Year'] = year
        
        if year == 2015:
            df = df.rename(columns={
                'Country': 'Country',
                'Happiness Score': 'Score',
                'Economy (GDP per Capita)': 'GDP per capita',
                'Family': 'Social support',
                'Health (Life Expectancy)': 'Healthy life expectancy',
                'Freedom': 'Freedom to make life choices',
                'Trust (Government Corruption)': 'Perceptions of corruption',
                'Generosity': 'Generosity'
            })
            
        elif year == 2016:
            df = df.rename(columns={
                'Country': 'Country',
                'Happiness Score': 'Score',
                'Economy (GDP per Capita)': 'GDP per capita',
                'Family': 'Social support',
                'Health (Life Expectancy)': 'Healthy life expectancy',
                'Freedom': 'Freedom to make life choices',
                'Trust (Government Corruption)': 'Perceptions of corruption',
                'Generosity': 'Generosity'
            })
            
        elif year == 2017:
            df = df.rename(columns={
                'Country': 'Country',
                'Happiness.Score': 'Score',
                'Economy..GDP.per.Capita.': 'GDP per capita',
                'Family': 'Social support',
                'Health..Life.Expectancy.': 'Healthy life expectancy',
                'Freedom': 'Freedom to make life choices',
                'Trust..Government.Corruption.': 'Perceptions of corruption',
                'Generosity': 'Generosity'
            })
            
        elif year in [2018, 2019]:
            df = df.rename(columns={
                'Country or region': 'Country'
            })
        
        return df
    
    def extract(self) -> None:
        """
        Fase EXTRACT: Carga los 5 archivos CSV
        """
        logger.info("="*80)
        logger.info("🔍 FASE 1: EXTRACT - Cargando archivos CSV")
        logger.info("="*80)
        
        years = [2015, 2016, 2017, 2018, 2019]
        
        for year in years:
            csv_path = os.path.join(self.csv_dir, f"{year}.csv")
            
            try:
                df = pd.read_csv(csv_path)
                registros = len(df)
                self.stats['registros_originales'] += registros
                
                self.dataframes[year] = df
                logger.info(f"   ✅ {year}.csv cargado: {registros} registros")
                
            except FileNotFoundError:
                logger.error(f"   ❌ Error: No se encontró {csv_path}")
                raise
            except Exception as e:
                logger.error(f"   ❌ Error al cargar {year}.csv: {e}")
                raise
        
        logger.info(f"\n✅ Total registros extraídos: {self.stats['registros_originales']}")
    
    def transform(self) -> None:
        """
        Fase TRANSFORM: Estandariza, limpia y valida los datos
        """
        logger.info("\n" + "="*80)
        logger.info("🔄 FASE 2: TRANSFORM - Transformando datos")
        logger.info("="*80)
        
        # Estandarizar columnas
        logger.info("\n📝 Estandarizando nombres de columnas...")
        transformed_dfs = []
        
        for year, df in self.dataframes.items():
            df_std = self.estandarizar_columnas(df, year)
            
            # Filtrar solo las columnas objetivo
            df_filtered = df_std[self.target_columns].copy()
            
            transformed_dfs.append(df_filtered)
            logger.info(f"   ✅ {year}: {len(df_filtered)} registros estandarizados")
        
        # Combinar todos los dataframes
        logger.info("\n🔗 Combinando dataframes...")
        self.combined_df = pd.concat(transformed_dfs, ignore_index=True)
        logger.info(f"   ✅ Dataframes combinados: {len(self.combined_df)} registros totales")
        
        # Limpiar valores nulos
        logger.info("\n🧹 Limpiando valores nulos...")
        registros_antes = len(self.combined_df)
        nulos_por_columna = self.combined_df[self.target_columns].isnull().sum()
        
        if nulos_por_columna.sum() > 0:
            logger.info(f"   ⚠️  Valores nulos encontrados:")
            for col, count in nulos_por_columna[nulos_por_columna > 0].items():
                logger.info(f"      - {col}: {count}")
            
            self.combined_df = self.combined_df.dropna(subset=self.target_columns)
            registros_despues = len(self.combined_df)
            self.stats['nulos_eliminados'] = registros_antes - registros_despues
            logger.info(f"   ✅ Registros con nulos eliminados: {self.stats['nulos_eliminados']}")
        else:
            logger.info(f"   ✅ No se encontraron valores nulos")
        
        # Eliminar duplicados
        logger.info("\n🔍 Verificando duplicados...")
        registros_antes = len(self.combined_df)
        duplicados = self.combined_df.duplicated(subset=['Country', 'Year']).sum()
        
        if duplicados > 0:
            logger.info(f"   ⚠️  Duplicados encontrados: {duplicados}")
            self.combined_df = self.combined_df.drop_duplicates(subset=['Country', 'Year'], keep='first')
            registros_despues = len(self.combined_df)
            self.stats['duplicados_eliminados'] = registros_antes - registros_despues
            logger.info(f"   ✅ Duplicados eliminados: {self.stats['duplicados_eliminados']}")
        else:
            logger.info(f"   ✅ No se encontraron duplicados")
        
        # Ordenar por año y país
        logger.info("\n📊 Ordenando datos...")
        self.combined_df = self.combined_df.sort_values(['Year', 'Country']).reset_index(drop=True)
        logger.info(f"   ✅ Datos ordenados por Year y Country")
        
        self.stats['registros_finales'] = len(self.combined_df)
        self.stats['registros_eliminados'] = (
            self.stats['registros_originales'] - self.stats['registros_finales']
        )
    
    def load(self) -> str:
        """
        Fase LOAD: Guarda el archivo CSV combinado
        
        Returns:
            Ruta del archivo guardado
        """
        logger.info("\n" + "="*80)
        logger.info("💾 FASE 3: LOAD - Guardando archivo consolidado")
        logger.info("="*80)
        
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ruta de salida
        output_path = os.path.join(self.output_dir, 'combined_data.csv')
        
        # Guardar CSV
        self.combined_df.to_csv(output_path, index=False)
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        logger.info(f"   ✅ Archivo guardado: {output_path}")
        logger.info(f"   📏 Tamaño: {file_size:.2f} KB")
        logger.info(f"   📊 Registros: {len(self.combined_df)}")
        
        return output_path
    
    def generate_report(self) -> None:
        """
        Genera un reporte detallado del proceso ETL
        """
        logger.info("\n" + "="*80)
        logger.info("📈 REPORTE ETL - ESTADÍSTICAS FINALES")
        logger.info("="*80)
        
        # Estadísticas generales
        logger.info(f"\n📊 ESTADÍSTICAS GENERALES:")
        logger.info(f"   • Registros originales:    {self.stats['registros_originales']}")
        logger.info(f"   • Registros finales:       {self.stats['registros_finales']}")
        logger.info(f"   • Registros eliminados:    {self.stats['registros_eliminados']}")
        logger.info(f"   • Nulos eliminados:        {self.stats['nulos_eliminados']}")
        logger.info(f"   • Duplicados eliminados:   {self.stats['duplicados_eliminados']}")
        
        # Registros por año
        logger.info(f"\n📅 REGISTROS POR AÑO:")
        year_counts = self.combined_df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            logger.info(f"   • {year}: {count} países")
        
        # Estadísticas de features
        logger.info(f"\n📊 ESTADÍSTICAS DE FEATURES:")
        feature_cols = [col for col in self.target_columns if col not in ['Country', 'Year']]
        
        for col in feature_cols:
            mean_val = self.combined_df[col].mean()
            min_val = self.combined_df[col].min()
            max_val = self.combined_df[col].max()
            logger.info(f"   • {col}:")
            logger.info(f"      - Media: {mean_val:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f}")
        
        # Top 5 países más felices (promedio 2015-2019)
        logger.info(f"\n🏆 TOP 5 PAÍSES MÁS FELICES (Promedio 2015-2019):")
        top_countries = (
            self.combined_df.groupby('Country')['Score']
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        for i, (country, score) in enumerate(top_countries.items(), 1):
            logger.info(f"   {i}. {country}: {score:.3f}")
        
        # Calidad de datos
        logger.info(f"\n✅ CALIDAD DE DATOS:")
        completeness = (self.stats['registros_finales'] / self.stats['registros_originales']) * 100
        logger.info(f"   • Completitud: {completeness:.2f}%")
        logger.info(f"   • Columnas: {len(self.target_columns)}")
        logger.info(f"   • Países únicos: {self.combined_df['Country'].nunique()}")
        logger.info(f"   • Años: {self.combined_df['Year'].nunique()} (2015-2019)")
    
    def run(self) -> str:
        """
        Ejecuta el proceso ETL completo
        
        Returns:
            Ruta del archivo generado
        """
        start_time = datetime.now()
        
        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*20 + "🌍 WORLD HAPPINESS REPORT ETL" + " "*29 + "║")
        logger.info("║" + " "*25 + "Data Pipeline 2015-2019" + " "*30 + "║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("\n")
        
        try:
            # Fase 1: Extract
            self.extract()
            
            # Fase 2: Transform
            self.transform()
            
            # Fase 3: Load
            output_path = self.load()
            
            # Reporte final
            self.generate_report()
            
            # Tiempo de ejecución
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*80)
            logger.info(f"✅ PROCESO ETL COMPLETADO EXITOSAMENTE")
            logger.info("="*80)
            logger.info(f"   ⏱️  Tiempo de ejecución: {duration:.2f} segundos")
            logger.info(f"   📁 Archivo generado: {output_path}")
            logger.info("\n")
            
            return output_path
            
        except Exception as e:
            logger.error("\n" + "="*80)
            logger.error(f"❌ ERROR EN PROCESO ETL: {e}")
            logger.error("="*80)
            raise


def main():
    """
    Función principal para ejecutar el ETL
    """
    # Crear instancia del ETL
    etl = HappinessETL(csv_dir='csv', output_dir='data')
    
    # Ejecutar proceso ETL
    output_file = etl.run()
    
    return output_file


if __name__ == "__main__":
    main()
