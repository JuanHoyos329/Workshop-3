# World Happiness Report - Sistema de Streaming con Kafka y Machine Learning

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Kafka](https://img.shields.io/badge/Apache_Kafka-7.5.0-orange.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Latest-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)

## Descripción del Proyecto

Sistema de streaming en tiempo real que integra Apache Kafka, Machine Learning y MySQL para predecir el Happiness Score de países basándose en datos del World Happiness Report (2015-2019).

### Objetivo
Implementar un pipeline ETL completo con streaming de datos, predicción en tiempo real usando un modelo de Regresión Lineal Multiple con One-Hot Encoding para países, y almacenamiento persistente en base de datos con visualización de KPIs.

---

## Arquitectura del Sistema

```
CSV Files (2015-2019)
    ↓
Kafka Producer (Python)
    ↓
Apache Kafka Topic: "happiness-data"
    ↓
Kafka Consumer (Python)
    ├─ Load ML Model + Preprocessor (.pkl)
    ├─ Predict Happiness Score
    └─ Store in MySQL
    ↓
MySQL Database: predictions
    ↓
KPI Dashboard Generator (HTML)
```

---

## Estructura del Proyecto

```
Workshop 3/
│
├── csv/                                   
│   ├── 2015.csv
│   ├── 2016.csv
│   ├── 2017.csv
│   ├── 2018.csv
│   └── 2019.csv
│
├── data/                                  
│   └── predictions_streaming.csv
│
├── kafka/                                
│   ├── kafka_consumer.py                   
│   └── kafka_producer.py                  
│
├── kpis/                                  
│   ├── generar_kpis.py                    
│   └── dashboard_kpis.html                
│
├── model_regresion/                        
│   ├── modelo_regresion_lineal.pkl         
│   └── model_utils.py                      
│
├── notebooks/                              
│   ├── EDA_Happiness_Report.ipynb         
│   └── Modelos_Regresion_Happiness.ipynb   
│
├── docker-compose.yml                      
├── requirements.txt                       
└── README.md                               
```

---

## Instalación y Setup

### 1. Prerequisitos

- Python 3.12+
- Docker Desktop
- MySQL Server (local o Docker)
- Git

### 2. Clonar el Repositorio

```bash
git clone https://github.com/JuanHoyos329/Workshop-3.git
cd "Workshop 3"
```

### 3. Crear Entorno Virtual

```powershell
# Windows PowerShell
python -m venv .kafka
.\.kafka\Scripts\Activate
```

### 4. Instalar Dependencias

```powershell
pip install -r requirements.txt
```

### 5. Levantar Kafka con Docker

```powershell
docker-compose up -d
```

Verifica que los contenedores estén corriendo:

```powershell
docker ps
```

Deberías ver:
- `zookeeper` (puerto 2181)
- `kafka` (puerto 9092)

---

## Ejecución del Sistema

### PASO 1: Entrenar el Modelo

```powershell
cd model_regresion
python model_utils.py
```

Salida esperada:
- Modelo entrenado con One-Hot Encoding para Country
- Archivo `modelo_regresion_lineal.pkl` generado (contiene modelo + preprocessor)
- Métricas de evaluación mostradas

### PASO 2: Iniciar el Consumidor Kafka

En una nueva terminal:

```powershell
cd kafka
python kafka_consumer.py
```


### PASO 3: Ejecutar el Productor Kafka

En otra terminal:

```powershell
cd kafka
python kafka_producer.py
```

El productor:
- Extrae datos de CSV (2015-2019)
- Divide en train/test (70-30) con estratificación por país
- Envía registros a Kafka topic `happiness-data`

### PASO 4: Generar Dashboard de KPIs

```powershell
cd kpis
python generar_kpis.py
```

Abre el archivo generado `dashboard_kpis.html` en tu navegador. El dashboard incluye:
- Métricas de rendimiento (R², MAE, RMSE, MAPE)
- Comparación Train vs Test
- Mapa mundial interactivo con filtros (Actual/Train/Test)
- Top 10 países más felices
- Evolución temporal (escala fija 5-6)
- Distribución de errores
- Análisis por región

### PASO 5: Verificar Datos en MySQL

```sql
USE happiness_db;

-- Ver total de predicciones
SELECT COUNT(*) FROM predictions;

-- Ver predicciones por tipo
SELECT type_model, COUNT(*) as total 
FROM predictions 
GROUP BY type_model;

-- Top 10 países con mejor predicción
SELECT country, AVG(actual_score) as avg_actual, 
       AVG(predicted_score) as avg_predicted,
       AVG(prediction_error) as avg_error
FROM predictions 
GROUP BY country 
ORDER BY avg_actual DESC 
LIMIT 10;
```

---

## Resultados del Modelo

### Modelo: Regresión Lineal Múltiple

**Características del modelo:**
- 6 features numéricas
- 1 variable categórica (Country) → ~157 variables dummy
- Total: 163 features después del One-Hot Encoding
- División: 70% train / 30% test con estratificación por país

### Features Utilizadas

**Numéricas (6):**
1. GDP per capita - PIB per cápita
2. Social support - Soporte social
3. Healthy life expectancy - Esperanza de vida saludable
4. Freedom to make life choices - Libertad para elegir
5. Generosity - Generosidad
6. Perceptions of corruption - Percepción de corrupción

**Categórica (1):**
- Country - País 

---

## Decisiones Técnicas Clave

### 1. ¿Por qué Regresión Lineal?
- Simplicidad e interpretabilidad mantenida
- R² de mayor a 90% con la inclusión de Country
- Entrenamiento rápido (ideal para streaming)
- Captura efectos específicos por país
- El preprocessor (ColumnTransformer) maneja automáticamente la transformación

### 2. ¿Por qué Kafka?
- Streaming en tiempo real
- Escalabilidad horizontal
- Tolerancia a fallos
- Procesamiento asíncrono
- Desacoplamiento productor-consumidor

### 3. ¿Por qué MySQL?
- Persistencia de predicciones
- Queries SQL para análisis
- Compatibilidad con herramientas BI
- Índices para consultas rápidas
- Fácil integración con dashboards

### 4. División 70-30 Estratificada
- 70% Training (~547 registros)
- 30% Test (~234 registros)
- `random_state=42` para reproducibilidad
- Estratificación por país para asegurar representación en ambos conjuntos
- Países con 1 solo registro se dividen aleatoriamente

### 5. Arquitectura de Archivos .pkl
- Un solo archivo contiene modelo + preprocessor
- Estructura: `{'modelo': LinearRegression, 'preprocessor': ColumnTransformer}`
- Facilita deployment y garantiza consistencia en transformaciones

---

## Troubleshooting

### Error: Kafka no inicia

```powershell
# Ver logs de Kafka
docker logs kafka

# Reiniciar contenedores
docker-compose down
docker-compose up -d

# Esperar 10-15 segundos para que Kafka esté listo
```

### Error: NoBrokersAvailable

Este error indica que Kafka aún no está completamente iniciado. Espera 10-15 segundos después de `docker-compose up -d` antes de ejecutar el consumer.

### Error: MySQL connection refused

Verifica las credenciales en `kafka_consumer.py`:

```python
mysql_config = {
    'host': 'localhost',
    'port': 3306,
    'database': 'happiness_db',
    'user': 'root',
    'password': 'tu_password'  # Cambiar
}
```

### Error: Modelo no cargado

Asegúrate de haber ejecutado primero:

```powershell
cd model_regresion
python model_utils.py
```

Esto genera el archivo `modelo_regresion_lineal.pkl` necesario para las predicciones.


## Dashboard de KPIs

El sistema incluye un dashboard interactivo HTML con visualizaciones consolidadas:

### Características del Dashboard

- **Métricas principales:** R², MAE, RMSE, MAPE, Records, Countries, Years
- **Tabla comparativa:** Train vs Test (sin columna Total)
- **Mapa mundial interactivo:** Filtros para Actual/Train/Test
- **Scatter plots:** Predicciones vs Actual con filtros
- **Top 10 países:** Comparación de barras agrupadas
- **Evolución temporal:** Escala fija 5-6 con intervalos de 0.2
- **Análisis por región:** Performance por área geográfica
- **Distribución de errores:** Histogramas Train vs Test

### Tecnologías

- Plotly para gráficos interactivos
- HTML/CSS/JavaScript para interfaz
- Pandas para procesamiento de datos
- MySQL como fuente de datos

## Hallazgos del EDA

### Correlaciones Principales

- GDP per capita tiene la correlación más fuerte con Happiness Score (~0.78)
- Social support y Healthy life expectancy también muy correlacionadas
- Generosity tiene la correlación más débil
- Corruption tiene correlación negativa (más corrupción = menos felicidad)

### Patrones Encontrados

1. Países nórdicos (Finlandia, Dinamarca, Noruega) consistentemente en top 10
2. GDP alto no garantiza felicidad, pero ayuda significativamente
3. Soporte social es crítico incluso en países con GDP bajo
4. La esperanza de vida saludable es más importante que la expectativa total
5. La inclusión de Country como variable categórica captura efectos culturales/geográficos específicos

---

## Tecnologías Utilizadas

- **Python 3.12+:** Lenguaje principal
- **Apache Kafka 7.5.0:** Streaming de datos
- **MySQL 8.0+:** Base de datos
- **Scikit-Learn:** Machine Learning (LinearRegression, OneHotEncoder, ColumnTransformer)
- **Pandas & NumPy:** Procesamiento de datos
- **Plotly:** Visualizaciones interactivas
- **Docker:** Contenedorización de Kafka y Zookeeper
- **Kafka-Python:** Cliente de Kafka para Python

## Autor

Juan A. Hoyos  
Workshop 3 - ETL con Kafka y Machine Learning


## Referencias

- [World Happiness Report](https://worldhappiness.report/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Plotly Documentation](https://plotly.com/python/)
