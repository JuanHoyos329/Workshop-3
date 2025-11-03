# 🌍 World Happiness Report - Sistema de Streaming con Kafka y Machine Learning

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Kafka](https://img.shields.io/badge/Apache_Kafka-7.5.0-orange.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Latest-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)

## 📋 Descripción del Proyecto

Sistema de **streaming en tiempo real** que integra **Apache Kafka**, **Machine Learning** y **MySQL** para predecir el **Happiness Score** de países basándose en datos del World Happiness Report (2015-2019).

### 🎯 Objetivo
Implementar un pipeline ETL completo con streaming de datos, predicción en tiempo real usando un modelo de Regresión Lineal, y almacenamiento persistente en base de datos.

---

## 🏗️ Arquitectura del Sistema

```
📊 CSV Files (2015-2019)
    ↓
🔄 Kafka Producer (Python)
    ↓
📡 Apache Kafka Topic: "happiness-data"
    ↓
🤖 Kafka Consumer (Python)
    ├─ Load ML Model (.pkl)
    ├─ Predict Happiness Score
    └─ Store in MySQL
    ↓
💾 MySQL Database: predictions
    ↓
📈 Analysis & Visualization
```

---

## 🗂️ Estructura del Proyecto

```
Workshop 3/
│
├── csv/                                    # Datos originales
│   ├── 2015.csv
│   ├── 2016.csv
│   ├── 2017.csv
│   ├── 2018.csv
│   └── 2019.csv
│
├── EDA_Happiness_Report.ipynb              # 📊 Análisis exploratorio
├── Modelos_Regresion_Happiness.ipynb      # 🤖 Entrenamiento del modelo
├── Evaluacion_Streaming_Kafka.ipynb       # 📈 Evaluación y visualizaciones
│
├── model_utils.py                          # 🛠️ Utilidades para el modelo
├── kafka_producer.py                       # 📤 Productor Kafka
├── kafka_consumer.py                       # 📥 Consumidor Kafka
│
├── modelo_regresion_lineal.pkl            # 💾 Modelo entrenado
├── combined_data.csv                       # 📊 Datos combinados (2015-2019)
│
├── docker-compose.yml                      # 🐳 Configuración Docker
├── requirements_kafka.txt                  # 📦 Dependencias
│
└── README.md                               # 📖 Este archivo
```

---

## 🚀 Instalación y Setup

### **1. Prerequisitos**

- Python 3.12+
- Docker Desktop
- MySQL Server (local o Docker)
- Git

### **2. Clonar el Repositorio**

```bash
git clone <tu-repositorio>
cd "Workshop 3"
```

### **3. Crear Entorno Virtual**

```powershell
# Windows PowerShell
python -m venv .kafka
.\.kafka\Scripts\Activate.ps1
```

### **4. Instalar Dependencias**

```powershell
pip install -r requirements_kafka.txt
pip install kafka-python-ng  # Importante: Para Python 3.12+
pip install mysql-connector-python
```

### **5. Levantar Kafka con Docker**

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

## 🎯 Ejecución del Sistema

### **PASO 1: Entrenar el Modelo** 🤖

```powershell
python model_utils.py
```

**Salida esperada:**
```
✅ Modelo entrenado y guardado exitosamente en: modelo_regresion_lineal.pkl

📊 MÉTRICAS DEL MODELO:
   R² Score: 0.7519
   MAE:      0.4321
   RMSE:     0.5566
   MAPE:     8.68%
```

---

### **PASO 2: Iniciar el Consumidor Kafka** 📥

**En una nueva terminal:**

```powershell
python kafka_consumer.py
```

**Salida esperada:**
```
✅ Conectado a MySQL: happiness_db
✅ Tabla 'predictions' verificada
✅ Modelo cargado exitosamente
🎯 Consumer iniciado. Esperando mensajes...
```

---

### **PASO 3: Ejecutar el Productor Kafka** 📤

**En otra terminal:**

```powershell
python kafka_producer.py
```

**Salida esperada:**
```
🔄 Iniciando Kafka Producer...
✅ Conectado a Kafka: localhost:9092
📤 Enviando mensajes...
   [1/100] Finland 2015 ✅
   [2/100] Denmark 2015 ✅
   ...
✅ Transmisión completada: 100 mensajes enviados
```

---

### **PASO 4: Verificar Datos en MySQL** 💾

```sql
USE happiness_db;

-- Ver total de predicciones
SELECT COUNT(*) FROM predictions;

-- Ver primeras 10 predicciones
SELECT country, year, actual_score, predicted_score, prediction_error 
FROM predictions 
LIMIT 10;

-- Países con mayor error de predicción
SELECT country, year, actual_score, predicted_score, 
       ABS(prediction_error) as error_absoluto
FROM predictions
ORDER BY error_absoluto DESC
LIMIT 10;
```

---

### **PASO 5: Análisis y Visualizaciones** 📊

Abre el notebook:

```powershell
jupyter notebook Evaluacion_Streaming_Kafka.ipynb
```

Este notebook genera:
- ✅ Gráficos de predicciones vs valores reales
- ✅ Distribución de errores
- ✅ Top 10 países con mejor Happiness Score
- ✅ Evolución temporal
- ✅ Performance del streaming

---

## 📊 Resultados del Modelo

### **Métricas de Evaluación**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R²** | 0.7519 | El modelo explica el 75.19% de la variabilidad |
| **MAE** | 0.4321 | Error absoluto promedio de 0.43 puntos |
| **RMSE** | 0.5566 | Raíz del error cuadrático medio |
| **MAPE** | 8.68% | Error porcentual relativo bajo |

### **Features Utilizadas (6)**

1. **GDP per capita** - PIB per cápita
2. **Social support** - Soporte social
3. **Healthy life expectancy** - Esperanza de vida saludable
4. **Freedom to make life choices** - Libertad para elegir
5. **Generosity** - Generosidad
6. **Perceptions of corruption** - Percepción de corrupción

---

## 🔧 Decisiones Técnicas Clave

### **1. ¿Por qué Regresión Lineal?**
- ✅ Simplicidad e interpretabilidad
- ✅ R² de 75.19% es excelente para ciencias sociales
- ✅ Entrenamiento rápido (ideal para streaming)
- ✅ Relaciones lineales claras entre variables

### **2. ¿Por qué Kafka?**
- ✅ Streaming en tiempo real
- ✅ Escalabilidad horizontal
- ✅ Tolerancia a fallos
- ✅ Procesamiento asíncrono

### **3. ¿Por qué MySQL?**
- ✅ Persistencia de predicciones
- ✅ Queries SQL para análisis
- ✅ Compatibilidad con herramientas BI
- ✅ Índices para consultas rápidas

### **4. División 70-30**
- **70% Training** (546 registros)
- **30% Test** (235 registros)
- `random_state=42` para reproducibilidad

---

## 🐛 Troubleshooting

### **Error: Kafka no inicia**

```powershell
# Ver logs de Kafka
docker logs kafka

# Reiniciar contenedores
docker-compose down
docker-compose up -d
```

### **Error: MySQL connection refused**

```python
# Verificar credenciales en kafka_consumer.py
mysql_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'tu_password'  # ⚠️ Cambiar aquí
}
```

### **Error: ModuleNotFoundError kafka**

```powershell
# Instalar kafka-python-ng (no kafka-python)
pip install kafka-python-ng
```

### **Error: Puerto 3306 en uso**

```powershell
# Verificar MySQL local
netstat -ano | findstr :3306

# Detener servicio MySQL local si es necesario
net stop MySQL80
```

---

## 📈 Dashboard de KPIs

El sistema incluye un dashboard ejecutivo con visualizaciones consolidadas:

### Ejecución del Dashboard

```powershell
cd dashboard
python dashboard_kpis.py
```

### Visualizaciones Generadas

1. **dashboard_kpis_cards.png** - 8 tarjetas de KPIs principales
2. **dashboard_performance.png** - Dashboard consolidado con 5 gráficos

Ver `dashboard/README.md` para más detalles.

---

## 📦 Entregables

- ✅ **README.md** - Documentación completa
- ✅ **EDA_Happiness_Report.ipynb** - Análisis exploratorio
- ✅ **Modelos_Regresion_Happiness.ipynb** - Entrenamiento
- ✅ **Evaluacion_Streaming_Kafka.ipynb** - Evaluación
- ✅ **kafka_producer.py** - Código del productor
- ✅ **kafka_consumer.py** - Código del consumidor
- ✅ **modelo_regresion_lineal.pkl** - Modelo entrenado
- ✅ **predictions_streaming.csv** - Predicciones exportadas
- ✅ **metricas_resumen.csv** - KPIs del modelo
- ✅ **Visualizaciones PNG** - Gráficos del desempeño

---

## 🎓 Hallazgos del EDA

### **Correlaciones Principales**

- **GDP per capita** tiene la correlación más fuerte con Happiness Score (~0.78)
- **Social support** y **Healthy life expectancy** también muy correlacionadas
- **Generosity** tiene la correlación más débil
- **Corruption** tiene correlación negativa (más corrupción = menos felicidad)

### **Patrones Encontrados**

1. Países nórdicos (Finlandia, Dinamarca, Noruega) consistentemente en top 10
2. GDP alto no garantiza felicidad, pero ayuda significativamente
3. Soporte social es crítico incluso en países con GDP bajo
4. La esperanza de vida saludable es más importante que la expectativa total

---

## 👥 Autor

**Juan A.**  
Workshop 3 - ETL con Kafka y Machine Learning

---

## 📄 Licencia

Este proyecto es parte de un workshop educativo.

---

## 🔗 Referencias

- [World Happiness Report](https://worldhappiness.report/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)

---

## 🚀 Próximos Pasos

- [ ] Implementar más modelos (Random Forest, XGBoost)
- [ ] Dashboard en tiempo real con Streamlit
- [ ] Deployment en la nube (AWS, Azure)
- [ ] CI/CD con GitHub Actions
- [ ] Monitoreo con Prometheus + Grafana

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**
