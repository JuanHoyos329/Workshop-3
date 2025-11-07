# World Happiness Report - Real-Time Streaming System with Kafka and Machine Learning

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Kafka](https://img.shields.io/badge/Apache_Kafka-7.5.0-orange.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Latest-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)

## Project Description

Real-time streaming system that integrates Apache Kafka, Machine Learning, and MySQL to predict country Happiness Scores based on World Happiness Report data (2015-2019).

### Objective
Implement a complete ETL pipeline with data streaming, real-time prediction using Multiple Linear Regression with One-Hot Encoding to capture country-specific patterns, and persistent storage with interactive KPI visualization.

---

## System Architecture

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

## Project Structure

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

## Installation and Setup

### 1. Prerequisites

- Python 3.12+
- Docker Desktop
- MySQL Server (local or Docker)
- Git

### 2. Clone Repository

```bash
git clone https://github.com/JuanHoyos329/Workshop-3.git
cd "Workshop 3"
```

### 3. Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv .kafka
.\.kafka\Scripts\Activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Start Kafka with Docker

```powershell
docker-compose up -d
```

Verify containers are running:

```powershell
docker ps
```

You should see:
- `zookeeper` (port 2181)
- `kafka` (port 9092)

---

## System Execution

### STEP 1: Train the Model

```powershell
cd model_regresion
python model_utils.py
```

Expected output:
- Model trained with One-Hot Encoding for Country
- `modelo_regresion_lineal.pkl` file generated (contains model + preprocessor)
- Evaluation metrics displayed

### STEP 2: Start Kafka Consumer

In a new terminal:

```powershell
cd kafka
python kafka_consumer.py
```


### STEP 3: Run Kafka Producer

In another terminal:

```powershell
cd kafka
python kafka_producer.py
```

The producer:
- Extracts data from CSV files (2015-2019)
- Splits into train/test (70-30) with country stratification
- Sends records to Kafka topic `happiness-data`

### STEP 4: Generate KPI Dashboard

```powershell
cd kpis
python generar_kpis.py
```

Open the generated `dashboard_kpis.html` file in your browser. The dashboard includes:
- Performance metrics (R², MAE, RMSE, MAPE)
- Train vs Test comparison
- Interactive world map with filters (Actual/Train/Test)
- Top 10 happiest countries
- Temporal evolution (fixed scale 5-6)
- Error distribution
- Regional analysis

---

## Machine Learning Model

### Architecture: Multiple Linear Regression with One-Hot Encoding

The model uses **Multiple Linear Regression** to predict Happiness Score based on 6 numeric features and the categorical variable `Country`.

### Feature Selection and Extraction

#### Numeric Features (6)

Numeric features come directly from the World Happiness Report and represent key factors influencing a country's happiness:

1. **GDP per capita** - Normalized GDP per capita
2. **Social support** - Social support (family and community support network)
3. **Healthy life expectancy** - Healthy life expectancy
4. **Freedom to make life choices** - Freedom to make life choices
5. **Generosity** - Generosity (donations and mutual aid)
6. **Perceptions of corruption** - Perception of corruption (governmental and business)

**Extraction process:**
- CSV files from different years have inconsistent column names
- A mapping dictionary (`COLUMN_MAPPINGS`) was used to normalize columns by year
- Example: `'Economy (GDP per Capita)'` (2015) → `'GDP per capita'` (standardized)
- Records with null values in critical features were removed

#### Categorical Feature: Country with One-Hot Encoding

**Initial problem:** Linear regressions only accept numeric values, but `Country` is a categorical text variable (e.g., "Finland", "Somalia", "Denmark").

**Implemented solution: One-Hot Encoding**

One-Hot Encoding converts the categorical variable `Country` into multiple binary columns (0 or 1), one for each unique country:

```python
Original:
Country
--------
Finland
Denmark
Somalia

After One-Hot Encoding (drop='first'):
Denmark  Somalia  (Finland is removed as reference)
-------  -------
0        0        → Finland (base category)
1        0        → Denmark
0        1        → Somalia
```

**Technical implementation:**

```python
# Preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', FEATURE_COLUMNS),  # 6 numeric features unchanged
        ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                             handle_unknown='ignore'), CATEGORICAL_COLUMNS)  # Country → dummies
    ])
```

**Key OneHotEncoder parameters:**
- `drop='first'`: Removes the first category (country) to avoid perfect multicollinearity
  - If there are 159 countries, it generates 158 dummy variables
  - The removed country serves as the reference category (baseline)
- `sparse_output=False`: Returns dense matrix instead of sparse for scikit-learn compatibility
- `handle_unknown='ignore'`: If a new country appears in production that wasn't seen during training, it's encoded as zeros (like the reference category)

**Benefits of including Country:**
1. **Captures country-specific patterns** not explained by the 6 numeric features alone
2. **Significant R² improvement**: From ~0.75 (without Country) to ~0.95 (with Country)
3. **Reduces systematic errors**: Countries with similar cultural/historical characteristics have specific effects
4. **Enables adjusted predictions**: The model "learns" that Finland tends to be happier even when controlling for GDP

**Model interpretation:**
- Each country has a **specific coefficient** that adjusts the base prediction
- Example: If Finland has a coefficient of +0.5, being Finland adds 0.5 points to the Happiness Score after considering GDP, Social Support, etc.

**Result:** A single `.pkl` file containing both the trained model and configured preprocessor, ensuring that transformations in production are identical to those in training.

### Evaluation Metrics

The model is evaluated **exclusively on the test set (30%)** to avoid overfitting:

- **R² Score:** ~0.9548 (95.48% of variance explained)
- **MAE:** ~0.2471 (average error of 0.25 points on 0-10 scale)
- **RMSE:** ~0.3221 (penalizes large errors)
- **MAPE:** ~4.2% (average percentage error) 

---

## Key Technical Decisions

### 1. Why Linear Regression with One-Hot Encoding?
- **Interpretability:** Each coefficient represents the direct impact of each feature
- **Performance:** R² of ~95% with Country inclusion as categorical variable
- **Speed:** Instantaneous training (~1 second), ideal for streaming
- **Captures specific effects:** One-Hot Encoding allows the model to learn unique patterns per country
- **Simplicity:** Doesn't require complex hyperparameters or additional regularization

### 2. Why Kafka?
- **Real-time streaming:** Processes data as it arrives without batch processing
- **Horizontal scalability:** Allows multiple consumers for load distribution
- **Fault tolerance:** Message persistence on disk
- **Decoupling:** Producer and Consumer operate independently

### 3. Why MySQL?
- **Reliable persistence:** Structured storage of predictions with indexes
- **Analytical queries:** SQL enables complex aggregations for KPIs
- **Data integrity:** ACID transactions and database constraints
- **BI compatibility:** Easy connection with visualization tools

### 4. .pkl File Architecture
- **Single file contains model + preprocessor**
- **Structure:** `{'modelo': LinearRegression, 'preprocessor': ColumnTransformer}`
- **Advantage:** Guarantees that transformations in production are identical to training
- **Deployment:** Single file facilitates distribution and model versioning

---

## KPI Dashboard

The system includes an interactive HTML dashboard with consolidated visualizations:

### Dashboard Features

- **Main metrics:** R², MAE, RMSE, MAPE, Records, Countries, Years
- **Comparative table:** Train vs Test
- **Interactive world map:** Filters for Actual/Train/Test
- **Scatter plots:** Predictions vs Actual with filters
- **Top 10 countries:** Grouped bar comparison
- **Temporal evolution:** Fixed scale
- **Regional analysis:** Performance by geographic area
- **Error distribution:** Train vs Test histograms

### Technologies

- Plotly for interactive charts
- HTML/CSS/JavaScript for interface
- Pandas for data processing
- MySQL as data source

## Exploratory Analysis Findings

### Correlations with Happiness Score

1. **Nordic countries dominate:** Finland, Denmark, Norway consistently in top 10
2. **GDP is important but not sufficient:** Strong correlation (0.78) but countries with similar GDP have different scores
3. **Significant Country effect:** Including Country as categorical variable improves R² from ~0.75 to ~0.95
4. **Social support is critical:** Second strongest correlation, essential even in low GDP countries
5. **Corruption impacts negatively:** Only strong negative correlation with happiness

---

## Technologies Used

- **Python 3.12+:** Main language
- **Apache Kafka 7.5.0:** Data streaming
- **MySQL 8.0+:** Database
- **Scikit-Learn:** Machine Learning (LinearRegression, OneHotEncoder, ColumnTransformer)
- **Pandas & NumPy:** Data processing
- **Plotly:** Interactive visualizations
- **Docker:** Kafka and Zookeeper containerization
- **Kafka-Python:** Kafka client for Python

## Author

Juan A. Hoyos  
Workshop 3 - ETL with Kafka and Machine Learning


## References

- [World Happiness Report](https://worldhappiness.report/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Plotly Documentation](https://plotly.com/python/)
