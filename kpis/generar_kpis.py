"""
Interactive KPI Generator - HTML Dashboard
Happiness Prediction System with Kafka
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KPIGenerator:
    """Interactive KPI Generator with HTML visualizations"""
    
    def __init__(self, mysql_config=None):
        self.mysql_config = mysql_config or self._default_mysql_config()
        self.df_predictions = None
        self.metrics_train = {}
        self.metrics_test = {}
        self.metrics_total = {}
        self.fixed_score_range = None # Added for fixed axis range
        
    def _default_mysql_config(self):
        return {
            'host': 'localhost',
            'port': 3306,
            'database': 'happiness_db',
            'user': 'root',
            'password': 'root'
        }
    
    def load_data_from_mysql(self):
        try:
            logger.info(" Connecting to MySQL...")
            conn = mysql.connector.connect(**self.mysql_config)
            
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
            
            self.df_predictions = pd.read_sql(query, conn)
            conn.close()
            
            if len(self.df_predictions) == 0:
                logger.error(" No data found in the predictions table.")
                logger.info(" Please run the Kafka producer and consumer first.")
                return False
            
            logger.info(f" Data loaded: {len(self.df_predictions)} records")
            logger.info(f"    - Train: {len(self.df_predictions[self.df_predictions['type_model']=='train'])}")
            logger.info(f"    - Test: {len(self.df_predictions[self.df_predictions['type_model']=='test'])}")
            return True
            
        except Error as e:
            logger.error(f" Error connecting to MySQL: {e}")
            return False
    
    def calculate_metrics(self, df, name="Total"):
        if len(df) == 0:
            return {}
        
        y_true = df['actual_score']
        y_pred = df['predicted_score']
        
        metrics = {
            'name': name,
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'total_records': len(df),
            'unique_countries': df['country'].nunique(),
            'years_covered': df['year'].nunique()
        }
        
        return metrics
    
    def calculate_all_metrics(self):
        logger.info(" Calculating metrics...")
        df_train = self.df_predictions[self.df_predictions['type_model'] == 'train']
        df_test = self.df_predictions[self.df_predictions['type_model'] == 'test']
        
        self.metrics_train = self.calculate_metrics(df_train, "Train")
        self.metrics_test = self.calculate_metrics(df_test, "Test")
        self.metrics_total = self.calculate_metrics(self.df_predictions, "Total")
        logger.info(" Metrics calculated successfully")
    
    def create_kpi_cards(self):
        """Creates KPI cards"""
        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('R² Score', 'MAE', 'RMSE', 'MAPE',
                          'Total Records', 'Unique Countries', 'Years Covered', 'Average Error')
        )
        
        m = self.metrics_total
        
        fig.add_trace(go.Indicator(mode="number", value=m['r2'], number={'valueformat': '.4f'}), row=1, col=1)
        fig.add_trace(go.Indicator(mode="number", value=m['mae'], number={'valueformat': '.4f'}), row=1, col=2)
        fig.add_trace(go.Indicator(mode="number", value=m['rmse'], number={'valueformat': '.4f'}), row=1, col=3)
        fig.add_trace(go.Indicator(mode="number", value=m['mape'], number={'valueformat': '.2f'}), row=1, col=4)
        fig.add_trace(go.Indicator(mode="number", value=m['total_records'], number={'valueformat': '.0f'}), row=2, col=1)
        fig.add_trace(go.Indicator(mode="number", value=m['unique_countries'], number={'valueformat': '.0f'}), row=2, col=2)
        fig.add_trace(go.Indicator(mode="number", value=m['years_covered'], number={'valueformat': '.0f'}), row=2, col=3)
        fig.add_trace(go.Indicator(mode="number", value=self.df_predictions['prediction_error'].abs().mean(), number={'valueformat': '.4f'}), row=2, col=4)
        
        fig.update_layout(height=500, showlegend=False)
        return fig
    
    def create_predictions_vs_actual(self, df_data=None, name='All', fixed_range=None):
        """
        Scatter plot of predictions.
        Modified to accept an optional fixed_range for axis limits.
        """
        df = df_data if df_data is not None else self.df_predictions
        
        fig = go.Figure()

        if name == 'All':
            df_train = df[df['type_model'] == 'train']
            df_test = df[df['type_model'] == 'test']
            
            fig.add_trace(go.Scatter(
                x=df_train['actual_score'], y=df_train['predicted_score'],
                mode='markers', name='Train',
                marker=dict(size=8, color='#2E86AB', opacity=0.6),
                text=df_train['country'],
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_test['actual_score'], y=df_test['predicted_score'],
                mode='markers', name='Test',
                marker=dict(size=8, color='#F18F01', opacity=0.6),
                text=df_test['country'],
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
            ))
        else: # For 'Train' or 'Test' filtered views
            color = '#2E86AB' if name == 'Train' else '#F18F01'
            fig.add_trace(go.Scatter(
                x=df['actual_score'], y=df['predicted_score'],
                mode='markers', name=name,
                marker=dict(size=8, color=color, opacity=0.6),
                text=df['country'],
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
            ))

        # Use fixed range if provided, otherwise calculate based on all data
        if fixed_range:
            min_val, max_val = fixed_range
        else:
            min_val = min(df['actual_score'].min(), df['predicted_score'].min())
            max_val = max(df['actual_score'].max(), df['predicted_score'].max())
            # Add a small buffer
            min_val -= 0.1
            max_val += 0.1
        
        # Perfect Prediction line
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        layout_updates = dict(
            xaxis_title='Actual Happiness Score',
            yaxis_title='Predicted Happiness Score',
            height=600,
            showlegend=True
        )

        # IMPORTANT: Set fixed ranges
        if fixed_range:
             # Apply fixed range calculated from ALL data to prevent resizing
            layout_updates['xaxis_range'] = [min_val, max_val]
            layout_updates['yaxis_range'] = [min_val, max_val]

        fig.update_layout(**layout_updates)
        return fig
    
    def create_top10_countries(self):
        """Top 10 Happiest Countries - Actual, Train, Test"""
        # Actual Data
        df_actual = self.df_predictions.groupby('country').agg({
            'actual_score': 'mean'
        }).reset_index()
        
        # Train
        df_train = self.df_predictions[self.df_predictions['type_model'] == 'train'].groupby('country').agg({
            'predicted_score': 'mean'
        }).reset_index()
        
        # Test
        df_test = self.df_predictions[self.df_predictions['type_model'] == 'test'].groupby('country').agg({
            'predicted_score': 'mean'
        }).reset_index()
        
        # Merge
        df_avg = df_actual.merge(df_train, on='country', how='left', suffixes=('', '_train'))
        df_avg = df_avg.merge(df_test, on='country', how='left', suffixes=('', '_test'))
        
        # Fill NaN values with 0 for countries with no test/train data (optional, but robust)
        df_avg['predicted_score'] = df_avg['predicted_score'].fillna(0)
        df_avg['predicted_score_test'] = df_avg['predicted_score_test'].fillna(0)
        
        # Top 10
        top10 = df_avg.nlargest(10, 'actual_score').sort_values('actual_score')
        
        fig = go.Figure()
        
        # Actual
        fig.add_trace(go.Bar(
            y=top10['country'], 
            x=top10['actual_score'], 
            name='Actual',
            orientation='h', 
            marker=dict(color='#6A994E')
        ))
        
        # Train
        fig.add_trace(go.Bar(
            y=top10['country'], 
            x=top10['predicted_score'], 
            name='Train Prediction',
            orientation='h', 
            marker=dict(color='#2E86AB')
        ))
        
        # Test
        fig.add_trace(go.Bar(
            y=top10['country'], 
            x=top10['predicted_score_test'], 
            name='Test Prediction',
            orientation='h', 
            marker=dict(color='#F18F01')
        ))
        
        fig.update_layout(
            xaxis_title='Happiness Score',
            barmode='group', 
            height=500,
            showlegend=True
        )
        return fig
    
    def create_time_series_evolution(self):
        """Temporal evolution by year - Only 3 lines: Actual, Train, Test"""
        # Actual Data (average of all)
        df_actual = self.df_predictions.groupby('year').agg({
            'actual_score': 'mean'
        }).reset_index()
        
        # Train
        df_train = self.df_predictions[self.df_predictions['type_model'] == 'train'].groupby('year').agg({
            'predicted_score': 'mean'
        }).reset_index()
        
        # Test
        df_test = self.df_predictions[self.df_predictions['type_model'] == 'test'].groupby('year').agg({
            'predicted_score': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        # Actual Line
        fig.add_trace(go.Scatter(
            x=df_actual['year'], 
            y=df_actual['actual_score'],
            mode='lines+markers', 
            name='Actual',
            line=dict(color='#6A994E', width=4),
            marker=dict(size=10)
        ))
        
        # Train Line
        fig.add_trace(go.Scatter(
            x=df_train['year'], 
            y=df_train['predicted_score'],
            mode='lines+markers', 
            name='Train Prediction',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        # Test Line
        fig.add_trace(go.Scatter(
            x=df_test['year'], 
            y=df_test['predicted_score'],
            mode='lines+markers', 
            name='Test Prediction',
            line=dict(color='#F18F01', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Average Happiness Score',
            height=500,
            legend=dict(x=0.02, y=0.98),
            showlegend=True
        )
        return fig
    
    def create_happiness_by_region(self):
        """Happiest and least happy regions - 3 bars: Actual, Train, Test"""
        # Actual Data
        df_actual = self.df_predictions.groupby('region').agg({
            'actual_score': 'mean'
        }).reset_index()
        
        # Train
        df_train = self.df_predictions[self.df_predictions['type_model'] == 'train'].groupby('region').agg({
            'predicted_score': 'mean'
        }).reset_index()
        
        # Test
        df_test = self.df_predictions[self.df_predictions['type_model'] == 'test'].groupby('region').agg({
            'predicted_score': 'mean'
        }).reset_index()
        
        # Merge all into one DataFrame
        df_region = df_actual.merge(df_train, on='region', how='left', suffixes=('', '_train'))
        df_region = df_region.merge(df_test, on='region', how='left', suffixes=('', '_test'))
        df_region = df_region.sort_values('actual_score', ascending=False)
        
        fig = go.Figure()
        
        # Actual Bar
        fig.add_trace(go.Bar(
            x=df_region['region'], 
            y=df_region['actual_score'],
            name='Actual', 
            marker=dict(color='#6A994E')
        ))
        
        # Train Bar
        fig.add_trace(go.Bar(
            x=df_region['region'], 
            y=df_region['predicted_score'],
            name='Train Prediction', 
            marker=dict(color='#2E86AB')
        ))
        
        # Test Bar
        fig.add_trace(go.Bar(
            x=df_region['region'], 
            y=df_region['predicted_score_test'],
            name='Test Prediction', 
            marker=dict(color='#F18F01')
        ))
        
        fig.update_layout(
            xaxis_title='Region',
            yaxis_title='Average Happiness Score',
            barmode='group', 
            height=500,
            xaxis_tickangle=-45,
            showlegend=True
        )
        return fig
    
    def create_error_distribution(self):
        """Error distribution"""
        fig = go.Figure()
        
        df_train = self.df_predictions[self.df_predictions['type_model'] == 'train']
        fig.add_trace(go.Histogram(x=df_train['prediction_error'], name='Train',
                                     marker=dict(color='#2E86AB', opacity=0.7), nbinsx=30))
        
        df_test = self.df_predictions[self.df_predictions['type_model'] == 'test']
        fig.add_trace(go.Histogram(x=df_test['prediction_error'], name='Test',
                                     marker=dict(color='#F18F01', opacity=0.7), nbinsx=30))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Error = 0")
        
        fig.update_layout(xaxis_title='Prediction Error',
                          yaxis_title='Frequency',
                          barmode='overlay', 
                          height=500,
                          showlegend=True)
        return fig
    
    def create_happiness_map(self):
        """Geographical map of happiness score by country"""
        # Group by country and get average
        df_map = self.df_predictions.groupby('country').agg({
            'actual_score': 'mean',
            'predicted_score': 'mean',
            'region': 'first'
        }).reset_index()
        
        # Create map with Choropleth
        fig = go.Figure(data=go.Choropleth(
            locations=df_map['country'],
            locationmode='country names',
            z=df_map['actual_score'],
            text=df_map['country'],
            colorscale='RdYlGn',  # Red-Yellow-Green
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title='Happiness<br>Score',
            hovertemplate='<b>%{text}</b><br>Happiness Score: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=600
        )
        return fig
    
    def create_metrics_table(self):
        """Comparative metrics table"""
        html_table = f"""
        <div style="margin: 20px;">
            <h2 style="text-align: center; color: #2E86AB;">Metrics Comparison: Train vs Test</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                <thead>
                    <tr style="background-color: #2E86AB; color: white;">
                        <th style="padding: 12px; border: 1px solid #ddd;">Metric</th>
                        <th style="padding: 12px; border: 1px solid #ddd;">Train</th>
                        <th style="padding: 12px; border: 1px solid #ddd;">Test</th>
                        <th style="padding: 12px; border: 1px solid #ddd;">Total</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background-color: #f9f9f9;">
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">R² Score</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_train.get('r2', 0):.4f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_test.get('r2', 0):.4f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_total.get('r2', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">MAE</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_train.get('mae', 0):.4f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_test.get('mae', 0):.4f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_total.get('mae', 0):.4f}</td>
                    </tr>
                    <tr style="background-color: #f9f9f9;">
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">RMSE</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_train.get('rmse', 0):.4f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_test.get('rmse', 0):.4f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_total.get('rmse', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">MAPE (%)</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_train.get('mape', 0):.2f}%</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_test.get('mape', 0):.2f}%</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_total.get('mape', 0):.2f}%</td>
                    </tr>
                    <tr style="background-color: #f9f9f9;">
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Records</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_train.get('total_records', 0)}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_test.get('total_records', 0)}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{self.metrics_total.get('total_records', 0)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        return html_table
    
    def create_filtered_visualizations(self, filter_type='all', fixed_range=None):
        """Creates filtered versions of the scatter plot visualization (Predictions vs Actuals)"""
        if filter_type == 'train':
            df = self.df_predictions[self.df_predictions['type_model'] == 'train']
        elif filter_type == 'test':
            df = self.df_predictions[self.df_predictions['type_model'] == 'test']
        else:
            df = self.df_predictions

        # Use the general method with filter specific data and name
        if filter_type == 'all':
            return self.create_predictions_vs_actual(df, 'All', fixed_range)
        else:
            return self.create_predictions_vs_actual(df, filter_type.title(), fixed_range)
    
    def create_visualizations_by_year(self, year):
        """Creates visualizations filtered by year"""
        df_year = self.df_predictions[self.df_predictions['year'] == year]
        
        # Top 10 by year
        df_actual = df_year.groupby('country').agg({'actual_score': 'mean'}).reset_index()
        df_train = df_year[df_year['type_model'] == 'train'].groupby('country').agg({'predicted_score': 'mean'}).reset_index()
        df_test = df_year[df_year['type_model'] == 'test'].groupby('country').agg({'predicted_score': 'mean'}).reset_index()
        
        df_avg = df_actual.merge(df_train, on='country', how='left', suffixes=('', '_train'))
        df_avg = df_avg.merge(df_test, on='country', how='left', suffixes=('', '_test'))
        top10 = df_avg.nlargest(10, 'actual_score').sort_values('actual_score')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top10['country'], x=top10['actual_score'], name='Actual', orientation='h', marker=dict(color='#6A994E')))
        fig.add_trace(go.Bar(y=top10['country'], x=top10['predicted_score'], name='Train Prediction', orientation='h', marker=dict(color='#2E86AB')))
        fig.add_trace(go.Bar(y=top10['country'], x=top10['predicted_score_test'], name='Test Prediction', orientation='h', marker=dict(color='#F18F01')))
        
        fig.update_layout(title=f'Top 10 Happiest Countries - Year {year}', xaxis_title='Happiness Score', barmode='group', height=500)
        return fig
    
    def create_visualizations_by_region(self, region):
        """Creates visualizations filtered by region"""
        df_region = self.df_predictions[self.df_predictions['region'] == region]
        
        # Top countries in the region
        df_actual = df_region.groupby('country').agg({'actual_score': 'mean'}).reset_index()
        df_train = df_region[df_region['type_model'] == 'train'].groupby('country').agg({'predicted_score': 'mean'}).reset_index()
        df_test = df_region[df_region['type_model'] == 'test'].groupby('country').agg({'predicted_score': 'mean'}).reset_index()
        
        df_avg = df_actual.merge(df_train, on='country', how='left', suffixes=('', '_train'))
        df_avg = df_avg.merge(df_test, on='country', how='left', suffixes=('', '_test'))
        df_avg = df_avg.sort_values('actual_score', ascending=False).head(10).sort_values('actual_score')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df_avg['country'], x=df_avg['actual_score'], name='Actual', orientation='h', marker=dict(color='#6A994E')))
        fig.add_trace(go.Bar(y=df_avg['country'], x=df_avg['predicted_score'], name='Train Prediction', orientation='h', marker=dict(color='#2E86AB')))
        fig.add_trace(go.Bar(y=df_avg['country'], x=df_avg['predicted_score_test'], name='Test Prediction', orientation='h', marker=dict(color='#F18F01')))
        
        fig.update_layout(title=f'Top Countries - {region}', xaxis_title='Happiness Score', barmode='group', height=500)
        return fig

    def generate_html_dashboard(self):
        """Generates the complete HTML dashboard"""
        logger.info("Generating HTML dashboard...")

        # Calculate fixed range for scatter plot once (to keep plot size constant)
        min_score = min(self.df_predictions['actual_score'].min(), self.df_predictions['predicted_score'].min())
        max_score = max(self.df_predictions['actual_score'].max(), self.df_predictions['predicted_score'].max())
        # Add a small buffer to the max/min values
        self.fixed_score_range = (min_score - 0.1, max_score + 0.1)
        
        # Create main visualizations
        fig_kpis = self.create_kpi_cards()
        fig_mapa = self.create_happiness_map()
        fig_top10 = self.create_top10_countries()
        fig_temporal = self.create_time_series_evolution()
        fig_regiones = self.create_happiness_by_region()
        fig_errores = self.create_error_distribution()
        tabla_metricas = self.create_metrics_table()
        
        # Create filtered versions of predictions vs actuals (using fixed range)
        fixed_range = self.fixed_score_range
        fig_pred_all = self.create_filtered_visualizations('all', fixed_range)
        fig_pred_train = self.create_filtered_visualizations('train', fixed_range)
        fig_pred_test = self.create_filtered_visualizations('test', fixed_range)
        
        # Create filtered versions by year
        years = sorted(self.df_predictions['year'].unique())
        figs_by_year = {}
        for year in years:
            figs_by_year[year] = self.create_visualizations_by_year(year)
        
        # Create filtered versions by region
        regions = sorted(self.df_predictions['region'].unique())
        figs_by_region = {}
        for region in regions:
            figs_by_region[region] = self.create_visualizations_by_region(region)
        
        # Convert figures to HTML using to_html()
        kpis_html = fig_kpis.to_html(full_html=False, include_plotlyjs='cdn')
        mapa_html = fig_mapa.to_html(full_html=False, include_plotlyjs=False)
        top10_html = fig_top10.to_html(full_html=False, include_plotlyjs=False)
        temporal_html = fig_temporal.to_html(full_html=False, include_plotlyjs=False)
        regiones_html = fig_regiones.to_html(full_html=False, include_plotlyjs=False)
        errores_html = fig_errores.to_html(full_html=False, include_plotlyjs=False)
        
        # Filtered versions
        pred_all_html = fig_pred_all.to_html(full_html=False, include_plotlyjs=False)
        pred_train_html = fig_pred_train.to_html(full_html=False, include_plotlyjs=False)
        pred_test_html = fig_pred_test.to_html(full_html=False, include_plotlyjs=False)
        
        # By year
        top10_by_year_html = {}
        for year in years:
            top10_by_year_html[year] = figs_by_year[year].to_html(full_html=False, include_plotlyjs=False)
        
        # By region
        top10_by_region_html = {}
        for region in regions:
            top10_by_region_html[region] = figs_by_region[region].to_html(full_html=False, include_plotlyjs=False)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KPI Dashboard - Happiness Prediction</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #1a1a1a;
            font-size: 2.2em;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            font-size: 1.1em;
            margin-bottom: 40px;
            font-weight: 400;
        }}
        .filters-panel {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 25px;
            margin: 30px 0;
        }}
        .filter-group {{
            margin-bottom: 20px;
        }}
        .filter-group:last-child {{
            margin-bottom: 0;
        }}
        .filter-label {{
            font-size: 1em;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            display: block;
        }}
        .btn {{
            padding: 10px 18px;
            margin: 4px;
            font-size: 0.95em;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            background: white;
            color: #495057;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }}
        .btn:hover {{
            background-color: #e9ecef;
        }}
        .btn.active {{
            background-color: #0066cc;
            color: white;
            border-color: #0066cc;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 25px;
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
        }}
        .section-title {{
            color: #1a1a1a;
            font-size: 1.4em;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #0066cc;
            font-weight: 600;
        }}
        .info-box {{
            background-color: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .info-box p {{
            margin: 5px 0;
            color: #333;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>KPI Dashboard - Happiness Prediction System</h1>
        <div class="subtitle">Streaming System with Apache Kafka</div>
        <div style="text-align: center; margin-bottom: 30px; color: #999; font-size: 0.9em;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">Key Performance Indicators</h2>
            {kpis_html}
        </div>
        
        {tabla_metricas}
        
        <div class="chart-container">
            <h2 class="section-title">World Happiness Map</h2>
            """ + mapa_html + """
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">Predictions</h2>
            
            <div style="margin-bottom: 20px;">
                <label class="filter-label">Data Type:</label>
                <button onclick="filterPredictions('all')" id="btnPredAll" class="btn active">All</button>
                <button onclick="filterPredictions('train')" id="btnPredTrain" class="btn">Train</button>
                <button onclick="filterPredictions('test')" id="btnPredTest" class="btn">Test</button>
            </div>
            
            <div id="predAll">""" + pred_all_html + """</div>
            <div id="predTrain" style="display: none;">""" + pred_train_html + """</div>
            <div id="predTest" style="display: none;">""" + pred_test_html + """</div>
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">Top 10 Happiest Countries</h2>
            """ + top10_html + """
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">Temporal Evolution of Happiness Score</h2>
""" + temporal_html + """
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">Happiness Score by Region</h2>
""" + regiones_html + """
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">Prediction Error Distribution</h2>
""" + errores_html + """
        </div>
    </div>
    
    <script>
        // Filter for Predictions
        function filterPredictions(type) {{
            // Hide all
            document.getElementById('predAll').style.display = 'none';
            document.getElementById('predTrain').style.display = 'none';
            document.getElementById('predTest').style.display = 'none';

            // Deactivate all buttons
            document.getElementById('btnPredAll').classList.remove('active');
            document.getElementById('btnPredTrain').classList.remove('active');
            document.getElementById('btnPredTest').classList.remove('active');

            // Show selected and activate button
            document.getElementById('pred' + type.charAt(0).toUpperCase() + type.slice(1)).style.display = 'block';
            document.getElementById('btnPred' + type.charAt(0).toUpperCase() + type.slice(1)).classList.add('active');
        }}
    </script>
</body>
</html>
        """
        
        # Save HTML in the kpis folder
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'dashboard_kpis.html')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"✅ Dashboard HTML generated successfully at {file_path}")
        return file_path

if __name__ == '__main__':
    # Usage example
    kpi_generator = KPIGenerator()
    if kpi_generator.load_data_from_mysql():
        kpi_generator.calculate_all_metrics()
        kpi_generator.generate_html_dashboard()