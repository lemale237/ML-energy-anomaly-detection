"""
Utilitaires pour la visualisation des données énergétiques et des résultats
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_energy_consumption_overview(df, save_path=None):
    """
    Vue d'ensemble de la consommation énergétique
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Vue d\'ensemble de la Consommation Énergétique', fontsize=16, fontweight='bold')
    
    # 1. Série temporelle complète
    axes[0, 0].plot(df['timestamp'], df['energy_consumption'], alpha=0.7, linewidth=0.8)
    anomalies = df[df['is_anomaly'] == True]
    if len(anomalies) > 0:
        axes[0, 0].scatter(anomalies['timestamp'], anomalies['energy_consumption'], 
                          color='red', s=30, alpha=0.8, label='Anomalies')
    axes[0, 0].set_title('Évolution Temporelle de la Consommation')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Consommation (kWh)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution de la consommation
    axes[0, 1].hist(df['energy_consumption'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['energy_consumption'].mean(), color='red', linestyle='--', 
                      label=f'Moyenne: {df["energy_consumption"].mean():.1f} kWh')
    axes[0, 1].set_title('Distribution de la Consommation')
    axes[0, 1].set_xlabel('Consommation (kWh)')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Pattern hebdomadaire
    df['day_name'] = df['timestamp'].dt.day_name()
    daily_avg = df.groupby('day_name')['energy_consumption'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = daily_avg.reindex(day_order)
    
    bars = axes[1, 0].bar(range(len(daily_avg)), daily_avg.values, alpha=0.8)
    axes[1, 0].set_title('Consommation Moyenne par Jour de la Semaine')
    axes[1, 0].set_xlabel('Jour')
    axes[1, 0].set_ylabel('Consommation Moyenne (kWh)')
    axes[1, 0].set_xticks(range(len(day_order)))
    axes[1, 0].set_xticklabels([day[:3] for day in day_order], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Coloration différente pour les weekends
    for i, bar in enumerate(bars):
        if i >= 5:  # Weekend
            bar.set_color('lightcoral')
        else:
            bar.set_color('skyblue')
    
    # 4. Pattern horaire
    hourly_avg = df.groupby('hour')['energy_consumption'].mean()
    axes[1, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
    axes[1, 1].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
    axes[1, 1].set_title('Consommation Moyenne par Heure')
    axes[1, 1].set_xlabel('Heure')
    axes[1, 1].set_ylabel('Consommation Moyenne (kWh)')
    axes[1, 1].set_xticks(range(0, 24, 3))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    """
    Matrice de corrélation des variables
    """
    # Sélection des variables numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    
    plt.title('Matrice de Corrélation des Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_seasonal_analysis(df, save_path=None):
    """
    Analyse saisonnière de la consommation
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Analyse Saisonnière de la Consommation', fontsize=16, fontweight='bold')
    
    # 1. Consommation par saison
    seasonal_stats = df.groupby('season')['energy_consumption'].agg(['mean', 'std', 'median'])
    
    x_pos = range(len(seasonal_stats))
    axes[0, 0].bar(x_pos, seasonal_stats['mean'], yerr=seasonal_stats['std'], 
                   capsize=5, alpha=0.8, color=['lightblue', 'lightgreen', 'gold', 'orange'])
    axes[0, 0].set_title('Consommation Moyenne par Saison')
    axes[0, 0].set_xlabel('Saison')
    axes[0, 0].set_ylabel('Consommation (kWh)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(seasonal_stats.index)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Boxplot par saison
    sns.boxplot(data=df, x='season', y='energy_consumption', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution par Saison')
    axes[0, 1].set_xlabel('Saison')
    axes[0, 1].set_ylabel('Consommation (kWh)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Relation température-consommation
    scatter = axes[1, 0].scatter(df['temperature'], df['energy_consumption'], 
                                alpha=0.6, c=df['hour'], cmap='viridis')
    axes[1, 0].set_title('Consommation vs Température')
    axes[1, 0].set_xlabel('Température (°C)')
    axes[1, 0].set_ylabel('Consommation (kWh)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Heure')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Anomalies par saison
    anomaly_by_season = df.groupby('season')['is_anomaly'].agg(['sum', 'count'])
    anomaly_by_season['rate'] = anomaly_by_season['sum'] / anomaly_by_season['count'] * 100
    
    bars = axes[1, 1].bar(anomaly_by_season.index, anomaly_by_season['rate'], 
                         alpha=0.8, color=['lightcoral', 'lightpink', 'salmon', 'indianred'])
    axes[1, 1].set_title('Taux d\'Anomalies par Saison')
    axes[1, 1].set_xlabel('Saison')
    axes[1, 1].set_ylabel('Taux d\'Anomalies (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Ajout des valeurs sur les barres
    for bar, rate in zip(bars, anomaly_by_season['rate']):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_interactive_time_series(df):
    """
    Crée un graphique interactif de la série temporelle
    """
    fig = go.Figure()
    
    # Ligne principale de consommation
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['energy_consumption'],
        mode='lines',
        name='Consommation',
        line=dict(color='blue', width=1),
        hovertemplate='<b>%{x}</b><br>Consommation: %{y:.1f} kWh<extra></extra>'
    ))
    
    # Points d'anomalies
    anomalies = df[df['is_anomaly'] == True]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['energy_consumption'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='diamond'),
            hovertemplate='<b>ANOMALIE</b><br>%{x}<br>Consommation: %{y:.1f} kWh<extra></extra>'
        ))
    
    fig.update_layout(
        title='Évolution de la Consommation Énergétique',
        xaxis_title='Date',
        yaxis_title='Consommation (kWh)',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig

def create_anomaly_detection_dashboard(df, predictions_dict):
    """
    Crée un dashboard interactif pour la détection d'anomalies
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Série Temporelle avec Anomalies', 'Distribution des Scores',
                       'Matrice de Confusion', 'Performance des Modèles'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "heatmap"}, {"type": "bar"}]]
    )
    
    # 1. Série temporelle
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['energy_consumption'],
                  mode='lines', name='Consommation', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 2. Distribution des scores (exemple avec un modèle)
    if 'isolation_forest' in predictions_dict:
        scores = predictions_dict['isolation_forest']['scores']
        fig.add_trace(
            go.Histogram(x=scores, name='Scores Anomalies', 
                        marker_color='orange', opacity=0.7),
            row=1, col=2
        )
    
    # 3. Performance comparative (exemple)
    models = list(predictions_dict.keys())
    f1_scores = [predictions_dict[model]['f1_score'] for model in models]
    
    fig.add_trace(
        go.Bar(x=models, y=f1_scores, name='F1-Score',
               marker_color=['skyblue', 'lightgreen', 'gold', 'coral'][:len(models)]),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True,
                     title_text="Dashboard de Détection d'Anomalies")
    
    return fig

def plot_model_predictions_comparison(df, predictions_dict, save_path=None):
    """
    Compare les prédictions de différents modèles
    """
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 4*n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, results) in enumerate(predictions_dict.items()):
        ax = axes[i]
        
        # Série temporelle de base
        ax.plot(df['timestamp'], df['energy_consumption'], 
               alpha=0.7, color='blue', label='Consommation')
        
        # Vraies anomalies
        true_anomalies = df[df['is_anomaly'] == True]
        ax.scatter(true_anomalies['timestamp'], true_anomalies['energy_consumption'],
                  color='red', s=50, alpha=0.8, label='Vraies Anomalies', marker='o')
        
        # Prédictions du modèle
        predicted_indices = np.where(results['predictions'] == 1)[0]
        if len(predicted_indices) > 0:
            pred_times = df.iloc[predicted_indices]['timestamp']
            pred_values = df.iloc[predicted_indices]['energy_consumption']
            ax.scatter(pred_times, pred_values, color='orange', s=30, 
                      alpha=0.8, label='Prédictions', marker='x')
        
        ax.set_title(f'{model_name.replace("_", " ").title()} - '
                    f'F1: {results["f1_score"]:.3f}, AUC: {results["auc"]:.3f}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consommation (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_feature_importance_plot(model, feature_names, save_path=None):
    """
    Visualise l'importance des features (pour les modèles qui le supportent)
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Importance des Features')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print("❌ Le modèle ne supporte pas l'importance des features")

def plot_anomaly_timeline(df, save_path=None):
    """
    Timeline des anomalies détectées
    """
    anomalies = df[df['is_anomaly'] == True].copy()
    
    if len(anomalies) == 0:
        print("❌ Aucune anomalie à afficher")
        return
    
    # Groupement par jour
    anomalies['date'] = anomalies['timestamp'].dt.date
    daily_anomalies = anomalies.groupby('date').size()
    
    plt.figure(figsize=(15, 6))
    plt.plot(daily_anomalies.index, daily_anomalies.values, 
             marker='o', linewidth=2, markersize=6)
    plt.fill_between(daily_anomalies.index, daily_anomalies.values, alpha=0.3)
    
    plt.title('Timeline des Anomalies Détectées', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Nombre d\'Anomalies')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Test des fonctions de visualisation
    print("🎨 Test des utilitaires de visualisation...")
    
    # Génération de données de test
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    test_df = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': np.random.normal(100, 20, 100),
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.normal(60, 10, 100),
        'hour': dates.hour,
        'season': np.random.choice(['Hiver', 'Printemps', 'Été', 'Automne'], 100),
        'is_anomaly': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    })
    
    print("✅ Données de test générées")
    print("✅ Utilitaires de visualisation prêts")
