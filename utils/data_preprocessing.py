"""
Utilitaires pour la génération et le préprocessing des données énergétiques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def generate_energy_data(n_days=365, anomaly_rate=0.05):
    """
    Génère des données synthétiques de consommation énergétique
    
    Args:
        n_days (int): Nombre de jours à générer
        anomaly_rate (float): Taux d'anomalies à introduire
    
    Returns:
        pd.DataFrame: DataFrame avec les données générées
    """
    # Génération des dates
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days*24, freq='H')
    
    # Initialisation du DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    # Ajout de la saison
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Hiver'
        elif month in [3, 4, 5]:
            return 'Printemps'
        elif month in [6, 7, 8]:
            return 'Été'
        else:
            return 'Automne'
    
    df['season'] = df['month'].apply(get_season)
    
    # Génération de la température (influencée par la saison et l'heure)
    base_temp = {
        'Hiver': 5, 'Printemps': 15, 'Été': 25, 'Automne': 12
    }
    
    df['temperature'] = df['season'].map(base_temp)
    df['temperature'] += np.sin(2 * np.pi * df['hour'] / 24) * 5  # Variation journalière
    df['temperature'] += np.random.normal(0, 3, len(df))  # Bruit
    
    # Génération de l'humidité
    df['humidity'] = 60 + np.sin(2 * np.pi * df['day_of_year'] / 365) * 20
    df['humidity'] += np.random.normal(0, 10, len(df))
    df['humidity'] = np.clip(df['humidity'], 20, 95)
    
    # Génération de la consommation énergétique de base
    # Pattern journalier: plus élevé le matin et le soir
    hourly_pattern = np.array([
        0.6, 0.5, 0.4, 0.4, 0.5, 0.7, 1.0, 1.2,  # 0-7h
        1.1, 0.9, 0.8, 0.8, 0.9, 0.9, 0.8, 0.9,  # 8-15h  
        1.0, 1.2, 1.4, 1.3, 1.1, 1.0, 0.9, 0.7   # 16-23h
    ])
    
    df['base_consumption'] = df['hour'].map(lambda h: hourly_pattern[h])
    
    # Influence de la température (climatisation/chauffage)
    df['temp_effect'] = np.where(
        df['temperature'] > 22, 
        (df['temperature'] - 22) * 0.05,  # Climatisation
        np.where(df['temperature'] < 18, 
                (18 - df['temperature']) * 0.04, 0)  # Chauffage
    )
    
    # Effet weekend (consommation plus faible)
    df['weekend_effect'] = np.where(df['is_weekend'], -0.2, 0)
    
    # Effet saisonnier
    seasonal_effect = {
        'Hiver': 0.3, 'Printemps': 0, 'Été': 0.2, 'Automne': 0.1
    }
    df['seasonal_effect'] = df['season'].map(seasonal_effect)
    
    # Calcul de la consommation finale
    df['energy_consumption'] = (
        (df['base_consumption'] + 
         df['temp_effect'] + 
         df['weekend_effect'] + 
         df['seasonal_effect']) * 100  # Échelle en kWh
    )
    
    # Ajout de bruit réaliste
    df['energy_consumption'] += np.random.normal(0, 5, len(df))
    df['energy_consumption'] = np.maximum(df['energy_consumption'], 10)  # Min 10 kWh
    
    # Introduction d'anomalies contrôlées
    n_anomalies = int(len(df) * anomaly_rate)
    anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
    
    df['is_anomaly'] = False
    df.loc[anomaly_indices, 'is_anomaly'] = True
    
    # Types d'anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'drop', 'sustained_high'], 
                                       p=[0.4, 0.3, 0.3])
        
        if anomaly_type == 'spike':
            # Pic de consommation (2-5x normal)
            multiplier = np.random.uniform(2, 5)
            df.loc[idx, 'energy_consumption'] *= multiplier
            
        elif anomaly_type == 'drop':
            # Chute de consommation (10-30% du normal)
            multiplier = np.random.uniform(0.1, 0.3)
            df.loc[idx, 'energy_consumption'] *= multiplier
            
        elif anomaly_type == 'sustained_high':
            # Consommation élevée maintenue (1.5-2x sur plusieurs heures)
            duration = np.random.randint(2, 6)
            multiplier = np.random.uniform(1.5, 2)
            end_idx = min(idx + duration, len(df))
            df.loc[idx:end_idx, 'energy_consumption'] *= multiplier
            df.loc[idx:end_idx, 'is_anomaly'] = True
    
    # Nettoyage et sélection des colonnes finales
    final_columns = [
        'timestamp', 'energy_consumption', 'temperature', 'humidity',
        'day_of_week', 'hour', 'is_weekend', 'season', 'is_anomaly'
    ]
    
    return df[final_columns].round(2)

def preprocess_data(df, scale_features=True):
    """
    Préprocesse les données pour l'entraînement des modèles
    
    Args:
        df (pd.DataFrame): DataFrame avec les données brutes
        scale_features (bool): Si True, normalise les features numériques
    
    Returns:
        tuple: (X_scaled, y, scaler, feature_names)
    """
    # Copie pour éviter de modifier l'original
    data = df.copy()
    
    # Feature engineering
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Encoding des variables catégorielles
    season_dummies = pd.get_dummies(data['season'], prefix='season')
    data = pd.concat([data, season_dummies], axis=1)
    
    # Sélection des features
    feature_columns = [
        'energy_consumption', 'temperature', 'humidity', 'hour', 
        'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 
        'day_sin', 'day_cos'
    ] + list(season_dummies.columns)
    
    X = data[feature_columns]
    y = data['is_anomaly'].astype(int)
    
    # Normalisation si demandée
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = X
    
    return X_scaled, y, scaler, feature_columns

def create_time_features(df):
    """
    Crée des features temporelles avancées
    """
    df = df.copy()
    
    # Features temporelles circulaires
    df['minute_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.minute / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.minute / 60)
    
    # Features de tendance
    df['energy_lag_1'] = df['energy_consumption'].shift(1)
    df['energy_lag_24'] = df['energy_consumption'].shift(24)  # Même heure hier
    df['energy_rolling_mean_24'] = df['energy_consumption'].rolling(24).mean()
    df['energy_rolling_std_24'] = df['energy_consumption'].rolling(24).std()
    
    # Ratio par rapport à la moyenne mobile
    df['energy_ratio_mean'] = df['energy_consumption'] / df['energy_rolling_mean_24']
    
    return df

def detect_and_handle_outliers(df, columns, method='iqr', factor=1.5):
    """
    Détecte et traite les outliers dans les données
    
    Args:
        df (pd.DataFrame): DataFrame
        columns (list): Colonnes à analyser
        method (str): 'iqr' ou 'zscore'
        factor (float): Facteur de seuil
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Marquage des outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > factor
        
        # Traitement par cap aux percentiles
        df_clean.loc[outliers, col] = np.where(
            df_clean.loc[outliers, col] > df[col].quantile(0.95),
            df[col].quantile(0.95),
            df[col].quantile(0.05)
        )
    
    return df_clean

if __name__ == "__main__":
    # Test des fonctions
    print("🔄 Génération des données de test...")
    df = generate_energy_data(n_days=30, anomaly_rate=0.05)
    print(f"✅ Données générées: {len(df)} lignes")
    print(f"📊 Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.2%})")
    
    # Préprocessing
    X, y, scaler, features = preprocess_data(df)
    print(f"🔧 Features préparées: {len(features)} colonnes")
    
    print("\n📈 Aperçu des données:")
    print(df.head())
