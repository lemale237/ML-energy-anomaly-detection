"""
Script pour générer les données synthétiques au format LEAD (Large-scale Energy Anomaly Detection)
Adapté pour correspondre au format Kaggle
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

def generate_lead_format_data(n_buildings=50, n_days=365, anomaly_rate=0.05, random_state=42):
    """
    Génère des données synthétiques au format LEAD pour la détection d'anomalies énergétiques
    
    Args:
        n_buildings (int): Nombre de bâtiments à simuler
        n_days (int): Nombre de jours à générer par bâtiment
        anomaly_rate (float): Taux d'anomalies global à introduire
        random_state (int): Seed pour la reproductibilité
    
    Returns:
        pd.DataFrame: DataFrame au format LEAD (building_id, timestamp, meter_reading, anomaly)
    """
    np.random.seed(random_state)
    
    # Génération des dates (lectures horaires)
    start_date = datetime(2023, 1, 1)
    hours_total = n_days * 24
    
    all_data = []
    
    for building_id in range(1, n_buildings + 1):
        print(f"Génération des données pour le bâtiment {building_id}/{n_buildings}")
        
        # Dates pour ce bâtiment
        dates = pd.date_range(start=start_date, periods=hours_total, freq='H')
        
        # Caractéristiques du bâtiment (chaque bâtiment a ses propres patterns)
        building_type = np.random.choice(['office', 'residential', 'commercial', 'industrial'], 
                                       p=[0.4, 0.3, 0.2, 0.1])
        base_consumption = {
            'office': 150, 'residential': 80, 'commercial': 200, 'industrial': 300
        }[building_type]
        
        # DataFrame temporaire pour ce bâtiment
        building_df = pd.DataFrame({
            'building_id': building_id,
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
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        building_df['season'] = building_df['month'].apply(get_season)
        
        # Génération de la température (varie selon la saison et l'heure)
        base_temp = {'Winter': 5, 'Spring': 15, 'Summer': 25, 'Fall': 12}
        building_df['temperature'] = building_df['season'].map(base_temp)
        building_df['temperature'] += np.sin(2 * np.pi * building_df['hour'] / 24) * 5
        building_df['temperature'] += np.random.normal(0, 3, len(building_df))
        
        # Pattern de consommation selon le type de bâtiment
        if building_type == 'office':
            # Bureaux: pic en journée, faible la nuit et weekend
            hourly_pattern = np.array([
                0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0,  # 0-7h
                1.2, 1.1, 1.0, 1.0, 1.1, 1.1, 1.0, 1.1,  # 8-15h  
                1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3   # 16-23h
            ])
        elif building_type == 'residential':
            # Résidentiel: pics matin/soir, plus élevé weekend
            hourly_pattern = np.array([
                0.6, 0.5, 0.4, 0.4, 0.5, 0.7, 1.0, 1.2,  # 0-7h
                0.8, 0.6, 0.5, 0.6, 0.7, 0.7, 0.6, 0.7,  # 8-15h  
                0.9, 1.1, 1.3, 1.2, 1.0, 0.9, 0.8, 0.7   # 16-23h
            ])
        elif building_type == 'commercial':
            # Commercial: ouvert tard, fermé tôt le matin
            hourly_pattern = np.array([
                0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7,  # 0-7h
                0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0,  # 8-15h  
                1.1, 1.2, 1.3, 1.2, 1.0, 0.8, 0.5, 0.3   # 16-23h
            ])
        else:  # industrial
            # Industriel: plus constant, légère baisse la nuit
            hourly_pattern = np.array([
                0.8, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1,  # 0-7h
                1.2, 1.1, 1.0, 1.0, 1.1, 1.1, 1.0, 1.1,  # 8-15h  
                1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8   # 16-23h
            ])
        
        building_df['hourly_factor'] = building_df['hour'].map(lambda h: hourly_pattern[h])
        
        # Effet de la température (climatisation/chauffage)
        building_df['temp_effect'] = np.where(
            building_df['temperature'] > 22, 
            (building_df['temperature'] - 22) * 0.03,  # Climatisation
            np.where(building_df['temperature'] < 18, 
                    (18 - building_df['temperature']) * 0.04, 0)  # Chauffage
        )
        
        # Effet weekend (varie selon le type de bâtiment)
        weekend_factors = {
            'office': -0.6, 'residential': 0.1, 'commercial': -0.2, 'industrial': -0.1
        }
        building_df['weekend_effect'] = np.where(
            building_df['is_weekend'], weekend_factors[building_type], 0
        )
        
        # Effet saisonnier
        seasonal_effects = {
            'Winter': 0.2, 'Spring': 0, 'Summer': 0.15, 'Fall': 0.05
        }
        building_df['seasonal_effect'] = building_df['season'].map(seasonal_effects)
        
        # Calcul de la consommation (meter_reading)
        building_df['meter_reading'] = (
            base_consumption * (
                building_df['hourly_factor'] * 
                (1 + building_df['temp_effect'] + 
                 building_df['weekend_effect'] + 
                 building_df['seasonal_effect'])
            )
        )
        
        # Ajout de bruit réaliste
        noise_std = base_consumption * 0.05  # 5% de bruit
        building_df['meter_reading'] += np.random.normal(0, noise_std, len(building_df))
        building_df['meter_reading'] = np.maximum(building_df['meter_reading'], 1)  # Min 1 kWh
        
        # Introduction d'anomalies spécifiques au bâtiment
        n_anomalies = int(len(building_df) * anomaly_rate)
        anomaly_indices = np.random.choice(len(building_df), n_anomalies, replace=False)
        
        building_df['anomaly'] = 0
        building_df.loc[building_df.index[anomaly_indices], 'anomaly'] = 1
        
        # Types d'anomalies réalistes
        for idx in anomaly_indices:
            actual_idx = building_df.index[idx]
            anomaly_type = np.random.choice(
                ['spike', 'drop', 'equipment_failure', 'sustained_high'], 
                p=[0.3, 0.2, 0.3, 0.2]
            )
            
            current_reading = building_df.loc[actual_idx, 'meter_reading']
            
            if anomaly_type == 'spike':
                # Pic soudain (2-4x normal)
                multiplier = np.random.uniform(2, 4)
                building_df.loc[actual_idx, 'meter_reading'] = current_reading * multiplier
                
            elif anomaly_type == 'drop':
                # Chute (10-40% du normal)
                multiplier = np.random.uniform(0.1, 0.4)
                building_df.loc[actual_idx, 'meter_reading'] = current_reading * multiplier
                
            elif anomaly_type == 'equipment_failure':
                # Panne d'équipement (consommation très faible sur plusieurs heures)
                duration = np.random.randint(2, 8)
                end_idx = min(idx + duration, len(building_df))
                failure_consumption = base_consumption * 0.1
                for j in range(idx, end_idx):
                    if j < len(building_df):
                        actual_j = building_df.index[j]
                        building_df.loc[actual_j, 'meter_reading'] = failure_consumption
                        building_df.loc[actual_j, 'anomaly'] = 1
                        
            elif anomaly_type == 'sustained_high':
                # Consommation élevée maintenue
                duration = np.random.randint(3, 12)
                multiplier = np.random.uniform(1.5, 2.5)
                end_idx = min(idx + duration, len(building_df))
                for j in range(idx, end_idx):
                    if j < len(building_df):
                        actual_j = building_df.index[j]
                        building_df.loc[actual_j, 'meter_reading'] *= multiplier
                        building_df.loc[actual_j, 'anomaly'] = 1
        
        # Sélection des colonnes finales au format LEAD
        final_columns = ['building_id', 'timestamp', 'meter_reading', 'anomaly']
        building_data = building_df[final_columns].copy()
        building_data['meter_reading'] = building_data['meter_reading'].round(2)
        
        all_data.append(building_data)
    
    # Combinaison de tous les bâtiments
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Tri par timestamp puis building_id
    final_df = final_df.sort_values(['timestamp', 'building_id']).reset_index(drop=True)
    
    return final_df

def create_additional_features(df):
    """
    Crée des features additionnelles comme dans le dataset LEAD
    """
    df_features = df.copy()
    
    # Features temporelles
    df_features['hour'] = df_features['timestamp'].dt.hour
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['month'] = df_features['timestamp'].dt.month
    df_features['is_weekend'] = (df_features['timestamp'].dt.dayofweek >= 5).astype(int)
    
    # Features cycliques
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # Features de lag et rolling statistics par bâtiment
    df_features = df_features.sort_values(['building_id', 'timestamp'])
    df_features['meter_reading_lag_1'] = df_features.groupby('building_id')['meter_reading'].shift(1)
    df_features['meter_reading_lag_24'] = df_features.groupby('building_id')['meter_reading'].shift(24)
    df_features['meter_reading_rolling_mean_24'] = df_features.groupby('building_id')['meter_reading'].rolling(24).mean().values
    df_features['meter_reading_rolling_std_24'] = df_features.groupby('building_id')['meter_reading'].rolling(24).std().values
    
    return df_features

def main():
    print("🔄 Génération des données synthétiques au format LEAD...")
    print("📊 Large-scale Energy Anomaly Detection Dataset")
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    # Génération des données principales
    print("\n1️⃣ Génération du dataset d'entraînement...")
    train_df = generate_lead_format_data(
        n_buildings=50, 
        n_days=365, 
        anomaly_rate=0.05,
        random_state=42
    )
    
    # Génération des données de test (sans labels d'anomalies)
    print("\n2️⃣ Génération du dataset de test...")
    test_df = generate_lead_format_data(
        n_buildings=30, 
        n_days=180, 
        anomaly_rate=0.05,
        random_state=123
    )
    
    # Pour le test, on retire les labels d'anomalies et on ajoute row_id
    test_df_submission = test_df[['building_id', 'timestamp', 'meter_reading']].copy()
    test_df_submission.insert(0, 'row_id', range(len(test_df_submission)))
    
    # Création des features additionnelles
    print("\n3️⃣ Génération des features additionnelles...")
    train_features = create_additional_features(train_df)
    test_features = create_additional_features(test_df)
    
    # Sauvegarde des fichiers
    print("\n4️⃣ Sauvegarde des fichiers...")
    train_df.to_csv('data/train.csv', index=False)
    test_df_submission.to_csv('data/test.csv', index=False)
    train_features.to_csv('data/train_features.csv', index=False)
    test_features[train_features.columns[:-1]].to_csv('data/test_features.csv', index=False)  # Sans anomaly
    
    # Sample submission
    sample_submission = pd.DataFrame({
        'row_id': test_df_submission['row_id'],
        'anomaly': 0  # Prédiction par défaut
    })
    sample_submission.to_csv('data/sample_submission.csv', index=False)
    
    print(f"\n✅ Génération terminée!")
    print(f"📁 Fichiers créés:")
    print(f"   - train.csv: {len(train_df):,} lignes ({train_df['building_id'].nunique()} bâtiments)")
    print(f"   - test.csv: {len(test_df_submission):,} lignes ({test_df_submission['building_id'].nunique()} bâtiments)")
    print(f"   - train_features.csv: {len(train_features):,} lignes, {len(train_features.columns)} colonnes")
    print(f"   - test_features.csv: {len(test_features[train_features.columns[:-1]]):,} lignes")
    print(f"   - sample_submission.csv: {len(sample_submission):,} lignes")
    
    print(f"\n📊 Statistiques des anomalies:")
    anomaly_stats = train_df['anomaly'].value_counts()
    print(f"   - Normal (0): {anomaly_stats[0]:,} ({anomaly_stats[0]/len(train_df):.1%})")
    print(f"   - Anomalie (1): {anomaly_stats[1]:,} ({anomaly_stats[1]/len(train_df):.1%})")
    
    print(f"\n📈 Aperçu des données d'entraînement:")
    print(train_df.head(10))
    
    print(f"\n📊 Statistiques par bâtiment (sample):")
    building_stats = train_df.groupby('building_id').agg({
        'meter_reading': ['mean', 'std', 'min', 'max'],
        'anomaly': 'sum'
    }).round(2)
    print(building_stats.head())

if __name__ == "__main__":
    main()
