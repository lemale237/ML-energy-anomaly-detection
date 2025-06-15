# Energy Anomaly Detection Analysis
# Authors: Aubin KAMTSA & Sonia KOM
# Project: Large-scale Energy Anomaly Detection (LEAD) Dataset Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('ggplot')
sns.set_palette("husl")

print("🔋 Energy Anomaly Detection Analysis")
print("=" * 50)

# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# =======================================================

def load_and_explore_data():
    """
    Charge et explore les données d'anomalies énergétiques depuis les fichiers CSV locaux.
    Cette fonction remplace le code Kaggle original pour fonctionner avec nos données locales.
    """
    print("\n📊 1. CHARGEMENT DES DONNÉES")
    print("-" * 30)
    
    # Chargement des données principales depuis le dossier local data/
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        print(f"✅ Données d'entraînement chargées: {train_df.shape}")
        print(f"✅ Données de test chargées: {test_df.shape}")
    except FileNotFoundError as e:
        print(f"❌ Erreur de chargement des fichiers: {e}")
        print("Vérifiez que les fichiers CSV sont dans le dossier 'data/'")
        return None, None, None, None
    
    # Chargement des features additionnelles si elles existent
    try:
        train_features = pd.read_csv('data/train_features.csv')
        test_features = pd.read_csv('data/test_features.csv')
        print(f"✅ Features d'entraînement chargées: {train_features.shape}")
        print(f"✅ Features de test chargées: {test_features.shape}")
    except FileNotFoundError:
        print("⚠️ Fichiers de features non trouvés, utilisation des données de base uniquement")
        train_features = None
        test_features = None
    
    # Affichage des informations de base
    print(f"\n📈 Aperçu des données d'entraînement:")
    print(train_df.head())
    print(f"\n📋 Informations sur les colonnes:")
    print(train_df.info())
    
    # Vérification des valeurs manquantes
    print(f"\n🔍 Valeurs manquantes:")
    print(train_df.isnull().sum())
    
    # Distribution des anomalies
    if 'anomaly' in train_df.columns:
        anomaly_dist = train_df['anomaly'].value_counts()
        print(f"\n⚡ Distribution des anomalies:")
        print(f"Normal (0): {anomaly_dist[0]} ({anomaly_dist[0]/len(train_df)*100:.1f}%)")
        print(f"Anomale (1): {anomaly_dist[1]} ({anomaly_dist[1]/len(train_df)*100:.1f}%)")
    
    return train_df, test_df, train_features, test_features

# 2. PRÉPROCESSING ET INGÉNIERIE DES FEATURES
# =======================================================

def preprocess_data(df, is_train=True):
    """
    Préprocesse les données en créant des features temporelles et en nettoyant les données.
    Version personnalisée et améliorée du preprocessing original.
    """
    print(f"\n🔧 2. PRÉPROCESSING DES DONNÉES {'(TRAIN)' if is_train else '(TEST)'}")
    print("-" * 40)
    
    # Copie pour éviter de modifier l'original
    df_processed = df.copy()
    
    # Conversion du timestamp si présent
    if 'timestamp' in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        
        # Création des features temporelles (version améliorée)
        df_processed['year'] = df_processed['timestamp'].dt.year
        df_processed['month'] = df_processed['timestamp'].dt.month
        df_processed['day'] = df_processed['timestamp'].dt.day
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['minute'] = df_processed['timestamp'].dt.minute
        df_processed['dayofweek'] = df_processed['timestamp'].dt.dayofweek
        df_processed['is_weekend'] = (df_processed['dayofweek'] >= 5).astype(int)
        
        # Features cycliques pour capturer la périodicité
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
        df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
        
        print("✅ Features temporelles créées")
    
    # Gestion des valeurs manquantes dans meter_reading
    if 'meter_reading' in df_processed.columns:
        missing_count = df_processed['meter_reading'].isnull().sum()
        if missing_count > 0:
            print(f"⚠️ {missing_count} valeurs manquantes dans meter_reading")
            # Imputation par la médiane par bâtiment
            df_processed['meter_reading'] = df_processed.groupby('building_id')['meter_reading'].fillna(
                df_processed.groupby('building_id')['meter_reading'].transform('median')
            )
            # Si encore des valeurs manquantes, utiliser la médiane globale
            df_processed['meter_reading'].fillna(df_processed['meter_reading'].median(), inplace=True)
            print("✅ Valeurs manquantes imputées")
    
    # Création de features de consommation
    if 'meter_reading' in df_processed.columns and 'building_id' in df_processed.columns:
        # Features statistiques par bâtiment
        building_stats = df_processed.groupby('building_id')['meter_reading'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).add_prefix('building_')
        
        df_processed = df_processed.merge(building_stats, on='building_id', how='left')
        
        # Ratio par rapport à la consommation moyenne du bâtiment
        df_processed['consumption_ratio'] = (
            df_processed['meter_reading'] / df_processed['building_mean']
        ).fillna(1.0)
        
        print("✅ Features de consommation créées")
    
    print(f"📊 Shape finale: {df_processed.shape}")
    return df_processed

# 3. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =======================================================

def perform_eda(df):
    """
    Effectue une analyse exploratoire complète des données.
    Version personnalisée avec des visualisations avancées.
    """
    print("\n📈 3. ANALYSE EXPLORATOIRE DES DONNÉES")
    print("-" * 40)
    
    # Distribution de la consommation énergétique
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogramme de la consommation
    axes[0, 0].hist(df['meter_reading'].dropna(), bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution de la Consommation Énergétique')
    axes[0, 0].set_xlabel('Meter Reading (kWh)')
    axes[0, 0].set_ylabel('Fréquence')
    
    # Box plot par anomalie
    if 'anomaly' in df.columns:
        df.boxplot(column='meter_reading', by='anomaly', ax=axes[0, 1])
        axes[0, 1].set_title('Consommation par Type (Normal vs Anomale)')
        axes[0, 1].set_xlabel('Type (0=Normal, 1=Anomale)')
    
    # Évolution temporelle
    if 'month' in df.columns:
        monthly_consumption = df.groupby('month')['meter_reading'].mean()
        axes[1, 0].plot(monthly_consumption.index, monthly_consumption.values, marker='o')
        axes[1, 0].set_title('Consommation Moyenne par Mois')
        axes[1, 0].set_xlabel('Mois')
        axes[1, 0].set_ylabel('Consommation Moyenne')
    
    # Distribution des anomalies par heure
    if 'hour' in df.columns and 'anomaly' in df.columns:
        hourly_anomalies = df.groupby('hour')['anomaly'].mean()
        axes[1, 1].bar(hourly_anomalies.index, hourly_anomalies.values, alpha=0.7, color='coral')
        axes[1, 1].set_title('Taux d\'Anomalies par Heure')
        axes[1, 1].set_xlabel('Heure')
        axes[1, 1].set_ylabel('Taux d\'Anomalies')
    
    plt.tight_layout()
    plt.show()
    
    # Analyse des corrélations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de Corrélation des Variables Numériques')
    plt.tight_layout()
    plt.show()
    
    print("✅ Analyse exploratoire terminée")

# 4. MODÉLISATION ET DÉTECTION D'ANOMALIES
# =======================================================

def train_anomaly_detection_models(df):
    """
    Entraîne plusieurs modèles de détection d'anomalies.
    Approche multi-modèle pour une analyse robuste.
    """
    print("\n🤖 4. MODÉLISATION - DÉTECTION D'ANOMALIES")
    print("-" * 40)
    
    # Préparation des features
    feature_cols = [col for col in df.columns if col not in ['anomaly', 'timestamp', 'building_id']]
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    # Gestion des valeurs manquantes dans les features
    X = X.fillna(X.mean())
    
    if 'anomaly' in df.columns:
        y = df['anomaly']
        
        # Division train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 4.1 Random Forest Classifier
        print("\n🌳 Entraînement Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # Prédictions Random Forest
        rf_pred = rf_model.predict(X_val)
        rf_proba = rf_model.predict_proba(X_val)[:, 1]
        
        print(f"Random Forest - Accuracy: {accuracy_score(y_val, rf_pred):.3f}")
        print(f"Random Forest - AUC-ROC: {roc_auc_score(y_val, rf_proba):.3f}")
        
        # 4.2 Isolation Forest (non supervisé)
        print("\n🌲 Entraînement Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.1,  # Environ 10% d'anomalies
            random_state=42
        )
        iso_predictions = iso_forest.fit_predict(X_train_scaled)
        iso_val_pred = iso_forest.predict(X_val_scaled)
        
        # Conversion des prédictions (-1 -> 1, 1 -> 0)
        iso_val_pred_binary = np.where(iso_val_pred == -1, 1, 0)
        
        print(f"Isolation Forest - Accuracy: {accuracy_score(y_val, iso_val_pred_binary):.3f}")
        
        # Matrice de confusion pour Random Forest
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        cm_rf = confusion_matrix(y_val, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion - Random Forest')
        plt.ylabel('Valeurs Réelles')
        plt.xlabel('Prédictions')
        
        plt.subplot(1, 2, 2)
        cm_iso = confusion_matrix(y_val, iso_val_pred_binary)
        sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Greens')
        plt.title('Matrice de Confusion - Isolation Forest')
        plt.ylabel('Valeurs Réelles')
        plt.xlabel('Prédictions')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance pour Random Forest
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Features les Plus Importantes - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return rf_model, iso_forest, scaler, feature_importance
    
    else:
        print("⚠️ Pas de colonne 'anomaly' trouvée, entraînement non supervisé uniquement")
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest uniquement
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_predictions = iso_forest.fit_predict(X_scaled)
        
        anomaly_rate = np.sum(iso_predictions == -1) / len(iso_predictions)
        print(f"Taux d'anomalies détectées: {anomaly_rate:.1%}")
        
        return None, iso_forest, scaler, None

# 5. VISUALISATIONS AVANCÉES
# =======================================================

def create_advanced_visualizations(df, models=None):
    """
    Crée des visualisations avancées pour l'analyse des anomalies.
    """
    print("\n📊 5. VISUALISATIONS AVANCÉES")
    print("-" * 30)
    
    # Visualisation interactive avec Plotly
    if 'timestamp' in df.columns and 'meter_reading' in df.columns:
        # Série temporelle interactive
        fig = go.Figure()
        
        # Données normales
        normal_data = df[df['anomaly'] == 0] if 'anomaly' in df.columns else df
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['meter_reading'],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4, opacity=0.6)
        ))
        
        # Données anomales
        if 'anomaly' in df.columns:
            anomaly_data = df[df['anomaly'] == 1]
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['meter_reading'],
                mode='markers',
                name='Anomalie',
                marker=dict(color='red', size=6, opacity=0.8)
            ))
        
        fig.update_layout(
            title='Série Temporelle des Consommations Énergétiques',
            xaxis_title='Temps',
            yaxis_title='Consommation (kWh)',
            height=500
        )
        
        fig.show()
    
    # Analyse PCA pour la visualisation dimensionnelle
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X_pca = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    if len(X_pca.columns) > 2:
        pca = PCA(n_components=2)
        X_pca_transformed = pca.fit_transform(StandardScaler().fit_transform(X_pca))
        
        plt.figure(figsize=(10, 6))
        if 'anomaly' in df.columns:
            colors = ['blue' if x == 0 else 'red' for x in df['anomaly']]
            labels = ['Normal' if x == 0 else 'Anomalie' for x in df['anomaly']]
        else:
            colors = 'blue'
            labels = 'Data Points'
        
        plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], 
                   c=colors, alpha=0.6, s=30)
        plt.title('Visualisation PCA des Données')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        if 'anomaly' in df.columns:
            # Légende personnalisée
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', label='Normal'),
                             Patch(facecolor='red', label='Anomalie')]
            plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
    
    print("✅ Visualisations créées")

# 6. FONCTION PRINCIPALE
# =======================================================

def main():
    """
    Fonction principale qui orchestre toute l'analyse.
    """
    print("🚀 Démarrage de l'analyse complète...")
    
    # Chargement des données
    train_df, test_df, train_features, test_features = load_and_explore_data()
    
    if train_df is None:
        print("❌ Impossible de continuer sans données")
        return
    
    # Préprocessing
    train_processed = preprocess_data(train_df, is_train=True)
    test_processed = preprocess_data(test_df, is_train=False)
    
    # Analyse exploratoire
    perform_eda(train_processed)
    
    # Modélisation
    rf_model, iso_model, scaler, feature_importance = train_anomaly_detection_models(train_processed)
    
    # Visualisations avancées
    create_advanced_visualizations(train_processed, (rf_model, iso_model))
    
    print("\n🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("=" * 50)
    
    # Sauvegarde des résultats si souhaité
    if input("\nSouhaitez-vous sauvegarder les modèles? (y/n): ").lower() == 'y':
        import joblib
        if rf_model:
            joblib.dump(rf_model, 'random_forest_model.pkl')
        joblib.dump(iso_model, 'isolation_forest_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("✅ Modèles sauvegardés")

# Exécution du script
if __name__ == "__main__":
    main()
