# Energy Anomaly Detection Dashboard - Version Améliorée
# Authors: Aubin KAMTSA & Sonia KOM
# Interactive Dash Application for LEAD Dataset Analysis

# Imports
import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# Imports pour la modélisation
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "🔋 Energy Anomaly Detection - LEAD Dataset"

# Variables globales
data_store = {
    'train_df': None,
    'test_df': None,
    'processed_data': None,
    'models': None,
    'loading_status': {}
}

def load_data():
    """Chargement optimisé des données avec gestion d'erreurs"""
    try:
        print("🔄 Chargement des données...")
        
        # Vérification de l'existence des fichiers
        data_files = {
            'train': 'data/train.csv',
            'test': 'data/test.csv'
        }
        
        for name, path in data_files.items():
            if not os.path.exists(path):
                return False, f"❌ Fichier manquant: {path}"
        
        # Chargement optimisé avec types de données
        dtype_dict = {
            'building_id': 'int32',
            'meter_reading': 'float32',
            'anomaly': 'int8'
        }
        
        train_df = pd.read_csv(data_files['train'], dtype=dtype_dict)
        test_df = pd.read_csv(data_files['test'])
        
        # Conversion des timestamps
        if 'timestamp' in train_df.columns:
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        if 'timestamp' in test_df.columns:
            test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            
        data_store['train_df'] = train_df
        data_store['test_df'] = test_df
        
        print(f"✅ Données chargées: Train {train_df.shape}, Test {test_df.shape}")
        return True, f"✅ Données chargées avec succès! Train: {train_df.shape}, Test: {test_df.shape}"
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {str(e)}")
        return False, f"❌ Erreur: {str(e)}"

# Layout principal avec design amélioré
app.layout = dbc.Container([
    # En-tête avec informations détaillées
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🔋 Energy Anomaly Detection Dashboard", 
                       className="text-center mb-3",
                       style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                html.H5("Analyse Interactive du Dataset LEAD - Détection d'Anomalies Énergétiques", 
                       className="text-center text-muted mb-3"),
                html.P("Par Aubin KAMTSA & Sonia KOM - Projet Académique d'Analyse de Données", 
                       className="text-center text-secondary mb-4")
            ])
        ])
    ]),
    
    # Description du projet
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📋 À Propos de ce Projet", className="card-title text-primary"),
                    html.P([
                        "Ce dashboard interactif présente une analyse complète du dataset LEAD (Large-scale Energy Anomaly Detection). ",
                        "L'objectif est de détecter automatiquement les anomalies dans la consommation énergétique de bâtiments ",
                        "en utilisant des techniques d'apprentissage automatique supervisées et non-supervisées."
                    ], className="card-text"),
                    html.Hr(),
                    html.H5("🎯 Objectifs du Projet", className="text-info"),
                    html.Ul([
                        html.Li("Analyser les patterns de consommation énergétique"),
                        html.Li("Identifier les anomalies dans les données temporelles"),
                        html.Li("Comparer différents modèles de détection d'anomalies"),
                        html.Li("Créer une interface interactive pour l'exploration des résultats")
                    ]),
                    html.Hr(),
                    html.H5("🔧 Technologies Utilisées", className="text-success"),
                    html.P("Python • Dash • Plotly • Scikit-learn • Pandas • NumPy", className="font-monospace")
                ])
            ], color="light", outline=True)
        ])
    ], className="mb-4"),
    
    # Barre de navigation améliorée
    dbc.Row([
        dbc.Col([
            html.H4("🗂️ Navigation", className="mb-3"),
            dbc.ButtonGroup([
                dbc.Button("📊 Données", id="btn-data", color="primary", className="me-2"),
                dbc.Button("🔧 Préprocessing", id="btn-preprocessing", color="info", className="me-2"),
                dbc.Button("📈 Exploration", id="btn-eda", color="success", className="me-2"),
                dbc.Button("🤖 Modélisation", id="btn-modeling", color="warning", className="me-2"),
                dbc.Button("📋 Résultats", id="btn-results", color="danger")
            ], className="d-grid gap-2 d-md-flex justify-content-md-center mb-4")
        ])
    ]),
    
    # Zone de progression
    html.Div(id="progress-container", className="mb-3"),
    
    # Zone de contenu principal
    html.Div(id="main-content", className="mb-5"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("© 2024 - Projet Académique - Aubin KAMTSA & Sonia KOM", 
                   className="text-center text-muted")
        ])
    ]),
    
    # Stockage des états
    dcc.Store(id='data-store'),
    dcc.Store(id='processed-store'),
    dcc.Store(id='model-store'),
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0, disabled=True)
    
], fluid=True, style={'backgroundColor': '#f8f9fa'})

def create_progress_bar(label, progress, color="primary"):
    """Créer une barre de progression avec label"""
    return dbc.Card([
        dbc.CardBody([
            html.H6(label, className="mb-2"),
            dbc.Progress(value=progress, color=color, className="mb-2"),
            html.Small(f"{progress}% complété", className="text-muted")
        ])
    ], className="mb-2")

def preprocess_data(df):
    """Preprocessing avancé avec features temporelles"""
    print("🔄 Démarrage du preprocessing...")
    df_processed = df.copy()
    
    if 'timestamp' in df_processed.columns:
        # Features temporelles
        df_processed['year'] = df_processed['timestamp'].dt.year
        df_processed['month'] = df_processed['timestamp'].dt.month
        df_processed['day'] = df_processed['timestamp'].dt.day
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['dayofweek'] = df_processed['timestamp'].dt.dayofweek
        df_processed['is_weekend'] = (df_processed['dayofweek'] >= 5).astype(int)
        
        # Features cycliques (importantes pour les données temporelles)
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
        df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
        df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['dayofweek'] / 7)
        df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['dayofweek'] / 7)
    
    # Gestion intelligente des valeurs manquantes
    if 'meter_reading' in df_processed.columns:
        # Remplacement par la médiane par bâtiment (plus robuste)
        df_processed['meter_reading'] = df_processed.groupby('building_id')['meter_reading'].transform(
            lambda x: x.fillna(x.median())
        )
        # Si encore des NaN, utiliser la médiane globale
        df_processed['meter_reading'].fillna(df_processed['meter_reading'].median(), inplace=True)
    
    print("✅ Preprocessing terminé")
    return df_processed

# Callback principal pour la navigation
@app.callback(
    [Output('main-content', 'children'),
     Output('progress-container', 'children')],
    [Input('btn-data', 'n_clicks'),
     Input('btn-preprocessing', 'n_clicks'),
     Input('btn-eda', 'n_clicks'),
     Input('btn-modeling', 'n_clicks'),
     Input('btn-results', 'n_clicks')]
)
def update_content(btn_data, btn_prep, btn_eda, btn_model, btn_results):
    ctx = callback_context
    
    # Déterminer quelle section afficher
    if not ctx.triggered:
        section = 'data'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        section = button_id.split('-')[1]
    
    # Créer la barre de progression selon la section
    progress_bars = create_section_progress(section)
    
    # Retourner le contenu et la progression
    if section == 'data':
        return get_data_loading_content(), progress_bars
    elif section == 'preprocessing':
        return get_preprocessing_content(), progress_bars
    elif section == 'eda':
        return get_eda_content(), progress_bars
    elif section == 'modeling':
        return get_modeling_content(), progress_bars
    elif section == 'results':
        return get_results_content(), progress_bars
    
    return get_data_loading_content(), progress_bars

def create_section_progress(section):
    """Créer les barres de progression pour chaque section"""
    sections = ['data', 'preprocessing', 'eda', 'modeling', 'results']
    current_index = sections.index(section) if section in sections else 0
    
    progress_items = []
    for i, sec in enumerate(sections):
        if i < current_index:
            progress = 100
            color = "success"
        elif i == current_index:
            progress = 100
            color = "primary"
        else:
            progress = 0
            color = "light"
            
        labels = {
            'data': '📊 Chargement des Données',
            'preprocessing': '🔧 Préprocessing',
            'eda': '📈 Analyse Exploratoire', 
            'modeling': '🤖 Modélisation',
            'results': '📋 Résultats'
        }
        
        progress_items.append(create_progress_bar(labels[sec], progress, color))
    
    return html.Div(progress_items)

def get_data_loading_content():
    """Interface de chargement des données avec explications détaillées"""
    success, message = load_data()
    
    if success and data_store['train_df'] is not None:
        train_df = data_store['train_df']
        
        # Analyse statistique approfondie
        stats_analysis = {
            'total_records': train_df.shape[0],
            'features': train_df.shape[1],
            'buildings': train_df['building_id'].nunique() if 'building_id' in train_df.columns else 0,
            'anomalies': train_df['anomaly'].sum() if 'anomaly' in train_df.columns else 0,
            'anomaly_rate': (train_df['anomaly'].mean() * 100) if 'anomaly' in train_df.columns else 0,
            'time_span': None,
            'missing_values': train_df.isnull().sum().sum()
        }
        
        if 'timestamp' in train_df.columns:
            stats_analysis['time_span'] = f"{train_df['timestamp'].min().date()} à {train_df['timestamp'].max().date()}"
        
        # Cards avec statistiques détaillées
        stats_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Volume de Données"),
                    dbc.CardBody([
                        html.H3(f"{stats_analysis['total_records']:,}", className="text-primary"),
                        html.P("Observations totales"),
                        html.Small(f"{stats_analysis['features']} variables", className="text-muted")
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🏢 Couverture"),
                    dbc.CardBody([
                        html.H3(f"{stats_analysis['buildings']:,}", className="text-info"),
                        html.P("Bâtiments uniques"),
                        html.Small(f"Moyenne: {stats_analysis['total_records']//stats_analysis['buildings']:,} obs/bâtiment" if stats_analysis['buildings'] > 0 else "N/A", className="text-muted")
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚠️ Anomalies"),
                    dbc.CardBody([
                        html.H3(f"{stats_analysis['anomalies']:,}", className="text-warning"),
                        html.P("Anomalies détectées"),
                        html.Small(f"{stats_analysis['anomaly_rate']:.2f}% du total", className="text-muted")
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📅 Période"),
                    dbc.CardBody([
                        html.P(stats_analysis['time_span'] or "Non disponible", className="font-monospace"),
                        html.P("Période couverte"),
                        html.Small(f"{stats_analysis['missing_values']} valeurs manquantes", className="text-muted")
                    ])
                ])
            ], md=3)
        ], className="mb-4")
        
        # Aperçu des données avec informations sur les colonnes
        column_info = []
        for col in train_df.columns:
            dtype = str(train_df[col].dtype)
            null_count = train_df[col].isnull().sum()
            unique_vals = train_df[col].nunique()
            
            column_info.append({
                'Colonne': col,
                'Type': dtype,
                'Valeurs Uniques': unique_vals,
                'Valeurs Manquantes': null_count,
                'Taux de Complétude': f"{((len(train_df) - null_count) / len(train_df) * 100):.1f}%"
            })
        
        column_df = pd.DataFrame(column_info)
        
        return html.Div([
            dbc.Alert([
                html.H4("✅ Données Chargées avec Succès", className="alert-heading"),
                html.P(message),
                html.Hr(),
                html.P("Les données sont maintenant prêtes pour l'analyse. Cliquez sur 'Préprocessing' pour continuer.", className="mb-0")
            ], color="success", className="mb-4"),
            
            html.H3("📊 1. EXPLORATION INITIALE DES DONNÉES", className="mb-4"),
            
            stats_cards,
            
            dbc.Row([
                dbc.Col([
                    html.H4("📋 Informations sur les Colonnes", className="mb-3"),
                    dbc.Table.from_dataframe(
                        column_df, 
                        striped=True, 
                        bordered=True, 
                        hover=True,
                        responsive=True,
                        size="sm"
                    )
                ], md=6),
                dbc.Col([
                    html.H4("👀 Aperçu des Données", className="mb-3"),
                    dbc.Table.from_dataframe(
                        train_df.head(8), 
                        striped=True, 
                        bordered=True, 
                        hover=True,
                        responsive=True,
                        size="sm"
                    )
                ], md=6)
            ]),
            
            html.Hr(),
            dbc.Alert([
                html.H5("💡 Prochaines Étapes", className="mb-2"),
                html.Ul([
                    html.Li("Cliquez sur '🔧 Préprocessing' pour créer de nouvelles features"),
                    html.Li("Les features temporelles cycliques seront générées automatiquement"),
                    html.Li("Les valeurs manquantes seront traitées intelligemment")
                ])
            ], color="info")
        ])
    else:
        return dbc.Alert([
            html.H4("❌ Erreur de Chargement", className="alert-heading"),
            html.P(message),
            html.Hr(),
            html.P("Vérifiez que les fichiers CSV sont présents dans le dossier 'data/'")
        ], color="danger")

def get_preprocessing_content():
    """Interface de preprocessing avec explications détaillées"""
    if data_store['train_df'] is None:
        return dbc.Alert("⚠️ Veuillez d'abord charger les données", color="warning")
    
    # Preprocessing des données
    train_processed = preprocess_data(data_store['train_df'])
    test_processed = preprocess_data(data_store['test_df'])
    data_store['processed_data'] = {'train': train_processed, 'test': test_processed}
    
    # Nouvelles features créées
    new_features = [col for col in train_processed.columns if col not in data_store['train_df'].columns]
    
    # Graphique des valeurs manquantes
    missing_data = train_processed.isnull().sum()
    fig_missing = px.bar(
        x=missing_data.index[:10],
        y=missing_data.values[:10],
        title="🔍 Valeurs Manquantes par Variable",
        color=missing_data.values[:10],
        color_continuous_scale='Reds'
    )
    fig_missing.update_layout(showlegend=False)
    
    # Distribution des nouvelles features cycliques
    cyclical_features = [f for f in new_features if any(x in f for x in ['sin', 'cos'])]
    fig_cyclical = make_subplots(rows=2, cols=2, 
                                subplot_titles=['Features Temporelles Cycliques'] * 4)
    
    for i, feature in enumerate(cyclical_features[:4]):
        row = (i // 2) + 1
        col = (i % 2) + 1
        fig_cyclical.add_trace(
            go.Histogram(x=train_processed[feature], name=feature, nbinsx=50),
            row=row, col=col
        )
    
    return html.Div([
        html.H3("🔧 2. PRÉPROCESSING AVANCÉ DES DONNÉES", className="mb-4"),
        
        dbc.Alert([
            html.H4("✅ Preprocessing Terminé avec Succès!", className="alert-heading"),
            html.P(f"Shape finale: {train_processed.shape} | {len(new_features)} nouvelles features créées"),
            html.Hr(),
            html.P("Toutes les transformations ont été appliquées. Vous pouvez maintenant passer à l'exploration.")
        ], color="success", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.H4("🆕 Nouvelles Features Créées", className="mb-3"),
                html.Div([
                    dbc.Badge(f"✅ {feature}", color="primary", className="me-2 mb-2") 
                    for feature in new_features[:15]
                ]),
                html.P(f"... et {max(0, len(new_features)-15)} autres features", className="text-muted") if len(new_features) > 15 else ""
            ], md=6),
            dbc.Col([
                html.H4("📊 Transformations Appliquées", className="mb-3"),
                html.Ul([
                    html.Li("✅ Features temporelles (année, mois, jour, heure)"),
                    html.Li("✅ Features cycliques (sin/cos pour saisonnalité)"),
                    html.Li("✅ Indicateurs weekend/semaine"),
                    html.Li("✅ Traitement des valeurs manquantes"),
                    html.Li("✅ Optimisation des types de données")
                ])
            ], md=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_missing)], md=6),
            dbc.Col([dcc.Graph(figure=fig_cyclical)], md=6)
        ])
    ])

def get_eda_content():
    """Interface d'analyse exploratoire enrichie"""
    if data_store['processed_data'] is None:
        return dbc.Alert("⚠️ Veuillez d'abord effectuer le preprocessing", color="warning")
    
    train_df = data_store['processed_data']['train']
    
    # Distribution de la consommation avec stats
    fig_dist = px.histogram(
        train_df, x='meter_reading', nbins=50,
        title="📊 Distribution de la Consommation Énergétique",
        marginal="box"
    )
    fig_dist.add_vline(x=train_df['meter_reading'].mean(), 
                      line_dash="dash", line_color="red",
                      annotation_text=f"Moyenne: {train_df['meter_reading'].mean():.1f}")
    
    # Comparaison anomalies vs normal
    fig_box = px.box(
        train_df, x='anomaly', y='meter_reading',
        title="⚠️ Consommation: Normal vs Anomalies",
        color='anomaly'
    )
    
    # Série temporelle interactive
    sample_df = train_df.sample(min(10000, len(train_df))) if 'timestamp' in train_df.columns else train_df
    fig_time = px.scatter(
        sample_df, x='timestamp', y='meter_reading', 
        color='anomaly', title="📈 Évolution Temporelle des Consommations",
        hover_data=['building_id']
    )
    
    # Patterns par heure et jour
    hourly_pattern = train_df.groupby('hour')['meter_reading'].mean()
    daily_pattern = train_df.groupby('dayofweek')['meter_reading'].mean()
    
    fig_patterns = make_subplots(rows=1, cols=2, 
                                subplot_titles=['Consommation par Heure', 'Consommation par Jour'])
    fig_patterns.add_trace(go.Scatter(x=hourly_pattern.index, y=hourly_pattern.values, name='Heure'), row=1, col=1)
    fig_patterns.add_trace(go.Scatter(x=daily_pattern.index, y=daily_pattern.values, name='Jour'), row=1, col=2)
    
    return html.Div([
        html.H3("📈 3. ANALYSE EXPLORATOIRE APPROFONDIE", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📊 Statistiques Clés", className="text-primary"),
                        html.P(f"Consommation moyenne: {train_df['meter_reading'].mean():.2f} kWh"),
                        html.P(f"Écart-type: {train_df['meter_reading'].std():.2f} kWh"),
                        html.P(f"Médiane: {train_df['meter_reading'].median():.2f} kWh")
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("⚠️ Anomalies", className="text-warning"),
                        html.P(f"Total: {train_df['anomaly'].sum():,}"),
                        html.P(f"Pourcentage: {train_df['anomaly'].mean()*100:.2f}%"),
                        html.P(f"Répartition équilibrée: {'✅' if 0.01 <= train_df['anomaly'].mean() <= 0.15 else '⚠️'}")
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("🏢 Couverture", className="text-info"),
                        html.P(f"Bâtiments: {train_df['building_id'].nunique()}"),
                        html.P(f"Période: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days} jours"),
                        html.P(f"Fréquence: {len(train_df) // train_df['building_id'].nunique():.0f} obs/bâtiment")
                    ])
                ])
            ], md=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_dist)], md=6),
            dbc.Col([dcc.Graph(figure=fig_box)], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_time)], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_patterns)], md=12)
        ])
    ])

def get_modeling_content():
    """Interface de modélisation avec métriques détaillées"""
    if data_store['processed_data'] is None:
        return dbc.Alert("⚠️ Veuillez d'abord effectuer le preprocessing", color="warning")
    
    train_df = data_store['processed_data']['train']
    feature_cols = [col for col in train_df.columns if col not in ['anomaly', 'timestamp', 'building_id']]
    X = train_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = train_df['anomaly']
    
    # Division stratifiée
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest optimisé
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_accuracy = accuracy_score(y_val, rf_pred)
    rf_auc = roc_auc_score(y_val, rf_proba)
    
    # Isolation Forest
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train_scaled)
    iso_pred = iso_forest.predict(X_val_scaled)
    iso_pred_binary = np.where(iso_pred == -1, 1, 0)
    iso_accuracy = accuracy_score(y_val, iso_pred_binary)
    
    # Sauvegarde des modèles
    data_store['models'] = {
        'rf': rf_model, 'iso': iso_forest, 'scaler': scaler,
        'feature_cols': feature_cols, 'metrics': {
            'rf_accuracy': rf_accuracy, 'rf_auc': rf_auc, 'iso_accuracy': iso_accuracy
        }
    }
    
    # Visualisations
    feature_importance = pd.DataFrame({
        'feature': X.columns, 'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance.head(10), x='importance', y='feature',
        orientation='h', title="🎯 Top 10 Features les Plus Importantes"
    )
    
    # Matrices de confusion
    cm_rf = confusion_matrix(y_val, rf_pred)
    cm_iso = confusion_matrix(y_val, iso_pred_binary)
    
    fig_cm = make_subplots(rows=1, cols=2, subplot_titles=('Random Forest', 'Isolation Forest'))
    fig_cm.add_trace(go.Heatmap(z=cm_rf, colorscale='Blues', showscale=False), row=1, col=1)
    fig_cm.add_trace(go.Heatmap(z=cm_iso, colorscale='Greens', showscale=False), row=1, col=2)
    fig_cm.update_layout(title_text="🎯 Matrices de Confusion - Performance des Modèles")
    
    return html.Div([
        html.H3("🤖 4. MODÉLISATION - DÉTECTION D'ANOMALIES", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌳 Random Forest (Supervisé)"),
                    dbc.CardBody([
                        html.H4(f"{rf_accuracy:.3f}", className="text-primary"),
                        html.P("Accuracy"),
                        html.Small(f"AUC-ROC: {rf_auc:.3f}", className="text-muted")
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔍 Isolation Forest (Non-supervisé)"),
                    dbc.CardBody([
                        html.H4(f"{iso_accuracy:.3f}", className="text-success"),
                        html.P("Accuracy"),
                        html.Small("Détection d'outliers", className="text-muted")
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Comparaison"),
                    dbc.CardBody([
                        html.H4("✅" if rf_accuracy > iso_accuracy else "🔄", className="text-info"),
                        html.P("Meilleur modèle"),
                        html.Small(f"RF: {rf_accuracy:.3f} vs ISO: {iso_accuracy:.3f}", className="text-muted")
                    ])
                ])
            ], md=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_importance)], md=6),
            dbc.Col([dcc.Graph(figure=fig_cm)], md=6)
        ])
    ])

def get_results_content():
    """Interface des résultats finaux avec recommandations"""
    if data_store['models'] is None:
        return dbc.Alert("⚠️ Veuillez d'abord entraîner les modèles", color="warning")
    
    test_df = data_store['processed_data']['test']
    models = data_store['models']
    X_test = test_df[models['feature_cols']].select_dtypes(include=[np.number]).fillna(0)
    
    # Prédictions finales
    rf_test_proba = models['rf'].predict_proba(X_test)[:, 1]
    rf_test_pred = models['rf'].predict(X_test)
    X_test_scaled = models['scaler'].transform(X_test)
    iso_test_pred = models['iso'].predict(X_test_scaled)
    iso_test_pred_binary = np.where(iso_test_pred == -1, 1, 0)
    
    # Analyse des résultats
    anomaly_rate_rf = np.mean(rf_test_pred) * 100
    anomaly_rate_iso = np.mean(iso_test_pred_binary) * 100
    
    # Visualisation finale
    sample_indices = np.random.choice(len(test_df), min(5000, len(test_df)), replace=False)
    fig_final = px.scatter(
        x=test_df.iloc[sample_indices]['meter_reading'],
        y=rf_test_proba[sample_indices],
        color=rf_test_pred[sample_indices],
        title="🎯 Prédictions Finales - Probabilités d'Anomalies",
        labels={'x': 'Consommation (kWh)', 'y': 'Probabilité d\'Anomalie', 'color': 'Prédiction'}
    )
    
    return html.Div([
        html.H3("📋 5. RÉSULTATS FINAUX ET RECOMMANDATIONS", className="mb-4"),
        
        dbc.Alert([
            html.H4("🎉 Analyse Terminée avec Succès!", className="alert-heading"),
            html.P("Votre modèle de détection d'anomalies énergétiques est prêt à être déployé."),
            html.Hr(),
            html.P(f"🔍 {anomaly_rate_rf:.1f}% d'anomalies détectées sur les données de test")
        ], color="success", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📊 Résumé de Performance", className="text-primary"),
                        html.P(f"✅ Random Forest: {models['metrics']['rf_accuracy']:.3f} accuracy"),
                        html.P(f"✅ AUC-ROC: {models['metrics']['rf_auc']:.3f}"),
                        html.P(f"✅ Isolation Forest: {models['metrics']['iso_accuracy']:.3f} accuracy"),
                        html.P(f"🎯 Anomalies détectées: {anomaly_rate_rf:.1f}%")
                    ])
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("💡 Recommandations", className="text-info"),
                        html.P("✅ Modèle prêt pour la production"),
                        html.P("✅ Monitoring en temps réel recommandé"),
                        html.P("✅ Réentraînement mensuel suggéré"),
                        html.P("✅ Validation continue nécessaire")
                    ])
                ])
            ], md=6)
        ], className="mb-4"),
        
        dcc.Graph(figure=fig_final),
        
        html.Hr(),
        html.H4("🏆 CONCLUSION", className="mb-3 text-center"),
        dbc.Row([
            dbc.Col([
                html.P("✅ Dataset LEAD analysé avec succès", className="text-success font-weight-bold"),
                html.P("✅ Features temporelles optimisées", className="text-success"),
                html.P("✅ Modèles multiples entraînés et évalués", className="text-success"),
                html.P("✅ Dashboard interactif déployé", className="text-success"),
                html.P("✅ Système de détection opérationnel", className="text-success")
            ], className="text-center")
        ])
    ])

# ================================
# CALLBACK PRINCIPAL
# ================================

@app.callback(
    [Output('main-content', 'children'),
     Output('progress-container', 'children')],
    [Input('btn-data', 'n_clicks'),
     Input('btn-preprocessing', 'n_clicks'), 
     Input('btn-eda', 'n_clicks'),
     Input('btn-modeling', 'n_clicks'),
     Input('btn-results', 'n_clicks')]
)
def update_content(btn_data, btn_prep, btn_eda, btn_model, btn_results):
    """Callback principal pour la navigation entre sections"""
    ctx = callback_context
    
    if not ctx.triggered:
        section = 'data'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        section = button_id.split('-')[1]
    
    # Création des barres de progression selon la section
    progress_bars = create_section_progress(section)
    
    try:
        if section == 'data':
            content = get_data_loading_content()
        elif section == 'preprocessing':
            content = get_preprocessing_content()
        elif section == 'eda':
            content = get_eda_content()
        elif section == 'modeling':
            print(f"🤖 Section modélisation activée")  # Debug
            content = get_modeling_content()
        elif section == 'results':
            content = get_results_content()
        else:
            content = get_data_loading_content()
            
        return content, progress_bars
        
    except Exception as e:
        print(f"❌ Erreur dans section {section}: {str(e)}")  # Debug
        error_content = dbc.Alert([
            html.H4("❌ Erreur", className="alert-heading"),
            html.P(f"Une erreur s'est produite dans la section {section}: {str(e)}"),
            html.Hr(),
            html.P("Veuillez réessayer ou contacter le support.")
        ], color="danger")
        
        return error_content, progress_bars

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
