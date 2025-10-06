#!/usr/bin/env python3
"""
Script para usar o modelo Random Forest de exoplanetas
- Lê um CSV novo
- Faz previsões (0=FALSE POSITIVE, 1=CONFIRMED)
- Mostra probabilidade de CONFIRMED
"""

import joblib
import pandas as pd
import os

# Caminhos
MODEL_PATH = r"C:\Users\thecr\Desktop\ProjetoNasaWorkspace\IA\exoplanet_rf.joblib"
NEW_DATA_CSV = r"C:\Users\thecr\Desktop\ProjetoNasaWorkspace\IA\data.csv"  # CSV fornecido

# Verifica se arquivos existem
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
if not os.path.exists(NEW_DATA_CSV):
    raise FileNotFoundError(f"CSV de dados não encontrado em {NEW_DATA_CSV}")

# Carrega o modelo
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
features = model_data["features"]

# Lê novos dados
data = pd.read_csv(NEW_DATA_CSV)

# Confere se todas as features existem
missing = [f for f in features if f not in data.columns]
if missing:
    raise ValueError(f"Faltando colunas no CSV: {missing}")

# Prepara dados
X_new = data[features].to_numpy()

# Faz previsões
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)[:, 1]  # probabilidade de CONFIRMED

# Adiciona resultados ao DataFrame
data["predicao"] = y_pred
data["prob_CONFIRMED"] = y_proba

# Converte predicao para texto
data["predicao_texto"] = data["predicao"].map({0: "FALSE POSITIVE", 1: "CONFIRMED"})

# Colunas para mostrar
cols_to_show = ["loc_rowid", "kepid", "kepler_name", "predicao_texto", "prob_CONFIRMED"]

# Confere se as colunas existem
for col in ["loc_rowid", "kepid", "kepler_name"]:
    if col not in data.columns:
        raise ValueError(f"Coluna {col} não encontrada no CSV")

# Mostra resultado
print(data[cols_to_show])
