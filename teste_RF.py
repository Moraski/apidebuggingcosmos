from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo
MODEL_PATH = "exoplanet_rf.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
features = model_data["features"]

# Pasta para CSVs
CSV_DIR = "tmp"
os.makedirs(CSV_DIR, exist_ok=True)

@app.post("/predict_exoplanets")
async def predict_exoplanets(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        missing = [f for f in features if f not in df.columns]
        if missing:
            return {"erro": f"Faltando colunas no CSV: {missing}"}

        X = df[features].to_numpy()
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        df = df.where(pd.notnull(df), None)
        df["resultado"] = ["CONFIRMED" if p == 1 else "FALSE POSITIVE" for p in y_pred]
        df["prob_CONFIRMED"] = y_proba

        output_path = os.path.join(CSV_DIR, f"predicoes_{file.filename}")
        df.to_csv(output_path, index=False)

        total = len(df)
        confirmados = df[df["resultado"] == "CONFIRMED"]
        qtd_confirmados = len(confirmados)
        perc_confirmados = (qtd_confirmados / total * 100) if total > 0 else 0
        estrelas = df["kepid"].nunique() if "kepid" in df.columns else "N/A"
        media_por_estrela = (qtd_confirmados / estrelas) if isinstance(estrelas, (int, float)) and estrelas > 0 else "N/A"

        return {
            "filename": file.filename,
            "total_entries": total,
            "confirmed_exoplanets": qtd_confirmados,
            "percent_confirmed": perc_confirmados,
            "stars_with_exoplanets": estrelas,
            "avg_exoplanets_per_star": media_por_estrela,
            "csv_gerado": output_path,
            "download_csv": f"/download_csv?file={os.path.basename(output_path)}"
        }

    except Exception as e:
        return {"erro": f"Erro ao processar CSV: {str(e)}"}

@app.get("/download_csv")
async def download_csv(file: str):
    file_path = os.path.join(CSV_DIR, file)
    if not os.path.exists(file_path):
        return {"erro": "Arquivo não encontrado."}
    return FileResponse(file_path, media_type='text/csv', filename=file)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
