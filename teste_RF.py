from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

# ==== FastAPI setup ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Definição do modelo (igual ao treino) ====
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)
        self.fc2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = getattr(nn, activation_fn)()

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.activation(out + residual)

class SpectraMLP(nn.Module):
    def __init__(self, config, num_inputs, num_outputs):
        super().__init__()
        size = config['model']['num_neurons']
        act = config['model']['activation_fn']
        self.input_layer = nn.Linear(num_inputs, size)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(size, config['model']['dropout_rate'], act)
              for _ in range(config['model']['num_res_blocks'])]
        )
        self.output_layer = nn.Linear(size, num_outputs)
        self.activation = getattr(nn, act)()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x

# ==== Configurações do modelo ====
config = {
    "model": {
        "num_neurons": 512,
        "num_res_blocks": 4,
        "dropout_rate": 0.1,
        "activation_fn": "SiLU"
    }
}

num_inputs = 7673   # ajuste para o seu caso real
num_outputs = 16    # número de saídas

# ==== Carrega o modelo ====
MODEL_PATH = "best_model.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

model = SpectraMLP(config, num_inputs, num_outputs)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ==== Endpoints ====
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recebe CSV com features, retorna previsões e probabilidades.
    """
    try:
        df = pd.read_csv(file.file)

        # Confere se há colunas suficientes
        if df.shape[1] != num_inputs:
            return JSONResponse(
                status_code=400,
                content={"erro": f"Número de colunas do CSV ({df.shape[1]}) diferente do esperado ({num_inputs})"}
            )

        # Converte para tensor
        X = torch.from_numpy(df.to_numpy(dtype=np.float32))

        with torch.no_grad():
            preds = model(X).numpy()

        # Retorna previsões como lista
        return {"predictions": preds.tolist(), "num_samples": len(df)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
