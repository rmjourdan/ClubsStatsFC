from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict

app = FastAPI(title="FC25 Calculator API", version="3.4.0")

# Variables globales
stats_data = None
positions_list = []

# Función para cargar datos al inicio
def load_csv_data():
    global stats_data, positions_list
    try:
        # Obtener directorio de datos
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data"
        
        # Cargar CSV básico para probar
        positions_file = data_dir / "Datos FC25 - POSICIONES.csv"
        if positions_file.exists():
            df_positions = pd.read_csv(positions_file)
            positions_list = df_positions.iloc[:, 0].tolist()  # Primera columna como posiciones
            stats_data = df_positions
            return True
        return False
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return False

# Cargar datos al iniciar
load_csv_data()

@app.get("/")
async def root():
    """Servir frontend básico"""
    static_dir = Path(__file__).parent.parent / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "FC25 Calculator API", "status": "online"}

@app.get("/api/health")
async def health():
    """Check de salud"""
    return {
        "status": "ok", 
        "data_loaded": stats_data is not None,
        "positions_count": len(positions_list)
    }

@app.get("/api/positions")
async def get_positions():
    """Obtener posiciones"""
    return {"positions": positions_list}

# Modelo simple para testing
class PlayerRequest(BaseModel):
    position: str
    height: int
    weight: int

@app.post("/api/calculate")
async def calculate_basic(request: PlayerRequest):
    """Cálculo básico para testing"""
    return {
        "position": request.position,
        "height": request.height,
        "weight": request.weight,
        "message": "Cálculo básico funcionando"
    }

# Handler para Vercel
def handler(event, context):
    return app(event, context)

# Para testing local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)