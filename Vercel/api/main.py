from fastapi import FastAPI, HTTPException

app = FastAPI(title="FC25 Calculator API", version="1.0.0")

@app.get("/")
def read_root():
    return {
        "message": "FC25 Calculator API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/health", "/api/health", "/api/calculate"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "message": "API funcionando correctamente"}

@app.get("/api/health")  
def api_health():
    return {"status": "ok", "message": "API funcionando correctamente"}

@app.post("/api/calculate")
def calculate_stats(player_data: dict):
    """
    Endpoint básico para cálculos de FC25
    """
    try:
        # Por ahora retornamos datos de ejemplo
        # Aquí integraremos la lógica del analizador_fc.py
        return {
            "status": "success",
            "message": "Cálculo realizado",
            "data": {
                "overall": 85,
                "pace": player_data.get("pace", 80),
                "shooting": player_data.get("shooting", 75),
                "passing": player_data.get("passing", 70),
                "defending": player_data.get("defending", 60),
                "physical": player_data.get("physical", 85)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e