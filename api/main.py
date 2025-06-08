from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel
from supabase import create_client, Client
import traceback

# Configuración de Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="FC25 Calculator API", version="3.4.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class PlayerRequest(BaseModel):
    position: str
    height: int
    weight: int
    unlocked_nodes: List[str] = []
    unlocked_facilities: List[str] = []
    apply_facilities: bool = True

class BuildResponse(BaseModel):
    base_stats: Dict[str, float]
    complete_stats: Dict[str, float]
    accelerate: str
    igs_total: float
    weak_foot: int = 2
    skill_moves: int = 3

class BuildSave(BaseModel):
    name: str
    description: str = ""
    author: str = "Anónimo"
    position: str
    height: int
    weight: int
    unlocked_nodes: List[str] = []
    unlocked_facilities: List[str] = []
    apply_facilities: bool = True
    stats_preview: Dict[str, any] = {}
    tags: List[str] = []

# Variables globales para datos
stats_base_lb_rb = None
modificadores_altura = None
modificadores_peso = None
diferenciales_posicion = None
df_skill_trees_global = None
df_instalaciones_global = None
lista_posiciones = None
stat_cols_order = None

# Constantes (copiadas de tu archivo original)
STAR_EMOJI = "⭐"
BASE_WF, BASE_SM, MAX_STARS, MAX_STAT_VAL = 2, 3, 5, 99
TOTAL_SKILL_POINTS = 184
DEFAULT_CLUB_BUDGET = 1750000

MAIN_CATEGORIES = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
SUB_STATS_MAPPING = {
    'PAC': ['Acc', 'Spr'],
    'SHO': ['Fin', 'FKA', 'HAcc', 'SPow', 'Lon S', 'Vol', 'Pen'],
    'PAS': ['Vis', 'Cros', 'LP', 'SP', 'Cur'],
    'DRI': ['AGI', 'BAL', 'APOS', 'BCON', 'REG'],
    'DEF': ['INT', 'AWA', 'STAN', 'SLID'],
    'PHY': ['JUMP', 'STA', 'STR', 'REA', 'AGGR', 'COMP']
}

def get_base_dir():
    """Obtener directorio base del proyecto"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def cargar_datos_csv():
    """Cargar todos los datos CSV necesarios"""
    base_dir = get_base_dir()
    data_dir = base_dir / "data"
    
    try:
        # Cargar archivos CSV
        df_altura = pd.read_csv(data_dir / "Datos FC25 - ALTURA.csv").set_index(pd.read_csv(data_dir / "Datos FC25 - ALTURA.csv").columns[0])
        df_peso = pd.read_csv(data_dir / "Datos FC25 - PESO.csv").set_index(pd.read_csv(data_dir / "Datos FC25 - PESO.csv").columns[0])
        df_posiciones = pd.read_csv(data_dir / "Datos FC25 - POSICIONES.csv").set_index(pd.read_csv(data_dir / "Datos FC25 - POSICIONES.csv").columns[0])
        
        # Cargar skill trees
        df_skills = pd.read_csv(data_dir / "ARBOL DE HABILIDAD - ARBOL.csv")
        
        # Cargar instalaciones
        df_facilities = pd.read_csv(data_dir / "INSTALACIONES - ARBOL.csv")
        
        # Procesar datos base
        expected_cols = df_posiciones.columns.tolist()
        df_altura = df_altura.reindex(columns=expected_cols, fill_value=0).astype(int)
        df_peso = df_peso.reindex(columns=expected_cols, fill_value=0).astype(int)
        
        # Stats base de referencia
        pos_base = 'LB' if 'LB' in df_posiciones.index else df_posiciones.index[0]
        stats_base = df_posiciones.loc[pos_base]
        
        # Modificadores
        alt_base = 180 if 180 in df_altura.index else df_altura.index[0]
        peso_base = 75 if 75 in df_peso.index else df_peso.index[0]
        
        mod_alt = df_altura.subtract(df_altura.loc[alt_base], axis=1)
        mod_peso = df_peso.subtract(df_peso.loc[peso_base], axis=1)
        diff_pos = df_posiciones.subtract(stats_base, axis=1)
        
        return stats_base, mod_alt, mod_peso, diff_pos, df_posiciones.index.tolist(), expected_cols, df_skills, df_facilities
        
    except Exception as e:
        print(f"Error cargando datos CSV: {e}")
        traceback.print_exc()
        return None

def determinar_estilo_carrera(altura, agilidad, fuerza, aceleracion):
    """Determinar AcceleRATE basado en stats"""
    if altura >= 180 and fuerza >= 80:
        return "Largo"
    elif agilidad >= 75 and aceleracion >= 80:
        return "Explosivo"
    else:
        return "Controlado"

def calcular_stats_base_jugador(pos_sel, alt_sel, peso_sel):
    """Calcular stats base de un jugador"""
    global stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion
    
    try:
        # Stats base
        stats_base = stats_base_lb_rb.copy()
        
        # Aplicar modificadores de posición
        if pos_sel in diferenciales_posicion.index:
            stats_base += diferenciales_posicion.loc[pos_sel]
        
        # Aplicar modificadores de altura
        if alt_sel in modificadores_altura.index:
            stats_base += modificadores_altura.loc[alt_sel]
        
        # Aplicar modificadores de peso
        if peso_sel in modificadores_peso.index:
            stats_base += modificadores_peso.loc[peso_sel]
        
        # Añadir campos especiales
        stats_base['PIERNA_MALA'] = BASE_WF
        stats_base['FILIGRANAS'] = BASE_SM
        
        # Calcular IGS
        main_stats = [stats_base.get(cat, 0) for cat in MAIN_CATEGORIES if cat in stats_base]
        stats_base['IGS'] = int(sum(main_stats) / len(main_stats)) if main_stats else 0
        
        return stats_base
        
    except Exception as e:
        print(f"Error en calcular_stats_base_jugador: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Cargar datos al iniciar la aplicación"""
    global stats_base_lb_rb, modificadores_altura, modificadores_peso
    global diferenciales_posicion, df_skill_trees_global, df_instalaciones_global
    global lista_posiciones, stat_cols_order
    
    print("🚀 Iniciando FC25 Calculator API...")
    
    try:
        datos = cargar_datos_csv()
        if datos:
            (stats_base_lb_rb, modificadores_altura, modificadores_peso, 
             diferenciales_posicion, lista_posiciones, stat_cols_order,
             df_skill_trees_global, df_instalaciones_global) = datos
            print("✅ Datos CSV cargados exitosamente")
        else:
            print("❌ Error cargando datos CSV")
    except Exception as e:
        print(f"❌ Error en startup: {e}")
        traceback.print_exc()

# === ENDPOINTS PRINCIPALES ===

@app.get("/")
async def root():
    """Página principal - servir frontend básico"""
    return {"message": "FC25 Calculator API v3.4.0", "status": "online"}

@app.get("/api/health")
async def health_check():
    """Check de salud de la API"""
    return {
        "status": "healthy",
        "version": "3.4.0",
        "data_loaded": stats_base_lb_rb is not None
    }

@app.get("/api/positions")
async def get_positions():
    """Obtener todas las posiciones disponibles"""
    if lista_posiciones:
        return {"positions": sorted(lista_posiciones)}
    return {"positions": []}

@app.get("/api/heights")
async def get_heights():
    """Obtener todas las alturas disponibles"""
    if modificadores_altura is not None:
        return {"heights": sorted(modificadores_altura.index.tolist())}
    return {"heights": list(range(150, 211))}

@app.get("/api/weights")
async def get_weights():
    """Obtener todos los pesos disponibles"""
    if modificadores_peso is not None:
        return {"weights": sorted(modificadores_peso.index.tolist())}
    return {"weights": list(range(50, 121))}

@app.post("/api/calculate-stats")
async def calculate_player_stats(request: PlayerRequest):
    """Calcular estadísticas completas de un jugador"""
    try:
        if not all([stats_base_lb_rb is not None, modificadores_altura is not None]):
            raise HTTPException(status_code=500, detail="Datos no cargados correctamente")
        
        # Calcular stats base
        stats_base = calcular_stats_base_jugador(
            request.position,
            request.height,
            request.weight
        )
        
        if stats_base is None:
            raise HTTPException(status_code=400, detail="Error calculando stats base")
        
        # Por ahora, stats completas = stats base (luego agregaremos habilidades)
        stats_completas = stats_base.copy()
        
        # Calcular AcceleRATE
        accelerate = determinar_estilo_carrera(
            request.height,
            stats_completas.get('AGI', 0),
            stats_completas.get('STR', 0),
            stats_completas.get('Acc', 0)
        )
        
        return BuildResponse(
            base_stats=stats_base.to_dict(),
            complete_stats=stats_completas.to_dict(),
            accelerate=accelerate,
            igs_total=float(stats_completas.get('IGS', 0)),
            weak_foot=int(stats_completas.get('PIERNA_MALA', 2)),
            skill_moves=int(stats_completas.get('FILIGRANAS', 3))
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# === ENDPOINTS SUPABASE ===

@app.post("/api/builds/save")
async def save_build(build: BuildSave):
    """Guardar un build en Supabase"""
    try:
        result = supabase.table('builds').insert({
            'name': build.name,
            'description': build.description,
            'author': build.author,
            'position': build.position,
            'height': build.height,
            'weight': build.weight,
            'unlocked_nodes': build.unlocked_nodes,
            'unlocked_facilities': build.unlocked_facilities,
            'apply_facilities': build.apply_facilities,
            'stats_preview': build.stats_preview,
            'tags': build.tags
        }).execute()
        
        return {"success": True, "build_id": result.data[0]['id'], "message": "Build guardado exitosamente"}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error guardando build: {str(e)}")

@app.get("/api/builds/popular")
async def get_popular_builds():
    """Obtener builds populares"""
    try:
        result = supabase.table('builds').select('*').order('rating', desc=True).order('downloads', desc=True).limit(20).execute()
        return {"builds": result.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo builds populares: {str(e)}")

@app.get("/api/builds/{build_id}")
async def get_build(build_id: int):
    """Obtener un build específico"""
    try:
        result = supabase.table('builds').select('*').eq('id', build_id).execute()
        if result.data:
            # Incrementar contador de descargas
            supabase.table('builds').update({'downloads': result.data[0]['downloads'] + 1}).eq('id', build_id).execute()
            return result.data[0]
        else:
            raise HTTPException(status_code=404, detail="Build no encontrado")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo build: {str(e)}")

@app.post("/api/builds/{build_id}/rate")
async def rate_build(build_id: int, rating: int, request: Request, comment: str = ""):
    """Calificar un build"""
    try:
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="Rating debe estar entre 1 y 5")
        
        user_ip = request.client.host
        
        # Insertar o actualizar rating
        result = supabase.table('build_ratings').upsert({
            'build_id': build_id,
            'user_ip': user_ip,
            'rating': rating,
            'comment': comment
        }).execute()
        
        return {"success": True, "message": "Rating guardado exitosamente"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error guardando rating: {str(e)}")

# Handler para Vercel (IMPORTANTE)
def handler(request, context):
    """Handler principal para Vercel"""
    return app(request, context)

# Aplicación principal
application = app