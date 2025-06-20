import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import json
from collections import defaultdict
from visualizacion_fc25_style import mostrar_visualizacion_fc25  # NUEVO: importar visualización FC25
import numpy as np


# Definir emoji de estrella AL INICIO
STAR_EMOJI = "⭐" 

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="FC Pro Clubs Builder v3.6", layout="wide") 

# --- Definiciones Constantes ---
APP_VERSION = "v3.6" 
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
IGS_SUB_STATS = [
    'Acc', 'Spr', 'Fin', 'FKA', 'HAcc', 'SPow', 'Lon S', 'Vol', 'Pen', 
    'Vis', 'Cros', 'LP', 'SP', 'Cur', 'AGI', 'BAL', 'APOS', 'BCON', 'REG', 
    'INT', 'AWA', 'STAN', 'SLID', 'JUMP', 'STA', 'STR', 'REA', 'AGGR', 'COMP'
]
ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE = IGS_SUB_STATS + ['PIERNA_MALA', 'FILIGRANAS']
ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES = IGS_SUB_STATS

# --- Lógica de AcceleRATE ---
def determinar_estilo_carrera(altura, agilidad, fuerza, aceleracion):
    try: altura, agilidad, fuerza, aceleracion = int(altura), int(agilidad), int(fuerza), int(aceleracion)
    except (ValueError, TypeError): return "N/A"
    agilidad_vs_fuerza, fuerza_vs_agilidad = agilidad - fuerza, fuerza - agilidad
    if altura <= 175 and agilidad >= 80 and aceleracion >= 80 and agilidad_vs_fuerza >= 20: return "Explosivo (Puro)"
    if altura >= 188 and fuerza >= 80 and aceleracion >= 55 and fuerza_vs_agilidad >= 20: return "Largo (Puro)"
    if altura <= 182 and agilidad >= 70 and aceleracion >= 80 and agilidad_vs_fuerza >= 12: return "Mayormente Explosivo"
    if altura >= 183 and fuerza >= 75 and aceleracion >= 55 and fuerza_vs_agilidad >= 12: return "Mayormente Largo"
    if altura <= 182 and agilidad >= 65 and aceleracion >= 70 and agilidad_vs_fuerza >= 4: return "Explosivo Controlado"
    if altura >= 181 and fuerza >= 65 and aceleracion >= 40 and fuerza_vs_agilidad >= 4: return "Largo Controlado"
    return "Controlado"

# --- Carga y Preparación de Datos Base ---
@st.cache_data
def cargar_y_preparar_datos_base():
    file_altura, file_peso, file_posiciones = "Datos FC25 - ALTURA.csv", "Datos FC25 - PESO.csv", "Datos FC25 - POSICIONES.csv"
    try:
        df_altura_raw, df_peso_raw, df_posiciones_raw = pd.read_csv(file_altura), pd.read_csv(file_peso), pd.read_csv(file_posiciones)
    except FileNotFoundError as e: st.error(f"Error crítico al cargar CSV base: {e}. Verifica nombres y ubicación."); return None, None, None, None, None, None
    except Exception as e_gen: st.error(f"Error crítico inesperado al cargar CSV base: {e_gen}"); return None, None, None, None, None, None
    try:
        df_altura, df_peso, df_posiciones = df_altura_raw.set_index('Altura'), df_peso_raw.set_index('Peso'), df_posiciones_raw.set_index('Posicion')
        if df_posiciones.empty or df_posiciones.columns.empty: st.error("Error: POSICIONES.csv vacío o sin encabezados."); return None, None, None, None, None, None
        expected_cols = df_posiciones.columns.tolist()
        df_altura = df_altura.reindex(columns=expected_cols, fill_value=0).astype(int)
        df_peso = df_peso.reindex(columns=expected_cols, fill_value=0).astype(int)
        df_posiciones = df_posiciones.reindex(columns=expected_cols, fill_value=0).astype(int)
        alt_base, peso_base, pos_base = 162, 45, 'LB/RB'
        if pos_base not in df_posiciones.index: st.error(f"Posición base '{pos_base}' no encontrada."); return None, None, None, None, None, None
        stats_base_lb_rb = df_posiciones.loc[pos_base]
        if alt_base not in df_altura.index: st.error(f"Altura base '{alt_base}cm' no encontrada."); return None, None, None, None, None, None
        stats_alt_ref = df_altura.loc[alt_base]
        if peso_base not in df_peso.index: st.error(f"Peso base '{peso_base}kg' no encontrada."); return None, None, None, None, None, None
        stats_peso_ref = df_peso.loc[peso_base]
        mod_alt = df_altura.subtract(stats_alt_ref, axis=1)
        mod_peso = df_peso.subtract(stats_peso_ref, axis=1)
        diff_pos = df_posiciones.subtract(stats_base_lb_rb, axis=1)
        lista_pos = df_posiciones.index.tolist()
        if not lista_pos: st.error("Lista de posiciones vacía (de POSICIONES.csv)."); return None, None, None, None, None, None
        return stats_base_lb_rb, mod_alt, mod_peso, diff_pos, lista_pos, expected_cols
    except Exception as e_proc: st.error(f"Error crítico al procesar DataFrames base: {e_proc}"); return None, None, None, None, None, None

# --- Carga y Preparación del Árbol de Habilidades ---
@st.cache_data
def cargar_arbol_habilidades():
    skill_tree_file = "ARBOL DE HABILIDAD - ARBOL.csv"
    try:
        df_skill_trees = pd.read_csv(skill_tree_file)
        for col in ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE: 
            if col in df_skill_trees.columns: 
                df_skill_trees[col] = pd.to_numeric(df_skill_trees[col], errors='coerce').fillna(0).astype(int)
            else: 
                df_skill_trees[col] = 0
        
        if 'Costo' in df_skill_trees.columns: 
            df_skill_trees['Costo'] = pd.to_numeric(df_skill_trees['Costo'], errors='coerce').fillna(0).astype(int)
        else: 
            df_skill_trees['Costo'] = 0
            st.warning("Columna 'Costo' no encontrada en árbol de hab., se usará 0.")
        
        text_cols_skill_tree = ['ID_Nodo', 'Arbol', 'Nombre_Visible', 'Prerrequisito', 'PlayStyle', 'Es_Arquetipo', 'Notas', 'Puntos_Req_Arbol']
        for col_extra in text_cols_skill_tree:
            if col_extra in df_skill_trees.columns:
                df_skill_trees[col_extra] = df_skill_trees[col_extra].fillna('').astype(str)
            else:
                df_skill_trees[col_extra] = ''
        return df_skill_trees
    except FileNotFoundError: st.error(f"ERROR CRÍTICO: No se encontró '{skill_tree_file}'."); return None
    except Exception as e: st.error(f"Error crítico al cargar CSV del árbol de hab: {e}"); return None

# --- Carga y Preparación de Instalaciones del Club ---
@st.cache_data
def cargar_instalaciones_club():
    instalaciones_file = "INSTALACIONES - ARBOL.csv"
    try:
        df_instalaciones = pd.read_csv(instalaciones_file)
        for col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
            if col in df_instalaciones.columns: 
                df_instalaciones[col] = pd.to_numeric(df_instalaciones[col], errors='coerce').fillna(0).astype(int)
            else: 
                df_instalaciones[col] = 0
        if 'Precio' in df_instalaciones.columns:
            df_instalaciones['Precio'] = pd.to_numeric(df_instalaciones['Precio'], errors='coerce').fillna(0).astype(int)
        else: st.error("Columna 'Precio' no encontrada en CSV de Instalaciones."); return None 
        
        text_cols_facilities = ['ID_Instalacion', 'Instalacion', 'Nombre_Instalacion', 'Prerrequisito', 'PlayStyle', 'EsPlus']
        for col_txt in text_cols_facilities:
            if col_txt in df_instalaciones.columns:
                df_instalaciones[col_txt] = df_instalaciones[col_txt].fillna('').astype(str)
            else:
                df_instalaciones[col_txt] = ''
        return df_instalaciones
    except FileNotFoundError: st.error(f"ERROR CRÍTICO: No se encontró '{instalaciones_file}'."); return None
    except Exception as e: st.error(f"Error crítico al cargar CSV de Instalaciones: {e}"); return None

# --- Función de Cálculo de Estadísticas Base ---
def calcular_stats_base_jugador(pos_sel, alt_sel, peso_sel, base_ref_stats, mod_alt_df, mod_peso_df, diff_pos_df):
    if not all(item is not None for item in [base_ref_stats, mod_alt_df, mod_peso_df, diff_pos_df]): return None
    if pos_sel not in diff_pos_df.index or alt_sel not in mod_alt_df.index or peso_sel not in mod_peso_df.index: return None
    diff = diff_pos_df.loc[pos_sel]; mod_a = mod_alt_df.loc[alt_sel]; mod_p = mod_peso_df.loc[peso_sel]
    final_base_stats = base_ref_stats.add(diff).add(mod_a).add(mod_p)
    return final_base_stats.round().astype(int)

# --- Pre-cálculo de todas las combinaciones base ---
@st.cache_data
def precompute_all_base_stats(_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df, _stat_cols_order):
    all_data = []
    if not all(isinstance(df, (pd.DataFrame, pd.Series)) for df in [_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df]): return pd.DataFrame()
    if not (_diff_pos_df.index.tolist() and _mod_alt_df.index.tolist() and _mod_peso_df.index.tolist()): return pd.DataFrame()
    for pos in _diff_pos_df.index:
        for alt_v in _mod_alt_df.index:
            for pes_v in _mod_peso_df.index:
                stats = calcular_stats_base_jugador(pos, alt_v, pes_v, _base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df)
                if stats is not None: 
                    entry = {'Posicion': pos, 'Altura': alt_v, 'Peso': pes_v, **stats.to_dict()}
                    all_data.append(entry)
    df = pd.DataFrame(all_data)
    if not df.empty and _stat_cols_order:
        valid_cols = [col for col in _stat_cols_order if col in df.columns]
        cols_df_order = ['Posicion', 'Altura', 'Peso'] + valid_cols 
        cols_df_order_existing = [col for col in cols_df_order if col in df.columns]
        df = df[cols_df_order_existing]
    return df

# --- Funciones Auxiliares para el Editor de Habilidades e Instalaciones ---
def calcular_stats_completas(stats_jugador_base, altura_jugador_actual, 
                             df_skill_tree, unlocked_skill_nodes, 
                             df_facilities, unlocked_facility_levels, apply_facilities_boost,
                             _stat_cols_order_ref): 
    
    if stats_jugador_base is None or df_skill_tree is None: 
        idx_name = "Stat"
        expected_index_list = (_stat_cols_order_ref or []) + ['PIERNA_MALA', 'FILIGRANAS', 'AcceleRATE', 'IGS']
        unique_expected_index = []
        [unique_expected_index.append(x) for x in expected_index_list if x not in unique_expected_index]
        return pd.Series(dtype=object, index=pd.Index(unique_expected_index, name=idx_name)).fillna(0)

    stats_modificadas = stats_jugador_base.copy().astype(float) 
    current_wf, current_sm = float(BASE_WF), float(BASE_SM)
    total_igs_boost_from_skills = 0 

    for node_id in unlocked_skill_nodes:
        if node_id not in df_skill_tree['ID_Nodo'].values: continue
        node_data = df_skill_tree[df_skill_tree['ID_Nodo'] == node_id].iloc[0]
        for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE:
            if stat_col in node_data.index and pd.notna(node_data[stat_col]) and node_data[stat_col] != 0:
                boost_val = int(node_data[stat_col])
                if stat_col == 'PIERNA_MALA': 
                    current_wf += boost_val
                elif stat_col == 'FILIGRANAS': 
                    current_sm += boost_val
                elif stat_col in stats_modificadas.index and pd.api.types.is_numeric_dtype(stats_modificadas[stat_col]):
                    stats_modificadas[stat_col] += boost_val
                    if stat_col in IGS_SUB_STATS: 
                        total_igs_boost_from_skills += boost_val
    
    for stat_name in IGS_SUB_STATS: 
        if stat_name in stats_modificadas.index and pd.api.types.is_numeric_dtype(stats_modificadas[stat_name]):
            stats_modificadas[stat_name] = min(stats_modificadas[stat_name], MAX_STAT_VAL)

    stats_modificadas['PIERNA_MALA'] = min(current_wf, MAX_STARS)
    stats_modificadas['FILIGRANAS'] = min(current_sm, MAX_STARS)
    
    stats_finales_con_todo = stats_modificadas.copy()
    total_igs_boost_from_facilities = 0

    if apply_facilities_boost and df_facilities is not None and 'unlocked_facility_levels' in st.session_state:
        for facility_id in st.session_state.unlocked_facility_levels: 
            if facility_id not in df_facilities['ID_Instalacion'].values: continue
            facility_data = df_facilities[df_facilities['ID_Instalacion'] == facility_id].iloc[0]
            for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                if stat_col in facility_data.index and pd.notna(facility_data[stat_col]) and facility_data[stat_col] != 0:
                    boost_val = int(facility_data[stat_col])
                    if stat_col in stats_finales_con_todo.index and pd.api.types.is_numeric_dtype(stats_finales_con_todo[stat_col]):
                        stats_finales_con_todo[stat_col] += boost_val
                        if stat_col in IGS_SUB_STATS: 
                            total_igs_boost_from_facilities += boost_val
        
        for stat_name in IGS_SUB_STATS:
            if stat_name in stats_finales_con_todo.index and pd.api.types.is_numeric_dtype(stats_finales_con_todo[stat_name]):
                stats_finales_con_todo[stat_name] = min(stats_finales_con_todo[stat_name], MAX_STAT_VAL)

    for main_cat, sub_stats_list in SUB_STATS_MAPPING.items():
        if main_cat in stats_finales_con_todo.index: 
            relevant_sub_stats_values = [stats_finales_con_todo.get(sub_stat, 0) for sub_stat in sub_stats_list if sub_stat in stats_finales_con_todo.index and pd.api.types.is_numeric_dtype(stats_finales_con_todo.get(sub_stat))]
            if relevant_sub_stats_values:
                stats_finales_con_todo[main_cat] = min(math.ceil(sum(relevant_sub_stats_values) / len(relevant_sub_stats_values)), MAX_STAT_VAL)

    stats_finales_con_todo['AcceleRATE'] = determinar_estilo_carrera(
        altura_jugador_actual, stats_finales_con_todo.get('AGI', 0), 
        stats_finales_con_todo.get('STR', 0), stats_finales_con_todo.get('Acc', 0)
    )
    
    base_igs_value = float(stats_jugador_base.get('IGS', 0)) if stats_jugador_base is not None else 0.0
    stats_finales_con_todo['IGS'] = base_igs_value + total_igs_boost_from_skills + (total_igs_boost_from_facilities if apply_facilities_boost else 0)
    
    final_dict_for_series = {}
    all_stats_to_process = (_stat_cols_order_ref if _stat_cols_order_ref else []) + ['PIERNA_MALA', 'FILIGRANAS', 'AcceleRATE', 'IGS']
    
    seen_keys_final = set()
    ordered_unique_keys_final = []
    for k in all_stats_to_process:
        if k not in seen_keys_final:
            ordered_unique_keys_final.append(k)
            seen_keys_final.add(k)

    for stat_name in ordered_unique_keys_final:
        value = stats_finales_con_todo.get(stat_name)

        if stat_name == 'AcceleRATE':
            final_dict_for_series[stat_name] = str(value if pd.notna(value) else "N/A")
        elif stat_name in ['PIERNA_MALA', 'FILIGRANAS']:
            default_value = BASE_WF if stat_name == 'PIERNA_MALA' else BASE_SM
            final_dict_for_series[stat_name] = int(float(value if pd.notna(value) else default_value))
        elif pd.api.types.is_numeric_dtype(value) or isinstance(value, (int, float)): 
            final_dict_for_series[stat_name] = int(round(float(value)))
        elif value is None: 
            final_dict_for_series[stat_name] = 0 
        else: 
            try: 
                final_dict_for_series[stat_name] = int(round(float(str(value))))
            except ValueError: 
                final_dict_for_series[stat_name] = str(value)
            
    return pd.Series(final_dict_for_series, name=stats_finales_con_todo.name if hasattr(stats_finales_con_todo, 'name') else "Build Stats")

def check_prerequisites_skills(node_id, df_skill_tree_data, unlocked_nodes_ids):
    if df_skill_tree_data is None or df_skill_tree_data.empty or 'ID_Nodo' not in df_skill_tree_data.columns:
        return False 
    node_info_series = df_skill_tree_data[df_skill_tree_data['ID_Nodo'] == node_id]
    if node_info_series.empty:
        return False 

    node_info = node_info_series.iloc[0]
    prereqs_from_df = node_info.get('Prerrequisito') 

    if pd.isna(prereqs_from_df) or str(prereqs_from_df).strip() == "":
        return True 

    # Lista de IDs de prerrequisitos, limpios de espacios
    prereq_ids_list = [pr_id.strip() for pr_id in str(prereqs_from_df).split(',') if pr_id.strip()]
    
    if not prereq_ids_list: 
        return True 

    # Lógica OR: si CUALQUIER prerrequisito está desbloqueado, se puede desbloquear el nodo.
    # Esto se alinea con la lógica en visualizacion_fc25_style.py (can_unlock_node)
    for pr_id in prereq_ids_list:
        if pr_id in unlocked_nodes_ids:
            return True # Se encontró un prerrequisito cumplido
            
    return False # Ninguno de los prerrequisitos está cumplido


def verificar_dependencias_nodo(nodo_id_a_evaluar, df_all_skills, currently_unlocked_nodes):
    dependent_nodes_names = []
    if df_all_skills is None or df_all_skills.empty or not currently_unlocked_nodes:
        return dependent_nodes_names

    for unlocked_node_id in currently_unlocked_nodes:
        if unlocked_node_id == nodo_id_a_evaluar: 
            continue
        
        node_data_series = df_all_skills[df_all_skills['ID_Nodo'] == unlocked_node_id]
        if not node_data_series.empty:
            node_data = node_data_series.iloc[0]
            prereqs_str = str(node_data.get('Prerrequisito', '')).strip()
            if prereqs_str:
                prereq_list_for_this_unlocked_node = [p.strip() for p in prereqs_str.split(',') if p.strip()]
                if nodo_id_a_evaluar in prereq_list_for_this_unlocked_node:
                    dependent_nodes_names.append(node_data.get('Nombre_Visible', unlocked_node_id))
    return list(set(dependent_nodes_names))


def check_prerequisites_facilities(facility_id, df_facilities_data, unlocked_facility_ids):
    if df_facilities_data is None or df_facilities_data.empty or 'ID_Instalacion' not in df_facilities_data.columns: 
        return False
    facility_info_series = df_facilities_data[df_facilities_data['ID_Instalacion'] == facility_id]
    if facility_info_series.empty:
        return False

    facility_info = facility_info_series.iloc[0]
    prereq_id_val = facility_info.get('Prerrequisito', '')
    
    if pd.isna(prereq_id_val): 
        prereq_id_cleaned = ""
    else:
        prereq_id_cleaned = str(prereq_id_val).strip()

    if prereq_id_cleaned == '' or prereq_id_cleaned.lower() in ['nan', 'none', 'ninguno', 'null', 'na']: 
        return True 
    
    return prereq_id_cleaned in unlocked_facility_ids

def verificar_dependencias_instalacion(instalacion_id_a_evaluar, df_all_facilities, currently_unlocked_levels):
    dependent_facilities_names = []
    if df_all_facilities is None or df_all_facilities.empty or not currently_unlocked_levels:
        return dependent_facilities_names

    facility_to_eval_info_series = df_all_facilities[df_all_facilities['ID_Instalacion'] == instalacion_id_a_evaluar]
    if facility_to_eval_info_series.empty:
        return dependent_facilities_names 

    for unlocked_level_id in currently_unlocked_levels:
        if unlocked_level_id == instalacion_id_a_evaluar:
            continue

        level_data_series = df_all_facilities[df_all_facilities['ID_Instalacion'] == unlocked_level_id]
        if not level_data_series.empty:
            level_data = level_data_series.iloc[0]
            prereq_str = str(level_data.get('Prerrequisito', '')).strip()
            
            if instalacion_id_a_evaluar == prereq_str:
                 dependent_facilities_names.append(level_data.get('Nombre_Instalacion', unlocked_level_id))

    return list(set(dependent_facilities_names))


# --- Función de Generación de Resumen Visual HTML ---
def generar_resumen_visual_html(pos, altura, peso, stats_completas, unlocked_nodes, df_skill_trees, 
                               unlocked_facilities, df_instalaciones, apply_facilities_boost, points_remaining):
    """
    Genera un resumen visual HTML del build actual con estilo FC25
    """
    
    # Calcular datos del build
    puntos_gastados = TOTAL_SKILL_POINTS - points_remaining
    progreso_build = (puntos_gastados / TOTAL_SKILL_POINTS) * 100
    
    # Obtener estadísticas principales
    main_stats = {}
    for stat in MAIN_CATEGORIES:
        main_stats[stat] = int(float(str(stats_completas.get(stat, 0))))
    
    # Obtener estadísticas especiales
    pierna_mala = int(float(str(stats_completas.get('PIERNA_MALA', BASE_WF))))
    filigranas = int(float(str(stats_completas.get('FILIGRANAS', BASE_SM))))
    igs_total = int(float(str(stats_completas.get('IGS', '0'))))
    accelerate = str(stats_completas.get('AcceleRATE', 'N/A'))
    
    # Función para obtener color de estadística con rangos fijos
    def get_stat_color(value):
        if value >= 95:
            return {"background": "#b4a7d6", "color": "#000000"}  # Morado + negro
        elif value >= 85:
            return {"background": "#b6d7a8", "color": "#003300"}  # Verde + verde oscuro
        elif value >= 75:
            return {"background": "#fce8b2", "color": "#594400"}  # Amarillo + marrón oscuro
        else:
            return {"background": "#f4c7c3", "color": "#800000"}  # Rosa + rojo oscuro
    
    # NUEVA FUNCIÓN: Generar coordenadas del mapa de nodos
    def generar_mapa_nodos():
        if not unlocked_nodes or df_skill_trees is None:
            return ""
        
        # Agrupar nodos por árbol, sub-árbol, nivel y columna
        arboles_con_nodos = {}
        for node_id in unlocked_nodes:
            node_data = df_skill_trees[df_skill_trees['ID_Nodo'] == node_id]
            if not node_data.empty:
                node_info = node_data.iloc[0]
                tree_name = node_info.get('Arbol', 'Sin Árbol')
                sub_tree_name = node_info.get('Sub_Arbol', '')
                nivel = node_info.get('Nivel', 0)
                
                # Manejar sub_tree_name como posible float NaN
                if pd.isna(sub_tree_name):
                    sub_tree_name = ''
                else:
                    sub_tree_name = str(sub_tree_name).strip()
                
                # Extraer la columna del ID_Nodo (parte después del último "_")
                node_id_parts = node_id.split('_')
                if len(node_id_parts) >= 2:
                    # Extraer la letra de la columna del último segmento (ej: "B3" -> "B")
                    last_part = node_id_parts[-1]
                    columna_letra = ""
                    for char in last_part:
                        if char.isalpha():
                            columna_letra = char
                            break
                    if not columna_letra:
                        columna_letra = "A"  # Default si no encuentra letra
                else:
                    columna_letra = "A"
                
                # Crear estructura jerárquica
                if tree_name not in arboles_con_nodos:
                    arboles_con_nodos[tree_name] = {}
                
                # Si hay sub-árbol, usarlo como división
                if sub_tree_name and sub_tree_name.strip():
                    if sub_tree_name not in arboles_con_nodos[tree_name]:
                        arboles_con_nodos[tree_name][sub_tree_name] = {}
                    sub_container = arboles_con_nodos[tree_name][sub_tree_name]
                else:
                    # Si no hay sub-árbol, usar directamente el árbol principal
                    if 'PRINCIPAL' not in arboles_con_nodos[tree_name]:
                        arboles_con_nodos[tree_name]['PRINCIPAL'] = {}
                    sub_container = arboles_con_nodos[tree_name]['PRINCIPAL']
                
                # Crear estructura de nivel y columna
                if nivel not in sub_container:
                    sub_container[nivel] = {}
                if columna_letra not in sub_container[nivel]:
                    sub_container[nivel][columna_letra] = []
                
                sub_container[nivel][columna_letra].append({
                    'id': node_id,
                    'nombre': node_info.get('Nombre_Visible', f'Nodo {node_id}'),
                    'costo': node_info.get('Costo', 0),
                    'columna_original': columna_letra,
                    'nivel_original': nivel
                })
        
        # Generar HTML del mapa
        mapa_html = '<div class="node-map-container">'
        
        for tree_name in sorted(arboles_con_nodos.keys()):
            mapa_html += f'<div class="tree-map-section">'
            mapa_html += f'<h4 class="tree-map-title">{tree_name}</h4>'
            
            # Crear contenedor horizontal para sub-árboles
            mapa_html += f'<div class="subtrees-container">'
            
            # Procesar cada sub-árbol
            for sub_tree_key in sorted(arboles_con_nodos[tree_name].keys()):
                sub_tree_data = arboles_con_nodos[tree_name][sub_tree_key]
                
                mapa_html += f'<div class="subtree-section">'
                
                # Solo mostrar título del sub-árbol si no es "PRINCIPAL"
                if sub_tree_key != 'PRINCIPAL':
                    mapa_html += f'<h5 class="subtree-title">{sub_tree_key}</h5>'
                
                mapa_html += f'<div class="tier-map">'
                
                # Obtener todos los niveles para este sub-árbol
                niveles = sorted(sub_tree_data.keys())
                
                # Crear grid de coordenadas para este sub-árbol
                for nivel in niveles:
                    mapa_html += f'<div class="tier-row" data-nivel="{nivel}">'
                    mapa_html += f'<div class="tier-label">N{nivel}</div>'
                    
                    # Obtener todas las columnas para este nivel y ordenarlas
                    columnas_en_nivel = sorted(sub_tree_data[nivel].keys())
                    
                    for columna_letra in columnas_en_nivel:
                        nodes_in_column = sub_tree_data[nivel][columna_letra]
                        for node in nodes_in_column:
                            mapa_html += f'''<div class="node-coord">
                                <div class="coord-label">{columna_letra}{nivel}</div>
                                <div class="node-mini-name">{node['nombre'][:10]}...</div>
                            </div>'''
                    
                    mapa_html += '</div>'
                
                mapa_html += '</div></div>'  # Cerrar tier-map y subtree-section
            
            mapa_html += '</div></div>'  # Cerrar subtrees-container y tree-map-section
        
        mapa_html += '</div>'
        return mapa_html
    
    # Generar sección de nodos desbloqueados (versión existente)
    nodos_html = ""
    if unlocked_nodes and df_skill_trees is not None:
        trees_with_nodes = {}
        for node_id in unlocked_nodes:
            node_data = df_skill_trees[df_skill_trees['ID_Nodo'] == node_id]
            if not node_data.empty:
                node_info = node_data.iloc[0]
                tree_name = node_info.get('Arbol', 'Sin Árbol')
                if tree_name not in trees_with_nodes:
                    trees_with_nodes[tree_name] = []
                trees_with_nodes[tree_name].append({
                    'nombre': node_info.get('Nombre_Visible', f'Nodo {node_id}'),
                    'costo': node_info.get('Costo', 0)
                })
        
        if trees_with_nodes:
            for tree_name, nodes in trees_with_nodes.items():
                nodos_html += f'<div class="tree-section"><h4 class="tree-title">{tree_name}</h4><div class="nodes-grid">'
                for node in nodes:
                    nodos_html += f'''
                    <div class="node-item">
                        <span class="node-check">✓</span>
                        <span class="node-name">{node['nombre']}</span>
                        <span class="node-cost">{node['costo']}pts</span>
                    </div>'''
                nodos_html += '</div></div>'
    
    # Generar sección de instalaciones activas
    instalaciones_html = ""
    if apply_facilities_boost and unlocked_facilities and df_instalaciones is not None:
        # Agrupar instalaciones por base de ID (sin el número de nivel)
        facility_groups = {}
        for facility_id in unlocked_facilities:
            facility_data = df_instalaciones[df_instalaciones['ID_Instalacion'] == facility_id]
            if not facility_data.empty:
                facility_info = facility_data.iloc[0]
                
                # Extraer la base del ID (ej: "FISICO_RITMO_1" -> "FISICO_RITMO")
                base_id = '_'.join(facility_id.split('_')[:-1]) if '_' in facility_id and facility_id.split('_')[-1].isdigit() else facility_id
                
                if base_id not in facility_groups:
                    facility_groups[base_id] = []
                facility_groups[base_id].append({
                    'id': facility_id,
                    'nombre': facility_info.get('Nombre_Instalacion', f'Instalación {facility_id}'),
                    'precio': facility_info.get('Precio', 0),
                    'nivel': int(facility_id.split('_')[-1]) if '_' in facility_id and facility_id.split('_')[-1].isdigit() else 1,
                    'info': facility_info
                })
        
        if facility_groups:
            for base_id, facilities in facility_groups.items():
                # Ordenar por nivel para obtener el más alto
                facilities_sorted = sorted(facilities, key=lambda x: x['nivel'])
                highest_tier = facilities_sorted[-1]
                
                # Calcular costos acumulados de todos los niveles desbloqueados
                costos_acumulados = [str(f['precio']) for f in facilities_sorted]
                costos_str = f"({', '.join(costos_acumulados)})" if len(costos_acumulados) > 1 else f"({costos_acumulados[0]})"
                
                # Calcular beneficios totales de todos los niveles desbloqueados
                total_benefits = {}
                playstyles_facility = []
                
                for facility in facilities_sorted:
                    facility_info = facility['info']
                    
                    # Sumar beneficios estadísticos
                    for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                        if stat_col in facility_info.index and pd.notna(facility_info[stat_col]) and facility_info[stat_col] != 0:
                            if stat_col not in total_benefits:
                                total_benefits[stat_col] = 0
                            total_benefits[stat_col] += int(facility_info[stat_col])
                    
                    # Recoger PlayStyles
                    playstyle_val = facility_info.get('PlayStyle', '')
                    is_plus = str(facility_info.get('EsPlus', '')).strip().lower() == 'si'
                    if pd.notna(playstyle_val) and playstyle_val.strip():
                        ps_display = f"{playstyle_val}{'+' if is_plus else ''}"
                        if ps_display not in playstyles_facility:
                            playstyles_facility.append(ps_display)
                
                # Construir string de beneficios
                benefits_str = ""
                if total_benefits:
                    benefits_list = [f"+{val} {stat}" for stat, val in total_benefits.items()]
                    benefits_str = f"[{', '.join(benefits_list)}]"
                
                if playstyles_facility:
                    if benefits_str:
                        benefits_str += f" [{', '.join(playstyles_facility)}]"
                    else:
                        benefits_str = f"[{', '.join(playstyles_facility)}]"
                
                # Mostrar nombre base y nivel más alto
                # nivel_display = f" {highest_tier['nivel']}" if highest_tier['nivel'] > 1 else ""
                # Mostrar nombre sin añadir nivel (ya está en Nombre_Instalacion)
                instalaciones_html += f'''
                <div class="facility-item">
                    <span class="facility-type">🏨 {highest_tier['nombre']}</span>
                    <span class="facility-price">${sum(f['precio'] for f in facilities_sorted):,} {costos_str}</span>
                    <span class="facility-benefits">{benefits_str}</span>
                </div>'''

    # Recopilar Playstyles
    playstyles = []
    if unlocked_nodes and df_skill_trees is not None:
        for node_id in unlocked_nodes:
            node_data = df_skill_trees[df_skill_trees['ID_Nodo'] == node_id]
            if not node_data.empty:
                playstyle = node_data.iloc[0].get('PlayStyle', '')
                if playstyle and playstyle.strip():
                    playstyles.append(playstyle.strip())
    
    if apply_facilities_boost and unlocked_facilities and df_instalaciones is not None:
        for facility_id in unlocked_facilities:
            facility_data = df_instalaciones[df_instalaciones['ID_Instalacion'] == facility_id]
            if not facility_data.empty:
                playstyle = facility_data.iloc[0].get('PlayStyle', '')
                if playstyle and playstyle.strip():
                    playstyles.append(playstyle.strip())
    
    playstyles_unique = sorted(list(set(playstyles)))
    
    # Generar HTML del gráfico radar
    radar_data = [main_stats[stat] for stat in MAIN_CATEGORIES]
    radar_labels = MAIN_CATEGORIES
    
    html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen Visual FC25 - {pos} {altura}cm {peso}kg</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 0 50px rgba(255, 140, 0, 0.3);
            border: 2px solid rgba(255, 140, 0, 0.5);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #ff8c00, #ffa500, #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin-bottom: 10px;
        }}
        
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .stats-section {{
            background: rgba(255, 140, 0, 0.1);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: scale(1.05);
        }}
        
        .stat-label {{
            font-size: 0.9em;
            margin-bottom: 5px;
            opacity: 0.9;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .special-stats {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        
        .special-stat {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .radar-section {{
            text-align: center;
        }}
        
        .radar-container {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .progress-section {{
            background: rgba(255, 140, 0, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .progress-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 15px;
        }}
        
        .progress-item {{
            text-align: center;
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff8c00, #ffa500);
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .nodes-section {{
            background: rgba(255, 140, 0, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .tree-section {{
            margin-bottom: 20px;
        }}
        
        .tree-title {{
            color: #ffa500;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .nodes-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 8px;
        }}
        
        .node-item {{
            background: rgba(0, 0, 0, 0.3);
            padding: 8px;
            border-radius: 6px;
            border-left: 3px solid #00ff00;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .node-check {{
            color: #00ff00;
            font-weight: bold;
        }}
        
        .node-name {{
            flex: 1;
            font-size: 0.9em;
        }}
        
        .node-cost {{
            color: #ffa500;
            font-size: 0.8em;
        }}
        
        /* NUEVOS ESTILOS PARA EL MAPA DE COORDENADAS */
        .node-map-container {{
            background: rgba(255, 140, 0, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 140, 0, 0.3);
        }}
        
        .tree-map-section {{
            margin-bottom: 25px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 15px;
        }}
        
        .tree-map-title {{
            color: #ffa500;
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.2em;
            border-bottom: 2px solid rgba(255, 140, 0, 0.5);
            padding-bottom: 5px;
        }}
        
        .subtrees-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: space-around;
        }}
        
        .subtree-section {{
            flex: 1;
            min-width: 200px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(255, 140, 0, 0.2);
        }}
        
        .subtree-title {{
            color: #ffcc00;
            font-size: 1em;
            text-align: center;
            margin-bottom: 10px;
            padding-bottom: 3px;
            border-bottom: 1px solid rgba(255, 204, 0, 0.3);
        }}
        
        .tier-map {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .tier-row {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 3px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
        }}
        
        .tier-label {{
            background: #ff8c00;
            color: black;
            padding: 3px 6px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 30px;
            text-align: center;
            font-size: 0.8em;
        }}
        
        .node-coord {{
            background: rgba(0, 255, 0, 0.2);
            border: 2px solid #00ff00;
            border-radius: 6px;
            padding: 3px;
            min-width: 60px;
            text-align: center;
            transition: transform 0.2s ease;
            font-size: 0.8em;
        }}
        
        .node-coord:hover {{
            transform: scale(1.05);
            background: rgba(0, 255, 0, 0.3);
        }}
        
        .node-coord.empty {{
            background: rgba(100, 100, 100, 0.1);
            border: 2px dashed rgba(100, 100, 100, 0.3);
            visibility: hidden;
        }}
        
        .coord-label {{
            font-weight: bold;
            color: #00ff00;
            font-size: 0.7em;
            margin-bottom: 1px;
        }}
        
        .node-mini-name {{
            font-size: 0.6em;
            color: #ffffff;
            opacity: 0.9;
            line-height: 1.1;
        }}
        
        .facilities-section {{
            background: rgba(0, 100, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(0, 100, 255, 0.3);
        }}
        
        .facility-item {{
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid #0066ff;
        }}
        
        .facility-type {{
            color: #0066ff;
            font-weight: bold;
        }}
        
        .facility-name {{
            margin-left: 10px;
        }}
        
        .facility-price {{
            float: right;
            color: #ffa500;
        }}
        
        .playstyles-section {{
            background: rgba(128, 0, 128, 0.1);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(128, 0, 128, 0.3);
        }}
        
        .playstyles-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        
        .playstyle-item {{
            background: rgba(128, 0, 128, 0.2);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(128, 0, 128, 0.4);
        }}
        
        .section-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #ffa500;
            font-weight: bold;
        }}
        
        @media (max-width: 768px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .subtrees-container {{
                flex-direction: column;
            }}
            
            .tier-row {{
                flex-wrap: wrap;
            }}
            
            .node-coord {{
                min-width: 50px;
                font-size: 0.7em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ BUILD FC25 ⚽</h1>
            <h2>{pos} | {altura}cm | {peso}kg</h2>
        </div>
        
        <div class="progress-section">
            <div class="section-title">📊 PROGRESO DEL BUILD</div>
            <div class="progress-grid">
                <div class="progress-item">
                    <div style="color: #ff8c00; font-weight: bold;">Puntos Gastados</div>
                    <div style="font-size: 1.5em;">{puntos_gastados} / {TOTAL_SKILL_POINTS}</div>
                </div>
                <div class="progress-item">
                    <div style="color: #ffd700; font-weight: bold;">Puntos Restantes</div>
                    <div style="font-size: 1.5em;">{points_remaining}</div>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progreso_build:.1f}%"></div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="stats-section">
                <div class="section-title">📈 ATRIBUTOS PRINCIPALES</div>
                <div class="stats-grid">
    """
    
    # Agregar estadísticas principales con colores de fondo y texto contrastante
    for stat, value in main_stats.items():
        colors = get_stat_color(value)
        html_content += f'''
                    <div class="stat-card" style="background: {colors["background"]}; color: {colors["color"]}; border-color: {colors["color"]};">
                        <div class="stat-label">{stat}</div>
                        <div class="stat-value">{value}</div>
                    </div>'''
    
    html_content += f"""
                </div>
                
                <div class="special-stats">
                    <div class="special-stat">
                        <div>⭐ Pierna Mala</div>
                        <div style="font-size: 1.5em;">{"⭐" * pierna_mala}</div>
                    </div>
                    <div class="special-stat">
                        <div>✨ Filigranas</div>
                        <div style="font-size: 1.5em;">{"⭐" * filigranas}</div>
                    </div>
                    <div class="special-stat">
                        <div>🏃 AcceleRATE</div>
                        <div style="font-size: 1.2em;">{accelerate}</div>
                    </div>
                    <div class="special-stat">
                        <div>🎯 IGS Total</div>
                        <div style="font-size: 1.5em; color: #ffa500;">{igs_total}</div>
                    </div>
                </div>
            </div>
            
            <div class="radar-section">
                <div class="section-title">🕸️ GRÁFICO RADAR</div>
                <div class="radar-container">
                    <canvas id="radarChart" width="300" height="300"></canvas>
                </div>
            </div>
        </div>
    """
    
    # NUEVA SECCIÓN: Mapa de coordenadas de nodos
    mapa_nodos_html = generar_mapa_nodos()
    if mapa_nodos_html:
        html_content += f'''
        <div class="section-title">🗺️ MAPA DE COORDENADAS DE NODOS DESBLOQUEADOS</div>
        {mapa_nodos_html}'''
    
    # Agregar secciones condicionales existentes
    # if nodos_html:
    #     html_content += f'''
    #     <div class="nodes-section">
    #         <div class="section-title">🌳 NODOS DE HABILIDAD DESBLOQUEADOS</div>
    #         {nodos_html}
    #     </div>'''
    
    if instalaciones_html:
        html_content += f'''
        <div class="facilities-section">
            <div class="section-title">🏨 INSTALACIONES ACTIVAS</div>
            {instalaciones_html}
        </div>'''
    
    if playstyles_unique:
        playstyles_html = ""
        for playstyle in playstyles_unique:
            playstyles_html += f'<div class="playstyle-item">{playstyle}</div>'
        
        html_content += f'''
        <div class="playstyles-section">
            <div class="section-title">🎮 PLAYSTYLES ACTIVOS</div>
            <div class="playstyles-grid">
                {playstyles_html}
            </div>
        </div>'''
    
    # JavaScript para el gráfico radar
    html_content += f"""
    </div>
    
    <script>
        const ctx = document.getElementById('radarChart').getContext('2d');
        const radarChart = new Chart(ctx, {{
            type: 'radar',
            data: {{
                labels: {radar_labels},
                datasets: [{{
                    label: 'Atributos',
                    data: {radar_data},
                    backgroundColor: 'rgba(255, 140, 0, 0.2)',
                    borderColor: 'rgba(255, 140, 0, 1)',
                    pointBackgroundColor: 'rgba(255, 140, 0, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(255, 140, 0, 1)',
                    borderWidth: 2,
                    pointRadius: 5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 99,
                        min: 0,
                        ticks: {{
                            stepSize: 20,
                            color: '#ffffff',
                            backdrop: 'rgba(0, 0, 0, 0.5)'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.2)'
                        }},
                        angleLines: {{
                            color: 'rgba(255, 255, 255, 0.2)'
                        }},
                        pointLabels: {{
                            color: '#ffffff',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    return html_content

def obtener_stats_jugador(posicion, altura, peso):
    """Obtiene las stats base para una combinación específica"""
    return calcular_stats_base_jugador(posicion, altura, peso, stats_base_lb_rb, 
                                     modificadores_altura, modificadores_peso, diferenciales_posicion)

def aplicar_boosts_a_stats(stats_base, boosts_dict):
    """Aplica un diccionario de boosts a las stats base"""
    if stats_base is None:
        return {}
    
    stats_finales = stats_base.copy()
    for stat, boost in boosts_dict.items():
        if stat in stats_finales:
            stats_finales[stat] = min(stats_finales[stat] + boost, 99)
    
    return stats_finales

# --- Carga de Datos Principal ---
if 'app_version' not in st.session_state or st.session_state.app_version != APP_VERSION:
    previous_app_version = st.session_state.get('app_version', None)
    st.session_state.clear() 
    st.session_state.app_version = APP_VERSION


stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, lista_posiciones, stat_cols_order = [None] * 6
all_stats_df_base = pd.DataFrame()
df_skill_trees_global = pd.DataFrame()
df_instalaciones_global = pd.DataFrame()
carga_completa_exitosa = False 

def calcular_tiers_nodos(df_nodos_a_evaluar):
    """
    Función auxiliar para calcular el tier/nivel de cada nodo basándose en sus prerrequisitos.
    
    Args:
        df_nodos_a_evaluar (pd.DataFrame): DataFrame con los nodos a evaluar
        
    Returns:
        dict: Mapeo de ID_Nodo a su tier/nivel (ej. {'RITMO_A1': 0, 'RITMO_A2': 1})
    """
    if df_nodos_a_evaluar is None or df_nodos_a_evaluar.empty:
        return {}
    
    nodos_por_tier = defaultdict(list)
    processed_nodes = set()
    tier_mapping = {}  # ID_Nodo -> tier_number
    
    # Tier 0: Nodos sin prerrequisitos
    for _, nodo in df_nodos_a_evaluar.iterrows():
        prereq_val = nodo.get('Prerrequisito', '')
        if pd.isna(prereq_val) or str(prereq_val).strip() == '':
            nodos_por_tier[0].append(nodo)
            processed_nodes.add(nodo['ID_Nodo'])
            tier_mapping[nodo['ID_Nodo']] = 0
    
    current_tier = 0
    max_iterations = len(df_nodos_a_evaluar) + 10  # Prevenir bucles infinitos
    iteration_count = 0
    
    # Procesar nodos restantes basándose en prerrequisitos
    while len(processed_nodes) < len(df_nodos_a_evaluar) and iteration_count < max_iterations:
        newly_added_to_tier = 0
        
        for _, nodo in df_nodos_a_evaluar.iterrows():
            if nodo['ID_Nodo'] in processed_nodes:
                continue
            
            prereqs_list = [pr_id.strip() for pr_id in str(nodo.get('Prerrequisito', '')).split(',') if pr_id.strip()]
            
            if prereqs_list and all(pr_id in processed_nodes for pr_id in prereqs_list):
                # Encontrar el tier máximo de los prerrequisitos
                max_prereq_tier = -1
                for pr_id in prereqs_list:
                    if pr_id in tier_mapping:
                        max_prereq_tier = max(max_prereq_tier, tier_mapping[pr_id])
                
                # Asignar al tier siguiente del prerrequisito más alto
                node_tier = max_prereq_tier + 1
                nodos_por_tier[node_tier].append(nodo)
                processed_nodes.add(nodo['ID_Nodo'])
                tier_mapping[nodo['ID_Nodo']] = node_tier
                newly_added_to_tier += 1
        
        if newly_added_to_tier == 0:
            # Si no se agregó ningún nodo, colocar los restantes en un tier especial
            remaining_tier = current_tier + 100
            for _, nodo_restante in df_nodos_a_evaluar.iterrows():
                if nodo_restante['ID_Nodo'] not in processed_nodes:
                    nodos_por_tier[remaining_tier].append(nodo_restante)
                    processed_nodes.add(nodo_restante['ID_Nodo'])
            break
        
        current_tier += 1
        iteration_count += 1
    
    return tier_mapping

datos_base_cargados_tuple = cargar_y_preparar_datos_base()
if datos_base_cargados_tuple:
    stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, lista_posiciones, stat_cols_order = datos_base_cargados_tuple
    if all(item is not None for item in datos_base_cargados_tuple) and lista_posiciones and stat_cols_order:
        all_stats_df_base = precompute_all_base_stats(stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, stat_cols_order)
        if all_stats_df_base is not None and not all_stats_df_base.empty:
            df_skill_trees_global = cargar_arbol_habilidades()
            if df_skill_trees_global is not None and not df_skill_trees_global.empty:
                df_instalaciones_global = cargar_instalaciones_club()
                if df_instalaciones_global is not None and not df_instalaciones_global.empty:
                    carga_completa_exitosa = True
                else: st.error("Fallo al cargar datos de Instalaciones.")
            else: st.error("Fallo al cargar datos del Árbol de Habilidades.")
        else: st.error("Fallo en el pre-cálculo de stats base.")
    else: st.error("Componentes de datos base faltantes o vacíos.")
else: st.error("Fallo crítico en la carga de datos base iniciales.")

if carga_completa_exitosa:
    for key_init in ['bc_unlocked_nodes', 'filters', 'unlocked_facility_levels', 'build_save_name_v33']: 
        if key_init not in st.session_state: 
            if key_init.endswith('nodes') or key_init.endswith('levels'):
                st.session_state[key_init] = set()
            elif key_init == 'build_save_name_v33': 
                 st.session_state[key_init] = ""
            else: 
                st.session_state[key_init] = []
    
    _sorted_lp_sb_init = sorted(lista_posiciones)
    _unique_alts_sb_init = sorted(modificadores_altura.index.unique().tolist())
    _unique_pesos_sb_init = sorted(modificadores_peso.index.unique().tolist())

    default_pos_init = _sorted_lp_sb_init[_sorted_lp_sb_init.index('ST') if 'ST' in _sorted_lp_sb_init else 0]
    default_alt_init = _unique_alts_sb_init[_unique_alts_sb_init.index(180) if 180 in _unique_alts_sb_init else 0]
    default_pes_init = _unique_pesos_sb_init[_unique_pesos_sb_init.index(75) if 75 in _unique_pesos_sb_init else 0]
    
    if 'bc_pos' not in st.session_state: st.session_state.bc_pos = default_pos_init
    if 'bc_alt' not in st.session_state: st.session_state.bc_alt = default_alt_init
    if 'bc_pes' not in st.session_state: st.session_state.bc_pes = default_pes_init
    
    if 'bc_points_total' not in st.session_state: st.session_state.bc_points_total = TOTAL_SKILL_POINTS
    if 'bc_points_remaining' not in st.session_state: st.session_state.bc_points_remaining = TOTAL_SKILL_POINTS
    if 'club_budget_total' not in st.session_state: st.session_state.club_budget_total = DEFAULT_CLUB_BUDGET 
    if 'club_budget_remaining' not in st.session_state: st.session_state.club_budget_remaining = st.session_state.club_budget_total 
    if 'apply_facility_boosts_toggle' not in st.session_state: st.session_state.apply_facility_boosts_toggle = True
    if 'next_filter_id' not in st.session_state: st.session_state.next_filter_id = 0
else: 
    st.error("La aplicación no pudo cargar todos los datos necesarios. Revisa los mensajes de error. La aplicación se detendrá.")
    st.stop()

# --- Interfaz de Usuario con Streamlit ---
st.title(f"FC Pro Clubs Builder ({APP_VERSION})")
PLAYER_COLORS = ['rgba(0,100,255,0.7)', 'rgba(220,50,50,0.7)', 'rgba(0,170,0,0.7)'] 
PLAYER_FILL_COLORS = [color.replace('0.7', '0.3') for color in PLAYER_COLORS]

def highlight_max_in_row(row_data):
    if row_data.dtype == 'object': return ['' for _ in row_data]
    if row_data.isnull().all(): return ['' for _ in row_data]
    try: max_val = row_data.max()
    except TypeError: return ['' for _ in row_data]
    return ['background-color: #d4edda; color: #155724; font-weight: bold;' if (pd.notna(v) and v == max_val) else '' for v in row_data]

def styled_metric(label, value):
    val_num = 0; apply_color_coding = label in MAIN_CATEGORIES
    if isinstance(value, (int, float)): val_num = value
    elif isinstance(value, str):
        try: val_num = float(value.replace(STAR_EMOJI,'')) 
        except ValueError: pass
    bg_color, text_color = "inherit", "inherit" 
    if apply_color_coding and val_num > 0 :
        color_map = {(95, 100): ("#b4a7d6", "black"), (85, 94): ("#b6d7a8", "#003300"), (75, 84): ("#fce8b2", "#594400"), (0, 74): ("#f4c7c3", "#800000"),}
        for (lower, upper), (bg, txt) in color_map.items():
            if lower <= val_num <= upper: bg_color, text_color = bg, txt; break
    st.markdown(f"""<div style="background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px; text-align: center;">
        <div style="font-size: 0.8em; {"color: #555555;" if bg_color == "inherit" else ""}">{label}</div>
        <div style="font-size: 1.7em; font-weight: bold;">{value}</div></div>""", unsafe_allow_html=True)


# Definición de Pestañas
tab_calc, tab_build_craft, tab_facilities, tab_explorer, tab_best_combo, tab_filters, tab_reverse = st.tabs([
    "🧮 Comparador", "🛠️ Build Craft", "🏨 Instalaciones Club", 
    "🔎 Explorador Mejoras", "🔍 Búsqueda Óptima", "📊 Filtros Múltiples", "🕵️‍♂️ Detective de Builds"
])

# --- Pestaña: Calculadora y Comparador ---
with tab_calc: 
    st.header("Comparador de Perfiles (Stats Base)")
    num_players_to_compare_calc = st.radio("Jugadores a definir/comparar:", (1, 2, 3), index=0, horizontal=True, key="num_players_radio_v33_calc") 
    cols_selectors_calc = st.columns(num_players_to_compare_calc)
    player_stats_list_base_calc, player_configs_base_calc = [], []

    for i in range(num_players_to_compare_calc):
        with cols_selectors_calc[i]:
            player_label = f"JUG {chr(65+i)}"
            st.subheader(player_label)
            pos_key, alt_key, pes_key = f"pos_p{i}_v33_calc", f"alt_p{i}_v33_calc", f"pes_p{i}_v33_calc" 
            
            if pos_key not in st.session_state: st.session_state[pos_key] = _sorted_lp_sb_init[0]
            idx_pos_val_calc = _sorted_lp_sb_init.index(st.session_state[pos_key]) if st.session_state.get(pos_key) in _sorted_lp_sb_init else 0
            pos_val_c = st.selectbox(f"Pos ({player_label}):", _sorted_lp_sb_init, index=idx_pos_val_calc, key=pos_key)

            if alt_key not in st.session_state: st.session_state[alt_key] = _unique_alts_sb_init[0]
            idx_alt_val_calc = _unique_alts_sb_init.index(st.session_state[alt_key]) if st.session_state.get(alt_key) in _unique_alts_sb_init else 0
            alt_val_c = st.selectbox(f"Alt ({player_label}):", _unique_alts_sb_init, index=idx_alt_val_calc, key=alt_key)

            if pes_key not in st.session_state: st.session_state[pes_key] = _unique_pesos_sb_init[0]
            idx_pes_val_calc = _unique_pesos_sb_init.index(st.session_state[pes_key]) if st.session_state.get(pes_key) in _unique_pesos_sb_init else 0
            pes_val_c = st.selectbox(f"Pes ({player_label}):", _unique_pesos_sb_init, index=idx_pes_val_calc, key=pes_key)
            
            player_configs_base_calc.append({'pos': pos_val_c, 'alt': alt_val_c, 'pes': pes_val_c, 'label': player_label})
            if pos_val_c and alt_val_c is not None and pes_val_c is not None:
                stats = calcular_stats_base_jugador(pos_val_c, alt_val_c, pes_val_c, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
                if stats is not None:
                    stats['AcceleRATE'] = determinar_estilo_carrera(alt_val_c, stats.get('AGI',0), stats.get('STR',0), stats.get('Acc',0))
                    player_stats_list_base_calc.append(stats)
                    if st.button(f"🛠️ Enviar a Build Craft ({player_label})", key=f"send_to_bc_calc_{i}_v33"): 
                        st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = pos_val_c, alt_val_c, pes_val_c
                        st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                        st.session_state.unlocked_facility_levels, st.session_state.club_budget_remaining = set(), st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                        st.success(f"Perfil de {player_label} enviado. Ve a '🛠️ Build Craft'.")
                else: player_stats_list_base_calc.append(None)
            else: player_stats_list_base_calc.append(None)
    st.divider(); st.subheader("Perfiles y Estadísticas (Base)")
    if any(ps is not None for ps in player_stats_list_base_calc):
        fig_radar_comp_calc = go.Figure(); radar_attrs_calc = MAIN_CATEGORIES
        cols_perfiles_calc = st.columns(num_players_to_compare_calc)
        for i, stats_item_calc in enumerate(player_stats_list_base_calc): 
            if stats_item_calc is not None:
                with cols_perfiles_calc[i]:
                    st.markdown(f"**{player_configs_base_calc[i]['label']} ({player_configs_base_calc[i]['pos']}, {player_configs_base_calc[i]['alt']}cm, {player_configs_base_calc[i]['pes']}kg)**")
                    styled_metric("Estilo de Carrera", stats_item_calc.get('AcceleRATE', "N/A"))
                valid_radar_attrs_calc = [attr for attr in radar_attrs_calc if attr in stats_item_calc.index]
                radar_values_calc = [stats_item_calc.get(attr, 0) for attr in valid_radar_attrs_calc]
                if len(valid_radar_attrs_calc) >= 3:
                    fig_radar_comp_calc.add_trace(go.Scatterpolar(r=radar_values_calc, theta=valid_radar_attrs_calc, fill='toself', name=f"{player_configs_base_calc[i]['label']} ({player_configs_base_calc[i]['pos']})",line_color=PLAYER_COLORS[i % len(PLAYER_COLORS)],fillcolor=PLAYER_FILL_COLORS[i % len(PLAYER_FILL_COLORS)]))
        if fig_radar_comp_calc.data:
            fig_radar_comp_calc.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Comparativa de Perfiles Base")
            st.plotly_chart(fig_radar_comp_calc, use_container_width=True)
        
        valid_player_stats_b_calc = [ps for ps in player_stats_list_base_calc if ps is not None] 
        
 
        valid_player_col_names_b_calc = [player_configs_base_calc[i]['label'] for i, ps in enumerate(player_stats_list_base_calc) if ps is not None] 
        if len(valid_player_stats_b_calc) > 1 :
            compare_dict_b_calc = {name: stats.drop('AcceleRATE', errors='ignore') for name, stats in zip(valid_player_col_names_b_calc, valid_player_stats_b_calc)} 
            df_compare_b_calc = pd.DataFrame(compare_dict_b_calc) 
            for col_df_compare_stable_calc in df_compare_b_calc.columns:
                df_compare_b_calc[col_df_compare_stable_calc] = pd.to_numeric(df_compare_b_calc[col_df_compare_stable_calc], errors='coerce').fillna(0)
            df_compare_styled_b_calc = df_compare_b_calc.style.apply(highlight_max_in_row, axis=1) 
            st.subheader("Tabla Comparativa de Estadísticas Base (Numéricas)")
            st.dataframe(df_compare_styled_b_calc)
            accele_rates_info_calc = {name: player_stats_list_base_calc[j].get('AcceleRATE', 'N/A') for j, name in enumerate(valid_player_col_names_b_calc)} 
            st.caption(f"Estilos de Carrera: {accele_rates_info_calc}")
        elif len(valid_player_stats_b_calc) == 1:
            st.subheader(f"Estadísticas Detalladas Base: {player_configs_base_calc[0]['label']}")
            stats_to_display_single_calc = valid_player_stats_b_calc[0] 
            accele_rate_single_calc = stats_to_display_single_calc.get('AcceleRATE', 'N/A')
            styled_metric("Estilo de Carrera", accele_rate_single_calc)
            display_series_calc_single = stats_to_display_single_calc.copy() 
            if 'AcceleRATE' in display_series_calc_single.index: display_series_calc_single['AcceleRATE'] = str(display_series_calc_single['AcceleRATE'])
            st.dataframe(display_series_calc_single.rename("Valor").astype(str))
        
    st.divider(); st.header("🏆 Top 5 Combinaciones por IGS (Stats Base)")
    if all_stats_df_base is not None and not all_stats_df_base.empty and 'IGS' in all_stats_df_base.columns:
        # Preparar datos del Top 5
        all_stats_df_base_display_top5_calc = all_stats_df_base.copy()
        all_stats_df_base_display_top5_calc['IGS'] = pd.to_numeric(all_stats_df_base_display_top5_calc['IGS'], errors='coerce').fillna(0)
        
        # Añadir AcceleRATE si no existe
        if 'AcceleRATE' not in all_stats_df_base_display_top5_calc.columns:
            all_stats_df_base_display_top5_calc['AcceleRATE'] = all_stats_df_base_display_top5_calc.apply(
                lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI', 0), r.get('STR', 0), r.get('Acc', 0)), 
                axis=1
            )
        
        # Obtener Top 5
        top_5_igs_calc_df = all_stats_df_base_display_top5_calc.sort_values(by='IGS', ascending=False).head(5)
        
        # Mostrar tabla del Top 5
        st.dataframe(
            top_5_igs_calc_df[['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']],
            use_container_width=True
        )
        
        # SECCIÓN DE BOTONES PARA ENVIAR A BUILD CRAFT
        st.markdown("**🛠️ Enviar Perfiles a Build Craft:**")
        
        # Crear columnas para los botones (máximo 5 columnas)
        num_top_cols_stable_v34 = min(len(top_5_igs_calc_df), 5)
        if num_top_cols_stable_v34 > 0:
            cols_top5_buttons_stable_v34 = st.columns(num_top_cols_stable_v34)
            
            # Crear un botón por cada perfil del Top 5
            for i, row_idx in enumerate(top_5_igs_calc_df.index):
                row = top_5_igs_calc_df.loc[row_idx]
                
                with cols_top5_buttons_stable_v34[i % num_top_cols_stable_v34]:
                    # Crear contenedor con información del perfil
                    with st.container(border=True):
                        st.markdown(f"**Top #{i+1}**")
                        st.markdown(f"**{row['Posicion']}** | {row['Altura']}cm | {row['Peso']}kg")
                        st.caption(f"IGS: {int(row['IGS'])}")
                        st.caption(f"AcceleRATE: {row['AcceleRATE']}")
                        
                        # Botón para enviar a Build Craft
                        if st.button(
                            f"🛠️ Editar", 
                            key=f"send_to_bc_top5_{row_idx}_v34",
                            help=f"Enviar {row['Posicion']} {row['Altura']}cm {row['Peso']}kg a Build Craft",
                            use_container_width=True
                        ):
                            # ACCIÓN: Actualizar el perfil base del Build Craft
                            st.session_state.bc_pos = row['Posicion']
                            st.session_state.bc_alt = row['Altura'] 
                            st.session_state.bc_pes = row['Peso']
                            
                            # ACCIÓN: Resetear el progreso del build del jugador
                            st.session_state.bc_unlocked_nodes = set()
                            st.session_state.bc_points_remaining = TOTAL_SKILL_POINTS
                            
                            # ACCIÓN: Resetear las instalaciones del club seleccionadas
                            st.session_state.unlocked_facility_levels = set()
                            st.session_state.club_budget_remaining = st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                            
                            # ACCIÓN: Informar al usuario
                            st.success(f"✅ Perfil Top #{i+1} ({row['Posicion']} {row['Altura']}cm {row['Peso']}kg) enviado a Build Craft!")
                            st.info("🔄 Ve a la pestaña '🛠️ Build Craft' para personalizar este perfil.")
                            
                            # ACCIÓN: Forzar actualización (opcional pero recomendado)
                            st.rerun()
        else:
            st.warning("No hay perfiles suficientes para mostrar el Top 5.")
            
    else: 
        st.warning("Datos para el Top 5 (base) no disponibles.")



    # --- Pestaña: Build Craft ---
with tab_build_craft:
    # --- Selectores de Perfil Base para Build Craft en la Barra Lateral ---
    st.markdown("### 🛠️ Perfil Base para Build Craft")
    # Agrupar selectores en una fila de 3 columnas
    col_pos, col_alt, col_pes = st.columns(3)
    with col_pos:
        idx_pos_sidebar_v32 = _sorted_lp_sb_init.index(st.session_state.bc_pos) if st.session_state.bc_pos in _sorted_lp_sb_init else 0
        st.session_state.bc_pos = st.selectbox("Posición", _sorted_lp_sb_init, index=idx_pos_sidebar_v32, key="sb_bc_pos_v33")
    with col_alt:
        idx_alt_sidebar_v32 = _unique_alts_sb_init.index(st.session_state.bc_alt) if st.session_state.bc_alt in _unique_alts_sb_init else 0
        st.session_state.bc_alt = st.selectbox("Altura (cm)", _unique_alts_sb_init, index=idx_alt_sidebar_v32, key="sb_bc_alt_v33")
    with col_pes:
        idx_pes_sidebar_v32 = _unique_pesos_sb_init.index(st.session_state.bc_pes) if st.session_state.bc_pes in _unique_pesos_sb_init else 0
        st.session_state.bc_pes = st.selectbox("Peso (kg)", _unique_pesos_sb_init, index=idx_pes_sidebar_v32, key="sb_bc_pes_v33")

    # Checkbox y métricas en una fila
    col_chk, col_pts, col_prog, col_cost = st.columns([2,1,1,1])
    with col_chk:
        st.session_state.apply_facility_boosts_toggle = st.checkbox(
            "Aplicar Boosts de Instalaciones del Club",
            value=st.session_state.get('apply_facility_boosts_toggle', True),
            key="facility_boost_toggle_v33"
        )

    if not carga_completa_exitosa: 
        st.error("Faltan datos para el Build Craft.")
    else:
       
        jugador_base_actual_bc = calcular_stats_base_jugador(st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
        if jugador_base_actual_bc is None:
            st.error("ERROR CRÍTICO (Build Craft): No se pudieron calcular las stats base para el perfil actual. Las stats se mostrarán en cero.")

        st.header(f"🛠️ Build Craft: {st.session_state.bc_pos} | {st.session_state.bc_alt}cm | {st.session_state.bc_pes}kg")
        jugador_base_actual_bc = calcular_stats_base_jugador(st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
        stats_completas_bc = calcular_stats_completas( 
            jugador_base_actual_bc, st.session_state.bc_alt,
            df_skill_trees_global, st.session_state.bc_unlocked_nodes,
            df_instalaciones_global, st.session_state.unlocked_facility_levels, 
            st.session_state.apply_facility_boosts_toggle,
            stat_cols_order 
        )
        st.subheader("Perfil Actual del Build")
        col_bc_info_main, col_bc_radar = st.columns([1,1]) 
        with col_bc_info_main: 
            styled_metric("Puntos Restantes", f"{st.session_state.bc_points_remaining} / {TOTAL_SKILL_POINTS}")
            wf_stars = int(float(str(stats_completas_bc.get('PIERNA_MALA', BASE_WF))))
            sm_stars = int(float(str(stats_completas_bc.get('FILIGRANAS', BASE_SM))))
            styled_metric("Pierna Mala (WF)", STAR_EMOJI * wf_stars)
            styled_metric("Filigranas (SM)", STAR_EMOJI * sm_stars)
            styled_metric("Estilo de Carrera", str(stats_completas_bc.get('AcceleRATE', "N/A")))
            styled_metric("IGS (Total Build)", f"{int(float(str(stats_completas_bc.get('IGS', '0'))))}")
            st.markdown("---"); st.markdown("**Atributos Generales (Total Build):**")
            cols_main_stats_bc = st.columns(2) 
            for idx, stat_name in enumerate(MAIN_CATEGORIES):
                with cols_main_stats_bc[idx % 2]:
                    styled_metric(stat_name, int(float(str(stats_completas_bc.get(stat_name, '0')))))
        with col_bc_radar: 
            radar_attrs = MAIN_CATEGORIES 
            valid_radar_attrs = [attr for attr in radar_attrs if attr in stats_completas_bc.index] 
            radar_values = [int(float(str(stats_completas_bc.get(attr,0)))) for attr in valid_radar_attrs] 
            if len(valid_radar_attrs) >=3:
                fig_radar_bc = go.Figure() 
                fig_radar_bc.add_trace(go.Scatterpolar(r=radar_values, theta=valid_radar_attrs, fill='toself', name="Build Completo", line_color=PLAYER_COLORS[0]))
                fig_radar_bc.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Perfil Actual del Build")
                st.plotly_chart(fig_radar_bc, use_container_width=True)

        with st.expander("Ver todas las estadísticas del Build Actual", expanded=False):
            df_display_build_stats = pd.DataFrame(stats_completas_bc.astype(str)).reset_index() 
            df_display_build_stats.columns = ["Atributo", "Valor Build"]
            st.dataframe(df_display_build_stats)
        
        st.divider() 
        with st.expander("📋 Generar/Copiar Resumen del Build Actual", expanded=False):
            # Crear dos columnas para los dos tipos de resúmenes
            col_texto, col_visual = st.columns(2)
            # === COLUMNA IZQUIERDA: RESUMEN DE TEXTO (FUNCIONALIDAD EXISTENTE) ===
            with col_texto:
                st.markdown("**📝 Resumen de Texto**")
                if st.button("Generar Resumen de Texto", key="btn_generate_summary_v33_bc"):
                    summary_text = f"**Resumen del Build: {st.session_state.bc_pos} | {st.session_state.bc_alt}cm | {st.session_state.bc_pes}kg**\n"
                    summary_text += f"Boosts de Instalaciones Aplicados: {'Sí' if st.session_state.apply_facility_boosts_toggle else 'No'}\n\n"
                    summary_text += f"- **Puntos de Habilidad:** {TOTAL_SKILL_POINTS - st.session_state.bc_points_remaining} / {TOTAL_SKILL_POINTS} (Restantes: {st.session_state.bc_points_remaining})\n"
                    summary_text += f"- **Pierna Mala:** {STAR_EMOJI * int(float(str(stats_completas_bc.get('PIERNA_MALA', BASE_WF))))}\n"
                    summary_text += f"- **Filigranas:** {STAR_EMOJI * int(float(str(stats_completas_bc.get('FILIGRANAS', BASE_SM))))}\n"
                    summary_text += f"- **AcceleRATE:** {stats_completas_bc.get('AcceleRATE', 'N/A')}\n"
                    summary_text += f"- **IGS (Total):** {int(float(str(stats_completas_bc.get('IGS', '0'))))}\n\n"
                    summary_text += "**Atributos Generales (Total):**\n"
                    for cat in MAIN_CATEGORIES: 
                        summary_text += f"  - {cat}: {int(float(str(stats_completas_bc.get(cat, '0'))))}\n"
                    
                    # NUEVA LÓGICA MEJORADA PARA NODOS DE HABILIDAD
                    summary_text += "\n**Nodos de Habilidad Desbloqueados:**\n"
                    
                    # Inicializar lista de playstyles antes del bloque if
                    playstyles_from_skills_sum = []
                    
                    if st.session_state.bc_unlocked_nodes:
                        # Paso A: Obtener el mapeo de tiers
                        mapa_de_tiers = calcular_tiers_nodos(df_skill_trees_global)
                        
                        # Paso B: Agrupar los nodos desbloqueados
                        nodos_agrupados = defaultdict(lambda: defaultdict(list))
                        
                        for node_id in st.session_state.bc_unlocked_nodes:
                            if node_id in df_skill_trees_global['ID_Nodo'].values:
                                node_info = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                                
                                # Obtener tier del nodo
                                tier = mapa_de_tiers.get(node_id, 999)  # 999 si no se encuentra
                                
                                # Agregar detalles del nodo a la estructura agrupada
                                arbol = node_info['Arbol']
                                detalles_nodo = {
                                    'nombre': node_info['Nombre_Visible'],
                                    'costo': node_info['Costo'],
                                    'id': node_id
                                }
                                nodos_agrupados[arbol][tier].append(detalles_nodo)
                                
                                # Recopilar PlayStyles
                                playstyle_node_val = node_info.get('PlayStyle', '')
                                is_plus_val_node = str(node_info.get('EsPlus', '')).strip().lower() == 'si'
                                if pd.notna(playstyle_node_val) and playstyle_node_val != '': 
                                    playstyles_from_skills_sum.append(f"{playstyle_node_val}{'+' if is_plus_val_node else ''}")
                        
                        # Paso C: Construir el String del Resumen con estructura jerárquica
                        for arbol_nombre in sorted(nodos_agrupados.keys()):
                            summary_text += f"  - **{arbol_nombre}:**\n"
                            
                            # Ordenar tiers dentro del árbol
                            tiers_del_arbol = sorted(nodos_agrupados[arbol_nombre].keys())

                            for tier_num in tiers_del_arbol:
                                nodos_del_tier = nodos_agrupados[arbol_nombre][tier_num]
                                
                                # Nombre del tier
                                if tier_num < 100:
                                    tier_name = f"Nivel {tier_num}"
                                else:
                                    tier_name = f"Otros (Grupo {tier_num})"
                                
                                summary_text += f"    - *{tier_name}:*\n"
                                  # Ordenar nodos dentro del tier por nombre para consistencia
                                nodos_del_tier_ordenados = sorted(nodos_del_tier, key=lambda x: x['nombre'])
                                
                                for nodo_detalle in nodos_del_tier_ordenados:
                                    summary_text += f"      - {nodo_detalle['nombre']} (Costo: {nodo_detalle['costo']})\n"
                    else:
                        summary_text += "  - Ningún nodo desbloqueado\n"
                    
                    playstyles_from_facilities_sum = []
                    if st.session_state.apply_facility_boosts_toggle and 'unlocked_facility_levels' in st.session_state:
                        for facility_id_sum in st.session_state.unlocked_facility_levels:
                            if facility_id_sum in df_instalaciones_global['ID_Instalacion'].values:
                                facility_info_sum = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id_sum].iloc[0]
                                ps_facility_val_sum = facility_info_sum.get('PlayStyle', '')
                                is_plus_val_fac_sum = facility_info_sum.get('EsPlus', '').strip().lower() == 'si' 
                                if pd.notna(ps_facility_val_sum) and ps_facility_val_sum != '':
                                     playstyles_from_facilities_sum.append(f"{ps_facility_val_sum}{'+' if is_plus_val_fac_sum else ''} (Inst.)")
                    
                    total_playstyles_sum = sorted(list(set(playstyles_from_skills_sum + playstyles_from_facilities_sum)))
                    if total_playstyles_sum:
                        summary_text += "\n**PlayStyles Totales Desbloqueados:**\n"
                        for ps_name_sum_val_disp in total_playstyles_sum: 
                            summary_text += f"  - {ps_name_sum_val_disp}\n"
                    
                    # Almacenar el resumen en session_state para persistencia
                    st.session_state.build_summary_text = summary_text
                
                # Mostrar el text_area con el resumen (usar valor por defecto si no existe)
                display_summary = st.session_state.get('build_summary_text', "Presiona 'Generar Resumen de Texto' para ver el resumen aquí.")
                st.text_area("Copia este resumen:", display_summary, height=450, key="build_summary_text_area_v33")
            
            # === COLUMNA DERECHA: RESUMEN VISUAL HTML (NUEVA FUNCIONALIDAD) ===
            with col_visual:
                st.markdown("**🎮 Resumen Visual FC25**")
                if st.button("🎮 Generar Resumen Visual HTML", key="btn_generate_visual_summary_v33_bc"):
                    # Generar el HTML visual del build
                    visual_html = generar_resumen_visual_html(
                        st.session_state.bc_pos, 
                        st.session_state.bc_alt, 
                        st.session_state.bc_pes,
                        stats_completas_bc,
                        st.session_state.bc_unlocked_nodes,
                        df_skill_trees_global,
                        st.session_state.unlocked_facility_levels,
                        df_instalaciones_global,
                        st.session_state.apply_facility_boosts_toggle,
                        st.session_state.bc_points_remaining
                    )
                    
                    # Preparar nombre del archivo
                    build_name_for_file = st.session_state.get("build_save_name_v33", "").strip()
                    if build_name_for_file:
                        file_name_html = f"{build_name_for_file.replace(' ', '_')}_resumen_visual_fc25.html"
                    else:
                        file_name_html = f"{st.session_state.bc_pos}_{st.session_state.bc_alt}cm_{st.session_state.bc_pes}kg_resumen_visual_fc25.html"
                    
                    # Botón de descarga
                    st.download_button(
                        label="📥 Descargar Resumen Visual HTML",
                        data=visual_html,
                        file_name=file_name_html,
                        mime="text/html",
                        key="download_visual_summary_btn_v33",
                        help="Descarga un archivo HTML que puedes abrir en cualquier navegador para tomar capturas de pantalla de alta calidad"
                    )
                    
                    st.success("✅ Resumen visual generado! Haz clic en 'Descargar' para obtener el archivo HTML.")
                    st.info("💡 Abre el archivo descargado en tu navegador y toma una captura de pantalla de alta calidad.")
        
        st.divider()
        
        with st.expander("💾 Gestión de Builds (Guardar/Cargar)"):
            
            # Verificar si venimos de una recarga después de cargar un archivo
            if 'just_loaded_file' in st.session_state and st.session_state.just_loaded_file:
                # Limpiar el flag para evitar un bucle infinito
                st.session_state.just_loaded_file = False
                # Crear un key diferente para el uploader para forzar que se limpie
                uploader_key = "build_file_uploader_cleared"
            else:
                uploader_key = "build_file_uploader_v33"
                
            uploaded_build_file = st.file_uploader("📤 Cargar Build desde Archivo (.json):", type=["json"], key=uploader_key)

            if uploaded_build_file is not None:
                try:
                    content_string = uploaded_build_file.read().decode()
                    data_cargada = json.loads(content_string)

                    required_keys = ['base_profile', 'nodos_habilidad_desbloqueados', 'instalaciones_desbloqueadas', 'aplicar_boost_instalaciones']
                    if not all(key in data_cargada for key in required_keys) or \
                       not all(key in data_cargada['base_profile'] for key in ['posicion', 'altura', 'peso']):
                        st.error("Formato de archivo de build incorrecto o claves faltantes.")
                    else:
                        # Actualizar st.session_state con los datos cargados
                        st.session_state.bc_pos = data_cargada['base_profile']['posicion']
                        st.session_state.bc_alt = data_cargada['base_profile']['altura']
                        st.session_state.bc_pes = data_cargada['base_profile']['peso']
                        st.session_state.bc_unlocked_nodes = set(data_cargada.get('nodos_habilidad_desbloqueados', []))
                        st.session_state.unlocked_facility_levels = set(data_cargada.get('instalaciones_desbloqueadas', []))
                        st.session_state.apply_facility_boosts_toggle = data_cargada.get('aplicar_boost_instalaciones', True)
                        
                        loaded_build_name = data_cargada.get("build_name", "Build Cargado Sin Nombre")
                        # No actualizar st.session_state.build_save_name_v33 aquí para evitar error de widget
                        
                        # Recalcular puntos de habilidad restantes
                        puntos_gastados_skills = 0
                        if df_skill_trees_global is not None and not df_skill_trees_global.empty:
                            for node_id_loaded in st.session_state.bc_unlocked_nodes:
                                node_data_cargado_series = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id_loaded]
                                if not node_data_cargado_series.empty:
                                    costo_nodo_loaded = node_data_cargado_series.iloc[0]['Costo']
                                    puntos_gastados_skills += costo_nodo_loaded
                        st.session_state.bc_points_remaining = TOTAL_SKILL_POINTS - puntos_gastados_skills
                        
                        # Recalcular presupuesto restante
                        costo_total_instalaciones_cargadas = 0
                        if df_instalaciones_global is not None and not df_instalaciones_global.empty:
                            for facility_id_loaded in st.session_state.unlocked_facility_levels:
                                facility_data_cargada_series = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id_loaded]
                                if not facility_data_cargada_series.empty:
                                    precio_facility_loaded = facility_data_cargada_series.iloc[0]['Precio']
                                    costo_total_instalaciones_cargadas += precio_facility_loaded
                        
                        current_total_budget_on_load = st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                        st.session_state.club_budget_remaining = current_total_budget_on_load - costo_total_instalaciones_cargadas
                        
                        st.success(f"¡Build '{loaded_build_name}' cargado exitosamente! La página se actualizará.")
                        # Guardar un flag para saber que acabamos de cargar un archivo
                        st.session_state.just_loaded_file = True
                        st.rerun() # Forzar rerun para que todos los widgets reflejen el nuevo estado

                except json.JSONDecodeError:
                    st.error("Error: El archivo proporcionado no es un JSON válido.")
                except Exception as e:
                    st.error(f"Error al procesar el build cargado: {e}")
            
            # Widgets para guardar build
            st.text_input("Nombre para este Build (opcional):", key="build_save_name_v33", value=st.session_state.get("build_save_name_v33",""))

            def prepare_build_data_to_save():
                # Obtener el nombre del build del input, con fallback
                build_name_input = st.session_state.get("build_save_name_v33", "").strip()
                final_build_name = build_name_input if build_name_input else f"{st.session_state.bc_pos}_{st.session_state.bc_alt}cm_{st.session_state.bc_pes}kg"
                
                build_data = {
                    "app_version": f"{APP_VERSION}_build_data", 
                    "build_name": final_build_name,  # Usar el nombre correcto
                    "base_profile": {
                        "posicion": st.session_state.bc_pos,
                        "altura": st.session_state.bc_alt,
                        "peso": st.session_state.bc_pes
                    },
                    "nodos_habilidad_desbloqueados": list(st.session_state.bc_unlocked_nodes),
                    "instalaciones_desbloqueadas": list(st.session_state.unlocked_facility_levels),
                    "aplicar_boost_instalaciones": st.session_state.apply_facility_boosts_toggle
                }
                return json.dumps(build_data, indent=2, ensure_ascii=False)

            # Generar nombre de archivo mejorado
            build_name_for_file = st.session_state.get("build_save_name_v33", "").strip()
            if build_name_for_file:
                # Si hay nombre personalizado, usarlo + datos del perfil
                file_name_to_save = f"{build_name_for_file.replace(' ', '_')}_{st.session_state.bc_pos}_{st.session_state.bc_alt}cm_{st.session_state.bc_pes}kg_build_fc25.json"
            else:
                # Si no hay nombre, usar solo los datos del perfil
                file_name_to_save = f"{st.session_state.bc_pos}_{st.session_state.bc_alt}cm_{st.session_state.bc_pes}kg_build_fc25.json"
            
            st.download_button(
                label="💾 Guardar Build Actual",
                data=prepare_build_data_to_save(),
                file_name=file_name_to_save,
                mime="application/json",
                key="download_build_btn_v33"
            )
        # --- FIN: Gestión de Builds (Guardar/Cargar) ---
        
        st.divider()

        st.subheader("Árboles de Habilidad para Personalizar")
        # NUEVO: Selector de visualización
        tipo_visualizacion_bc = st.radio(
            "Tipo de visualización de árbol:",
            ["Estilo EA FC25", "Lista de nodos"],
            horizontal=True,
            key="tipo_visualizacion_arbol_bc"
        )

        arbol_sel_bc = st.selectbox("Selecciona Árbol:", options=["Todos"] + sorted(df_skill_trees_global['Arbol'].unique()), key="skill_tree_select_bc_v33")
        
        st.divider()

        # NUEVA SECCIÓN: Mostrar tabla de impacto solo cuando se selecciona un árbol específico
        if arbol_sel_bc != "Todos":
            st.subheader(f"📊 Impacto del Árbol '{arbol_sel_bc}' y Instalaciones")
            
            # 1. Identificar las sub-stats a mostrar
            sub_stats_del_arbol = []
            
            # Primero intentar obtener del mapeo global si el árbol corresponde a una categoría principal
            if arbol_sel_bc in SUB_STATS_MAPPING:
                sub_stats_del_arbol = SUB_STATS_MAPPING[arbol_sel_bc]
            else:
                # Si no está en el mapeo, generar dinámicamente basado en los nodos del árbol
                nodos_del_arbol = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol_sel_bc]
                sub_stats_encontradas = set()
                
                for _, nodo in nodos_del_arbol.iterrows():
                    for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE:
                        if stat_col in nodo.index and pd.notna(nodo[stat_col]) and nodo[stat_col] != 0:
                            if stat_col in IGS_SUB_STATS:  # Solo sub-stats, no WF/SM
                                sub_stats_encontradas.add(stat_col)
                
                sub_stats_del_arbol = sorted(list(sub_stats_encontradas))
            
            if sub_stats_del_arbol:
                # 2. Calcular los boosts de instalaciones (una sola vez)
                boosts_instalaciones_totales = defaultdict(int)
                
                if st.session_state.apply_facility_boosts_toggle and 'unlocked_facility_levels' in st.session_state:
                    for facility_id in st.session_state.unlocked_facility_levels:
                        if facility_id in df_instalaciones_global['ID_Instalacion'].values:
                            facility_data = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id].iloc[0]
                            for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                                if stat_col in facility_data.index and pd.notna(facility_data[stat_col]) and facility_data[stat_col] != 0:
                                    boosts_instalaciones_totales[stat_col] += int(facility_data[stat_col])
                
                # 3. Calcular los boosts del árbol seleccionado
                boosts_solo_arbol_seleccionado = defaultdict(int)
                
                # Filtrar nodos que pertenecen al árbol seleccionado
                nodos_arbol_desbloqueados = []
                for node_id in st.session_state.bc_unlocked_nodes:
                    if node_id in df_skill_trees_global['ID_Nodo'].values:
                        node_info = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                        if node_info['Arbol'] == arbol_sel_bc:
                            nodos_arbol_desbloqueados.append(node_id)
                
                # Sumar boosts de los nodos del árbol seleccionado
                for node_id in nodos_arbol_desbloqueados:
                    node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                    for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE:
                        if stat_col in node_data.index and pd.notna(node_data[stat_col]) and node_data[stat_col] != 0:
                            if stat_col in IGS_SUB_STATS:  # Solo sub-stats, no WF/SM
                                boosts_solo_arbol_seleccionado[stat_col] += int(node_data[stat_col])
                
                # 4. Construir los datos para la tabla
                summary_data_impacto = []
                
                for stat_name in sub_stats_del_arbol:
                    # Valor base
                    base_val = int(jugador_base_actual_bc.get(stat_name, 0)) if jugador_base_actual_bc is not None else 0
                    
                    # Boost del árbol actual
                    total_boost_arbol = boosts_solo_arbol_seleccionado.get(stat_name, 0)
                    
                    # Boost de instalaciones
                    total_boost_instalacion = boosts_instalaciones_totales.get(stat_name, 0)
                    
                    # Valor final con tope de 99
                    final_val = min(base_val + total_boost_arbol + total_boost_instalacion, MAX_STAT_VAL)
                    
                    # Crear strings formateados con HTML para los boosts
                    boost_arbol_display = f"<span style='color:green; font-weight:bold;'>+{total_boost_arbol}</span>" if total_boost_arbol > 0 else "+0"
                    boost_instalacion_display = f"<span style='color:blue; font-weight:bold;'>+{total_boost_instalacion}</span>" if total_boost_instalacion > 0 else "+0"
                    
                    # Agregar datos a la lista
                    summary_data_impacto.append({
                        "Stat": stat_name,
                        "Base": base_val,
                        f"Boost ({arbol_sel_bc})": boost_arbol_display,
                        "Boost (Instalaciones)": boost_instalacion_display,
                        "Final": final_val
                    })
                
                # 5. Renderizar la tabla
                if summary_data_impacto:
                    df_summary_impacto = pd.DataFrame(summary_data_impacto).set_index("Stat")
                    column_order = ["Base", f"Boost ({arbol_sel_bc})", "Boost (Instalaciones)", "Final"]
                    
                    st.markdown("**Detalle del Impacto por Sub-Estadística:**")
                    st.markdown(df_summary_impacto[column_order].to_html(escape=False), unsafe_allow_html=True)
                    
                    # Mostrar información adicional
                    total_boost_arbol_general = sum(boosts_solo_arbol_seleccionado.values())
                    total_boost_instalaciones_general = sum(boosts_instalaciones_totales.get(stat, 0) for stat in sub_stats_del_arbol)
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric(f"Total Boost del Árbol {arbol_sel_bc}", f"+{total_boost_arbol_general}")
                    with col_info2:
                        st.metric("Total Boost de Instalaciones", f"+{total_boost_instalaciones_general}")
                else:
                    st.info(f"No hay sub-estadísticas relevantes para mostrar en el árbol '{arbol_sel_bc}'.")
            else:
                st.info(f"No se encontraron sub-estadísticas relevantes para el árbol '{arbol_sel_bc}'.")
        
        if st.button("Resetear Puntos de Habilidad", key="reset_skills_btn_bc"): 
            st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
            st.rerun()

        # NUEVO: Visualización EA FC25
        if tipo_visualizacion_bc == "Estilo EA FC25":
            mostrar_visualizacion_fc25(df_skill_trees_global, arbol_sel_bc, st.session_state.bc_unlocked_nodes)
            st.divider()
    
        # Si es "Lista de nodos", seguir con el código original de nodos:
        if tipo_visualizacion_bc == "Lista de nodos":
            nodos_a_mostrar_display = df_skill_trees_global  
            if arbol_sel_bc != "Todos": 
                nodos_a_mostrar_display = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol_sel_bc]
            
            st.markdown(f"**Nodos para '{arbol_sel_bc}':** ({len(nodos_a_mostrar_display)} nodos)")
            
            nodos_por_tier_display = defaultdict(list) 
            processed_nodes_display = set() 
            
            # Tier 0: Nodos sin prerrequisitos
            for _, nodo_disp in nodos_a_mostrar_display.iterrows(): 
                prereq_val = nodo_disp.get('Prerrequisito', '') 
                if pd.isna(prereq_val) or str(prereq_val).strip() == '': 
                    nodos_por_tier_display[0].append(nodo_disp)
                    processed_nodes_display.add(nodo_disp['ID_Nodo'])
            
            current_tier_display = 0 
            while len(processed_nodes_display) < len(nodos_a_mostrar_display):
                newly_added_to_tier_display = 0 
                for _, nodo_disp_loop in nodos_a_mostrar_display.iterrows(): 
                    if nodo_disp_loop['ID_Nodo'] in processed_nodes_display:
                        continue
                    
                    prereqs_list_disp = [pr_id.strip() for pr_id in str(nodo_disp_loop.get('Prerrequisito', '')).split(',') if pr_id.strip()] 
                    if prereqs_list_disp and all(pr_id in processed_nodes_display for pr_id in prereqs_list_disp):
                        max_prereq_tier_disp = -1 
                        for pr_id_disp in prereqs_list_disp: 
                            for t_disp, nodes_in_tier_disp in nodos_por_tier_display.items(): 
                                if any(n_disp['ID_Nodo'] == pr_id_disp for n_disp in nodes_in_tier_disp): 
                                    max_prereq_tier_disp = max(max_prereq_tier_disp, t_disp)
                                    break
                        
                        nodos_por_tier_display[max_prereq_tier_disp + 1].append(nodo_disp_loop)
                        processed_nodes_display.add(nodo_disp_loop['ID_Nodo'])
                        newly_added_to_tier_display += 1

                if newly_added_to_tier_display == 0 and len(processed_nodes_display) < len(nodos_a_mostrar_display):
                    remaining_tier_disp = current_tier_display + 100  
                    for _, nodo_restante_disp in nodos_a_mostrar_display.iterrows(): 
                        if nodo_restante_disp['ID_Nodo'] not in processed_nodes_display:
                            nodos_por_tier_display[remaining_tier_disp].append(nodo_restante_disp)
                            processed_nodes_display.add(nodo_restante_disp['ID_Nodo'])
                    break 
                current_tier_display += 1
                if current_tier_display > len(nodos_a_mostrar_display) + 5: 
                    for _, nodo_restante_fb_disp in nodos_a_mostrar_display.iterrows(): 
                        if nodo_restante_fb_disp['ID_Nodo'] not in processed_nodes_display:
                            nodos_por_tier_display[current_tier_display + 200].append(nodo_restante_fb_disp) 
                            processed_nodes_display.add(nodo_restante_fb_disp['ID_Nodo'])
                    break

            for tier_level_disp in sorted(nodos_por_tier_display.keys()):  
                if nodos_por_tier_display[tier_level_disp]:
                    tier_name_disp = f"Nivel {tier_level_disp}" if tier_level_disp < 100 else f"Otros/Restantes (Grupo {tier_level_disp})" 
                    st.markdown(f"--- *{tier_name_disp}* ---")
                    num_cols_display_nodes_disp = 3 
                    node_cards_cols_disp = st.columns(num_cols_display_nodes_disp) 
                    col_idx_node_disp = 0  
                    
                    for nodo_item_disp in nodos_por_tier_display[tier_level_disp]:  
                        with node_cards_cols_disp[col_idx_node_disp % num_cols_display_nodes_disp]:
                            node_id_disp = nodo_item_disp['ID_Nodo'] 
                            is_unlocked_disp = node_id_disp in st.session_state.bc_unlocked_nodes  
                            can_be_unlocked_disp = check_prerequisites_skills(node_id_disp, df_skill_trees_global, st.session_state.bc_unlocked_nodes) 
                            
                            beneficios_str_list_disp = []  
                            cols_para_beneficios_disp = (stat_cols_order if stat_cols_order else IGS_SUB_STATS) + ['PIERNA_MALA', 'FILIGRANAS']
                            for col_stat_disp in cols_para_beneficios_disp:
                                if col_stat_disp in nodo_item_disp.index and pd.notna(nodo_item_disp[col_stat_disp]) and nodo_item_disp[col_stat_disp] != 0: 
                                    valor_beneficio_disp = int(nodo_item_disp[col_stat_disp]) 
                                    if col_stat_disp == 'PIERNA_MALA': beneficios_str_list_disp.append(f"+{valor_beneficio_disp}⭐ WF")
                                    elif col_stat_disp == 'FILIGRANAS': beneficios_str_list_disp.append(f"+{valor_beneficio_disp}⭐ SM")
                                    else: beneficios_str_list_disp.append(f"+{valor_beneficio_disp} {col_stat_disp}")
                            
                            playstyle_val_node_disp = nodo_item_disp.get('PlayStyle', '')
                            if pd.notna(playstyle_val_node_disp) and playstyle_val_node_disp != '':
                                beneficios_str_list_disp.append(f"PlayStyle: {playstyle_val_node_disp}")
                            
                            beneficios_str_disp = ", ".join(beneficios_str_list_disp) if beneficios_str_list_disp else "Sin bonus directo"  
                            
                            with st.container(border=True):
                                st.markdown(f"**{nodo_item_disp['Nombre_Visible']}**")
                                st.caption(f"ID: {node_id_disp} | Costo: {nodo_item_disp['Costo']}")
                                st.caption(f"Beneficios: {beneficios_str_disp}")
                                prereq_display_val_node = nodo_item_disp.get('Prerrequisito', '') 
                                prereq_display_node = f"Prerreq: {str(prereq_display_val_node).strip() if str(prereq_display_val_node).strip() else 'Ninguno'}" 
                                st.caption(prereq_display_node)

                                if is_unlocked_disp:
                                    if st.button("↩️ Devolver", key=f"return_skill_node_{node_id_disp}_{arbol_sel_bc}_v33r"): 
                                        dependencias_node = verificar_dependencias_nodo(node_id_disp, df_skill_trees_global, st.session_state.bc_unlocked_nodes) 
                                        if not dependencias_node:
                                            st.session_state.bc_unlocked_nodes.remove(node_id_disp)
                                            st.session_state.bc_points_remaining += nodo_item_disp['Costo']
                                            st.rerun()
                                        else:
                                            nombres_dependencias_node = ", ".join(dependencias_node) 
                                            st.warning(f"No se puede devolver. Nodos dependientes: {nombres_dependencias_node}. Devuélvelos primero.")
                                elif can_be_unlocked_disp and st.session_state.bc_points_remaining >= nodo_item_disp['Costo']:
                                    if st.button("🔓 Desbloquear", key=f"unlock_skill_node_{node_id_disp}_{arbol_sel_bc}_v33u"): 
                                        st.session_state.bc_unlocked_nodes.add(node_id_disp)
                                        st.session_state.bc_points_remaining -= nodo_item_disp['Costo']
                                        st.rerun()
                                else:
                                    help_text_node_disp = "Prerrequisito no cumplido." if not can_be_unlocked_disp else f"Puntos insuficientes (Req: {nodo_item_disp['Costo']})" 
                                    st.button("🔒 Bloqueado", key=f"locked_skill_node_{node_id_disp}_{arbol_sel_bc}_v33l", disabled=True, help=help_text_node_disp) 
                        col_idx_node_disp += 1
        else: 
            st.info("Selecciona un árbol de habilidad para ver sus nodos.")
                
# --- Pestaña: Instalaciones Club ---
with tab_facilities:
    if not carga_completa_exitosa or df_instalaciones_global.empty:
        st.error("Datos de Instalaciones no cargados.")
    else:
        st.header("🏨 Gestión de Instalaciones del Club")
        current_budget_total_fac = st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET) 
        st.session_state.club_budget_total = st.number_input(
            "Presupuesto Total del Club:", 
            min_value=0, 
            value=int(current_budget_total_fac), 
            step=100000, 
            key='club_budget_total_input_v33_fac', 
            format="%d"
        )
        
        costo_instalaciones_desbloqueadas_fac = 0 
        for facility_id_loop_fac in st.session_state.get('unlocked_facility_levels', set()): 
            if facility_id_loop_fac in df_instalaciones_global['ID_Instalacion'].values:
                facility_row_fac = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id_loop_fac] 
                if not facility_row_fac.empty:
                     costo_instalaciones_desbloqueadas_fac += facility_row_fac.iloc[0]['Precio']
        
        st.session_state.club_budget_remaining = st.session_state.club_budget_total - costo_instalaciones_desbloqueadas_fac
        
        styled_metric("Presupuesto Restante:", f"{st.session_state.club_budget_remaining:,}")

        if st.button("Resetear Instalaciones Seleccionadas", key="reset_facilities_v33_fac"): 
            st.session_state.unlocked_facility_levels = set()
            st.rerun()
        st.markdown("---")
        
        facility_types_list_fac = sorted(df_instalaciones_global['Instalacion'].unique()) 
        for facility_type_item_fac in facility_types_list_fac: 
            st.subheader(f"{facility_type_item_fac}")
            facility_levels_df_fac = df_instalaciones_global[df_instalaciones_global['Instalacion'] == facility_type_item_fac].sort_values(by='Precio') 
            
            num_facility_cols_disp_fac = min(len(facility_levels_df_fac), 4)  
            cols_facilities_fac = st.columns(num_facility_cols_disp_fac) if num_facility_cols_disp_fac > 0 else [st] 
            
            for idx_fac_loop, (_, level_data_fac) in enumerate(facility_levels_df_fac.iterrows()): 
                level_id_fac_item = level_data_fac['ID_Instalacion'] 
                is_level_unlocked_fac_item = level_id_fac_item in st.session_state.unlocked_facility_levels 
                can_unlock_level_fac_item = check_prerequisites_facilities(level_id_fac_item, df_instalaciones_global, st.session_state.unlocked_facility_levels) 
                
                with cols_facilities_fac[idx_fac_loop % num_facility_cols_disp_fac]:
                    with st.container(border=True):
                        st.markdown(f"**{level_data_fac['Nombre_Instalacion']}**")
                        st.caption(f"Costo: {level_data_fac['Precio']:,}")
                        prereq_fac_val_item = level_data_fac.get('Prerrequisito', '') 
                        prereq_fac_display_item = f"Prerreq: {prereq_fac_val_item if str(prereq_fac_val_item).strip() else 'Ninguno'}" 
                        st.caption(prereq_fac_display_item)
                        
                        benefits_list_facility_item = [] 
                        for stat_col_facility_item in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES: 
                            if stat_col_facility_item in level_data_fac.index and level_data_fac[stat_col_facility_item] > 0: 
                                benefits_list_facility_item.append(f"+{int(level_data_fac[stat_col_facility_item])} {stat_col_facility_item}") 
                        
                        playstyle_facility_val_item = level_data_fac.get('PlayStyle', '') 
                        is_plus_facility_item = str(level_data_fac.get('EsPlus', '')).strip().lower() == 'si' 
                        if pd.notna(playstyle_facility_val_item) and playstyle_facility_val_item != '':
                            benefits_list_facility_item.append(f"PlayStyle: {playstyle_facility_val_item}{'+' if is_plus_facility_item else ''}")
                        
                        if benefits_list_facility_item: st.caption("Beneficios: " + ", ".join(benefits_list_facility_item))
                        else: st.caption("Sin boosts directos este nivel.")

                        if is_level_unlocked_fac_item:
                            if st.button("💸 Vender", key=f"sell_facility_level_{level_id_fac_item}_v33r"): 
                                dependencias_fac_item = verificar_dependencias_instalacion(level_id_fac_item, df_instalaciones_global, st.session_state.unlocked_facility_levels) 
                                if not dependencias_fac_item:
                                    st.session_state.unlocked_facility_levels.remove(level_id_fac_item)
                                    st.session_state.bc_points_remaining += level_data_fac['Precio']
                                    st.rerun()
                                else:
                                    nombres_dependencias_fac_item = ", ".join(dependencias_fac_item) 
                                    st.warning(f"No se puede vender. Niveles dependientes: {nombres_dependencias_fac_item}. Véndelos primero.")
                        elif can_unlock_level_fac_item and st.session_state.club_budget_remaining >= level_data_fac['Precio']:
                            if st.button("🛒 Comprar", key=f"buy_facility_level_{level_id_fac_item}_v33u"): 
                                st.session_state.unlocked_facility_levels.add(level_id_fac_item)
                                st.session_state.club_budget_remaining -= level_data_fac['Precio']
                                st.rerun()
                        else:
                            help_text_facility_item = "Prerreq. no cumplido" if not can_unlock_level_fac_item else f"Presupuesto Insuf. ({level_data_fac['Precio']:,})" 
                            st.button("🔒 No Disponible", key=f"locked_facility_level_{level_id_fac_item}_v33l", disabled=True, help=help_text_facility_item) 
            st.markdown("---")


# --- Pestaña: Explorador de Mejoras ---
with tab_explorer:
    if not carga_completa_exitosa or (df_skill_trees_global.empty and df_instalaciones_global.empty):
        st.error("Datos para Explorador no cargados.")
    else:
        st.header("🔎 Explorador de Mejoras (Habilidades e Instalaciones)")
        col_exp_type_exp, col_exp_selector_exp = st.columns(2) 
        with col_exp_type_exp:
            search_type_explorer_exp = st.radio("Buscar por:", ["Estadística", "PlayStyle"], key="explorer_search_type_v33_exp", horizontal=True) 
        
        results_found_explorer_exp = pd.DataFrame() 
        
        if search_type_explorer_exp == "Estadística":
            all_boostable_stats_explorer_exp = sorted(list(set(ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE + ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES))) 
            all_boostable_stats_explorer_filtered_exp = [s for s in all_boostable_stats_explorer_exp if s not in ['PIERNA_MALA', 'FILIGRANAS']] 

            stat_to_find_explorer_exp = col_exp_selector_exp.selectbox("Selecciona Estadística:", options=all_boostable_stats_explorer_filtered_exp, key="explorer_stat_select_v33_exp") 
            
            if st.button("Buscar Mejoras de Estadística", key="explorer_search_stat_btn_v33_exp"): 
                temp_results_explorer_exp = [] 
                if stat_to_find_explorer_exp in df_skill_trees_global.columns:
                    skill_tree_results_explorer_exp = df_skill_trees_global[df_skill_trees_global[stat_to_find_explorer_exp] > 0].copy() 
                    if not skill_tree_results_explorer_exp.empty:
                        for _, r_skill_exp in skill_tree_results_explorer_exp.iterrows(): 
                            temp_results_explorer_exp.append({
                                'Fuente': f"Árbol Hab. ({r_skill_exp['Arbol']})", 'Nombre Mejora': r_skill_exp['Nombre_Visible'],
                                'Costo/Precio': r_skill_exp['Costo'], 'Prerrequisito': r_skill_exp.get('Prerrequisito',''),
                                'Beneficio Específico': f"+{int(r_skill_exp[stat_to_find_explorer_exp])} {stat_to_find_explorer_exp}"})
                if stat_to_find_explorer_exp in df_instalaciones_global.columns:
                    facility_results_explorer_exp = df_instalaciones_global[df_instalaciones_global[stat_to_find_explorer_exp] > 0].copy() 
                    if not facility_results_explorer_exp.empty:
                        for _, r_fac_exp in facility_results_explorer_exp.iterrows(): 
                            temp_results_explorer_exp.append({
                                'Fuente': f"Instalación ({r_fac_exp['Instalacion']})", 'Nombre Mejora': r_fac_exp['Nombre_Instalacion'],
                                'Costo/Precio': r_fac_exp['Precio'], 'Prerrequisito': r_fac_exp.get('Prerrequisito',''),
                                'Beneficio Específico': f"+{int(r_fac_exp[stat_to_find_explorer_exp])} {stat_to_find_explorer_exp}"})
                if temp_results_explorer_exp: results_found_explorer_exp = pd.DataFrame(temp_results_explorer_exp)
                    
        elif search_type_explorer_exp == "PlayStyle":
            available_playstyles_explorer_exp = [] 
            if 'PlayStyle' in df_skill_trees_global.columns: 
                available_playstyles_explorer_exp.extend(df_skill_trees_global[df_skill_trees_global['PlayStyle'].fillna('').str.strip() != '']['PlayStyle'].unique())
            if 'PlayStyle' in df_instalaciones_global.columns: 
                available_playstyles_explorer_exp.extend(df_instalaciones_global[df_instalaciones_global['PlayStyle'].fillna('').str.strip() != '']['PlayStyle'].unique())
            
            unique_playstyles_sorted_explorer_exp = sorted(list(set(ps for ps in available_playstyles_explorer_exp if ps))) 
            
            playstyle_to_find_explorer_exp = col_exp_selector_exp.selectbox("Selecciona PlayStyle:", options=unique_playstyles_sorted_explorer_exp, key="explorer_ps_select_v33_exp") 
            
            if st.button("Buscar Fuentes de PlayStyle", key="explorer_search_ps_btn_v33_exp"): 
                temp_results_ps_explorer_exp = [] 
                skill_tree_ps_results_exp = df_skill_trees_global[df_skill_trees_global['PlayStyle'] == playstyle_to_find_explorer_exp].copy() 
                if not skill_tree_ps_results_exp.empty:
                    for _, r_skill_ps_exp in skill_tree_ps_results_exp.iterrows(): 
                        temp_results_ps_explorer_exp.append({
                            'Fuente': f"Árbol Hab. ({r_skill_ps_exp['Arbol']})", 'Nombre Mejora': r_skill_ps_exp['Nombre_Visible'],
                            'Costo/Precio': r_skill_ps_exp['Costo'], 'Prerrequisito': r_skill_ps_exp.get('Prerrequisito',''),
                            'Beneficio Específico': f"PlayStyle: {r_skill_ps_exp['PlayStyle']}"})
                facility_ps_results_exp = df_instalaciones_global[df_instalaciones_global['PlayStyle'] == playstyle_to_find_explorer_exp].copy() 
                if not facility_ps_results_exp.empty:
                    for _, r_fac_ps_exp in facility_ps_results_exp.iterrows(): 
                        es_plus_str_explorer_exp = "+" if str(r_fac_ps_exp.get('EsPlus', '')).strip().lower() == 'si' else "" 
                        temp_results_ps_explorer_exp.append({
                            'Fuente': f"Instalación ({r_fac_ps_exp['Instalacion']})", 'Nombre Mejora': r_fac_ps_exp['Nombre_Instalacion'],
                            'Costo/Precio': r_fac_ps_exp['Precio'], 'Prerrequisito': r_fac_ps_exp.get('Prerrequisito',''),
                            'Beneficio Específico': f"PlayStyle: {r_fac_ps_exp['PlayStyle']}{es_plus_str_explorer_exp}"})
                if temp_results_ps_explorer_exp: results_found_explorer_exp = pd.DataFrame(temp_results_ps_explorer_exp)

        if not results_found_explorer_exp.empty:
            st.dataframe(results_found_explorer_exp.reset_index(drop=True))
        elif ('explorer_search_stat_btn_v33_exp' in st.session_state and st.session_state.explorer_search_stat_btn_v33_exp and results_found_explorer_exp.empty) or \
             ('explorer_search_ps_btn_v33_exp' in st.session_state and st.session_state.explorer_search_ps_btn_v33_exp and results_found_explorer_exp.empty):
            st.info("No se encontraron mejoras para tu selección.")


# --- Pestaña: Búsqueda Óptima ---
with tab_best_combo:
    if not carga_completa_exitosa or all_stats_df_base.empty:
        st.error("Datos para Búsqueda Óptima no disponibles.")
    else:
        st.header("🔍 Búsqueda de Mejor Combinación por Atributos (Priorizados) - Stats Base")
        queryable_stats_optimal_opt = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE']] 
        
        col1_opt_combo, col2_opt_combo, col3_opt_combo = st.columns(3) 
        with col1_opt_combo: 
            attr_primary_optimal_opt = st.selectbox("1er Atributo (Más importante):", options=queryable_stats_optimal_opt, key="attr_pri_v33_opt") 
        with col2_opt_combo: 
            attr_secondary_optimal_opt = st.selectbox("2do Atributo:", options=["(Ninguno)"] + [s for s in queryable_stats_optimal_opt if s != attr_primary_optimal_opt], key="attr_sec_v33_opt") 
        with col3_opt_combo: 
            attr_tertiary_optimal_opt = st.selectbox("3er Atributo:", options=["(Ninguno)"] + [s for s in queryable_stats_optimal_opt if s != attr_primary_optimal_opt and s != attr_secondary_optimal_opt], key="attr_ter_v33_opt") 
        
        if st.button("Buscar Mejor Combinación Priorizada (Stats Base)", key="btn_multi_attr_find_v33_opt"): 
            df_optimal_search_opt = all_stats_df_base.copy() 
            sort_by_attributes_optimal_opt = [attr_primary_optimal_opt] 
            if attr_secondary_optimal_opt != "(Ninguno)": sort_by_attributes_optimal_opt.append(attr_secondary_optimal_opt)
            if attr_tertiary_optimal_opt != "(Ninguno)": sort_by_attributes_optimal_opt.append(attr_tertiary_optimal_opt)
            
            for attr_opt_check_loop in sort_by_attributes_optimal_opt: 
                if attr_opt_check_loop not in df_optimal_search_opt.columns:
                    st.error(f"Atributo de ordenamiento '{attr_opt_check_loop}' no encontrado en los datos base.")
                    st.stop()
                df_optimal_search_opt[attr_opt_check_loop] = pd.to_numeric(df_optimal_search_opt[attr_opt_check_loop], errors='coerce').fillna(0)

            df_optimal_search_opt = df_optimal_search_opt.sort_values(by=sort_by_attributes_optimal_opt, ascending=[False]*len(sort_by_attributes_optimal_opt))
            
            if not df_optimal_search_opt.empty:
                best_player_combination_opt = df_optimal_search_opt.iloc[0] 
                st.subheader(f"Mejor Jugador (Stats Base): {best_player_combination_opt['Posicion']} | {best_player_combination_opt['Altura']}cm | {best_player_combination_opt['Peso']}kg")
                
                if st.button(f"🛠️ Personalizar Build para esta Combinación", key=f"send_to_bc_best_opt_v33_opt_btn"):  
                    st.session_state.bc_pos = best_player_combination_opt['Posicion']
                    st.session_state.bc_alt = best_player_combination_opt['Altura']
                    st.session_state.bc_pes = best_player_combination_opt['Peso']
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.session_state.unlocked_facility_levels = set()
                    st.session_state.club_budget_remaining = st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET) 
                    st.success(f"Perfil de Mejor Combinación Base enviado. Ve a '🛠️ Build Craft'.")
                
                if 'AcceleRATE' not in best_player_combination_opt or pd.isna(best_player_combination_opt['AcceleRATE']):
                     accel_rate_best_optimal_opt = determinar_estilo_carrera(best_player_combination_opt['Altura'],best_player_combination_opt.get('AGI',0),best_player_combination_opt.get('STR',0),best_player_combination_opt.get('Acc',0)) 
                else:
                    accel_rate_best_optimal_opt = best_player_combination_opt['AcceleRATE']
                
                styled_metric("Estilo de Carrera (Base)", accel_rate_best_optimal_opt)
                st.divider()
                
                # CORRECCIÓN: Verificar que sort_by_attributes_optimal_opt esté definido
                if 'sort_by_attributes_optimal_opt' in locals() and sort_by_attributes_optimal_opt:
                    key_metrics_to_display_optimal_opt = sort_by_attributes_optimal_opt
                else:
                    key_metrics_to_display_optimal_opt = [attr_primary_optimal_opt]
                
                metrics_to_show = key_metrics_to_display_optimal_opt + ['IGS']
                if len(metrics_to_show) > 0:
                    cols_metrics = st.columns(min(len(metrics_to_show), 4))
                    for idx, metric_name in enumerate(metrics_to_show):
                        if metric_name in best_player_combination_opt:
                            with cols_metrics[idx % len(cols_metrics)]:
                                metric_value = int(float(str(best_player_combination_opt[metric_name])))
                                styled_metric(metric_name, metric_value)
                        
                with st.expander("Ver todos los atributos base de esta combinación"):
                    display_series_optimal_opt = best_player_combination_opt.drop(['Posicion', 'Altura', 'Peso'], errors='ignore').astype(str) 
                    st.dataframe(display_series_optimal_opt)
        
        st.divider()
        
        st.subheader("Filtrar y Buscar Combinaciones Óptimas")
        st.markdown("Define criterios de filtro avanzados para encontrar combinaciones de estadísticas base que se ajusten a tus necesidades.")
        
        col_filtro_attr, col_filtro_cond, col_filtro_valor = st.columns(3) 
        with col_filtro_attr:
            attr_filter_opt = st.selectbox("Atributo a filtrar:", options=queryable_stats_optimal_opt, key="filtro_attr_v33_opt") 
        with col_filtro_cond:
            condition_options_filter_opt = ['>=', '<=', '==', '>', '<', '!='] 
            cond_filter_opt = st.selectbox("Condición:", options=condition_options_filter_opt, key="filtro_cond_v33_opt") 
        with col_filtro_valor:
            val_filter_opt = st.number_input("Valor:", value=70, step=1, key="filtro_valor_v33_opt") 
        
        if st.button("Buscar Combinaciones Óptimas", key="btn_search_optimal_combos_v33"): 
            df_filtered_results_opt = all_stats_df_base.copy() 
            
            # Aplicar filtro
            if attr_filter_opt in df_filtered_results_opt.columns:
                df_filtered_results_opt[attr_filter_opt] = pd.to_numeric(df_filtered_results_opt[attr_filter_opt], errors='coerce').fillna(0)
                
                # Aplicar la condición
                if cond_filter_opt == '>=':
                    df_filtered_results_opt = df_filtered_results_opt[df_filtered_results_opt[attr_filter_opt] >= val_filter_opt]
                elif cond_filter_opt == '<=':
                    df_filtered_results_opt = df_filtered_results_opt[df_filtered_results_opt[attr_filter_opt] <= val_filter_opt]
                elif cond_filter_opt == '==':
                    df_filtered_results_opt = df_filtered_results_opt[df_filtered_results_opt[attr_filter_opt] == val_filter_opt]
                elif cond_filter_opt == '>':
                    df_filtered_results_opt = df_filtered_results_opt[df_filtered_results_opt[attr_filter_opt] > val_filter_opt]
                elif cond_filter_opt == '<':
                    df_filtered_results_opt = df_filtered_results_opt[df_filtered_results_opt[attr_filter_opt] < val_filter_opt]
                elif cond_filter_opt == '!=':
                    df_filtered_results_opt = df_filtered_results_opt[df_filtered_results_opt[attr_filter_opt] != val_filter_opt]

            if 'AcceleRATE' not in df_filtered_results_opt.columns:
                df_filtered_results_opt['AcceleRATE'] = df_filtered_results_opt.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI', 0), r.get('STR', 0), r.get('Acc', 0)), axis=1)

            # Ordenar por los atributos seleccionados si existen
            if 'sort_by_attributes_optimal_opt' in locals() and sort_by_attributes_optimal_opt:
                df_filtered_results_opt = df_filtered_results_opt.sort_values(by=sort_by_attributes_optimal_opt, ascending=[False]*len(sort_by_attributes_optimal_opt))
            
            if not df_filtered_results_opt.empty:
                st.write(f"Combinaciones encontradas ({len(df_filtered_results_opt)}):")
                st.dataframe(df_filtered_results_opt[['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']])
            else:
                st.info("No se encontraron combinaciones que cumplan con los criterios.")
# --- Pestaña: Filtros Múltiples ---
with tab_filters:
    if not carga_completa_exitosa or all_stats_df_base.empty:
        st.error("Datos para Filtros Múltiples no disponibles.")
    else:
        st.header("📊 Filtros Múltiples - Explorador de Perfiles")
        st.markdown("""
        **Encuentra todos los jugadores que cumplan con tus criterios específicos.**
        
        A diferencia de la "Búsqueda Óptima" que te da el mejor jugador, esta herramienta te muestra 
        **todos los perfiles** que encajan dentro de tus requisitos mínimos.
        """)
        
        # Inicializar filtros en session_state si no existen
        if 'filter_criteria' not in st.session_state:
            st.session_state.filter_criteria = []
        
        # Estadísticas disponibles para filtrar
        filterable_stats = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE']]
        
        # --- SECCIÓN: CONSTRUCTOR DE FILTROS ---
        st.subheader("🔧 Constructor de Filtros")
        
        # Botón para añadir nuevo criterio
        if st.button("➕ Añadir Criterio de Filtro", key="add_filter_btn_v33"):
            new_filter_id = st.session_state.get('next_filter_id', 0)
            st.session_state.filter_criteria.append({
                'id': new_filter_id,
                'attribute': filterable_stats[0] if filterable_stats else 'IGS',
                'condition': '>=',
                'value': 70
            })
            st.session_state.next_filter_id = new_filter_id + 1
            st.rerun()
        
        # Mostrar filtros existentes
        if st.session_state.filter_criteria:
            st.markdown("**Criterios Activos:**")
            
            # Crear columnas para los filtros
            for i, filter_item in enumerate(st.session_state.filter_criteria):
                cols_filter = st.columns([3, 2, 2, 1])
                
                with cols_filter[0]:
                    # Selector de atributo
                    current_attr_idx = filterable_stats.index(filter_item['attribute']) if filter_item['attribute'] in filterable_stats else 0
                    new_attribute = st.selectbox(
                        "Atributo:", 
                        options=filterable_stats,
                        index=current_attr_idx,
                        key=f"filter_attr_{filter_item['id']}_v33"
                    )
                    st.session_state.filter_criteria[i]['attribute'] = new_attribute
                
                with cols_filter[1]:
                    # Selector de condición
                    condition_options = ['>=', '<=', '==', '>', '<', '!=']
                    current_cond_idx = condition_options.index(filter_item['condition']) if filter_item['condition'] in condition_options else 0
                    new_condition = st.selectbox(
                        "Condición:",
                        options=condition_options,
                        index=current_cond_idx,
                        key=f"filter_cond_{filter_item['id']}_v33"
                    )
                    st.session_state.filter_criteria[i]['condition'] = new_condition
                
                with cols_filter[2]:
                    # Input de valor
                    new_value = st.number_input(
                        "Valor:",
                        value=filter_item['value'],
                        min_value=0,
                        max_value=99,
                        step=1,
                        key=f"filter_val_{filter_item['id']}_v33"
                    )
                    st.session_state.filter_criteria[i]['value'] = new_value
                
                with cols_filter[3]:
                    # Botón para eliminar este filtro
                    if st.button("🗑️", key=f"remove_filter_{filter_item['id']}_v33", help="Eliminar este filtro"):
                        st.session_state.filter_criteria.pop(i)
                        st.rerun()
            
            # Mostrar resumen de filtros activos
            st.markdown("**Resumen de Filtros:**")
            filter_summary = []
            for filter_item in st.session_state.filter_criteria:
                filter_summary.append(f"**{filter_item['attribute']}** {filter_item['condition']} **{filter_item['value']}**")
            st.markdown(" **Y** ".join(filter_summary))
            
        else:
            st.info("No hay filtros activos. Haz clic en '➕ Añadir Criterio de Filtro' para comenzar.")
        
        # --- SECCIÓN: APLICAR FILTROS Y RESULTADOS ---
        if st.session_state.filter_criteria:
            col_apply, col_clear = st.columns(2)
            
            with col_apply:
                if st.button("🔍 Aplicar Filtros", key="apply_filters_btn_v33", type="primary"):
                    # Aplicar todos los filtros a los datos base
                    df_filtered = all_stats_df_base.copy()
                    
                    # Añadir AcceleRATE si no existe
                    if 'AcceleRATE' not in df_filtered.columns:
                        df_filtered['AcceleRATE'] = df_filtered.apply(
                            lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI', 0), r.get('STR', 0), r.get('Acc', 0)), 
                            axis=1
                        )
                    
                    # Aplicar cada filtro
                    for filter_item in st.session_state.filter_criteria:
                        attr = filter_item['attribute']
                        condition = filter_item['condition']
                        value = filter_item['value']
                        
                        if attr in df_filtered.columns:
                            # Asegurar que la columna es numérica
                            df_filtered[attr] = pd.to_numeric(df_filtered[attr], errors='coerce').fillna(0)
                            
                            # Aplicar la condición
                            if condition == '>=':
                                df_filtered = df_filtered[df_filtered[attr] >= value]
                            elif condition == '<=':
                                df_filtered = df_filtered[df_filtered[attr] <= value]
                            elif condition == '==':
                                df_filtered = df_filtered[df_filtered[attr] == value]
                            elif condition == '>':
                                df_filtered = df_filtered[df_filtered[attr] > value]
                            elif condition == '<':
                                df_filtered = df_filtered[df_filtered[attr] < value]
                            elif condition == '!=':
                                df_filtered = df_filtered[df_filtered[attr] != value]
                    
                    # Guardar resultados en session_state
                    st.session_state.filtered_results = df_filtered
                    
                    # Mostrar resultados inmediatamente
                    if not df_filtered.empty:
                        st.success(f"✅ Se encontraron **{len(df_filtered)}** perfiles que cumplen todos los criterios.")
                    else:
                        st.warning("⚠️ No se encontraron perfiles que cumplan con todos los criterios. Prueba relajar algunos filtros.")
            
            with col_clear:
                if st.button("🧹 Limpiar Todos los Filtros", key="clear_filters_btn_v33"):
                    st.session_state.filter_criteria = []
                    if 'filtered_results' in st.session_state:
                        del st.session_state.filtered_results
                    st.rerun()
        
        # --- SECCIÓN: MOSTRAR RESULTADOS ---
        if 'filtered_results' in st.session_state and not st.session_state.filtered_results.empty:
            st.divider()
            st.subheader(f"🎯 Resultados Encontrados ({len(st.session_state.filtered_results)})")
            
            # Ordenar por IGS descendente por defecto
            df_results = st.session_state.filtered_results.sort_values('IGS', ascending=False)
            
            # Selector de columnas a mostrar
            available_columns = ['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS'] + filterable_stats
            default_columns = ['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS'] + [f['attribute'] for f in st.session_state.filter_criteria]
            
            # Eliminar duplicados y mantener orden
            columns_to_show = []
            for col in default_columns:
                if col in available_columns and col not in columns_to_show:
                    columns_to_show.append(col)
            
            # Añadir columnas restantes importantes
            for col in ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']:
                if col in available_columns and col not in columns_to_show:
                    columns_to_show.append(col)
            
            # Mostrar tabla de resultados
            st.dataframe(
                df_results[columns_to_show].reset_index(drop=True),
                use_container_width=True,
                height=400
            )
            
            # --- SECCIÓN: ENVIAR PERFILES A BUILD CRAFT ---
            st.subheader("🛠️ Enviar a Build Craft")
            
            # Mostrar Top 5 para envío rápido
            top_5_results = df_results.head(5)
            st.markdown("**Top 5 por IGS (Envío Rápido):**")
            
            cols_top5 = st.columns(min(len(top_5_results), 5))
            
            for i, (idx, row) in enumerate(top_5_results.iterrows()):
                with cols_top5[i]:
                    with st.container(border=True):
                        st.markdown(f"**#{i+1}**")
                        st.markdown(f"**{row['Posicion']}** | {row['Altura']}cm | {row['Peso']}kg")
                        st.caption(f"IGS: {int(row['IGS'])}")
                        st.caption(f"AcceleRATE: {row['AcceleRATE']}")
                        
                        if st.button(
                            "🛠️ Editar", 
                            key=f"send_filtered_result_{idx}_v33",
                            use_container_width=True
                        ):
                            # Enviar a Build Craft
                            st.session_state.bc_pos = row['Posicion']
                            st.session_state.bc_alt = row['Altura'] 
                            st.session_state.bc_pes = row['Peso']
                            st.session_state.bc_unlocked_nodes = set()
                            st.session_state.bc_points_remaining = TOTAL_SKILL_POINTS
                            st.session_state.unlocked_facility_levels = set()
                            st.session_state.club_budget_remaining = st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                            st.success(f"Perfil #{i+1} enviado a Build Craft. Ve a la pestaña '🛠️ Build Craft'.")
            
            # Estadísticas del conjunto de resultados
            st.divider()
            st.subheader("📊 Estadísticas del Conjunto")
            
            cols_stats = st.columns(4)
            
            with cols_stats[0]:
                st.metric("Total de Perfiles", len(df_results))
            
            with cols_stats[1]:
                igs_promedio = df_results['IGS'].mean()
                st.metric("IGS Promedio", f"{igs_promedio:.1f}")
            
            with cols_stats[2]:
                posiciones_unicas = df_results['Posicion'].nunique()
                st.metric("Posiciones Diferentes", posiciones_unicas)
            
            with cols_stats[3]:
                mejor_igs = df_results['IGS'].max()
                st.metric("Mejor IGS", int(mejor_igs))
            
            # Distribución por posiciones
            if len(df_results) > 0:
                st.markdown("**Distribución por Posiciones:**")
                pos_counts = df_results['Posicion'].value_counts()
                st.bar_chart(pos_counts)
            elif 'filtered_results' in st.session_state and st.session_state.filtered_results.empty:
                st.info("Los filtros aplicados no produjeron resultados. Intenta ajustar los criterios.")


# NUEVA FUNCIONALIDAD: Exportación mejorada
def generar_exportacion_avanzada():
    """
    Funciones de exportación mejoradas:
    - Excel con múltiples hojas
    - PDF con resumen visual
    - JSON con metadatos completos
    - CSV para análisis estadístico
    """
    
    # Formato Excel multi-hoja
    def export_to_excel():
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Hoja 1: Stats del build
            stats_df.to_excel(writer, sheet_name='Stats_Build')
            # Hoja 2: Nodos desbloqueados
            nodes_df.to_excel(writer, sheet_name='Nodos_Habilidad')
            # Hoja 3: Instalaciones
            facilities_df.to_excel(writer, sheet_name='Instalaciones')
            # Hoja 4: Comparativa con builds populares
            comparison_df.to_excel(writer, sheet_name='Comparativa')
        return buffer.getvalue()
    
    # Formato PDF con gráficos
    def export_to_pdf():
        # Usar reportlab o matplotlib para generar PDF
        # con gráficos radar, tablas y estadísticas
        pass

# NUEVA FUNCIONALIDAD: Filtros múltiples avanzados
def implementar_filtros_avanzados():
    """
    Sistema de filtros múltiples con:
    - Filtros AND/OR combinados
    - Filtros por rango (ej: PAC entre 80-90)
    - Filtros por tags (Delantero, Defensivo, etc.)
    - Filtros por presupuesto
    - Guardado de filtros favoritos
    """
    
    # Widget de construcción de filtros
    st.subheader("🔍 Constructor de Filtros Avanzados")
    
    # Sistema de filtros anidados
    filter_groups = st.container()
    with filter_groups:
        for i in range(st.session_state.get('num_filter_groups', 1)):
            with st.expander(f"Grupo de Filtros {i+1}"):
                # Filtros individuales dentro del grupo
                pass


def calcular_boosts_nodos(nodos_ids):
    """Calcula los boosts totales de una lista de nodos"""
    boosts = defaultdict(int)
    
    if df_skill_trees_global is not None:
        for node_id in nodos_ids:
            node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id]
            if not node_data.empty:
                node_info = node_data.iloc[0]
                for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE:
                    if stat_col in node_info.index and pd.notna(node_info[stat_col]) and node_info[stat_col] != 0:
                        if stat_col in IGS_SUB_STATS:  # Solo sub-stats
                            boosts[stat_col] += int(node_info[stat_col])
    
    return dict(boosts)

def calcular_boosts_instalaciones(instalaciones_ids):
    """Calcula los boosts totales de una lista de instalaciones"""
    boosts = defaultdict(int)
    
    if df_instalaciones_global is not None:
        for facility_id in instalaciones_ids:
            facility_data = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id]
            if not facility_data.empty:
                facility_info = facility_data.iloc[0]
                for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                    if stat_col in facility_info.index and pd.notna(facility_info[stat_col]) and facility_info[stat_col] != 0:
                        boosts[stat_col] += int(facility_info[stat_col])
    
    return dict(boosts)

def combinar_boosts(boosts1, boosts2):
    """Combina dos diccionarios de boosts"""
    combined = defaultdict(int)
    
    for stat, boost in boosts1.items():
        combined[stat] += boost
    
    for stat, boost in boosts2.items():
        combined[stat] += boost
    
    return dict(combined)

def calcular_diferencias(stats_calculadas, stats_objetivo, substats_objetivo=None):
    """Calcula las diferencias entre estadísticas calculadas y objetivo"""
    diferencias = {}
    
    # Estadísticas principales
    for stat, valor_objetivo in stats_objetivo.items():
        if stat in stats_calculadas:
            diferencias[stat] = stats_calculadas[stat] - valor_objetivo
    
    # Sub-estadísticas
    if substats_objetivo:
        for stat, valor_objetivo in substats_objetivo.items():
            if valor_objetivo is not None and stat in stats_calculadas:
                diferencias[stat] = stats_calculadas[stat] - valor_objetivo
    
    return diferencias
# Nueva pestaña: "🔍 Detective De Builds"
def implementar_detector_build_inverso():
    # === PRE-CARGA DE DATOS ANTES DE WIDGETS ===
    # Verificar si hay datos pendientes de carga
    if 'build_data_to_load' in st.session_state and st.session_state.get('auto_load_build', False):
        build_data = st.session_state.build_data_to_load
        
        # Cargar datos de entrada si existen
        if 'datos_entrada_deteccion' in build_data:
            entrada = build_data['datos_entrada_deteccion']
            
            # Pre-cargar sub-stats en session_state ANTES de crear widgets
            if 'substats_objetivo' in entrada and entrada['substats_objetivo']:
                for substat, value in entrada['substats_objetivo'].items():
                    if substat in IGS_SUB_STATS and value is not None:
                        st.session_state[f"reverse_substat_{substat}"] = int(value)
                
                # Activar checkbox
                st.session_state.show_substats_reverse = True
        
        # Limpiar flags
        del st.session_state.build_data_to_load
        st.session_state.auto_load_build = False
        st.rerun()  
                            # Recargar página con datos
    # ... resto del código
    # === ENCABEZADO ATRACTIVO ===
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">
            🕵️‍♂️DETECTIVE DE BUILDS
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 0.5rem 0;">
            🎮 ¿Viste un build increíble en el juego? ¡Descubre sus secretos!
        </p>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">
            Ingresa lo que puedes ver → Nosotros calculamos lo que falta
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === FORZAR ACTIVACIÓN DE SUB-STATS SI HAY DATOS CARGADOS ===
    # Verificar si hay sub-stats cargadas en session_state
    substats_detectadas = any(
        key.startswith('reverse_substat_') and st.session_state.get(key) is not None 
        for key in st.session_state.keys()
    )
    
    if substats_detectadas and not st.session_state.get('show_substats_reverse', False):
        st.session_state.show_substats_reverse = True
        st.info("✅ Sub-estadísticas detectadas - Activando automáticamente")
    # === SECCIÓN DE CARGA DE BUILDS PREVIOS ===
    with st.expander("📤 **CARGA RÁPIDA DE BUILD DETECTADO PREVIO**", expanded=False):
        st.markdown("""
        <div style="
            background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        ">
            <h4 style="color: #2c3e50; margin-top: 0;">💡 ¿Ya tienes un build detectado?</h4>
            <p style="color: #34495e; margin-bottom: 0;">
                Carga tu archivo JSON para refinarlo o modificarlo fácilmente
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar si venimos de una recarga después de cargar un archivo
        if 'just_loaded_reverse_file' in st.session_state and st.session_state.just_loaded_reverse_file:
            st.session_state.just_loaded_reverse_file = False
            uploader_key_reverse = "reverse_build_file_uploader_cleared"
        else:
            uploader_key_reverse = "reverse_build_file_uploader_v33"
        
        uploaded_reverse_file = st.file_uploader(
            "📤 **Arrastra tu archivo JSON aquí:**", 
            type=["json"], 
            key=uploader_key_reverse,
            help="Acepta archivos de builds detectados previamente o builds normales"
        )
        
        if uploaded_reverse_file is not None:
            try:
                content_string_reverse = uploaded_reverse_file.read().decode()
                data_reverse_cargada = json.loads(content_string_reverse)
                
                # Verificar si es un archivo de build detectado
                is_detected_build = (
                    'metodo_deteccion' in data_reverse_cargada and 
                    data_reverse_cargada['metodo_deteccion'] == 'Detector_Reverso'
                )
                
                # También aceptar builds normales
                is_normal_build = (
                    'base_profile' in data_reverse_cargada and
                    'nodos_habilidad_desbloqueados' in data_reverse_cargada
                )
                
                if is_detected_build or is_normal_build:
                    # Extraer datos del perfil base
                    if 'base_profile' in data_reverse_cargada:
                        base_profile = data_reverse_cargada['base_profile']
                        
                        # Establecer posición y dimensiones
                        if 'posicion' in base_profile:
                            st.session_state.reverse_pos = base_profile['posicion']
                        if 'altura' in base_profile:
                            st.session_state.reverse_altura = base_profile['altura']
                        
                        # Si es un build detectado, puede tener el peso detectado
                        if is_detected_build and 'peso' in base_profile:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(45deg, #ffecd2 0%, #fcb69f 100%);
                                padding: 1rem;
                                border-radius: 8px;
                                border-left: 4px solid #ff6b35;
                                margin: 1rem 0;
                            ">
                                <p style="margin: 0; color: #2c3e50;">
                                    💡 <strong>Este build fue detectado con peso:</strong> 
                                    <span style="color: #e74c3c; font-weight: bold; font-size: 1.2em;">
                                        {base_profile['peso']}kg
                                    </span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Cargar nodos de habilidad
                    if 'nodos_habilidad_desbloqueados' in data_reverse_cargada:
                        nodos_cargados = data_reverse_cargada['nodos_habilidad_desbloqueados']
                        
                        if nodos_cargados and df_skill_trees_global is not None:
                            # Limpiar selecciones anteriores
                            arboles_disponibles = sorted(df_skill_trees_global['Arbol'].unique())
                            for arbol in arboles_disponibles:
                                st.session_state[f"reverse_nodes_{arbol}"] = []
                            
                            # Agrupar nodos por árbol y cargar
                            nodos_por_arbol_cargados = {}
                            for node_id in nodos_cargados:
                                node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == str(node_id)]
                                if not node_data.empty:
                                    node_info = node_data.iloc[0]
                                    arbol_name = node_info['Arbol']
                                    nombre_visible = node_info['Nombre_Visible']
                                    
                                    if arbol_name not in nodos_por_arbol_cargados:
                                        nodos_por_arbol_cargados[arbol_name] = []
                                    nodos_por_arbol_cargados[arbol_name].append(nombre_visible)
                            
                            # Establecer los nodos en session_state
                            for arbol_name, nodos_nombres in nodos_por_arbol_cargados.items():
                                st.session_state[f"reverse_nodes_{arbol_name}"] = nodos_nombres
                    
                    # Si es un build detectado, cargar datos de entrada originales
                    if is_detected_build and 'datos_entrada_deteccion' in data_reverse_cargada:
                        datos_entrada = data_reverse_cargada['datos_entrada_deteccion']
                        
                        # Cargar estadísticas principales si existen
                        if 'stats_objetivo' in datos_entrada and datos_entrada['stats_objetivo']:
                            for stat, value in datos_entrada['stats_objetivo'].items():
                                if stat in ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']:
                                    st.session_state[f"reverse_stat_{stat}"] = value
                            st.session_state.usar_stats_finales_reverse = True
                        
                                                # Cargar sub-estadísticas si existen (VERSIÓN CORREGIDA)
                        if 'substats_objetivo' in datos_entrada and datos_entrada['substats_objetivo']:
                            substats_cargadas = 0
                            substats_info = datos_entrada['substats_objetivo']
                            
                            # Debug: Mostrar las substats encontradas
                            st.info(f"🔍 DEBUG: Sub-stats encontradas: {list(substats_info.keys())}")
                            
                            # Primero, activar el checkbox de sub-stats
                            st.session_state.show_substats_reverse = True
                            
                            # Luego cargar cada sub-stat
                            for substat, value in substats_info.items():
                                if substat in IGS_SUB_STATS and value is not None:
                                    try:
                                        # Asegurar que el valor sea numérico
                                        numeric_value = int(float(value))
                                        st.session_state[f"reverse_substat_{substat}"] = numeric_value
                                        substats_cargadas += 1
                                        
                                        # Debug: Confirmar cada carga
                                        st.success(f"✅ Cargada: {substat} = {numeric_value}")
                                        
                                    except (ValueError, TypeError) as e:
                                        st.warning(f"⚠️ Error cargando {substat}: {e}")
                            
                            if substats_cargadas > 0:
                                st.success(f"✅ Se cargaron {substats_cargadas} sub-estadísticas")
                            else:
                                st.warning("⚠️ No se pudieron cargar sub-estadísticas válidas")
                        else:
                            st.info("ℹ️ No se encontraron sub-estadísticas en el archivo")
                    # Mostrar información de lo que se cargó
                    loaded_build_name_reverse = data_reverse_cargada.get("build_name", "Build Sin Nombre")
                    precision_info = ""
                    
                    if is_detected_build and 'precision_deteccion' in data_reverse_cargada:
                        precision_info = f" (Precisión: {data_reverse_cargada['precision_deteccion']})"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(45deg, #a8edea 0%, #fed6e3 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 2px solid #2ecc71;
                        margin: 1rem 0;
                    ">
                        <h4 style="color: #27ae60; margin: 0;">
                            ✅ Build Cargado Exitosamente
                        </h4>
                        <p style="color: #2c3e50; margin: 0.5rem 0;">
                            <strong>{loaded_build_name_reverse}</strong>{precision_info}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar resumen de lo cargado
                    col_loaded_info1, col_loaded_info2 = st.columns(2)
                    
                    with col_loaded_info1:
                        if 'base_profile' in data_reverse_cargada:
                            profile = data_reverse_cargada['base_profile']
                            st.markdown(f"""
                            <div style="
                                background: #e8f5e8;
                                padding: 1rem;
                                border-radius: 8px;
                                text-align: center;
                            ">
                                <p style="margin: 0; color: #2c3e50;">
                                    📋 <strong>Perfil:</strong> {profile.get('posicion', 'N/A')} | {profile.get('altura', 'N/A')}cm
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_loaded_info2:
                        nodos_count = len(data_reverse_cargada.get('nodos_habilidad_desbloqueados', []))
                        st.markdown(f"""
                        <div style="
                            background: #e8f5e8;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                        ">
                            <p style="margin: 0; color: #2c3e50;">
                                🌳 <strong>Nodos:</strong> {nodos_count} cargados
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Marcar flag para limpiar el uploader
                    st.session_state.just_loaded_reverse_file = True
                    
                    st.markdown("""
                    <div style="
                        background: #fff3cd;
                        padding: 1rem;
                        border-radius: 8px;
                        border-left: 4px solid #ffc107;
                        margin: 1rem 0;
                    ">
                        <p style="margin: 0; color: #856404;">
                            🔄 La página se actualizará para mostrar los datos cargados...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                    
                else:
                    st.markdown("""
                    <div style="
                        background: #f8d7da;
                        padding: 1rem;
                        border-radius: 8px;
                        border-left: 4px solid #dc3545;
                        margin: 1rem 0;
                    ">
                        <p style="margin: 0; color: #721c24;">
                            ❌ El archivo no es un build detectado válido o un build normal. Verifica el formato.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except json.JSONDecodeError:
                st.markdown("""
                <div style="
                    background: #f8d7da;
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 4px solid #dc3545;
                    margin: 1rem 0;
                ">
                    <p style="margin: 0; color: #721c24;">
                        ❌ Error: El archivo no es un JSON válido.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div style="
                    background: #f8d7da;
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 4px solid #dc3545;
                    margin: 1rem 0;
                ">
                    <p style="margin: 0; color: #721c24;">
                        ❌ Error al procesar el archivo: {e}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # === SEPARADOR VISUAL ===
    st.markdown("""
    <div style="
        height: 4px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        margin: 2rem 0;
        border-radius: 2px;
    "></div>
    """, unsafe_allow_html=True)
    
    # === TÍTULO DE SECCIÓN PRINCIPAL ===
    st.markdown("""
    <div style="
        background: linear-gradient(45deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
    ">
        <h2 style="color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📋 DATOS CONOCIDOS
        </h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Ingresa toda la información que puedes ver en el juego
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === COLUMNAS PRINCIPALES ===
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # === SECCIÓN: DATOS BÁSICOS ===
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <h3 style="color: white; margin: 0; text-align: center;">
                ⚙️ CONFIGURACIÓN BASE
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Datos básicos con mejor styling
        reverse_pos = st.selectbox(
            "🎯 **Posición del Jugador:**", 
            _sorted_lp_sb_init, 
            key="reverse_pos",
            help="La posición que ves en el juego"
        )
        
        # Altura con opciones disponibles
        idx_altura_reverse = _unique_alts_sb_init.index(180) if 180 in _unique_alts_sb_init else 0
        reverse_altura = st.selectbox(
            "📏 **Altura (cm):**", 
            _unique_alts_sb_init, 
            index=idx_altura_reverse,
            key="reverse_altura",
            help="Rangos: 160-162, 163-167, 168-172, 173-177, 178-182, 183-187, 188-192, 193-195"
        )
        
        # === SECCIÓN: NODOS DE HABILIDAD ===
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1.5rem 0 1rem 0;
        ">
            <h3 style="color: white; margin: 0; text-align: center;">
                🌳 NODOS DE HABILIDAD
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        reverse_nodes = []
        
        if df_skill_trees_global is not None:
            arboles_disponibles = sorted(df_skill_trees_global['Arbol'].unique())
            
            mostrar_selector_nodos = st.checkbox(
                "📋 **Activar Selector de Nodos por Árbol**", 
                value=True,
                key="show_node_selector",
                help="Marca esta casilla para poder seleccionar los nodos que ves desbloqueados en el juego"
            )
            
            if mostrar_selector_nodos:
                # Pre-crear el mapeo de nombres a IDs para evitar búsquedas repetidas
                mapeo_nombres_ids = {}
                for arbol in arboles_disponibles:
                    nodos_del_arbol = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol]
                    for _, nodo in nodos_del_arbol.iterrows():
                        nombre_visible = nodo['Nombre_Visible']
                        id_nodo = nodo['ID_Nodo']
                        mapeo_nombres_ids[nombre_visible] = id_nodo
                
                # Botón global para limpiar con mejor styling
                if st.button(
                    "🧹 **LIMPIAR TODOS LOS ÁRBOLES**", 
                    key="clear_all_trees_reverse",
                    help="Elimina todas las selecciones de nodos en todos los árboles"
                ):
                    for arbol in arboles_disponibles:
                        st.session_state[f"reverse_nodes_{arbol}"] = []
                
                st.markdown("---")
                
                # Procesar árbol por árbol con mejor diseño
                for idx, arbol in enumerate(arboles_disponibles):
                    # Color diferente para cada árbol
                    colores_arboles = [
                        "linear-gradient(45deg, #667eea 0%, #764ba2 100%)",
                        "linear-gradient(45deg, #f093fb 0%, #f5576c 100%)",
                        "linear-gradient(45deg, #4facfe 0%, #00f2fe 100%)",
                        "linear-gradient(45deg, #43e97b 0%, #38f9d7 100%)",
                        "linear-gradient(45deg, #fa709a 0%, #fee140 100%)",
                        "linear-gradient(45deg, #a8edea 0%, #fed6e3 100%)"
                    ]
                    color_arbol = colores_arboles[idx % len(colores_arboles)]
                    
                    st.markdown(f"""
                    <div style="
                        background: {color_arbol};
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 1rem 0;
                    ">
                        <h4 style="color: white; margin: 0; text-align: center;">
                            🌳 {arbol.upper()}
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Botones de acción del árbol en una sola fila
                    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
                    
                    with col_btn1:
                        if st.button(
                            "✅ **Todo**", 
                            key=f"select_all_{arbol}_reverse", 
                            help=f"Seleccionar todos los nodos de {arbol}"
                        ):
                            nodos_del_arbol = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol]
                            opciones_nodos = []
                            for _, nodo in nodos_del_arbol.iterrows():
                                nombre_visible = nodo['Nombre_Visible']
                                opciones_nodos.append(nombre_visible)
                            st.session_state[f"reverse_nodes_{arbol}"] = opciones_nodos
                    
                    with col_btn2:
                        if st.button(
                            "❌ **Limpiar**", 
                            key=f"clear_all_{arbol}_reverse", 
                            help=f"Limpiar todos los nodos de {arbol}"
                        ):
                            st.session_state[f"reverse_nodes_{arbol}"] = []
                    
                    # Obtener nodos del árbol
                    nodos_del_arbol = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol]
                    
                    # Crear opciones con nombre visible + ID
                    opciones_nodos = []
                    for _, nodo in nodos_del_arbol.iterrows():
                        nombre_visible = nodo['Nombre_Visible']
                        id_nodo = nodo['ID_Nodo']
                        opciones_nodos.append((f"{nombre_visible}", id_nodo))
                    
                    # Multiselect con contador y mejor descripción
                    current_selection = st.session_state.get(f"reverse_nodes_{arbol}", [])
                    nodos_seleccionados = st.multiselect(
                        f"**Nodos desbloqueados en {arbol}:**",
                        options=[opcion[0] for opcion in opciones_nodos],
                        default=current_selection,
                        key=f"reverse_nodes_{arbol}",
                        help=f"Selecciona los nodos que ves desbloqueados en el árbol {arbol}. "
                             f"Actualmente: {len(current_selection)}/{len(opciones_nodos)} seleccionados"
                    )
                    
                    # Extraer los IDs de los nodos seleccionados
                    for nombre_seleccionado in nodos_seleccionados:
                        if nombre_seleccionado in mapeo_nombres_ids:
                            reverse_nodes.append(mapeo_nombres_ids[nombre_seleccionado])
                    
                    # Mostrar información del árbol si hay nodos seleccionados
                    if nodos_seleccionados:
                        costo_arbol = 0
                        for nombre_sel in nodos_seleccionados:
                            if nombre_sel in mapeo_nombres_ids:
                                node_id = mapeo_nombres_ids[nombre_sel]
                                node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id]
                                if not node_data.empty:
                                    costo_arbol += node_data.iloc[0]['Costo']
                        
                        # Mostrar información con colores y mejor diseño
                        if costo_arbol <= 40:
                            color_info = "#d4edda"
                            border_color = "#28a745"
                            icono = "🟢"
                        elif costo_arbol <= 80:
                            color_info = "#fff3cd"
                            border_color = "#ffc107"
                            icono = "🟡"
                        else:
                            color_info = "#f8d7da"
                            border_color = "#dc3545"
                            icono = "🔴"
                        
                        st.markdown(f"""
                        <div style="
                            background: {color_info};
                            padding: 1rem;
                            border-radius: 8px;
                            border-left: 4px solid {border_color};
                            margin: 0.5rem 0;
                        ">
                            <p style="margin: 0; color: #2c3e50;">
                                {icono} <strong>Árbol {arbol}:</strong> 
                                {len(nodos_seleccionados)} nodos | {costo_arbol} puntos
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # Mostrar resumen general de puntos con mejor diseño
                if reverse_nodes:
                    puntos_totales_nodos = 0
                    if df_skill_trees_global is not None:
                        for node_id in reverse_nodes:
                            node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id]
                            if not node_data.empty:
                                puntos_totales_nodos += node_data.iloc[0]['Costo']
                    
                    progreso_nodos = (puntos_totales_nodos / TOTAL_SKILL_POINTS) * 100
                    
                    # Color del progreso basado en el porcentaje
                    if progreso_nodos <= 50:
                        color_progreso = "#28a745"
                    elif progreso_nodos <= 80:
                        color_progreso = "#ffc107"
                    else:
                        color_progreso = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin: 1rem 0;
                        text-align: center;
                    ">
                        <h4 style="color: #2c3e50; margin: 0;">📊 RESUMEN DE NODOS</h4>
                        <p style="color: #2c3e50; margin: 0.5rem 0;">
                            <strong>{len(reverse_nodes)} nodos seleccionados</strong>
                        </p>
                        <p style="color: #2c3e50; margin: 0;">
                            <strong>{puntos_totales_nodos}/{TOTAL_SKILL_POINTS} puntos</strong> 
                            ({progreso_nodos:.1f}% del total)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barra de progreso visual
                    st.markdown(f"""
                    <div style="
                        width: 100%;
                        background: #e9ecef;
                        border-radius: 10px;
                        height: 20px;
                        margin: 1rem 0;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {progreso_nodos}%;
                            background: {color_progreso};
                            height: 100%;
                            border-radius: 10px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background: #e2e3e5;
                    padding: 1.5rem;
                    border-radius: 10px;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <p style="margin: 0; color: #6c757d;">
                        ℹ️ Activa el selector para elegir nodos por árbol
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # === SECCIÓN: ESTADÍSTICAS ===
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <h3 style="color: white; margin: 0; text-align: center;">
                📊 ESTADÍSTICAS OBSERVADAS
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Estadísticas finales opcionales con mejor diseño
        usar_stats_finales = st.checkbox(
            "📈 **Usar estadísticas Generales (PAC, SHO, etc.)**", 
            key="usar_stats_finales_reverse",
            help="Activa si puedes ver las estadísticas principales en el juego. "
                 "Desmarca si están buggeadas o no son confiables"
        )
        
        reverse_stats = {}
        if usar_stats_finales:
            st.markdown("""
            <div style="
                background: #e8f5e8;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            ">
                <h4 style="color: #2c3e50; margin: 0;">📊 Estadísticas Globales</h4>
                <p style="color: #6c757d; margin: 0.5rem 0 0 0; font-size: 0.9em;">
                    Ingresa los valores exactos que ves en la carta del jugador
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Crear las estadísticas en una cuadrícula 2x3
            cols_stats = st.columns(2)
            stats_principales = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
            
            for idx, stat in enumerate(stats_principales):
                with cols_stats[idx % 2]:
                    reverse_stats[stat] = st.number_input(
                        f"⚡ **{stat}:**", 
                        min_value=1, 
                        max_value=99, 
                        value=75, 
                        key=f"reverse_stat_{stat}",
                        help=f"Valor de {stat} que aparece en la carta del jugador"
                    )
        else:
            st.markdown("""
            <div style="
                background: #fff3cd;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid #ffc107;
            ">
                <p style="margin: 0; color: #856404;">
                    📊 Stats principales deshabilitadas. Solo se usarán sub-estadísticas.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # === SECCIÓN: SUB-ESTADÍSTICAS ===
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1.5rem 0 1rem 0;
        ">
            <h3 style="color: white; margin: 0; text-align: center;">
                🎯 SUB-ESTADÍSTICAS
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sub-estadísticas organizadas por categoría
        mostrar_substats = st.checkbox(
            "🔬 **Usar sub-estadísticas (máxima precisión)**",
            key="show_substats_reverse",
            help="Activa para ingresar sub-estadísticas específicas. "
                 "Esto dará resultados mucho más precisos en la detección"
        )
        
        reverse_substats = {}
        if mostrar_substats:
            # DEBUG: Mostrar las claves relevantes de session_state
            st.markdown("---")
            st.subheader("🕵️ DEBUG: Contenido de st.session_state para sub-stats (antes de inputs)")
            relevant_ss_keys_debug = {k: v for k, v in st.session_state.items() if k.startswith("reverse_substat_")}
            if relevant_ss_keys_debug:
                st.json(relevant_ss_keys_debug)
            else:
                st.warning("No hay claves 'reverse_substat_' en st.session_state en este punto.")
            st.markdown("---")
            # FIN DEBUG
            st.markdown("""
            <div style="
                background: #e8f4f8;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            ">
                <p style="color: #2c3e50; margin: 0; font-size: 0.9em;">
                    💡 <strong>Tip:</strong> Solo completa las sub-estadísticas que puedes ver claramente. 
                    Deja en blanco las que no conoces.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            for main_cat, sub_stats_list in SUB_STATS_MAPPING.items():
                # Color para cada categoría principal
                colores_categorias = {
                    'PAC': '#ff6b6b',
                    'SHO': '#4ecdc4', 
                    'PAS': '#45b7d1',
                    'DRI': '#96ceb4',
                    'DEF': '#feca57',
                    'PHY': '#ff9ff3'
                }
                color_cat = colores_categorias.get(main_cat, '#74b9ff')
                
                st.markdown(f"""
                <div style="
                    background: {color_cat};
                    padding: 0.8rem;
                    border-radius: 8px;
                    margin: 1rem 0 0.5rem 0;
                ">
                    <h5 style="color: white; margin: 0; text-align: center;">
                        {main_cat}
                    </h5>
                </div>
                """, unsafe_allow_html=True)
                
                # Crear columnas para las sub-stats de esta categoría
                num_cols = min(len(sub_stats_list), 2)
                if num_cols > 0:
                    cols_substats = st.columns(num_cols)
                    
                    for idx, substat in enumerate(sub_stats_list):
                        with cols_substats[idx % num_cols]:
                            reverse_substats[substat] = st.number_input(
                                f"📊 **{substat}:**", 
                                min_value=0, 
                                max_value=99, 
                                value=st.session_state.get(f"reverse_substat_{substat}", None),
                                key=f"reverse_substat_{substat}",
                                help=f"Sub-estadística {substat} de la categoría {main_cat}"
                            )
        else:
            st.markdown("""
            <div style="
                background: #e2e3e5;
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                margin: 1rem 0;
            ">
                <p style="margin: 0; color: #6c757d;">
                    🔬 Activa las sub-estadísticas para mayor precisión en la detección
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # === SECCIÓN DE DETECCIÓN ===
    st.markdown("""
    <div style="
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        margin: 2rem 0;
        border-radius: 2px;
    "></div>
    """, unsafe_allow_html=True)
    
    # Validación con mejor diseño
    tiene_datos_suficientes = reverse_nodes and (reverse_stats or any(v is not None for v in reverse_substats.values()))
    
    if tiene_datos_suficientes:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
        ">
            <h2 style="color: white; margin: 0;">
                🚀 ¡LISTO PARA ENCONTRAR!
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(
            "🔍 **INICIAR DETECCIÓN**", 
            key="detect_reverse_build", 
            type="primary",
            help="Analiza todas las combinaciones posibles para encontrar el peso exacto y las instalaciones",
            use_container_width=True
        ):
            with st.spinner("🔄 Analizando todas las combinaciones posibles..."):
                # Añadir una pequeña animación de progreso
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress((i + 1))
                    if i % 20 == 0:  # Actualizar cada 20%
                        import time
                        time.sleep(0.05)
                
                resultados = detectar_build_reverso(
                    reverse_pos, 
                    reverse_altura, 
                    reverse_stats if usar_stats_finales else {}, 
                    reverse_nodes, 
                    reverse_substats
                )
                
                progress_bar.empty()  # Limpiar la barra de progreso
                mostrar_resultados_deteccion(resultados, reverse_pos, reverse_altura, reverse_nodes, reverse_stats, reverse_substats)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            border: 3px dashed #ff6b6b;
        ">
            <h3 style="color: #c53030; margin: 0;">
                ⚠️ DATOS INCOMPLETOS
            </h3>
            <p style="color: #e53e3e; margin: 0.5rem 0 0 0;">
                Necesitas completar: <strong>Posición + Altura + Algunos nodos</strong><br>
                <strong>Y al menos:</strong> Estadísticas principales <strong>O</strong> sub-estadísticas
            </p>
        </div>
        """, unsafe_allow_html=True)
def calcular_boosts_nodos(nodos_ids):
    """Calcula los boosts totales de una lista de nodos"""
    boosts = defaultdict(int)
    
    if df_skill_trees_global is not None:
        for node_id in nodos_ids:
            node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id]
            if not node_data.empty:
                node_info = node_data.iloc[0]
                for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_SKILL_TREE:
                    if stat_col in node_info.index and pd.notna(node_info[stat_col]) and node_info[stat_col] != 0:
                        if stat_col in IGS_SUB_STATS:  # Solo sub-stats
                            boosts[stat_col] += int(node_info[stat_col])
    
    return dict(boosts)

def calcular_boosts_instalaciones(instalaciones_ids):
    """Calcula los boosts totales de una lista de instalaciones"""
    boosts = defaultdict(int)
    
    if df_instalaciones_global is not None:
        for facility_id in instalaciones_ids:
            facility_data = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id]
            if not facility_data.empty:
                facility_info = facility_data.iloc[0]
                for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                    if stat_col in facility_info.index and pd.notna(facility_info[stat_col]) and facility_info[stat_col] != 0:
                        boosts[stat_col] += int(facility_info[stat_col])
    
    return dict(boosts)

def hacer_json_serializable(obj):
    """Convierte objetos NumPy/Pandas a tipos nativos de Python para JSON"""
    
    if isinstance(obj, dict):
        return {key: hacer_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [hacer_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Para otros tipos de NumPy
        return obj.item()
    else:
        return obj

def detectar_build_reverso(posicion, altura, stats_objetivo, nodos_conocidos, substats_objetivo=None):
    """
    🕵️‍♂️ DETECTIVE DE BUILDS - Versión Corregida con Presupuesto Real
    """
    candidatos = []
    pesos_reales = [45, 55, 69, 80, 91, 103]
    PRESUPUESTO_MAX = 1750000
    
    st.info(f"🔍 Analizando: {posicion} + {altura}cm + {len(nodos_conocidos)} nodos (Presupuesto: ${PRESUPUESTO_MAX:,})")
    
    # PASO 1: Evaluar cada peso sin instalaciones primero
    for peso in pesos_reales:
        # Stats base + nodos
        stats_base = calcular_stats_base_jugador(posicion, altura, peso, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
        if stats_base is None:
            continue
        
        boosts_nodos = calcular_boosts_nodos(nodos_conocidos)
        stats_con_nodos = aplicar_boosts_a_stats(stats_base, boosts_nodos)
        
        # Comparar con objetivo
        if substats_objetivo and any(v is not None for v in substats_objetivo.values()):
            precision = calcular_precision_substats(stats_con_nodos, substats_objetivo)
            diferencias = calcular_diferencias_substats(stats_con_nodos, substats_objetivo)
        elif stats_objetivo:
            precision = calcular_precision_stats_globales(stats_con_nodos, stats_objetivo)
            diferencias = calcular_diferencias_stats_globales(stats_con_nodos, stats_objetivo)
        else:
            continue
        
        # NUEVO: Calcular desvío máximo para análisis
        max_desvio = max(abs(diff) for diff in diferencias.values()) if diferencias else 0
        
        # Candidato base (sin instalaciones)
        candidatos.append({
            'peso': peso,
            'instalaciones': [],
            'precision': precision,
            'stats_calculadas': stats_con_nodos,
            'stats_base_puras': stats_base,
            'diferencias': diferencias,
            'costo_total': 0,
            'max_desvio': max_desvio,
            'tipo_deteccion': 'Solo Nodos'
        })
        
        # PASO 2: OBLIGATORIO - Probar instalaciones si hay desvíos >= 2 puntos
        stats_desviadas = {stat: diff for stat, diff in diferencias.items() if abs(diff) >= 2}
        
        if stats_desviadas:
            st.info(f"🔧 Peso {peso}kg: Detectados desvíos de hasta {max_desvio:.0f} puntos. Probando instalaciones...")
            
            # Buscar combinaciones realistas de instalaciones
            combinaciones_instalaciones = generar_combinaciones_instalaciones_realistas(stats_desviadas, PRESUPUESTO_MAX)
            
            st.info(f"🧪 Probando {len(combinaciones_instalaciones)} combinaciones de instalaciones...")
            
            for combo_instalaciones in combinaciones_instalaciones:
                # Calcular costo total de la combinación
                costo_total = calcular_costo_combo_instalaciones(combo_instalaciones)
                
                if costo_total <= PRESUPUESTO_MAX:
                    # Calcular boosts acumulativos
                    boosts_instalaciones = calcular_boosts_instalaciones_acumulativo(combo_instalaciones)
                    boosts_totales = combinar_boosts(boosts_nodos, boosts_instalaciones)
                    stats_finales = aplicar_boosts_a_stats(stats_base, boosts_totales)
                    
                    # Recalcular precisión
                    if substats_objetivo and any(v is not None for v in substats_objetivo.values()):
                        precision_con_inst = calcular_precision_substats(stats_finales, substats_objetivo)
                        diferencias_con_inst = calcular_diferencias_substats(stats_finales, substats_objetivo)
                    else:
                        precision_con_inst = calcular_precision_stats_globales(stats_finales, stats_objetivo)
                        diferencias_con_inst = calcular_diferencias_stats_globales(stats_finales, stats_objetivo)
                    
                    # NUEVO: Calcular nuevo desvío máximo
                    nuevo_max_desvio = max(abs(diff) for diff in diferencias_con_inst.values()) if diferencias_con_inst else 0
                    
                    # CRITERIO MÁS FLEXIBLE: Agregar si mejora precisión O reduce desvío máximo
                    mejora_precision = precision_con_inst > precision + 0.02  # Reducido de 0.05 a 0.02
                    reduce_desvio = nuevo_max_desvio < max_desvio - 1  # Reduce al menos 1 punto de desvío
                    
                    if mejora_precision or reduce_desvio:
                        candidatos.append({
                            'peso': peso,
                            'instalaciones': combo_instalaciones,
                            'precision': precision_con_inst,
                            'stats_calculadas': stats_finales,
                            'diferencias': diferencias_con_inst,
                            'costo_total': costo_total,
                            'max_desvio': nuevo_max_desvio,
                            'desvio_corregido': max_desvio - nuevo_max_desvio,
                            'tipo_deteccion': f'{len(combo_instalaciones)} Instalaciones (${costo_total:,}) - Desvío: {nuevo_max_desvio:.0f}pts'
                        })
                        
                        st.success(f"✅ Mejora encontrada: Precisión {precision_con_inst:.1%} (antes {precision:.1%}), Desvío {nuevo_max_desvio:.0f}pts (antes {max_desvio:.0f}pts)")
    
    # Ordenamiento inteligente usando la función existente pero mejorada
    def calcular_score_candidato(candidato):
        precision = candidato['precision']
        max_desvio = candidato.get('max_desvio', 999)
        costo = candidato.get('costo_total', 0)
        
        # Bonificar alta precisión
        score_precision = precision * 100
        
        # Penalizar desvíos altos
        penalizacion_desvio = max_desvio * 5
        
        # Bonificar corrección de desvíos
        correccion_bonus = candidato.get('desvio_corregido', 0) * 10
        
        # Penalizar costo (ligeramente)
        penalizacion_costo = costo / 100000
        
        score_final = score_precision + correccion_bonus - penalizacion_desvio - penalizacion_costo
        
        return score_final
    
    candidatos.sort(key=calcular_score_candidato, reverse=True)
    return candidatos[:8]  # Top 8 candidatos
def generar_combinaciones_instalaciones_realistas(stats_desviadas, presupuesto_max):
    """
    Versión corregida que respeta las instalaciones acumulativas
    """
    if df_instalaciones_global is None:
        return []
    
    # Mapeo igual que antes...
    mapeo_stats_instalaciones = {
        'Acc': 'CientD', 'Spr': 'Parac', 'STR': 'SalaPes', 'STA': 'PrepFis',
        'AGI': 'Yog', 'BAL': 'Yog', 'JUMP': 'DTequipo', 'Fin': 'EntTir',
        'SPow': 'Tacos', 'HAcc': 'RedTenis', 'Vol': 'Rebot', 'Vis': 'AnRen',
        'LP': 'EntPas', 'SP': 'AnRen', 'Cros': 'CanchaE', 'INT': 'EntTacDef',
        'AWA': 'Reclu', 'STAN': 'EntEntr', 'SLID': 'EntEntr', 'REA': 'TeraD',
        'COMP': 'TeraD', 'AGGR': 'TeraD'
    }
    
    tipos_necesarios = []
    for stat_desviada in stats_desviadas.keys():
        if stat_desviada in mapeo_stats_instalaciones:
            tipo_inst = mapeo_stats_instalaciones[stat_desviada]
            if tipo_inst not in tipos_necesarios:
                tipos_necesarios.append(tipo_inst)
    
    combinaciones = []
    
    # ESTRATEGIA CORREGIDA: Instalaciones acumulativas por tipo
    for tipo in tipos_necesarios:
        # Obtener TODOS los niveles de este tipo, ordenados por precio
        instalaciones_tipo = df_instalaciones_global[
            df_instalaciones_global['ID_Instalacion'].str.contains(tipo, na=False)
        ].sort_values('Precio')
        
        # NUEVO: Generar cadenas acumulativas
        cadena_acumulativa = []
        costo_acumulativo = 0
        
        for _, instalacion in instalaciones_tipo.iterrows():
            # Agregar esta instalación a la cadena
            cadena_acumulativa.append(instalacion['ID_Instalacion'])
            costo_acumulativo += instalacion['Precio']
            
            # Si la cadena completa cabe en el presupuesto, agregarla
            if costo_acumulativo <= presupuesto_max:
                combinaciones.append(cadena_acumulativa.copy())
            else:
                break  # No agregar más niveles de este tipo
    
    # ESTRATEGIA 2 CORREGIDA: Combinaciones de 2 tipos (ambos acumulativos)
    from itertools import combinations
    for combo_tipos in combinations(tipos_necesarios, 2):
        # Para cada combinación de tipos, encontrar los niveles máximos acumulativos
        instalaciones_combo = []
        costo_total = 0
        
        for tipo in combo_tipos:
            instalaciones_tipo = df_instalaciones_global[
                df_instalaciones_global['ID_Instalacion'].str.contains(tipo, na=False)
            ].sort_values('Precio')
            
            # Encontrar el nivel máximo acumulativo que podemos permitirnos
            cadena_tipo = []
            for _, instalacion in instalaciones_tipo.iterrows():
                costo_con_esta = costo_total + instalacion['Precio']
                if costo_con_esta <= presupuesto_max:
                    cadena_tipo.append(instalacion['ID_Instalacion'])
                    costo_total = costo_con_esta
                else:
                    break
            
            instalaciones_combo.extend(cadena_tipo)
        
        if instalaciones_combo:
            combinaciones.append(instalaciones_combo)
    
    # ESTRATEGIA 3 CORREGIDA: 3 tipos con niveles más conservadores
    if presupuesto_max >= 800000:
        for combo_tipos in combinations(tipos_necesarios, min(3, len(tipos_necesarios))):
            instalaciones_combo = []
            costo_total = 0
            
            for tipo in combo_tipos:
                instalaciones_tipo = df_instalaciones_global[
                    df_instalaciones_global['ID_Instalacion'].str.contains(tipo, na=False)
                ].sort_values('Precio')
                
                # Para 3 tipos, ser más conservador en los niveles
                cadena_tipo = []
                for _, instalacion in instalaciones_tipo.iterrows():
                    costo_con_esta = costo_total + instalacion['Precio']
                    if costo_con_esta <= presupuesto_max:
                        cadena_tipo.append(instalacion['ID_Instalacion'])
                        costo_total = costo_con_esta
                    else:
                        break
                
                instalaciones_combo.extend(cadena_tipo)
            
            if len(instalaciones_combo) > 0:
                combinaciones.append(instalaciones_combo)
    
    # Eliminar duplicados y limitar
    combinaciones_unicas = []
    for combo in combinaciones:
        combo_sorted = sorted(combo)
        if combo_sorted not in combinaciones_unicas:
            combinaciones_unicas.append(combo_sorted)
    
    return combinaciones_unicas[:20]
def calcular_costo_combo_instalaciones(combo_instalaciones):
    """Calcula el costo total de una combinación de instalaciones"""
    costo_total = 0
    if df_instalaciones_global is not None:
        for inst_id in combo_instalaciones:
            inst_data = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == inst_id]
            if not inst_data.empty:
                costo_total += inst_data.iloc[0]['Precio']
    return costo_total

def calcular_boosts_instalaciones_acumulativo(instalaciones_ids):
    """
    Calcula boosts acumulativos correctamente.
    Si tienes CientD_1, CientD_2, CientD_3 = suma todos los boosts
    """
    boosts = defaultdict(int)
    
    if df_instalaciones_global is not None:
        for facility_id in instalaciones_ids:
            facility_data = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == facility_id]
            if not facility_data.empty:
                facility_info = facility_data.iloc[0]
                for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                    if stat_col in facility_info.index and pd.notna(facility_info[stat_col]) and facility_info[stat_col] != 0:
                        boosts[stat_col] += int(facility_info[stat_col])
    
    return dict(boosts)

def calcular_precision_substats(stats_calculadas, substats_objetivo):
    """Calcula precisión basándose solo en sub-estadísticas"""
    total_diferencias = 0
    total_stats = 0
    
    for stat, valor_objetivo in substats_objetivo.items():
        if valor_objetivo is not None and stat in stats_calculadas:
            diferencia = abs(stats_calculadas[stat] - valor_objetivo)
            total_diferencias += diferencia
            total_stats += 1
    
    if total_stats == 0:
        return 0
    
    # Precisión más estricta para sub-stats (son más específicas)
    precision = max(0, 1 - (total_diferencias / (total_stats * 3)))
    return precision

def calcular_diferencias_substats(stats_calculadas, substats_objetivo):
    """Calcula diferencias solo para sub-estadísticas"""
    diferencias = {}
    
    for stat, valor_objetivo in substats_objetivo.items():
        if valor_objetivo is not None and stat in stats_calculadas:
            diferencias[stat] = stats_calculadas[stat] - valor_objetivo
    
    return diferencias

def calcular_precision_stats_globales(stats_calculadas, stats_objetivo):
    """Calcula precisión basándose en stats globales (menos confiable)"""
    total_diferencias = 0
    total_stats = 0
    
    for stat, valor_objetivo in stats_objetivo.items():
        if stat in stats_calculadas:
            diferencia = abs(stats_calculadas[stat] - valor_objetivo)
            total_diferencias += diferencia
            total_stats += 1
    
    if total_stats == 0:
        return 0
    
    # Precisión más relajada para stats globales (pueden estar buggeadas)
    precision = max(0, 1 - (total_diferencias / (total_stats * 8)))
    return precision

def calcular_diferencias_stats_globales(stats_calculadas, stats_objetivo):
    """Calcula diferencias solo para estadísticas globales"""
    diferencias = {}
    
    for stat, valor_objetivo in stats_objetivo.items():
        if stat in stats_calculadas:
            diferencias[stat] = stats_calculadas[stat] - valor_objetivo
    
    return diferencias
def calcular_precision_match(stats_calculadas, stats_objetivo, substats_objetivo=None):
    """
    Calcula qué tan bien coinciden las estadísticas calculadas con las objetivo
    """
    total_diferencias = 0
    total_stats = 0
    
    # Estadísticas principales
    for stat, valor_objetivo in stats_objetivo.items():
        if stat in stats_calculadas:
            diferencia = abs(stats_calculadas[stat] - valor_objetivo)
            total_diferencias += diferencia
            total_stats += 1
    
    # Sub-estadísticas (peso mayor en la precisión)
    if substats_objetivo:
        for stat, valor_objetivo in substats_objetivo.items():
            if valor_objetivo is not None and stat in stats_calculadas:
                diferencia = abs(stats_calculadas[stat] - valor_objetivo)
                total_diferencias += diferencia * 1.5  # Peso mayor
                total_stats += 1.5
    
    # Convertir diferencias a precisión (0-1)
    if total_stats == 0:
        return 0
    
    precision = max(0, 1 - (total_diferencias / (total_stats * 10)))  # Normalizar
    return precision

def mostrar_resultados_deteccion(resultados, reverse_pos, reverse_altura, reverse_nodes, reverse_stats, reverse_substats):
    """
    Muestra los resultados de la detección de build inverso con funcionalidad mejorada
    """
    if not resultados:
        st.markdown("""
        <div style="
            background: linear-gradient(45deg, #ff6b6b 0%, #ee5a52 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            border: 2px solid #ff4444;
        ">
            <h3 style="color: white; margin: 0;">🕵️‍♂️ INVESTIGACIÓN SIN RESULTADOS</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                No se encontraron builds que coincidan con los datos proporcionados.<br>
                Verifica que los nodos y estadísticas sean correctos.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(45deg, #28a745 0%, #20c997 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #28a745;
        margin-bottom: 2rem;
    ">
        <h3 style="color: white; margin: 0;">🕵️‍♂️ ¡CASO RESUELTO!</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Se encontraron <strong>{len(resultados)} configuraciones sospechosas</strong> que coinciden con las evidencias.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar top 3 candidatos con mejor diseño
    for i, candidato in enumerate(resultados[:3]):
        # Emojis según el puesto
        ranking_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
        tipo_deteccion = candidato.get('tipo_deteccion', 'Detección Estándar')
        costo_total = candidato.get('costo_total', 0)
        costo_text = f" | Costo: ${costo_total:,}" if costo_total > 0 else ""
        with st.expander(f"{ranking_emoji} **SOSPECHOSO #{i+1}** | Precisión: {candidato['precision']:.1%} | {tipo_deteccion}{costo_text}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔍 EVIDENCIAS ENCONTRADAS:**")
                
                # Información del peso con estilo detective
                peso_emoji = "⚖️"
                st.markdown(f"""
                <div style="
                    background: linear-gradient(45deg, #ffd700 0%, #ffed4e 100%);
                    padding: 1rem;
                    border-radius: 8px;
                    text-align: center;
                    margin: 1rem 0;
                    border: 2px solid #ffd700;
                ">
                    <h4 style="color: #856404; margin: 0;">{peso_emoji} PESO DETECTADO</h4>
                    <h2 style="color: #856404; margin: 0.5rem 0 0 0;">{candidato['peso']} kg</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if candidato['instalaciones']:
                    st.markdown("**🏨 INSTALACIONES IDENTIFICADAS:**")
                    for inst_id in candidato['instalaciones']:
                        inst_info = df_instalaciones_global[df_instalaciones_global['ID_Instalacion'] == inst_id]
                        if not inst_info.empty:
                            nombre = inst_info.iloc[0]['Nombre_Instalacion']
                            precio = inst_info.iloc[0]['Precio']
                            st.markdown(f"""
                            <div style="
                                background: rgba(0, 123, 255, 0.1);
                                padding: 0.5rem;
                                border-radius: 6px;
                                margin: 0.3rem 0;
                                border-left: 4px solid #007bff;
                            ">
                                🏨 <strong>{nombre}</strong><br>
                                <small>💰 ${precio:,}</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="
                        background: rgba(108, 117, 125, 0.1);
                        padding: 1rem;
                        border-radius: 8px;
                        text-align: center;
                        border: 2px dashed #6c757d;
                    ">
                        <p style="margin: 0; color: #6c757d;">
                            🏢 <strong>Sin instalaciones detectadas</strong><br>
                            <small>Build básico usando solo nodos de habilidad</small>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**📊 ANÁLISIS DE DIFERENCIAS:**")
                diferencias_ordenadas = sorted(candidato['diferencias'].items(), key=lambda x: abs(x[1]))
                                # En el with col2: donde se muestran las diferencias, agrega esto antes del bucle existente:
                
                # NUEVO: Mostrar información de corrección de desvíos
                if 'desvio_corregido' in candidato and candidato['desvio_corregido'] > 0:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(45deg, #28a745 0%, #20c997 100%);
                        padding: 0.8rem;
                        border-radius: 8px;
                        margin: 0.5rem 0;
                        text-align: center;
                    ">
                        <strong style="color: white;">
                            🎯 CORRECCIÓN EXITOSA: -{candidato['desvio_corregido']:.1f} puntos de desvío
                        </strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar el desvío máximo actual
                max_desvio_actual = candidato.get('max_desvio', 0)
                if max_desvio_actual > 0:
                    if max_desvio_actual <= 2:
                        color_desvio = "#28a745"
                        nivel_desvio = "EXCELENTE"
                    elif max_desvio_actual <= 5:
                        color_desvio = "#ffc107"
                        nivel_desvio = "BUENO"
                    else:
                        color_desvio = "#dc3545"
                        nivel_desvio = "NECESITA MEJORA"
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba({color_desvio[1:]}, 0.1);
                        padding: 0.5rem;
                        border-radius: 6px;
                        margin: 0.5rem 0;
                        border-left: 4px solid {color_desvio};
                    ">
                        📏 <strong>Desvío Máximo:</strong> {max_desvio_actual:.1f} puntos <small>({nivel_desvio})</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**📊 DIFERENCIAS POR ESTADÍSTICA:**")
                for stat, diff in diferencias_ordenadas:
                    if abs(diff) <= 1:
                        color = "#28a745"
                        icono = "✅"
                        nivel = "EXACTO"
                    elif abs(diff) <= 2:
                        color = "#ffc107"
                        icono = "⚠️"
                        nivel = "CERCA"
                    else:
                        color = "#dc3545"
                        icono = "❌"
                        nivel = "DESVIADO"
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba({color[1:]}, 0.1);
                        padding: 0.5rem;
                        border-radius: 6px;
                        margin: 0.3rem 0;
                        border-left: 4px solid {color};
                    ">
                        {icono} <strong>{stat}:</strong> ±{diff} <small>({nivel})</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Botones de acción mejorados
            col_recrear, col_descargar = st.columns(2)
            
            with col_recrear:
                # NUEVA LÓGICA: Generar JSON y simular carga automática
                build_para_buildcraft = {
                    "app_version": f"{APP_VERSION}_build_data",
                    "build_name": f"Build_Detectado_{reverse_pos}_{reverse_altura}cm_{candidato['peso']}kg",
                    "base_profile": {
                        "posicion": str(reverse_pos),
                        "altura": int(reverse_altura),
                        "peso": int(candidato['peso'])
                    },
                    "nodos_habilidad_desbloqueados": [str(node) for node in reverse_nodes],
                    "instalaciones_desbloqueadas": [str(inst) for inst in candidato['instalaciones']],
                    "aplicar_boost_instalaciones": True
                }
                
                if st.button(f"🛠️ Recrear en Build Craft", key=f"recreate_build_{i}_{reverse_pos}_{reverse_altura}_{candidato['peso']}"):
                    try:
                        # Convertir a JSON serializable
                        build_serializable = hacer_json_serializable(build_para_buildcraft)
                        
                        # CLAVE: Guardar el JSON en session_state para que Build Craft lo detecte
                        st.session_state.build_data_to_load = build_serializable
                        st.session_state.auto_load_build = True
                        
                        # Mostrar mensaje de éxito
                        st.success("✅ Build preparado para cargar en Build Craft!")
                        st.info("🔄 Ve a la pestaña '🛠️ Build Craft' - el build se cargará automáticamente.")
                        
                        # Mostrar resumen de lo que se va a cargar
                        with st.container(border=True):
                            st.markdown("**📋 Resumen del Build a Cargar:**")
                            col_summary1, col_summary2 = st.columns(2)
                            
                            with col_summary1:
                                st.write(f"**Posición:** {reverse_pos}")
                                st.write(f"**Altura:** {reverse_altura}cm")
                                st.write(f"**Peso:** {candidato['peso']}kg")
                                st.write(f"**Precisión:** {candidato['precision']:.1%}")
                            
                            with col_summary2:
                                st.write(f"**Nodos:** {len(reverse_nodes)} seleccionados")
                                st.write(f"**Instalaciones:** {len(candidato['instalaciones'])} activas")
                                st.write(f"**Boost Instalaciones:** ✅ Activado")
                        
                    except Exception as e:
                        st.error(f"❌ Error al preparar el build:")
                        st.error(f"Detalles: {str(e)}")
                        
                        # Mostrar datos para copia manual
                        st.info("💡 **Datos para copia manual:**")
                        st.code(f"""
Posición: {reverse_pos}
Altura: {reverse_altura}cm
Peso: {candidato['peso']}kg
Nodos: {len(reverse_nodes)} seleccionados
Instalaciones: {len(candidato['instalaciones'])} activas
                        """)
            
            with col_descargar:
                # ===== BOTÓN DE DESCARGA COMPLETO =====
                build_detectado = {
                    "app_version": f"{APP_VERSION}_build_detectado",
                    "build_name": f"Build_Detectado_{reverse_pos}_{reverse_altura}cm_{candidato['peso']}kg",
                    "metodo_deteccion": "Detective_Builds",
                    "precision_deteccion": f"{candidato['precision']:.1%}",
                    "base_profile": {
                        "posicion": str(reverse_pos),
                        "altura": int(reverse_altura),
                        "peso": int(candidato['peso'])
                    },
                    "nodos_habilidad_desbloqueados": [str(node) for node in reverse_nodes],
                    "instalaciones_desbloqueadas": [str(inst) for inst in candidato['instalaciones']],
                    "aplicar_boost_instalaciones": True,
                    "datos_entrada_deteccion": {
                        "stats_objetivo": reverse_stats if reverse_stats else None,
                        "substats_objetivo": {k: v for k, v in reverse_substats.items() if v is not None} if reverse_substats else None
                    },
                    "diferencias_calculadas": {str(k): int(v) for k, v in candidato['diferencias'].items()}
                }
                
                try:
                    build_detectado_serializable = hacer_json_serializable(build_detectado)
                    build_json = json.dumps(build_detectado_serializable, indent=2, ensure_ascii=False)
                    
                    file_name = f"Build_Detectado_{reverse_pos}_{reverse_altura}cm_{candidato['peso']}kg_FC25.json"
                    
                    st.download_button(
                        label="💾 **Descargar JSON**",
                        data=build_json,
                        file_name=file_name,
                        mime="application/json",
                        key=f"download_detected_{i}_{reverse_pos}_{reverse_altura}_{candidato['peso']}",
                        help="Descarga el build detectado como archivo JSON",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Error al generar descarga: {e}")
def obtener_instalaciones_comunes():
    """
    Obtiene las instalaciones más comúnmente usadas para probar
    """
    # Lista de instalaciones comunes (puedes expandir según el meta del juego)
    instalaciones_populares = [
        'FISICO_RITMO_1', 'FISICO_RITMO_2', 'FISICO_RITMO_3',
        'TIRO_POTENCIA_1', 'TIRO_POTENCIA_2', 'TIRO_POTENCIA_3',
        'DRIBBLING_AGILIDAD_1', 'DRIBBLING_AGILIDAD_2',
        'PASE_VISION_1', 'PASE_VISION_2',
        'DEFENSA_MARCAJE_1', 'DEFENSA_MARCAJE_2'
    ]
    
    # Filtrar solo las que existen en el CSV
    if df_instalaciones_global is not None:
        existentes = df_instalaciones_global['ID_Instalacion'].tolist()
        return [inst for inst in instalaciones_populares if inst in existentes]
    
    return []

# --- Pestaña: Detector Reverso ---
with tab_reverse:
    if not carga_completa_exitosa:
        st.error("Datos necesarios para el Detector Reverso no disponibles.")
    else:
        implementar_detector_build_inverso()