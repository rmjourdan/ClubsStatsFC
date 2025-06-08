import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import json
from collections import defaultdict
from visualizacion_fc25_style import mostrar_visualizacion_fc25  # NUEVO: importar visualización FC25


# Definir emoji de estrella AL INICIO
STAR_EMOJI = "⭐" 

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Calculadora Stats FC25 v3.3", layout="wide") 

# --- Definiciones Constantes ---
APP_VERSION = "v3.3.5" 
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

    prereq_ids_list = [pr_id.strip() for pr_id in str(prereqs_from_df).split(',') if pr_id.strip()]
    
    if not prereq_ids_list: 
        return True 

    for pr_id in prereq_ids_list:
        if pr_id not in unlocked_nodes_ids:
            return False 
            
    return True

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
        # Agrupar instalaciones por tipo y obtener solo el nivel más alto
        facility_groups = {}
        for facility_id in unlocked_facilities:
            facility_data = df_instalaciones[df_instalaciones['ID_Instalacion'] == facility_id]
            if not facility_data.empty:
                facility_info = facility_data.iloc[0]
                facility_type = facility_info.get('Instalacion', 'Sin Tipo')
                
                if facility_type not in facility_groups:
                    facility_groups[facility_type] = []
                facility_groups[facility_type].append({
                    'id': facility_id,
                    'nombre': facility_info.get('Nombre_Instalacion', f'Instalación {facility_id}'),
                    'precio': facility_info.get('Precio', 0),
                    'info': facility_info
                })
        
        if facility_groups:
            for facility_type, facilities in facility_groups.items():
                # Ordenar por precio para obtener el tier más alto
                facilities_sorted = sorted(facilities, key=lambda x: x['precio'])
                highest_tier = facilities_sorted[-1]
                
                # Calcular costos acumulados
                costos_acumulados = [str(f['precio']) for f in facilities_sorted]
                costos_str = f"({', '.join(costos_acumulados)})"
                
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
                
                instalaciones_html += f'''
                <div class="facility-item">
                    <span class="facility-type">🏨 {highest_tier['nombre']}</span>
                    <span class="facility-price">${highest_tier['precio']:,} {costos_str}</span>
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
    if nodos_html:
        html_content += f'''
        <div class="nodes-section">
            <div class="section-title">🌳 NODOS DE HABILIDAD DESBLOQUEADOS</div>
            {nodos_html}
        </div>'''
    
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
st.title(f"Calculadora Avanzada de Estadísticas FC25 ({APP_VERSION})")
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

# --- Selectores de Perfil Base para Build Craft en la Barra Lateral ---
st.sidebar.markdown("--- \n ### 🛠️ Perfil Base para Build Craft:")
idx_pos_sidebar_v32 = _sorted_lp_sb_init.index(st.session_state.bc_pos) if st.session_state.bc_pos in _sorted_lp_sb_init else 0
st.session_state.bc_pos = st.sidebar.selectbox("Posición Base (BC):", _sorted_lp_sb_init, index=idx_pos_sidebar_v32, key="sb_bc_pos_v33") 
idx_alt_sidebar_v32 = _unique_alts_sb_init.index(st.session_state.bc_alt) if st.session_state.bc_alt in _unique_alts_sb_init else 0
st.session_state.bc_alt = st.sidebar.selectbox("Altura Base (cm) (BC):", _unique_alts_sb_init, index=idx_alt_sidebar_v32, key="sb_bc_alt_v33") 
idx_pes_sidebar_v32 = _unique_pesos_sb_init.index(st.session_state.bc_pes) if st.session_state.bc_pes in _unique_pesos_sb_init else 0
st.session_state.bc_pes = st.sidebar.selectbox("Peso Base (kg) (BC):", _unique_pesos_sb_init, index=idx_pes_sidebar_v32, key="sb_bc_pes_v33") 
st.sidebar.markdown("---")
st.session_state.apply_facility_boosts_toggle = st.sidebar.checkbox("Aplicar Boosts de Instalaciones del Club", 
                                                                    value=st.session_state.get('apply_facility_boosts_toggle', True), 
                                                                    key="facility_boost_toggle_v33") 

# Definición de Pestañas
tab_calc, tab_build_craft, tab_facilities, tab_explorer, tab_best_combo, tab_filters = st.tabs([
    "🧮 Calculadora", "🛠️ Build Craft", "🏨 Instalaciones Club", 
    "🔎 Explorador Mejoras", "🔍 Búsqueda Óptima", "📊 Filtros Múltiples"
])

# --- Pestaña: Calculadora y Comparador ---
with tab_calc: 
    st.header("Calculadora de Estadísticas y Comparador (Stats Base)")
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
            all_stats_df_base_display_top5_calc = all_stats_df_base.copy()
            all_stats_df_base_display_top5_calc['IGS'] = pd.to_numeric(all_stats_df_base_display_top5_calc['IGS'], errors='coerce').fillna(0)
            if 'AcceleRATE' not in all_stats_df_base_display_top5_calc.columns:
                 all_stats_df_base_display_top5_calc['AcceleRATE'] = all_stats_df_base_display_top5_calc.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI', 0), r.get('STR', 0), r.get('Acc', 0)), axis=1)
            top_5_igs_calc_df = all_stats_df_base_display_top5_calc.sort_values(by='IGS', ascending=False).head(5)
            st.dataframe(top_5_igs_calc_df[['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']])
            num_top_cols_stable_v33 = min(len(top_5_igs_calc_df), 5) 
            cols_top5_buttons_stable_v33 = st.columns(num_top_cols_stable_v33) if num_top_cols_stable_v33 > 0 else [st] 
            for i, row_idx in enumerate(top_5_igs_calc_df.index):
                row = top_5_igs_calc_df.loc[row_idx]
                with cols_top5_buttons_stable_v33[i % num_top_cols_stable_v33]:
                    if st.button(f"🛠️ Editar {row['Posicion']} {row['Altura']}cm {row['Peso']}kg", key=f"send_to_bc_top5_{row_idx}_v33"): 
                        st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = row['Posicion'], row['Altura'], row['Peso']
                        st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                        st.session_state.unlocked_facility_levels, st.session_state.club_budget_remaining = set(), st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                        st.success(f"Perfil Top {i+1} enviado. Ve a '🛠️ Build Craft'.")
        else: st.warning("Datos para el Top 5 (base) no disponibles.")
    else: st.info("Define parámetros de al menos un jugador para ver stats base.")

# --- Pestaña: Build Craft ---
with tab_build_craft:
    if not carga_completa_exitosa: 
        st.error("Faltan datos para el Build Craft.")
    else:
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
                    
                    if st.session_state.bc_unlocked_nodes:
                        # Paso A: Obtener el mapeo de tiers
                        mapa_de_tiers = calcular_tiers_nodos(df_skill_trees_global)
                        
                        # Paso B: Agrupar los nodos desbloqueados
                        nodos_agrupados = defaultdict(lambda: defaultdict(list))
                        playstyles_from_skills_sum = []
                        
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
                            summary_text += f"  - {ps_name_val_disp}\n"
                    
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
                build_data = {
                    "app_version": f"{APP_VERSION}_build_data", 
                    "build_name": st.session_state.get("build_save_name_v33", "Build Sin Nombre") or "Build Sin Nombre", # Asegurar que siempre haya un valor
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

            build_name_for_file = st.session_state.get("build_save_name_v33", "").strip()
            file_name_to_save = f"{build_name_for_file.replace(' ', '_')}_build_fc25.json" if build_name_for_file else "mi_build_fc25.json"
            
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
            ["Estilo EA FC25", "Lista de nodos", "Grafo interactivo"],
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
        elif tipo_visualizacion_bc == "Grafo interactivo":
            mostrar_visualizacion_arbol(df_skill_trees_global, arbol_sel_bc, st.session_state.bc_unlocked_nodes)
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
                        st.caption(f"Costo: {level_data_fac['Precio']:,} | ID: {level_id_fac_item}")
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
                    st.success("Perfil de Mejor Combinación Base enviado. Ve a '🛠️ Build Craft'.")
                
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
                
                # Aplicar filtro según la condición
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
                df_filtered_results_opt['AcceleRATE'] = df_filtered_results_opt.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI',0),r.get('STR',0),r.get('Acc',0)), axis=1)

            # Ordenar por los atributos seleccionados si existen
            if 'sort_by_attributes_optimal_opt' in locals() and sort_by_attributes_optimal_opt:
                df_filtered_results_opt = df_filtered_results_opt.sort_values(by=sort_by_attributes_optimal_opt, ascending=[False]*len(sort_by_attributes_optimal_opt))
            
            if not df_filtered_results_opt.empty:
                st.write(f"Combinaciones encontradas ({len(df_filtered_results_opt)}):")
                st.dataframe(df_filtered_results_opt[['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']])
            else:
                st.info("No se encontraron combinaciones que cumplan con los criterios.")

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