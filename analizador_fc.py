import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import math
import json 
from visualizacion_fc25_style import mostrar_visualizacion_fc25  # NUEVO: importar visualización FC25


# Definir emoji de estrella AL INICIO
STAR_EMOJI = "⭐" 

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Calculadora Stats FC25 v3.3", layout="wide") 

# --- Definiciones Constantes ---
APP_VERSION = "v3.3" 
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
            if col in df_skill_trees.columns: df_skill_trees[col] = pd.to_numeric(df_skill_trees[col], errors='coerce').fillna(0).astype(int)
            else: df_skill_trees[col] = 0 
        
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
            if col in df_instalaciones.columns: df_instalaciones[col] = pd.to_numeric(df_instalaciones[col], errors='coerce').fillna(0).astype(int)
            else: df_instalaciones[col] = 0 
        if 'Precio' in df_instalaciones.columns:
            df_instalaciones['Precio'] = pd.to_numeric(df_instalaciones['Precio'], errors='coerce').fillna(0).astype(int)
        else: st.error("Columna 'Precio' no encontrada en CSV de Instalaciones."); return None 
        
        text_cols_facilities = ['ID_Instalacion', 'Instalacion', 'Nombre_Instalacion', 'Prerrequisito', 'PlayStyle', 'EsPlus']
        for col_txt in text_cols_facilities:
            if col_txt in df_instalaciones.columns:
                df_instalaciones[col_txt] = df_instalaciones[col_txt].fillna('').astype(str)
            else:
                if col_txt not in ['PlayStyle', 'EsPlus']: st.warning(f"Columna '{col_txt}' no encontrada en {instalaciones_file}, se creará vacía.")
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
    all_data = []; 
    if not all(isinstance(df, (pd.DataFrame, pd.Series)) for df in [_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df]): return pd.DataFrame()
    if not (_diff_pos_df.index.tolist() and _mod_alt_df.index.tolist() and _mod_peso_df.index.tolist()): return pd.DataFrame()
    for pos in _diff_pos_df.index:
        for alt_v in _mod_alt_df.index:
            for pes_v in _mod_peso_df.index:
                stats = calcular_stats_base_jugador(pos, alt_v, pes_v, _base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df)
                if stats is not None: entry = {'Posicion': pos, 'Altura': alt_v, 'Peso': pes_v, **stats.to_dict()}; all_data.append(entry)
    df = pd.DataFrame(all_data)
    if not df.empty and _stat_cols_order:
        valid_cols = [col for col in _stat_cols_order if col in df.columns]; cols_df_order = ['Posicion', 'Altura', 'Peso'] + valid_cols 
        cols_df_order_existing = [col for col in cols_df_order if col in df.columns]; df = df[cols_df_order_existing]
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
                if stat_col == 'PIERNA_MALA': current_wf += boost_val
                elif stat_col == 'FILIGRANAS': current_sm += boost_val
                elif stat_col in stats_modificadas.index and pd.api.types.is_numeric_dtype(stats_modificadas[stat_col]):
                    stats_modificadas[stat_col] += boost_val
                    if stat_col in IGS_SUB_STATS: total_igs_boost_from_skills += boost_val
    
    for stat_name in IGS_SUB_STATS: 
        if stat_name in stats_modificadas.index and pd.api.types.is_numeric_dtype(stats_modificadas[stat_name]):
            stats_modificadas[stat_name] = min(stats_modificadas[stat_name], MAX_STAT_VAL)

    stats_modificadas['PIERNA_MALA'] = min(current_wf, MAX_STARS)
    stats_modificadas['FILIGRANAS'] = min(current_sm, MAX_STARS)
    
    stats_finales_con_todo = stats_modificadas.copy()
    total_igs_boost_from_facilities = 0

    if apply_facilities_boost and df_facilities is not None and 'unlocked_facility_levels' in st.session_state :
        for facility_id in st.session_state.unlocked_facility_levels: 
            if facility_id not in df_facilities['ID_Instalacion'].values: continue
            facility_data = df_facilities[df_facilities['ID_Instalacion'] == facility_id].iloc[0]
            for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS_FACILITIES:
                if stat_col in facility_data.index and pd.notna(facility_data[stat_col]) and facility_data[stat_col] != 0:
                    boost_val = int(facility_data[stat_col])
                    if stat_col in stats_finales_con_todo.index and pd.api.types.is_numeric_dtype(stats_finales_con_todo[stat_col]):
                        stats_finales_con_todo[stat_col] += boost_val
                        if stat_col in IGS_SUB_STATS: total_igs_boost_from_facilities += boost_val
        
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
                 all_stats_df_base_display_top5_calc['AcceleRATE'] = all_stats_df_base_display_top5_calc.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI',0),r.get('STR',0),r.get('Acc',0)), axis=1)
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
            if st.button("Generar Resumen de Texto", key="btn_generate_summary_v33_bc"):  
                summary_text = f"**Resumen del Build: {st.session_state.bc_pos} | {st.session_state.bc_alt}cm | {st.session_state.bc_pes}kg**\n"
                summary_text += f"Boosts de Instalaciones Aplicados: {'Sí' if st.session_state.apply_facility_boosts_toggle else 'No'}\n\n"
                summary_text += f"- **Puntos de Habilidad:** {TOTAL_SKILL_POINTS - st.session_state.bc_points_remaining} / {TOTAL_SKILL_POINTS} (Restantes: {st.session_state.bc_points_remaining})\n"
                summary_text += f"- **Pierna Mala:** {STAR_EMOJI * int(float(str(stats_completas_bc.get('PIERNA_MALA', BASE_WF))))}\n"
                summary_text += f"- **Filigranas:** {STAR_EMOJI * int(float(str(stats_completas_bc.get('FILIGRANAS', BASE_SM))))}\n"
                summary_text += f"- **AcceleRATE:** {stats_completas_bc.get('AcceleRATE', 'N/A')}\n"
                summary_text += f"- **IGS (Total):** {int(float(str(stats_completas_bc.get('IGS', '0'))))}\n\n"
                summary_text += "**Atributos Generales (Total):**\n"
                for cat in MAIN_CATEGORIES: summary_text += f"  - {cat}: {int(float(str(stats_completas_bc.get(cat, '0'))))}\n"
                
                summary_text += "\n**Nodos de Habilidad Desbloqueados:**\n"
                unlocked_nodes_details_sum = defaultdict(list)
                playstyles_from_skills_sum = []
                sorted_unlocked_nodes_sum = sorted(list(st.session_state.bc_unlocked_nodes), 
                                                   key=lambda x: (df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == x].iloc[0]['Arbol'] 
                                                                  if x in df_skill_trees_global['ID_Nodo'].values else "", x))
                for node_id_sum in sorted_unlocked_nodes_sum:
                    if node_id_sum in df_skill_trees_global['ID_Nodo'].values:
                        node_info_sum = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id_sum].iloc[0]
                        unlocked_nodes_details_sum[node_info_sum['Arbol']].append(f"{node_info_sum['Nombre_Visible']} (Costo: {node_info_sum['Costo']})")
                        playstyle_node_val_sum = node_info_sum.get('PlayStyle', '')
                        if pd.notna(playstyle_node_val_sum) and playstyle_node_val_sum != '': playstyles_from_skills_sum.append(playstyle_node_val_sum)
                
                for arbol_name_sum, nodos_list_sum in unlocked_nodes_details_sum.items():
                    summary_text += f"  - **{arbol_name_sum}:**\n"
                    for nodo_str_sum in nodos_list_sum: summary_text += f"    - {nodo_str_sum}\n"
                
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
                    for ps_name_sum_val_disp in total_playstyles_sum: summary_text += f"  - {ps_name_sum_val_disp}\n"
                
                st.text_area("Copia este resumen:", summary_text, height=450, key="build_summary_text_area_v33")
        
        # --- INICIO: Gestión de Builds (Guardar/Cargar) ---
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
                        node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                        if node_data['Arbol'] == arbol_sel_bc:
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
        
        if st.button("Resetear Puntos de Habilidad", key="reset_skills_btn_bc_v33"): 
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
            
            for _, nodo_disp in nodos_a_mostrar_display.iterrows(): 
                prereq_val_nodo_disp = nodo_disp.get('Prerrequisito', '') 
                if pd.isna(prereq_val_nodo_disp) or str(prereq_val_nodo_disp).strip() == '': 
                    nodos_por_tier_display[0].append(nodo_disp)
                    processed_nodes_display.add(nodo_disp['ID_Nodo'])
            
            current_tier_display = 0 
            while len(processed_nodes_display) < len(nodos_a_mostrar_display):
                newly_added_to_tier_disp = 0 
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
                        newly_added_to_tier_disp += 1

                if newly_added_to_tier_disp == 0 and len(processed_nodes_display) < len(nodos_a_mostrar_display):
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
                                prereq_display_node = f"Prerreq: {prereq_display_val_node if str(prereq_display_val_node).strip() else 'Ninguno'}" 
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
                                    st.rerun()
                                else:
                                    nombres_dependencias_fac_item = ", ".join(dependencias_fac_item) 
                                    st.warning(f"No se puede vender. Niveles dependientes: {nombres_dependencias_fac_item}. Véndelos primero.")
                        elif can_unlock_level_fac_item and st.session_state.club_budget_remaining >= level_data_fac['Precio']:
                            if st.button("🛒 Comprar", key=f"buy_facility_level_{level_id_fac_item}_v33u"): 
                                st.session_state.unlocked_facility_levels.add(level_id_fac_item)
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
                        es_plus_str_explorer_exp = "+" if str(r_fac_ps_exp.get('EsPlus','')).strip().lower() == 'si' else "" 
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
                
                styled_metric("Estilo de Carrera (Base)", accel_rate_best_optimal_opt); st.divider()
                
                key_metrics_to_display_optimal_opt = {} 
                for attr_disp_opt_loop in sort_by_attributes_optimal_opt + ['IGS']:  
                    if attr_disp_opt_loop in best_player_combination_opt and pd.notna(best_player_combination_opt[attr_disp_opt_loop]):
                        key_metrics_to_display_optimal_opt[attr_disp_opt_loop] = int(float(str(best_player_combination_opt[attr_disp_opt_loop])))
                
                num_metrics_cols_optimal_opt = min(len(key_metrics_to_display_optimal_opt), 4)  
                cols_metrics_cards_optimal_opt = st.columns(num_metrics_cols_optimal_opt) if num_metrics_cols_optimal_opt > 0 else [st] 
                
                metric_idx_opt_loop = 0 
                for label_opt_loop, value_opt_loop in key_metrics_to_display_optimal_opt.items(): 
                    with cols_metrics_cards_optimal_opt[metric_idx_opt_loop % num_metrics_cols_optimal_opt]:
                        styled_metric(label_opt_loop, value_opt_loop)
                    metric_idx_opt_loop +=1
                        
                with st.expander("Ver todos los atributos base de esta combinación"):
                    display_series_optimal_opt = best_player_combination_opt.drop(['Posicion', 'Altura', 'Peso'], errors='ignore').astype(str) 
                    st.dataframe(display_series_optimal_opt)
            else: 
                st.info("No se encontraron combinaciones que coincidan.")

# --- Pestaña: Filtros Múltiples ---
with tab_filters:
    if not carga_completa_exitosa or all_stats_df_base.empty:
        st.error("Datos para Filtros Múltiples no disponibles.")
    else:
        st.header("📊 Filtros Múltiples Avanzados (Stats Base)")
        
        queryable_stats_for_filter_fil = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']] 

        def add_filter_cb_v33_fil():  
            filter_id_fil = st.session_state.next_filter_id 
            st.session_state.filters.append({'id': filter_id_fil, 'attribute': queryable_stats_for_filter_fil[0], 'condition': '>=', 'value': 70})
            st.session_state.next_filter_id += 1

        def remove_filter_cb_v33_fil(filter_id_to_remove_fil):  
            st.session_state.filters = [f_item_fil for f_item_fil in st.session_state.filters if f_item_fil['id'] != filter_id_to_remove_fil] 

        st.button("➕ Añadir Criterio de Filtro", on_click=add_filter_cb_v33_fil, key="add_filter_btn_v33_fil") 
        
        filter_container_fil = st.container()  
        with filter_container_fil: 
            for filter_item_loop_fil in st.session_state.filters: 
                f_id_fil = filter_item_loop_fil['id'] 
                cols_filter_row_fil = st.columns([5,3,2,1])  
                
                try: current_attr_idx_filter_fil = queryable_stats_for_filter_fil.index(filter_item_loop_fil['attribute']) 
                except ValueError: current_attr_idx_filter_fil = 0 
                filter_item_loop_fil['attribute'] = cols_filter_row_fil[0].selectbox("Atributo:", options=queryable_stats_for_filter_fil, index=current_attr_idx_filter_fil, key=f"filter_attr_{f_id_fil}_v33_fil") 
                
                condition_options_filter_fil = ['>=', '<=', '==', '>', '<', '!='] 
                try: current_cond_idx_filter_fil = condition_options_filter_fil.index(filter_item_loop_fil['condition']) 
                except ValueError: current_cond_idx_filter_fil = 0
                filter_item_loop_fil['condition'] = cols_filter_row_fil[1].selectbox("Condición:", options=condition_options_filter_fil, index=current_cond_idx_filter_fil, key=f"filter_cond_{f_id_fil}_v33_fil") 
                
                filter_item_loop_fil['value'] = cols_filter_row_fil[2].number_input("Valor:", value=int(filter_item_loop_fil['value']), step=1, key=f"filter_val_{f_id_fil}_v33_fil") 
                
                if cols_filter_row_fil[3].button("➖", key=f"filter_remove_{f_id_fil}_v33_fil", help="Eliminar este criterio"): 
                    remove_filter_cb_v33_fil(f_id_fil)
                    st.rerun() 
        
        df_filtered_results_fil = pd.DataFrame()  
        if 'apply_filters_v33_fil_clicked' not in st.session_state: 
            st.session_state.apply_filters_v33_fil_clicked = False

        if st.button("Aplicar Filtros (sobre Stats Base)", key="btn_apply_filters_v33_fil_apply"):  
            st.session_state.apply_filters_v33_fil_clicked = True
            if not st.session_state.filters: 
                st.info("No hay criterios de filtro definidos. Mostrando todos los perfiles base.")
                df_filtered_results_fil = all_stats_df_base.copy()
                st.session_state.df_filtered_results_cache = df_filtered_results_fil 
            else:
                df_to_filter_fil = all_stats_df_base.copy() 
                
                if 'AcceleRATE' not in df_to_filter_fil.columns:
                    df_to_filter_fil['AcceleRATE'] = df_to_filter_fil.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI',0),r.get('STR',0),r.get('Acc',0)), axis=1)

                df_filtered_results_fil = df_to_filter_fil 
                query_is_valid_fil = True 
                active_filter_attributes_fil = []  
                
                for f_config_loop_fil in st.session_state.filters: 
                    attr_fil, cond_fil, val_fil = f_config_loop_fil['attribute'], f_config_loop_fil['condition'], f_config_loop_fil['value'] 
                    active_filter_attributes_fil.append(attr_fil)
                    if attr_fil not in df_filtered_results_fil.columns: 
                        st.error(f"Atributo de filtro '{attr_fil}' no encontrado en los datos."); query_is_valid_fil = False; break
                    
                    df_filtered_results_fil[attr_fil] = pd.to_numeric(df_filtered_results_fil[attr_fil], errors='coerce').fillna(0)

                    if cond_fil == '>=': df_filtered_results_fil = df_filtered_results_fil[df_filtered_results_fil[attr_fil] >= val_fil]
                    elif cond_fil == '<=': df_filtered_results_fil = df_filtered_results_fil[df_filtered_results_fil[attr_fil] <= val_fil]
                    elif cond_fil == '==': df_filtered_results_fil = df_filtered_results_fil[df_filtered_results_fil[attr_fil] == val_fil]
                    elif cond_fil == '>': df_filtered_results_fil = df_filtered_results_fil[df_filtered_results_fil[attr_fil] > val_fil]
                    elif cond_fil == '<': df_filtered_results_fil = df_filtered_results_fil[df_filtered_results_fil[attr_fil] < val_fil]
                    elif cond_fil == '!=': df_filtered_results_fil = df_filtered_results_fil[df_filtered_results_fil[attr_fil] != val_fil]
                
                if query_is_valid_fil:
                    if not df_filtered_results_fil.empty:
                        st.write(f"Jugadores que cumplen los criterios ({len(df_filtered_results_fil)}):")
                        cols_to_display_filtered_fil = ['Posicion', 'Altura', 'Peso']  
                        cols_to_display_filtered_fil.extend(sorted(list(set(active_filter_attributes_fil)))) 
                        cols_to_display_filtered_fil.extend(['AcceleRATE', 'IGS'])
                        
                        final_cols_to_display_filtered_fil = [col for col in cols_to_display_filtered_fil if col in df_filtered_results_fil.columns] 
                        
                        st.dataframe(df_filtered_results_fil[final_cols_to_display_filtered_fil])
                        st.session_state.df_filtered_results_cache = df_filtered_results_fil 
                    else: 
                        st.info("Ninguna combinación de stats base cumple con todos los criterios definidos.")
                        st.session_state.df_filtered_results_cache = pd.DataFrame() 
            
        if st.session_state.apply_filters_v33_fil_clicked and 'df_filtered_results_cache' in st.session_state : 
            cached_results_fil = st.session_state.df_filtered_results_cache 
            if not cached_results_fil.empty:
                st.markdown("---")
                st.subheader("Enviar Perfil Filtrado a Build Craft")
                if len(cached_results_fil) == 1: 
                    selected_profile_index_filter_fil = cached_results_fil.index[0] 
                    st.caption(f"Enviar: {cached_results_fil.loc[selected_profile_index_filter_fil, 'Posicion']} {cached_results_fil.loc[selected_profile_index_filter_fil, 'Altura']}cm {cached_results_fil.loc[selected_profile_index_filter_fil, 'Peso']}kg")
                    if st.button("Enviar este perfil a Build Craft", key="send_single_filtered_to_bc_v33_fil_send"): 
                        profile_to_send_fil = cached_results_fil.loc[selected_profile_index_filter_fil] 
                        st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = profile_to_send_fil['Posicion'], profile_to_send_fil['Altura'], profile_to_send_fil['Peso']
                        st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                        st.session_state.unlocked_facility_levels, st.session_state.club_budget_remaining = set(), st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                        st.success("Perfil enviado a Build Craft.")

                elif len(cached_results_fil) > 1 : 
                    options_for_bc_send_fil = [f"{idx}: {row['Posicion']} {row['Altura']}cm {row['Peso']}kg (IGS: {row.get('IGS',0)})" for idx, row in cached_results_fil.iterrows()] 
                    selected_option_bc_send_fil = st.selectbox("Selecciona un perfil de la lista filtrada para enviar a Build Craft:", options=options_for_bc_send_fil, key="select_filtered_for_bc_v33_fil_select") 
                    if selected_option_bc_send_fil and st.button("Enviar perfil seleccionado a Build Craft", key="send_selected_filtered_to_bc_v33_fil_send"): 
                        selected_idx_from_option_fil = int(selected_option_bc_send_fil.split(":")[0]) 
                        profile_to_send_fil_selected = cached_results_fil.loc[selected_idx_from_option_fil] 
                        st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = profile_to_send_fil_selected['Posicion'], profile_to_send_fil_selected['Altura'], profile_to_send_fil_selected['Peso']
                        st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                        st.session_state.unlocked_facility_levels, st.session_state.club_budget_remaining = set(), st.session_state.get('club_budget_total', DEFAULT_CLUB_BUDGET)
                        st.success(f"Perfil '{selected_option_bc_send_fil}' enviado a Build Craft.")
            elif st.session_state.apply_filters_v33_fil_clicked: 
                 st.info("No hay resultados filtrados para enviar.")