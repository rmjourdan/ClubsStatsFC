import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import math

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Calculadora Stats FC25 v3.0", layout="wide")

# --- Definiciones Constantes ---
BASE_WF, BASE_SM, MAX_STARS, MAX_STAT_VAL = 2, 3, 5, 99
TOTAL_SKILL_POINTS = 184
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
ALL_POSSIBLE_STAT_BOOST_COLS = IGS_SUB_STATS + ['PIERNA_MALA', 'FILIGRANAS']

# --- Lógica de AcceleRATE ---
def determinar_estilo_carrera(altura, agilidad, fuerza, aceleracion):
    # ... (igual que en v2.9) ...
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
    # ... (igual que en v2.9) ...
    file_altura, file_peso, file_posiciones = "Datos FC25 - ALTURA.csv", "Datos FC25 - PESO.csv", "Datos FC25 - POSICIONES.csv"
    try:
        df_altura_raw, df_peso_raw, df_posiciones_raw = pd.read_csv(file_altura), pd.read_csv(file_peso), pd.read_csv(file_posiciones)
    except FileNotFoundError as e: st.error(f"Error al cargar CSV base: {e}."); return None, None, None, None, None, None
    except Exception as e_gen: st.error(f"Error inesperado al cargar CSV base: {e_gen}"); return None, None, None, None, None, None
    try:
        df_altura, df_peso, df_posiciones = df_altura_raw.set_index('Altura'), df_peso_raw.set_index('Peso'), df_posiciones_raw.set_index('Posicion')
        expected_cols = df_posiciones.columns.tolist()
        df_altura = df_altura.reindex(columns=expected_cols).fillna(0).astype(int)
        df_peso = df_peso.reindex(columns=expected_cols).fillna(0).astype(int)
        df_posiciones = df_posiciones.reindex(columns=expected_cols).fillna(0).astype(int)
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
        return stats_base_lb_rb, mod_alt, mod_peso, diff_pos, df_posiciones.index.tolist(), expected_cols
    except KeyError as ke: st.error(f"Error de clave (KeyError) en datos base: '{ke}'. Revisa nombres de columna en CSVs base."); return None, None, None, None, None, None
    except Exception as e_proc: st.error(f"Error al procesar DataFrames base: {e_proc}"); return None, None, None, None, None, None

# --- Carga y Preparación del Árbol de Habilidades ---
@st.cache_data
def cargar_arbol_habilidades():
    # ... (igual que en v2.9) ...
    skill_tree_file = "ARBOL DE HABILIDAD - ARBOL.csv"
    try:
        df_skill_trees = pd.read_csv(skill_tree_file)
        for col in ALL_POSSIBLE_STAT_BOOST_COLS: 
            if col in df_skill_trees.columns: df_skill_trees[col] = pd.to_numeric(df_skill_trees[col], errors='coerce').fillna(0).astype(int)
            else: df_skill_trees[col] = 0 
        if 'Costo' in df_skill_trees.columns: df_skill_trees['Costo'] = pd.to_numeric(df_skill_trees['Costo'], errors='coerce').fillna(0).astype(int)
        if 'Prerrequisito' in df_skill_trees.columns: df_skill_trees['Prerrequisito'] = df_skill_trees['Prerrequisito'].fillna('').astype(str)
        for col_extra in ['ID_Nodo', 'Arbol', 'Nombre_Visible', 'PlayStyle', 'Es_Arquetipo', 'Notas', 'Puntos_Req_Arbol']:
            if col_extra in df_skill_trees.columns: df_skill_trees[col_extra] = df_skill_trees[col_extra].astype(str).fillna('')
            else: df_skill_trees[col_extra] = '' # Añadir columna si no existe para evitar KeyErrors más adelante
        return df_skill_trees
    except FileNotFoundError: st.error(f"ERROR: No se encontró '{skill_tree_file}'."); return None
    except Exception as e: st.error(f"Error al cargar CSV del árbol de hab: {e}"); return None

# --- Función de Cálculo de Estadísticas Base (sin árbol) ---
def calcular_stats_base_jugador(pos_sel, alt_sel, peso_sel, base_ref_stats, mod_alt_df, mod_peso_df, diff_pos_df):
    # ... (igual que en v2.9) ...
    diff = diff_pos_df.loc[pos_sel]; mod_a = mod_alt_df.loc[alt_sel]; mod_p = mod_peso_df.loc[peso_sel]
    final_base_stats = base_ref_stats.add(diff).add(mod_a).add(mod_p)
    return final_base_stats.round().astype(int)

# --- Pre-cálculo de todas las combinaciones base ---
@st.cache_data
def precompute_all_base_stats(_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df, _stat_cols_order):
    # ... (igual que en v2.9) ...
    all_data = []; 
    if not all(isinstance(df, (pd.DataFrame, pd.Series)) for df in [_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df]): return pd.DataFrame()
    for pos in _diff_pos_df.index:
        for alt_v in _mod_alt_df.index:
            for pes_v in _mod_peso_df.index:
                stats = calcular_stats_base_jugador(pos, alt_v, pes_v, _base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df)
                entry = {'Posicion': pos, 'Altura': alt_v, 'Peso': pes_v, **stats.to_dict()}
                all_data.append(entry)
    df = pd.DataFrame(all_data)
    if not df.empty:
        valid_cols = [col for col in _stat_cols_order if col in df.columns]; cols_df_order = ['Posicion', 'Altura', 'Peso'] + valid_cols 
        cols_df_order_existing = [col for col in cols_df_order if col in df.columns]; df = df[cols_df_order_existing]
    return df

# --- Funciones Auxiliares para el Editor de Habilidades ---
def aplicar_mejoras_habilidad(stats_jugador_base, df_skill_tree_data, unlocked_nodes_ids, altura_jugador_actual):
    # ... (igual que en v2.9 - con recálculo de IGS, Main Cats, y tope de 99) ...
    stats_modificadas = stats_jugador_base.copy().astype(float) 
    current_wf, current_sm = float(BASE_WF), float(BASE_SM)
    total_igs_boost_from_skills = 0 
    for node_id in unlocked_nodes_ids:
        if node_id not in df_skill_tree_data['ID_Nodo'].values: continue
        node_data = df_skill_tree_data[df_skill_tree_data['ID_Nodo'] == node_id].iloc[0]
        for stat_col in ALL_POSSIBLE_STAT_BOOST_COLS:
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
    for main_cat, sub_stats_list in SUB_STATS_MAPPING.items():
        if main_cat in stats_modificadas.index: 
            relevant_sub_stats_values = [stats_modificadas.get(sub_stat, 0) for sub_stat in sub_stats_list if sub_stat in stats_modificadas.index and pd.api.types.is_numeric_dtype(stats_modificadas.get(sub_stat))]
            if relevant_sub_stats_values: stats_modificadas[main_cat] = min(math.ceil(sum(relevant_sub_stats_values) / len(relevant_sub_stats_values)), MAX_STAT_VAL)
    stats_modificadas['AcceleRATE'] = determinar_estilo_carrera(altura_jugador_actual, stats_modificadas.get('AGI', 0), stats_modificadas.get('STR', 0), stats_modificadas.get('Acc', 0))
    base_igs_value = float(stats_jugador_base.get('IGS', 0)); stats_modificadas['IGS'] = base_igs_value + total_igs_boost_from_skills
    stats_final_dict = {}
    for stat_name, value in stats_modificadas.items():
        if pd.api.types.is_numeric_dtype(value) and stat_name not in ['PIERNA_MALA', 'FILIGRANAS']: stats_final_dict[stat_name] = int(round(float(value)))
        elif stat_name in ['PIERNA_MALA', 'FILIGRANAS']: stats_final_dict[stat_name] = int(value)
        else: stats_final_dict[stat_name] = value
    return pd.Series(stats_final_dict, name=stats_modificadas.name if hasattr(stats_modificadas, 'name') else None)

def check_prerequisites(node_id, df_skill_tree_data, unlocked_nodes_ids):
    # ... (igual que en v2.8) ...
    if node_id not in df_skill_tree_data['ID_Nodo'].values: return False
    node_info = df_skill_tree_data[df_skill_tree_data['ID_Nodo'] == node_id].iloc[0]
    prereqs_str = node_info.get('Prerrequisito', '')
    if pd.isna(prereqs_str) or prereqs_str == '': return True
    prereq_ids = prereqs_str.split(',')
    return any(pr_id.strip() in unlocked_nodes_ids for pr_id in prereq_ids)

# --- Carga de Datos Principal ---
APP_VERSION = "v3.0" 
if 'app_version' not in st.session_state or st.session_state.app_version != APP_VERSION:
    st.session_state.clear(); st.session_state.app_version = APP_VERSION
# ... (resto de lógica de carga y session_state igual que v2.8) ...
carga_base_exitosa = False
datos_base_cargados = cargar_y_preparar_datos_base()
df_skill_trees_global = cargar_arbol_habilidades()
if datos_base_cargados and all(d is not None for d in datos_base_cargados) and df_skill_trees_global is not None and not df_skill_trees_global.empty:
    stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, lista_posiciones, stat_cols_order = datos_base_cargados
    all_stats_df_base = precompute_all_base_stats(stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, stat_cols_order)
    if not all_stats_df_base.empty: carga_base_exitosa = True
    else: st.error("No se pudieron pre-calcular todas las estadísticas base.")
else: st.error("Fallo en la carga de datos base o del árbol de habilidades.")
if carga_base_exitosa:
    for key in ['bc_unlocked_nodes', 'filters']: 
        if key not in st.session_state: st.session_state[key] = set() if key.endswith('nodes') else []
    if 'bc_points_total' not in st.session_state: st.session_state.bc_points_total = TOTAL_SKILL_POINTS
    if 'bc_points_remaining' not in st.session_state: st.session_state.bc_points_remaining = TOTAL_SKILL_POINTS
    if 'next_filter_id' not in st.session_state: st.session_state.next_filter_id = 0
    default_pos = lista_posiciones[lista_posiciones.index('ST') if 'ST' in lista_posiciones else 0]
    unique_alts_sb_init_v30 = sorted(modificadores_altura.index.unique().tolist())
    default_alt = unique_alts_sb_init_v30[unique_alts_sb_init_v30.index(180) if 180 in unique_alts_sb_init_v30 else 0]
    unique_pesos_sb_init_v30 = sorted(modificadores_peso.index.unique().tolist())
    default_pes = unique_pesos_sb_init_v30[unique_pesos_sb_init_v30.index(75) if 75 in unique_pesos_sb_init_v30 else 0]
    if 'bc_pos' not in st.session_state: st.session_state.bc_pos = default_pos
    if 'bc_alt' not in st.session_state: st.session_state.bc_alt = default_alt
    if 'bc_pes' not in st.session_state: st.session_state.bc_pes = default_pes
else: st.stop()

# --- Interfaz de Usuario con Streamlit ---
st.title(f"Calculadora Avanzada de Estadísticas FC25 ({APP_VERSION})")

PLAYER_COLORS = ['rgba(0,100,255,0.7)', 'rgba(220,50,50,0.7)', 'rgba(0,170,0,0.7)']
PLAYER_FILL_COLORS = [color.replace('0.7', '0.3') for color in PLAYER_COLORS]
def highlight_max_in_row(row):
    if row.dtype == 'object': return ['' for _ in row]
    if row.isnull().all(): return ['' for _ in row]
    try: max_val = row.max()
    except TypeError: return ['' for _ in row] # Evitar error con tipos mixtos no comparables
    return ['background-color: #d4edda; color: #155724; font-weight: bold;' if (pd.notna(v) and v == max_val) else '' for v in row]

def styled_metric(label, value):
    val_num = 0; apply_color_coding = label in MAIN_CATEGORIES
    if isinstance(value, (int, float)): val_num = value
    elif isinstance(value, str):
        try: val_num = float(value.replace('⭐','')) # Intentar quitar estrellas si es WF/SM
        except ValueError: pass
    bg_color, text_color = "inherit", "inherit" 
    if apply_color_coding and val_num > 0 :
        color_map = {(95, 100): ("#b4a7d6", "black"), (85, 94): ("#b6d7a8", "#003300"), (75, 84): ("#fce8b2", "#594400"), (1, 74): ("#f4c7c3", "#800000"),}
        for (lower, upper), (bg, txt) in color_map.items():
            if lower <= val_num <= upper: bg_color, text_color = bg, txt; break
    st.markdown(f"""<div style="background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; margin-bottom: 5px; text-align: center;">
        <div style="font-size: 0.8em; {"color: #555555;" if bg_color == "inherit" else ""}">{label}</div>
        <div style="font-size: 1.7em; font-weight: bold;">{value}</div></div>""", unsafe_allow_html=True)

st.sidebar.markdown("--- \n ### 🛠️ Perfil Base para Build Craft:")
# ... (selectores de sidebar igual que v2.8, usando keys _v30) ...
current_bc_pos_idx_sb_v30 = lista_posiciones.index(st.session_state.bc_pos) if st.session_state.bc_pos in lista_posiciones else 0
st.session_state.bc_pos = st.sidebar.selectbox("Posición Base (BC):", sorted(lista_posiciones), index=current_bc_pos_idx_sb_v30, key="sb_bc_pos_v30")
unique_alts_sb_ui_v30 = sorted(modificadores_altura.index.unique().tolist())
current_bc_alt_idx_sb_v30 = unique_alts_sb_ui_v30.index(st.session_state.bc_alt) if st.session_state.bc_alt in unique_alts_sb_ui_v30 else 0
st.session_state.bc_alt = st.sidebar.selectbox("Altura Base (cm) (BC):", unique_alts_sb_ui_v30, index=current_bc_alt_idx_sb_v30, key="sb_bc_alt_v30")
unique_pesos_sb_ui_v30 = sorted(modificadores_peso.index.unique().tolist())
current_bc_pes_idx_sb_v30 = unique_pesos_sb_ui_v30.index(st.session_state.bc_pes) if st.session_state.bc_pes in unique_pesos_sb_ui_v30 else 0
st.session_state.bc_pes = st.sidebar.selectbox("Peso Base (kg) (BC):", unique_pesos_sb_ui_v30, index=current_bc_pes_idx_sb_v30, key="sb_bc_pes_v30")


tab_calc, tab_build_craft, tab_best_combo, tab_filters = st.tabs([
    "🧮 Calculadora y Comparador", "🛠️ Build Craft", "🔍 Búsqueda Óptima", "📊 Filtros Múltiples"
])

# --- Pestaña: Calculadora y Comparador ---
with tab_calc:
    # ... (código igual que v2.8, con keys _v30) ...
    st.header("Calculadora de Estadísticas y Comparador (Stats Base)")
    num_players_to_compare = st.radio("Jugadores a definir/comparar:", (1, 2, 3), index=0, horizontal=True, key="num_players_radio_v30_calc")
    cols_selectors_calc = st.columns(num_players_to_compare)
    player_stats_list_base_calc, player_configs_base_calc = [], []
    for i in range(num_players_to_compare):
        with cols_selectors_calc[i]:
            player_label = f"JUG {chr(65+i)}"
            st.subheader(player_label)
            pos_key, alt_key, pes_key = f"pos_p{i}_v30_calc", f"alt_p{i}_v30_calc", f"pes_p{i}_v30_calc"
            if pos_key not in st.session_state: st.session_state[pos_key] = lista_posiciones[0]
            if alt_key not in st.session_state: st.session_state[alt_key] = unique_alts_sb_ui_v30[0]
            if pes_key not in st.session_state: st.session_state[pes_key] = unique_pesos_sb_ui_v30[0]
            pos_val_c,alt_val_c,pes_val_c = st.selectbox(f"Pos ({player_label}):", sorted(lista_posiciones), key=pos_key), st.selectbox(f"Alt ({player_label}):", unique_alts_sb_ui_v30, key=alt_key), st.selectbox(f"Pes ({player_label}):", unique_pesos_sb_ui_v30, key=pes_key)
            player_configs_base_calc.append({'pos': pos_val_c, 'alt': alt_val_c, 'pes': pes_val_c, 'label': player_label})
            if pos_val_c and alt_val_c is not None and pes_val_c is not None:
                stats = calcular_stats_base_jugador(pos_val_c, alt_val_c, pes_val_c, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
                stats['AcceleRATE'] = determinar_estilo_carrera(alt_val_c, stats.get('AGI',0), stats.get('STR',0), stats.get('Acc',0))
                player_stats_list_base_calc.append(stats)
                if st.button(f"🛠️ Enviar a Build Craft ({player_label})", key=f"send_to_bc_calc_{i}_v30", help=f"Carga este perfil base ({pos_val_c}, {alt_val_c}cm, {pes_val_c}kg) en la pestaña 'Build Craft' y resetea los puntos de habilidad."):
                    st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = pos_val_c, alt_val_c, pes_val_c
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.success(f"Perfil de {player_label} enviado a Build Craft. (Pos: {pos_val_c}, Alt: {alt_val_c}cm, Pes: {pes_val_c}kg).")
            else: player_stats_list_base_calc.append(None)
    st.divider(); st.subheader("Perfiles y Estadísticas (Base)")
    if any(ps is not None for ps in player_stats_list_base_calc):
        # ... (Lógica de visualización de radar y tabla comparativa como antes, pero usando los nombres de variable _v30)
        fig_radar_comp_calc_disp_obj_v30 = go.Figure(); radar_attrs_calc_disp_list_val_v30 = MAIN_CATEGORIES 
        cols_perfiles_calc_disp_list_val_v30 = st.columns(num_players_to_compare) 
        for i, stats_item_calc_disp_val_v30 in enumerate(player_stats_list_base_calc): 
            if stats_item_calc_disp_val_v30 is not None:
                with cols_perfiles_calc_disp_list_val_v30[i]:
                    st.markdown(f"**{player_configs_base_calc[i]['label']} ({player_configs_base_calc[i]['pos']}, {player_configs_base_calc[i]['alt']}cm, {player_configs_base_calc[i]['pes']}kg)**")
                    styled_metric("Estilo de Carrera", stats_item_calc_disp_val_v30.get('AcceleRATE', "N/A"))
                valid_radar_attrs_calc_disp_list_val_set_v30 = [attr for attr in radar_attrs_calc_disp_list_val_v30 if attr in stats_item_calc_disp_val_v30.index] 
                radar_values_calc_disp_list_val_set_v30 = [stats_item_calc_disp_val_v30.get(attr, 0) for attr in valid_radar_attrs_calc_disp_list_val_set_v30] 
                if len(valid_radar_attrs_calc_disp_list_val_set_v30) >= 3:
                    fig_radar_comp_calc_disp_obj_v30.add_trace(go.Scatterpolar(r=radar_values_calc_disp_list_val_set_v30, theta=valid_radar_attrs_calc_disp_list_val_set_v30, fill='toself', name=f"{player_configs_base_calc[i]['label']} ({player_configs_base_calc[i]['pos']})",line_color=PLAYER_COLORS[i % len(PLAYER_COLORS)],fillcolor=PLAYER_FILL_COLORS[i % len(PLAYER_FILL_COLORS)]))
        if fig_radar_comp_calc_disp_obj_v30.data:
            fig_radar_comp_calc_disp_obj_v30.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Comparativa de Perfiles Base")
            st.plotly_chart(fig_radar_comp_calc_disp_obj_v30, use_container_width=True)
        valid_player_stats_b_calc_display_list_final_v30 = [ps for ps in player_stats_list_base_calc if ps is not None] 
        valid_player_col_names_b_calc_disp_list_final_v30 = [player_configs_base_calc[i]['label'] for i, ps in enumerate(player_stats_list_base_calc) if ps is not None] 
        if len(valid_player_stats_b_calc_display_list_final_v30) > 1 :
            compare_dict_b_calc_display_dict_final_v30 = {name: stats.drop('AcceleRATE', errors='ignore') for name, stats in zip(valid_player_col_names_b_calc_disp_list_final_v30, valid_player_stats_b_calc_display_list_final_v30)} 
            df_compare_b_calc_display_df_final_v30 = pd.DataFrame(compare_dict_b_calc_display_dict_final_v30) 
            df_compare_styled_b_calc_display_df_final_v30 = df_compare_b_calc_display_df_final_v30.style.apply(highlight_max_in_row, axis=1) 
            st.subheader("Tabla Comparativa de Estadísticas Base (Numéricas)")
            st.dataframe(df_compare_styled_b_calc_display_df_final_v30)
            accele_rates_info_calc_val_final_v30 = {name: player_stats_list_base_calc[j].get('AcceleRATE', 'N/A') for j, name in enumerate(valid_player_col_names_b_calc_disp_list_final_v30)} 
            st.caption(f"Estilos de Carrera: {accele_rates_info_calc_val_final_v30}")
        elif len(valid_player_stats_b_calc_display_list_final_v30) == 1:
            st.subheader(f"Estadísticas Detalladas Base: {player_configs_base_calc[0]['label']}")
            stats_to_display_single_calc_final_v30 = valid_player_stats_b_calc_display_list_final_v30[0] 
            accele_rate_single_calc_final_v30 = stats_to_display_single_calc_final_v30.get('AcceleRATE', 'N/A')
            styled_metric("Estilo de Carrera", accele_rate_single_calc_final_v30)
            st.dataframe(stats_to_display_single_calc_final_v30.drop('AcceleRATE', errors='ignore').rename("Valor").astype(str))
        
        st.divider(); st.header("🏆 Top 5 Combinaciones por IGS (Stats Base)")
        if not all_stats_df_base.empty and 'IGS' in all_stats_df_base.columns:
            all_stats_df_base_display_top5_calc_final_v30 = all_stats_df_base.copy()
            all_stats_df_base_display_top5_calc_final_v30['IGS'] = pd.to_numeric(all_stats_df_base_display_top5_calc_final_v30['IGS'], errors='coerce').fillna(0)
            if 'AcceleRATE' not in all_stats_df_base_display_top5_calc_final_v30.columns:
                 all_stats_df_base_display_top5_calc_final_v30['AcceleRATE'] = all_stats_df_base_display_top5_calc_final_v30.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI',0),r.get('STR',0),r.get('Acc',0)), axis=1)
            top_5_igs_calc_df_final_v30 = all_stats_df_base_display_top5_calc_final_v30.sort_values(by='IGS', ascending=False).head(5)
            st.dataframe(top_5_igs_calc_df_final_v30[['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']])
            for i, row_idx in enumerate(top_5_igs_calc_df_final_v30.index):
                row = top_5_igs_calc_df_final_v30.loc[row_idx]
                if st.button(f"🛠️ Editar {row['Posicion']} {row['Altura']}cm {row['Peso']}kg (Top {i+1})", key=f"send_to_bc_top5_{row_idx}_v30"):
                    st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = row['Posicion'], row['Altura'], row['Peso']
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.success(f"Perfil Top {i+1} ({row['Posicion']}) enviado. Ajusta selectores en sidebar y ve a '🛠️ Build Craft'.")
        else: st.warning("Datos para el Top 5 (base) no disponibles.")
    else: st.info("Define parámetros de al menos un jugador para ver stats base.")

# --- Pestaña: Build Craft ---
with tab_build_craft:
    # ... (código igual que v2.8, con la nueva tabla de impacto de árbol y visualización de nodos mejorada, y actualizando keys) ...
    st.header(f"🛠️ Build Craft: {st.session_state.bc_pos} | {st.session_state.bc_alt}cm | {st.session_state.bc_pes}kg")
    jugador_base_actual_bc = calcular_stats_base_jugador(st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
    stats_con_skills_bc = aplicar_mejoras_habilidad(jugador_base_actual_bc, df_skill_trees_global, st.session_state.bc_unlocked_nodes, st.session_state.bc_alt)
    
    st.subheader("Perfil Actual del Build")
    col_bc_info_main_craft_disp_v30, col_bc_radar_display_craft_disp_v30 = st.columns([1,1]) 
    with col_bc_info_main_craft_disp_v30:
        styled_metric("Puntos Restantes", f"{st.session_state.bc_points_remaining} / {TOTAL_SKILL_POINTS}")
        wf_stars_val_bc_disp_v30 = int(stats_con_skills_bc.get('PIERNA_MALA', BASE_WF)) 
        sm_stars_val_bc_disp_v30 = int(stats_con_skills_bc.get('FILIGRANAS', BASE_SM))
        styled_metric("Pierna Mala (WF)", "⭐" * wf_stars_val_bc_disp_v30)
        styled_metric("Filigranas (SM)", "⭐" * sm_stars_val_bc_disp_v30)
        styled_metric("Estilo de Carrera", str(stats_con_skills_bc.get('AcceleRATE', "N/A")))
        styled_metric("IGS (Con Habilidades)", f"{int(stats_con_skills_bc.get('IGS', 0))}")
        st.markdown("---"); st.markdown("**Atributos Generales (Recalculados):**")
        cols_main_stats_bc_display_list_v30 = st.columns(2) 
        for idx, stat_name in enumerate(MAIN_CATEGORIES):
            with cols_main_stats_bc_display_list_v30[idx % 2]:
                styled_metric(stat_name, int(stats_con_skills_bc.get(stat_name, 0)))
    with col_bc_radar_display_craft_disp_v30:
        radar_attrs_bc_disp_list_val_v30 = MAIN_CATEGORIES 
        valid_radar_attrs_bc_disp_list_val_set_v30 = [attr for attr in radar_attrs_bc_disp_list_val_v30 if attr in stats_con_skills_bc.index] 
        radar_values_bc_disp_list_val_set_v30 = [stats_con_skills_bc.get(attr,0) for attr in valid_radar_attrs_bc_disp_list_val_set_v30] 
        if len(valid_radar_attrs_bc_disp_list_val_set_v30) >=3:
            fig_radar_bc_disp_obj_val_v30 = go.Figure() 
            fig_radar_bc_disp_obj_val_v30.add_trace(go.Scatterpolar(r=radar_values_bc_disp_list_val_set_v30, theta=valid_radar_attrs_bc_disp_list_val_set_v30, fill='toself', name="Con Habilidades", line_color=PLAYER_COLORS[0]))
            fig_radar_bc_disp_obj_val_v30.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Perfil Actual del Build")
            st.plotly_chart(fig_radar_bc_disp_obj_val_v30, use_container_width=True)

    with st.expander("Ver todas las estadísticas del Build Actual (con habilidades)", expanded=False):
        # Convertir Series a DataFrame de dos columnas para mejor visualización y evitar error Arrow
        df_display_build_stats = pd.DataFrame(stats_con_skills_bc.astype(str)).reset_index()
        df_display_build_stats.columns = ["Atributo", "Valor Build"]
        st.dataframe(df_display_build_stats)
    
    st.divider()
    # --- Sección para Resumen de Build para Copiar ---
    if st.button("📋 Generar Resumen del Build para Copiar", key="btn_generate_summary_v30"):
        summary_text = f"**Resumen del Build: {st.session_state.bc_pos} | {st.session_state.bc_alt}cm | {st.session_state.bc_pes}kg**\n\n"
        summary_text += f"- **Puntos de Habilidad:** {TOTAL_SKILL_POINTS - st.session_state.bc_points_remaining} / {TOTAL_SKILL_POINTS} usados (Restantes: {st.session_state.bc_points_remaining})\n"
        summary_text += f"- **Pierna Mala:** {int(stats_con_skills_bc.get('PIERNA_MALA', BASE_WF))} ⭐\n"
        summary_text += f"- **Filigranas:** {int(stats_con_skills_bc.get('FILIGRANAS', BASE_SM))} ⭐\n"
        summary_text += f"- **Estilo de Carrera (AcceleRATE):** {stats_con_skills_bc.get('AcceleRATE', 'N/A')}\n"
        summary_text += f"- **IGS (Con Habilidades):** {int(stats_con_skills_bc.get('IGS', 0))}\n\n"
        summary_text += "**Atributos Generales Finales:**\n"
        for cat in MAIN_CATEGORIES:
            summary_text += f"  - {cat}: {int(stats_con_skills_bc.get(cat, 0))}\n"
        
        summary_text += "\n**Nodos Desbloqueados:**\n"
        unlocked_nodes_details = defaultdict(list)
        playstyles_unlocked = []
        for node_id in sorted(list(st.session_state.bc_unlocked_nodes)): # Ordenar para consistencia
            if node_id in df_skill_trees_global['ID_Nodo'].values:
                node_info = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                unlocked_nodes_details[node_info['Arbol']].append(f"{node_info['Nombre_Visible']} (Costo: {node_info['Costo']})")
                if 'PlayStyle' in node_info.index and pd.notna(node_info['PlayStyle']) and node_info['PlayStyle'] != '':
                    playstyles_unlocked.append(node_info['PlayStyle'])
        
        for arbol_name, nodos_list in unlocked_nodes_details.items():
            summary_text += f"  - **{arbol_name}:**\n"
            for nodo_str in nodos_list:
                summary_text += f"    - {nodo_str}\n"
        
        if playstyles_unlocked:
            summary_text += "\n**PlayStyles Desbloqueados:**\n"
            for ps_name in sorted(list(set(playstyles_unlocked))): # Únicos y ordenados
                summary_text += f"  - {ps_name}\n"
        
        st.text_area("Copia este resumen:", summary_text, height=400, key="build_summary_text_area")

    st.subheader("Árboles de Habilidad para Personalizar")
    # ... (resto de la pestaña Build Craft igual que v2.8, con la tabla de impacto mejorada y keys _v30) ...
    col_tree_selector_bc_edit_ui_v30, col_tree_summary_bc_edit_ui_v30 = st.columns([1,2]) 
    with col_tree_selector_bc_edit_ui_v30:
        arbol_sel_bc_edit_val_v30 = st.selectbox("Selecciona Árbol:", options=["Todos"] + sorted(df_skill_trees_global['Arbol'].unique()), key="skill_tree_select_bc_v30") 
        if st.button("Resetear Puntos de Habilidad", key="reset_skills_btn_bc_v30"):
            st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
            st.rerun()
    with col_tree_summary_bc_edit_ui_v30: 
        if arbol_sel_bc_edit_val_v30 != "Todos":
            st.markdown(f"**Impacto del Árbol '{arbol_sel_bc_edit_val_v30}' en Sub-Stats:**")
            sub_stats_del_arbol_sel_actual_v30 = [] 
            if arbol_sel_bc_edit_val_v30 in SUB_STATS_MAPPING:
                sub_stats_del_arbol_sel_actual_v30 = SUB_STATS_MAPPING[arbol_sel_bc_edit_val_v30]
            else:
                nodos_del_arbol_actual_df_v30 = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol_sel_bc_edit_val_v30]
                stats_potencialmente_mejoradas_v30 = []
                for s_col_v30 in IGS_SUB_STATS: 
                    if s_col_v30 in nodos_del_arbol_actual_df_v30.columns and nodos_del_arbol_actual_df_v30[s_col_v30].sum() > 0:
                        stats_potencialmente_mejoradas_v30.append(s_col_v30)
                sub_stats_del_arbol_sel_actual_v30 = sorted(list(set(stats_potencialmente_mejoradas_v30))) 
            summary_data_live_list_val_v30 = [] 
            if sub_stats_del_arbol_sel_actual_v30:
                boosts_solo_arbol_sel_dict_val_v30 = defaultdict(int) 
                nodos_desbloq_en_arbol_actual_list_val_v30 = [nid for nid in st.session_state.bc_unlocked_nodes if df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == nid].iloc[0]['Arbol'] == arbol_sel_bc_edit_val_v30] 
                for node_id in nodos_desbloq_en_arbol_actual_list_val_v30:
                    node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                    for col_stat_name in sub_stats_del_arbol_sel_actual_v30: 
                        if col_stat_name in node_data.index and node_data[col_stat_name] > 0:
                            boosts_solo_arbol_sel_dict_val_v30[col_stat_name] += node_data[col_stat_name]
                for stat_name in sub_stats_del_arbol_sel_actual_v30:
                    base_val = jugador_base_actual_bc.get(stat_name, 0)
                    total_boost_arbol = boosts_solo_arbol_sel_dict_val_v30.get(stat_name, 0)
                    parcial_val = min(base_val + total_boost_arbol, MAX_STAT_VAL)
                    boost_display_str = f"<span style='color:green; font-weight:bold;'>+{total_boost_arbol}</span>" if total_boost_arbol > 0 else "+0"
                    summary_data_live_list_val_v30.append({"Stat": stat_name, "Base": base_val, f"Boost ({arbol_sel_bc_edit_val_v30})": boost_display_str, "Resultado (con este árbol)": parcial_val}) # Columna "Total Build" eliminada
                if summary_data_live_list_val_v30:
                    df_summary_live_df_val_v30 = pd.DataFrame(summary_data_live_list_val_v30).set_index("Stat") 
                    st.markdown(df_summary_live_df_val_v30.to_html(escape=False), unsafe_allow_html=True)
                else: st.caption(f"No hay nodos desbloqueados o mejoras para las sub-stats de '{arbol_sel_bc_edit_val_v30}'.")
            else: st.caption(f"No se definieron sub-stats para el árbol '{arbol_sel_bc_edit_val_v30}' o no hay nodos que las mejoren.")
        else: st.caption("Selecciona un árbol para ver su impacto detallado.")
    
    if arbol_sel_bc_edit_val_v30:
        # ... (lógica de tiers y visualización de nodos como tarjeta igual que v2.8, actualizando keys _v30) ...
        # (Esta parte es bastante larga, asumo que la lógica de tiers y el display de cada nodo en tarjeta se mantiene, solo actualizando las keys de los widgets con _v30)
        # ... (Para brevedad, no repito toda la lógica de visualización de nodos aquí, pero debe estar presente y con keys actualizadas)
        nodos_a_mostrar_bc_disp_v30 = df_skill_trees_global 
        if arbol_sel_bc_edit_val_v30 != "Todos":
            nodos_a_mostrar_bc_disp_v30 = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol_sel_bc_edit_val_v30]
        st.markdown(f"**Nodos para '{arbol_sel_bc_edit_val_v30}':** ({len(nodos_a_mostrar_bc_disp_v30)} nodos)")
        nodos_por_tier_bc_disp_v30 = defaultdict(list); processed_nodes_bc_disp_v30 = set() 
        for _, nodo in nodos_a_mostrar_bc_disp_v30.iterrows(): 
            if not nodo['Prerrequisito'] or pd.isna(nodo['Prerrequisito']):
                nodos_por_tier_bc_disp_v30[0].append(nodo); processed_nodes_bc_disp_v30.add(nodo['ID_Nodo'])
        current_tier_bc_disp_v30 = 0 
        while len(processed_nodes_bc_disp_v30) < len(nodos_a_mostrar_bc_disp_v30) and current_tier_bc_disp_v30 < len(nodos_a_mostrar_bc_disp_v30) :
            newly_added_bc_disp_v30 = 0; next_tier_candidates_bc_disp_v30 = []
            for _, nodo in nodos_a_mostrar_bc_disp_v30.iterrows():
                if nodo['ID_Nodo'] in processed_nodes_bc_disp_v30: continue
                prereqs_list_disp_v30 = [pr_id.strip() for pr_id in nodo['Prerrequisito'].split(',') if pr_id.strip()]
                if prereqs_list_disp_v30 and all(pr_id in processed_nodes_bc_disp_v30 for pr_id in prereqs_list_disp_v30): 
                    next_tier_candidates_bc_disp_v30.append(nodo); newly_added_bc_disp_v30 +=1
            if not next_tier_candidates_bc_disp_v30 and newly_added_bc_disp_v30 == 0: 
                for _, nodo_restante in nodos_a_mostrar_bc_disp_v30.iterrows():
                    if nodo_restante['ID_Nodo'] not in processed_nodes_bc_disp_v30:
                        nodos_por_tier_bc_disp_v30[current_tier_bc_disp_v30 + 100].append(nodo_restante); processed_nodes_bc_disp_v30.add(nodo_restante['ID_Nodo'])
                break
            if newly_added_bc_disp_v30 > 0 :
                current_tier_bc_disp_v30 += 1
                for nodo in next_tier_candidates_bc_disp_v30:
                    if nodo['ID_Nodo'] not in processed_nodes_bc_disp_v30: 
                        nodos_por_tier_bc_disp_v30[current_tier_bc_disp_v30].append(nodo); processed_nodes_bc_disp_v30.add(nodo['ID_Nodo'])
            else: 
                for _, nodo_restante in nodos_a_mostrar_bc_disp_v30.iterrows():
                    if nodo_restante['ID_Nodo'] not in processed_nodes_bc_disp_v30:
                        nodos_por_tier_bc_disp_v30[current_tier_bc_disp_v30 + 100].append(nodo_restante); processed_nodes_bc_disp_v30.add(nodo_restante['ID_Nodo'])
                break
        for tier_level_val_disp_v30 in sorted(nodos_por_tier_bc_disp_v30.keys()): 
            if nodos_por_tier_bc_disp_v30[tier_level_val_disp_v30]:
                st.markdown(f"--- *Nivel/Grupo {tier_level_val_disp_v30 if tier_level_val_disp_v30 < 100 else 'Otros/Restantes'}* ---")
                num_cols_display_bc_val_v30 = 3; node_cards_cols_bc_val_v30 = st.columns(num_cols_display_bc_val_v30); col_idx_bc_val_v30 = 0 
                for nodo_item_bc_val_v30 in nodos_por_tier_bc_disp_v30[tier_level_val_disp_v30]: 
                    with node_cards_cols_bc_val_v30[col_idx_bc_val_v30 % num_cols_display_bc_val_v30]:
                        node_id_bc_val_v30 = nodo_item_bc_val_v30['ID_Nodo']; is_unlocked_bc_val_v30 = node_id_bc_val_v30 in st.session_state.bc_unlocked_nodes 
                        can_be_unlocked_bc_val_v30 = check_prerequisites(node_id_bc_val_v30, df_skill_trees_global, st.session_state.bc_unlocked_nodes)
                        beneficios_str_list_bc_val_v30 = [] 
                        cols_para_beneficios_bc_val_v30 = stat_cols_order + ['PIERNA_MALA', 'FILIGRANAS'] 
                        for col_stat_bc_val_v30 in cols_para_beneficios_bc_val_v30: 
                            if col_stat_bc_val_v30 in nodo_item_bc_val_v30.index and pd.notna(nodo_item_bc_val_v30[col_stat_bc_val_v30]) and nodo_item_bc_val_v30[col_stat_bc_val_v30] > 0:
                                if col_stat_bc_val_v30 == 'PIERNA_MALA': beneficios_str_list_bc_val_v30.append(f"+{int(nodo_item_bc_val_v30[col_stat_bc_val_v30])}⭐ WF")
                                elif col_stat_bc_val_v30 == 'FILIGRANAS': beneficios_str_list_bc_val_v30.append(f"+{int(nodo_item_bc_val_v30[col_stat_bc_val_v30])}⭐ SM")
                                else: beneficios_str_list_bc_val_v30.append(f"+{int(nodo_item_bc_val_v30[col_stat_bc_val_v30])} {col_stat_bc_val_v30}")
                        if 'PlayStyle' in nodo_item_bc_val_v30.index and pd.notna(nodo_item_bc_val_v30['PlayStyle']) and nodo_item_bc_val_v30['PlayStyle'] != '':
                            beneficios_str_list_bc_val_v30.append(f"PlayStyle: {nodo_item_bc_val_v30['PlayStyle']}")
                        beneficios_str_bc_val_v30 = ", ".join(beneficios_str_list_bc_val_v30) if beneficios_str_list_bc_val_v30 else "Sin bonus directo" 
                        with st.container(border=True):
                            st.markdown(f"**{nodo_item_bc_val_v30['Nombre_Visible']}**")
                            st.caption(f"ID: {node_id_bc_val_v30} | Costo: {nodo_item_bc_val_v30['Costo']}")
                            st.caption(f"Beneficios: {beneficios_str_bc_val_v30}")
                            if is_unlocked_bc_val_v30: st.button("✅ Desbloq.", key=f"skill_node_{node_id_bc_val_v30}_v30d", disabled=True)
                            elif can_be_unlocked_bc_val_v30 and st.session_state.bc_points_remaining >= nodo_item_bc_val_v30['Costo']:
                                if st.button("🔓 Desbloq.", key=f"skill_node_{node_id_bc_val_v30}_v30u"):
                                    st.session_state.bc_unlocked_nodes.add(node_id_bc_val_v30)
                                    st.session_state.bc_points_remaining -= nodo_item_bc_val_v30['Costo']
                                    st.rerun()
                            else:
                                help_t_bc_val_v30 = "Prerreq." if not can_be_unlocked_bc_val_v30 else f"Puntos ({nodo_item_bc_val_v30['Costo']})"
                                st.button("🔒 Bloq.", key=f"skill_node_{node_id_bc_val_v30}_v30l", disabled=True, help=help_t_bc_val_v30)
                    col_idx_bc_val_v30 +=1
    else: st.info("Selecciona un árbol de habilidad para ver sus nodos.")


with tab_best_combo:
    # ... (igual que v2.8, con botón "Enviar a Build Craft" y usando stat_cols_order, y actualizando keys) ...
    st.header("Búsqueda de Mejor Combinación por Atributos (Priorizados) - Stats Base")
    if not all_stats_df_base.empty:
        queryable_stats_ordered_bc_opt_disp_v30 = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE']]
        col1_bc_opt_disp_v30, col2_bc_opt_disp_v30, col3_bc_opt_disp_v30 = st.columns(3)
        with col1_bc_opt_disp_v30: attr_pri_opt_val_v30 = st.selectbox("1er Atributo:", options=queryable_stats_ordered_bc_opt_disp_v30, key="attr_pri_v30_opt")
        with col2_bc_opt_disp_v30: attr_sec_opt_val_v30 = st.selectbox("2do Atributo:", options=["(Ninguno)"] + queryable_stats_ordered_bc_opt_disp_v30, key="attr_sec_v30_opt")
        with col3_bc_opt_disp_v30: attr_ter_opt_val_v30 = st.selectbox("3er Atributo:", options=["(Ninguno)"] + queryable_stats_ordered_bc_opt_disp_v30, key="attr_ter_v30_opt")
        if st.button("Buscar Mejor Combinación Priorizada", key="btn_multi_attr_find_v30_opt"):
            temp_df_opt_res_v30 = all_stats_df_base.copy(); sort_by_attrs_opt_list_v30 = [attr_pri_opt_val_v30]
            if attr_sec_opt_val_v30 != "(Ninguno)": sort_by_attrs_opt_list_v30.append(attr_sec_opt_val_v30)
            if attr_ter_opt_val_v30 != "(Ninguno)": sort_by_attrs_opt_list_v30.append(attr_ter_opt_val_v30)
            temp_df_opt_res_v30 = temp_df_opt_res_v30.sort_values(by=sort_by_attrs_opt_list_v30, ascending=[False]*len(sort_by_attrs_opt_list_v30))
            if not temp_df_opt_res_v30.empty:
                best_player_opt_res_v30 = temp_df_opt_res_v30.iloc[0]
                st.subheader(f"Mejor Jugador (Stats Base): {best_player_opt_res_v30['Posicion']} | {best_player_opt_res_v30['Altura']}cm | {best_player_opt_res_v30['Peso']}kg")
                if st.button(f"🛠️ Personalizar Habilidades para este Mejor Build", key=f"send_to_bc_best_opt_v30"):
                    st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = best_player_opt_res_v30['Posicion'], best_player_opt_res_v30['Altura'], best_player_opt_res_v30['Peso']
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.success(f"Perfil de Mejor Build enviado. Ajusta selectores en sidebar y ve a '🛠️ Build Craft'.")
                accele_rate_best_opt_res_v30 = determinar_estilo_carrera(best_player_opt_res_v30['Altura'],best_player_opt_res_v30.get('AGI',0),best_player_opt_res_v30.get('STR',0),best_player_opt_res_v30.get('Acc',0))
                styled_metric("Estilo de Carrera (AcceleRATE Base)", accele_rate_best_opt_res_v30); st.divider()
                key_metrics_display_dict_v30 = {}
                if attr_pri_opt_val_v30 in best_player_opt_res_v30: key_metrics_display_dict_v30[attr_pri_opt_val_v30] = best_player_opt_res_v30[attr_pri_opt_val_v30]
                if attr_sec_opt_val_v30 != "(Ninguno)" and attr_sec_opt_val_v30 in best_player_opt_res_v30: key_metrics_display_dict_v30[attr_sec_opt_val_v30] = best_player_opt_res_v30[attr_sec_opt_val_v30]
                if attr_ter_opt_val_v30 != "(Ninguno)" and attr_ter_opt_val_v30 in best_player_opt_res_v30: key_metrics_display_dict_v30[attr_ter_opt_val_v30] = best_player_opt_res_v30[attr_ter_opt_val_v30]
                if 'IGS' in best_player_opt_res_v30: key_metrics_display_dict_v30['IGS'] = best_player_opt_res_v30['IGS']
                num_metrics_cols_disp_v30_val = min(len(key_metrics_display_dict_v30), 3)
                cols_metrics_cards_disp_v30_val = st.columns(num_metrics_cols_disp_v30_val) if num_metrics_cols_disp_v30_val > 0 else [st]
                for i, (label, value) in enumerate(key_metrics_display_dict_v30.items()):
                    with cols_metrics_cards_disp_v30_val[i % num_metrics_cols_disp_v30_val]: styled_metric(label, value)
                with st.expander("Ver todos los atributos base del mejor jugador"):
                    st.json(best_player_opt_res_v30.drop(['Posicion', 'Altura', 'Peso'], errors='ignore').to_dict())
            else: st.info("No se encontraron combinaciones.")
    else: st.warning("Datos para las búsquedas no disponibles.")

with tab_filters:
    # ... (igual que v2.8, usando stat_cols_order para options, y actualizando keys) ...
    st.header("Filtros Múltiples Avanzados (Stats Base)")
    if not all_stats_df_base.empty:
        queryable_stats_filter_ordered_tf_disp_v30 = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']]
        def add_filter_cb_v30():
            fid = st.session_state.next_filter_id
            st.session_state.filters.append({'id': fid, 'attribute': queryable_stats_filter_ordered_tf_disp_v30[0], 'condition': '>=', 'value': 70})
            st.session_state.next_filter_id += 1
        def remove_filter_cb_v30(fid_to_remove):
            st.session_state.filters = [f for f in st.session_state.filters if f['id'] != fid_to_remove]
        st.button("➕ Añadir Criterio", on_click=add_filter_cb_v30, key="add_filter_btn_v30_tab")
        filter_cont_disp_v30 = st.container()
        with filter_cont_disp_v30: 
            for item in st.session_state.filters:
                fid = item['id']; cols_f_disp_v30_val = st.columns([5,3,2,1]) # Renombrar
                try: current_attr_idx_f_v30_val = queryable_stats_filter_ordered_tf_disp_v30.index(item['attribute']) # Renombrar
                except ValueError: current_attr_idx_f_v30_val = 0 
                item['attribute'] = cols_f_disp_v30_val[0].selectbox("Atributo:", options=queryable_stats_filter_ordered_tf_disp_v30, index=current_attr_idx_f_v30_val, key=f"f_attr_{fid}_v30")
                cond_opts_f_v30_val = ['>=', '<=', '==', '>', '<'] # Renombrar
                try: current_cond_idx_f_v30_val = cond_opts_f_v30_val.index(item['condition']) # Renombrar
                except ValueError: current_cond_idx_f_v30_val = 0
                item['condition'] = cols_f_disp_v30_val[1].selectbox("Condición:", options=cond_opts_f_v30_val, index=current_cond_idx_f_v30_val, key=f"f_cond_{fid}_v30")
                item['value'] = cols_f_disp_v30_val[2].number_input("Valor:", value=int(item['value']), step=1, key=f"f_val_{fid}_v30")
                if cols_f_disp_v30_val[3].button("➖", key=f"f_rem_{fid}_v30", help="Eliminar"): remove_filter_cb_v30(fid); st.rerun() 
        
        if st.button("Aplicar Filtros (Stats Base)", key="btn_apply_adv_f_v30_tab"): 
            if not st.session_state.filters: st.info("No hay criterios definidos.")
            else:
                # ... (lógica de aplicación de filtros igual que v2.8) ...
                df_to_filter_res_v30_f_val = all_stats_df_base.copy()
                if 'AcceleRATE' not in df_to_filter_res_v30_f_val.columns:
                    df_to_filter_res_v30_f_val['AcceleRATE'] = df_to_filter_res_v30_f_val.apply(lambda r: determinar_estilo_carrera(r['Altura'], r.get('AGI',0),r.get('STR',0),r.get('Acc',0)), axis=1)
                df_adv_f_res_v30_f_val = df_to_filter_res_v30_f_val; valid_q_res_v30_f_val = True; act_f_attrs_list_v30_f_val = []
                for f_app_item_v30_f_val in st.session_state.filters:
                    attr_f_val_f_val, cond_f_val_f_val, val_f_val_f_val = f_app_item_v30_f_val['attribute'], f_app_item_v30_f_val['condition'], f_app_item_v30_f_val['value']
                    act_f_attrs_list_v30_f_val.append(attr_f_val_f_val)
                    if attr_f_val_f_val not in df_adv_f_res_v30_f_val.columns: st.error(f"Atributo '{attr_f_val_f_val}' no encontrado."); valid_q_res_v30_f_val = False; break
                    if cond_f_val_f_val == '>=': df_adv_f_res_v30_f_val = df_adv_f_res_v30_f_val[df_adv_f_res_v30_f_val[attr_f_val_f_val] >= val_f_val_f_val]
                    elif cond_f_val_f_val == '<=': df_adv_f_res_v30_f_val = df_adv_f_res_v30_f_val[df_adv_f_res_v30_f_val[attr_f_val_f_val] <= val_f_val_f_val]
                    elif cond_f_val_f_val == '==': df_adv_f_res_v30_f_val = df_adv_f_res_v30_f_val[df_adv_f_res_v30_f_val[attr_f_val_f_val] == val_f_val_f_val]
                    elif cond_f_val_f_val == '>': df_adv_f_res_v30_f_val = df_adv_f_res_v30_f_val[df_adv_f_res_v30_f_val[attr_f_val_f_val] > val_f_val_f_val]
                    elif cond_f_val_f_val == '<': df_adv_f_res_v30_f_val = df_adv_f_res_v30_f_val[df_adv_f_res_v30_f_val[attr_f_val_f_val] < val_f_val_f_val]
                if valid_q_res_v30_f_val:
                    if not df_adv_f_res_v30_f_val.empty:
                        st.write(f"Combinaciones ({len(df_adv_f_res_v30_f_val)}):")
                        cols_disp_res_list_f_val = ['Posicion', 'Altura', 'Peso'] + [col for col in queryable_stats_filter_ordered_tf_disp_v30 if col in act_f_attrs_list_v30_f_val] + ['AcceleRATE', 'IGS']
                        final_cols_disp_list_val_f_val = []; [final_cols_disp_list_val_f_val.append(col) for col in cols_disp_res_list_f_val if col not in final_cols_disp_list_val_f_val]
                        final_cols_disp_existing_list_val_f_val = [col for col in final_cols_disp_list_val_f_val if col in df_adv_f_res_v30_f_val.columns]
                        st.dataframe(df_adv_f_res_v30_f_val[final_cols_disp_existing_list_val_f_val])
                    else: st.info("Ninguna combinación cumple criterios.")
    else: st.warning("Datos para filtros no disponibles.")