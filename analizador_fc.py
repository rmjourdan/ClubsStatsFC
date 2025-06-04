import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Calculadora Stats FC25 v2.6", layout="wide")

# --- Lógica de AcceleRATE ---
def determinar_estilo_carrera(altura, agilidad, fuerza, aceleracion):
    try: altura, agilidad, fuerza, aceleracion = int(altura), int(agilidad), int(fuerza), int(aceleracion)
    except (ValueError, TypeError): return "N/A" # Cambiado para consistencia
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
    except FileNotFoundError as e: st.error(f"Error al cargar CSV base: {e}."); return None, None, None, None, None, None
    except Exception as e_gen: st.error(f"Error inesperado al cargar CSV base: {e_gen}"); return None, None, None, None, None, None
    try:
        df_altura, df_peso, df_posiciones = df_altura_raw.set_index('Altura'), df_peso_raw.set_index('Peso'), df_posiciones_raw.set_index('Posicion')
        expected_cols = df_posiciones.columns.tolist()
        for df, name in [(df_altura, "ALTURA"), (df_peso, "PESO")]:
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols: st.warning(f"Advertencia: Columnas {missing_cols} faltantes en {name}.csv (rellenadas con 0).")
        df_altura = df_altura.reindex(columns=expected_cols).fillna(0).astype(int)
        df_peso = df_peso.reindex(columns=expected_cols).fillna(0).astype(int)
        df_posiciones = df_posiciones.reindex(columns=expected_cols).fillna(0).astype(int)
        alt_base, peso_base, pos_base = 162, 45, 'LB/RB'
        if pos_base not in df_posiciones.index: st.error(f"Posición base '{pos_base}' no encontrada."); return None, None, None, None, None, None
        stats_base_lb_rb = df_posiciones.loc[pos_base]
        if alt_base not in df_altura.index: st.error(f"Altura base '{alt_base}cm' no encontrada."); return None, None, None, None, None, None
        stats_alt_ref = df_altura.loc[alt_base]
        if peso_base not in df_peso.index: st.error(f"Peso base '{peso_base}kg' no encontrado."); return None, None, None, None, None, None
        stats_peso_ref = df_peso.loc[peso_base]
        mod_alt = df_altura.subtract(stats_alt_ref, axis=1)
        mod_peso = df_peso.subtract(stats_peso_ref, axis=1)
        diff_pos = df_posiciones.subtract(stats_base_lb_rb, axis=1)
        return stats_base_lb_rb, mod_alt, mod_peso, diff_pos, df_posiciones.index.tolist(), expected_cols
    except KeyError as ke: st.error(f"Error de clave (KeyError) en datos base: '{ke}'."); return None, None, None, None, None, None
    except Exception as e_proc: st.error(f"Error al procesar DataFrames base: {e_proc}"); return None, None, None, None, None, None

# --- Carga y Preparación del Árbol de Habilidades ---
@st.cache_data
def cargar_arbol_habilidades():
    skill_tree_file = "ARBOL DE HABILIDAD - ARBOL.csv"
    try:
        df_skill_trees = pd.read_csv(skill_tree_file)
        stat_boost_cols = [
            'Acc', 'Spr', 'Fin', 'FKA', 'HAcc', 'SPow', 'Lon S', 'Vol', 'Pen', 
            'Vis', 'Cros', 'LP', 'SP', 'Cur', 'AGI', 'BAL', 'APOS', 'BCON', 
            'REG', 'INT', 'AWA', 'STAN', 'SLID', 'JUMP', 'STA', 'STR', 'REA', 
            'AGGR', 'COMP', 'PIERNA_MALA', 'FILIGRANAS'
        ]
        for col in stat_boost_cols:
            if col in df_skill_trees.columns:
                df_skill_trees[col] = pd.to_numeric(df_skill_trees[col], errors='coerce').fillna(0).astype(int)
            else: df_skill_trees[col] = 0 
        if 'Costo' in df_skill_trees.columns:
             df_skill_trees['Costo'] = pd.to_numeric(df_skill_trees['Costo'], errors='coerce').fillna(0).astype(int)
        if 'Prerrequisito' in df_skill_trees.columns:
            df_skill_trees['Prerrequisito'] = df_skill_trees['Prerrequisito'].fillna('').astype(str)
        return df_skill_trees
    except FileNotFoundError: st.error(f"ERROR: No se encontró '{skill_tree_file}'."); return None
    except Exception as e: st.error(f"Error al cargar CSV del árbol de hab: {e}"); return None

# --- Función de Cálculo de Estadísticas Base (sin árbol) ---
def calcular_stats_base_jugador(pos_sel, alt_sel, peso_sel, base_ref_stats, mod_alt_df, mod_peso_df, diff_pos_df):
    diff = diff_pos_df.loc[pos_sel]; mod_a = mod_alt_df.loc[alt_sel]; mod_p = mod_peso_df.loc[peso_sel]
    final_base_stats = base_ref_stats.add(diff).add(mod_a).add(mod_p)
    return final_base_stats.round().astype(int)

# --- Pre-cálculo de todas las combinaciones base ---
@st.cache_data
def precompute_all_base_stats(_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df, _stat_cols_order):
    all_data = []
    # ... (igual que v2.5) ...
    if not all(isinstance(df, (pd.DataFrame, pd.Series)) for df in [_base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df]):
         st.error("Datos base inválidos para precompute_all_base_stats."); return pd.DataFrame()
    for pos in _diff_pos_df.index:
        for alt_v in _mod_alt_df.index:
            for pes_v in _mod_peso_df.index:
                stats = calcular_stats_base_jugador(pos, alt_v, pes_v, _base_ref_stats, _mod_alt_df, _mod_peso_df, _diff_pos_df)
                entry = {'Posicion': pos, 'Altura': alt_v, 'Peso': pes_v, **stats.to_dict()}
                all_data.append(entry)
    df = pd.DataFrame(all_data)
    if not df.empty:
        valid_cols = [col for col in _stat_cols_order if col in df.columns]
        cols_df_order = ['Posicion', 'Altura', 'Peso'] + valid_cols 
        cols_df_order_existing = [col for col in cols_df_order if col in df.columns]
        df = df[cols_df_order_existing]
    return df

# --- Funciones Auxiliares para el Editor de Habilidades ---
BASE_WF, BASE_SM, MAX_STARS, MAX_STAT_VAL = 2, 3, 5, 99
TOTAL_SKILL_POINTS = 184
MAIN_CATEGORIES_FOR_IGS = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'] # Para recalcular IGS

def aplicar_mejoras_habilidad(stats_jugador_base, df_skill_tree_data, unlocked_nodes_ids, altura_jugador_actual):
    stats_modificadas = stats_jugador_base.copy().astype(float) # Trabajar con float para sumas
    
    non_stat_boost_cols_in_skill_tree = ['ID_Nodo', 'Arbol', 'Nombre_Visible', 'Costo', 'Prerrequisito', 
                                         'PlayStyle', 'Es_Arquetipo', 'Notas', 'Puntos_Req_Arbol'] 

    current_wf, current_sm = float(BASE_WF), float(BASE_SM)

    for node_id in unlocked_nodes_ids:
        if node_id not in df_skill_tree_data['ID_Nodo'].values: continue
        node_data = df_skill_tree_data[df_skill_tree_data['ID_Nodo'] == node_id].iloc[0]
        
        for stat_col, boost_val_obj in node_data.items():
            # Asegurarse que el boost_val sea numérico y no NaN antes de convertir a int
            if pd.notna(boost_val_obj) and isinstance(boost_val_obj, (int, float, str)) and str(boost_val_obj).replace('.','',1).isdigit() : # Verifica si es numérico
                boost_val = int(float(boost_val_obj)) # Convertir a float primero, luego a int
            elif pd.notna(boost_val_obj) and stat_col not in non_stat_boost_cols_in_skill_tree: # Si no es numérico pero es una stat que podría ser 0
                boost_val = 0
            else: # Si es NaN o no es un boost de stat válido, continuar
                continue


            if stat_col not in non_stat_boost_cols_in_skill_tree and boost_val != 0:
                if stat_col == 'PIERNA_MALA': current_wf += boost_val
                elif stat_col == 'FILIGRANAS': current_sm += boost_val
                elif stat_col in stats_modificadas.index:
                    # Asegurarse que la stat base sea numérica antes de sumar
                    if pd.api.types.is_numeric_dtype(stats_modificadas[stat_col]):
                        stats_modificadas[stat_col] = stats_modificadas[stat_col] + boost_val
                    else: # Si la stat base no es numérica (improbable para stats principales), inicializarla
                        stats_modificadas[stat_col] = boost_val 
                # else: stats_modificadas[stat_col] = boost_val # No deberíamos añadir nuevas stats aquí
    
    # Capar atributos individuales a MAX_STAT_VAL (99)
    for stat_name in stats_modificadas.index:
        if stat_name not in ['PIERNA_MALA', 'FILIGRANAS', 'AcceleRATE', 'IGS']: 
             if pd.api.types.is_numeric_dtype(stats_modificadas[stat_name]):
                stats_modificadas[stat_name] = min(stats_modificadas[stat_name], MAX_STAT_VAL)

    stats_modificadas['PIERNA_MALA'] = min(current_wf, MAX_STARS)
    stats_modificadas['FILIGRANAS'] = min(current_sm, MAX_STARS)
    
    # Determinar AcceleRATE con las stats ya potencialmente modificadas
    stats_modificadas['AcceleRATE'] = determinar_estilo_carrera(
        altura_jugador_actual, stats_modificadas.get('AGI', 0), 
        stats_modificadas.get('STR', 0), stats_modificadas.get('Acc', 0)
    )
    
    # Recalcular IGS como suma de las 6 categorías principales
    igs_sum = 0
    for cat in MAIN_CATEGORIES_FOR_IGS: # MAIN_CATEGORIES_FOR_IGS = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
        # Asegurarse que la categoría exista y sea numérica antes de sumar
        if cat in stats_modificadas.index and pd.api.types.is_numeric_dtype(stats_modificadas[cat]):
            igs_sum += stats_modificadas[cat]
        # else: st.warning(f"Categoría {cat} para IGS no es numérica o no encontrada en stats_modificadas.")
    stats_modificadas['IGS'] = igs_sum
    
    # Redondear y convertir a entero SOLO las estadísticas numéricas
    # AcceleRATE (string) y otras posibles columnas de texto se mantendrán como están.
    stats_final_dict = {}
    for stat_name, value in stats_modificadas.items():
        if isinstance(value, (int, float)):
            stats_final_dict[stat_name] = int(round(value))
        else:
            stats_final_dict[stat_name] = value # Mantener strings como AcceleRATE
            
    return pd.Series(stats_final_dict, name=stats_modificadas.name)

def check_prerequisites(node_id, df_skill_tree_data, unlocked_nodes_ids):
    if node_id not in df_skill_tree_data['ID_Nodo'].values: return False
    node_info = df_skill_tree_data[df_skill_tree_data['ID_Nodo'] == node_id].iloc[0]
    prereqs_str = node_info.get('Prerrequisito', '')
    if pd.isna(prereqs_str) or prereqs_str == '': return True
    prereq_ids = prereqs_str.split(',')
    # Condición OR: si CUALQUIERA de los prerrequisitos directos está desbloqueado
    return any(pr_id.strip() in unlocked_nodes_ids for pr_id in prereq_ids)


# --- Carga de Datos Principal ---
APP_VERSION = "v2.6" 
if 'app_version' not in st.session_state or st.session_state.app_version != APP_VERSION:
    st.session_state.clear(); st.session_state.app_version = APP_VERSION

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
    for key in ['bc_unlocked_nodes', 'filters']: # Inicializar listas/sets si no existen
        if key not in st.session_state: st.session_state[key] = set() if key.endswith('nodes') else []
    if 'bc_points_total' not in st.session_state: st.session_state.bc_points_total = TOTAL_SKILL_POINTS
    if 'bc_points_remaining' not in st.session_state: st.session_state.bc_points_remaining = TOTAL_SKILL_POINTS
    if 'next_filter_id' not in st.session_state: st.session_state.next_filter_id = 0
    
    # Perfil base para Build Craft
    default_pos = lista_posiciones[lista_posiciones.index('ST') if 'ST' in lista_posiciones else 0]
    default_alt = sorted(modificadores_altura.index.unique().tolist())[sorted(modificadores_altura.index.unique().tolist()).index(180) if 180 in modificadores_altura.index.unique() else 0]
    default_pes = sorted(modificadores_peso.index.unique().tolist())[sorted(modificadores_peso.index.unique().tolist()).index(75) if 75 in modificadores_peso.index.unique() else 0]

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
    except TypeError: return ['' for _ in row] # Si hay tipos mixtos no numéricos
    return ['background-color: #d4edda; color: #155724; font-weight: bold;' if (v == max_val and pd.notna(v)) else '' for v in row]

# Selectores de Perfil Base para Build Craft en la Barra Lateral
st.sidebar.markdown("--- \n ### 🛠️ Perfil Base para Build Craft:")
# Guardar selecciones de sidebar en session_state para que los botones "Enviar a Build Craft" puedan actualizarlas
current_bc_pos_idx = lista_posiciones.index(st.session_state.bc_pos) if st.session_state.bc_pos in lista_posiciones else 0
st.session_state.bc_pos = st.sidebar.selectbox("Posición Base (Build Craft):", options=sorted(lista_posiciones), index=current_bc_pos_idx, key="sb_bc_pos_v26")

unique_alts = sorted(modificadores_altura.index.unique().tolist())
current_bc_alt_idx = unique_alts.index(st.session_state.bc_alt) if st.session_state.bc_alt in unique_alts else 0
st.session_state.bc_alt = st.sidebar.selectbox("Altura Base (cm) (Build Craft):", options=unique_alts, index=current_bc_alt_idx, key="sb_bc_alt_v26")

unique_pesos = sorted(modificadores_peso.index.unique().tolist())
current_bc_pes_idx = unique_pesos.index(st.session_state.bc_pes) if st.session_state.bc_pes in unique_pesos else 0
st.session_state.bc_pes = st.sidebar.selectbox("Peso Base (kg) (Build Craft):", options=unique_pesos, index=current_bc_pes_idx, key="sb_bc_pes_v26")


tab_calc, tab_build_craft, tab_best_combo, tab_filters = st.tabs([
    "🧮 Calculadora y Comparador", 
    "🛠️ Build Craft",
    "🔍 Búsqueda Óptima", 
    "📊 Filtros Múltiples"
])

# --- Pestaña: Calculadora y Comparador ---
with tab_calc:
    st.header("Calculadora de Estadísticas y Comparador (Stats Base)")
    num_players_to_compare = st.radio("Jugadores a definir/comparar:", (1, 2, 3), index=0, horizontal=True, key="num_players_radio_v26_calc")
    cols_selectors_calc = st.columns(num_players_to_compare)
    player_stats_list_base_calc, player_configs_base_calc = [], []

    for i in range(num_players_to_compare):
        with cols_selectors_calc[i]:
            player_label = f"JUG {chr(65+i)}"
            st.subheader(player_label)
            # Claves únicas para los selectores de esta pestaña
            pos_key, alt_key, pes_key = f"pos_p{i}_v26_calc", f"alt_p{i}_v26_calc", f"pes_p{i}_v26_calc"
            # Inicializar estado para estos selectores si no existen
            if pos_key not in st.session_state: st.session_state[pos_key] = lista_posiciones[0]
            if alt_key not in st.session_state: st.session_state[alt_key] = unique_alts[0]
            if pes_key not in st.session_state: st.session_state[pes_key] = unique_pesos[0]

            pos = st.selectbox(f"Posición ({player_label}):", sorted(lista_posiciones), key=pos_key)
            alt = st.selectbox(f"Altura ({player_label}):", unique_alts, key=alt_key)
            pes = st.selectbox(f"Peso ({player_label}):", unique_pesos, key=pes_key)
            
            player_configs_base_calc.append({'pos': pos, 'alt': alt, 'pes': pes, 'label': player_label})
            if pos and alt is not None and pes is not None:
                stats = calcular_stats_base_jugador(pos, alt, pes, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
                stats['AcceleRATE'] = determinar_estilo_carrera(alt, stats.get('AGI',0), stats.get('STR',0), stats.get('Acc',0))
                player_stats_list_base_calc.append(stats)
                if st.button(f"🛠️ Editar Habilidades para {player_label}", key=f"send_to_bc_calc_{i}_v26"):
                    st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = pos, alt, pes
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.success(f"Perfil de {player_label} enviado a Build Craft. ¡Ve a la pestaña '🛠️ Build Craft' y actualiza los selectores de la barra lateral si es necesario para que coincidan!")
            else: player_stats_list_base_calc.append(None)
    
    st.divider(); st.subheader("Perfiles y Estadísticas (Base)")
    if any(ps is not None for ps in player_stats_list_base_calc):
        # ... (Lógica de visualización de radar y tabla comparativa como antes)
        # Para corregir el error de Arrow, convertiremos DataFrames/Series problemáticos a string antes de st.dataframe
        fig_radar_comp = go.Figure()
        radar_attrs = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
        cols_perfiles = st.columns(num_players_to_compare)

        for i, stats in enumerate(player_stats_list_base_calc):
            if stats is not None:
                with cols_perfiles[i]:
                    st.markdown(f"**{player_configs_base_calc[i]['label']} ({player_configs_base_calc[i]['pos']}, {player_configs_base_calc[i]['alt']}cm, {player_configs_base_calc[i]['pes']}kg)**")
                    st.metric(label="Estilo de Carrera", value=stats.get('AcceleRATE', "N/A"))
                # ... (código del radar igual)
                valid_radar_attrs = [attr for attr in radar_attrs if attr in stats.index]
                radar_values = stats[valid_radar_attrs].values.flatten().tolist()
                if len(valid_radar_attrs) >= 3:
                    fig_radar_comp.add_trace(go.Scatterpolar(r=radar_values, theta=valid_radar_attrs, fill='toself', 
                                                            name=f"{player_configs_base_calc[i]['label']} ({player_configs_base_calc[i]['pos']})",
                                                            line_color=PLAYER_COLORS[i % len(PLAYER_COLORS)],
                                                            fillcolor=PLAYER_FILL_COLORS[i % len(PLAYER_FILL_COLORS)]))
        if fig_radar_comp.data:
            fig_radar_comp.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Comparativa de Perfiles Base")
            st.plotly_chart(fig_radar_comp, use_container_width=True)
        
        valid_player_stats_b_calc = [ps for ps in player_stats_list_base_calc if ps is not None]
        valid_player_column_names_b_calc = [player_configs_base_calc[i]['label'] for i, ps in enumerate(player_stats_list_base_calc) if ps is not None]

        if len(valid_player_stats_b_calc) > 1 :
            compare_dict_short_names_b_calc = {valid_player_column_names_b_calc[j]: valid_player_stats_b_calc[j] for j in range(len(valid_player_stats_b_calc))}
            df_compare_b_calc = pd.DataFrame(compare_dict_short_names_b_calc)
            # Preparar para visualización (mover AcceleRATE y convertir a string para evitar error Arrow)
            if 'AcceleRATE' in df_compare_b_calc.index:
                accele_rate_row_b_calc = df_compare_b_calc.loc[['AcceleRATE']]
                df_compare_b_calc_numeric = df_compare_b_calc.drop('AcceleRATE')
                df_compare_b_calc_styled = df_compare_b_calc_numeric.style.apply(highlight_max_in_row, axis=1)
                st.subheader("Tabla Comparativa de Estadísticas Base")
                st.dataframe(df_compare_b_calc_styled)
                st.caption("Estilo de Carrera (AcceleRATE):")
                st.dataframe(accele_rate_row_b_calc) # Mostrar AcceleRATE por separado
            else: # Si no hay AcceleRATE, aplicar estilo directamente
                df_compare_b_calc_styled = df_compare_b_calc.style.apply(highlight_max_in_row, axis=1)
                st.subheader("Tabla Comparativa de Estadísticas Base")
                st.dataframe(df_compare_b_calc_styled)

        elif len(valid_player_stats_b_calc) == 1:
            st.subheader(f"Estadísticas Detalladas Base: {player_configs_base_calc[0]['label']}")
            stats_display_b_calc = valid_player_stats_b_calc[0]
            # Convertir a string para st.dataframe si AcceleRATE está presente y es problemático
            st.dataframe(stats_display_b_calc.astype(str).rename(player_configs_base_calc[0]['label']))
        
        st.divider(); st.header("🏆 Top 5 Combinaciones por IGS (Stats Base)")
        # ... (código igual que v2.4, con botón "Enviar a Build Craft") ...
        if not all_stats_df_base.empty and 'IGS' in all_stats_df_base.columns:
            all_stats_df_base_display = all_stats_df_base.copy() # Copia para añadir AcceleRATE si no está
            all_stats_df_base_display['IGS'] = pd.to_numeric(all_stats_df_base_display['IGS'], errors='coerce').fillna(0)
            if 'AcceleRATE' not in all_stats_df_base_display.columns:
                 all_stats_df_base_display['AcceleRATE'] = all_stats_df_base_display.apply(
                     lambda row: determinar_estilo_carrera(row['Altura'], row.get('AGI',0),row.get('STR',0),row.get('Acc',0)), axis=1)
            top_5_igs = all_stats_df_base_display.sort_values(by='IGS', ascending=False).head(5)
            
            for i, row_idx in enumerate(top_5_igs.index): # Usar index para asegurar que obtenemos la fila correcta
                row = top_5_igs.loc[row_idx]
                # Crear columnas dinámicamente para los botones del Top 5
                # Esto es un poco más complejo para layout, podríamos listarlos verticalmente
                if st.button(f"🛠️ Editar {row['Posicion']} {row['Altura']}cm {row['Peso']}kg (Top {i+1})", key=f"send_to_bc_top5_{row_idx}_v26"):
                    st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = row['Posicion'], row['Altura'], row['Peso']
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.success(f"Perfil Top {i+1} enviado a Build Craft. Ajusta selectores en sidebar si es necesario y ve a la pestaña '🛠️ Build Craft'.")
            st.dataframe(top_5_igs[['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']])
        else: st.warning("Datos para el Top 5 (base) no disponibles.")
    else: st.info("Define parámetros de al menos un jugador para ver stats base.")


# --- Pestaña: Build Craft ---
with tab_build_craft:
    st.header(f"🛠️ Build Craft: {st.session_state.bc_pos} | {st.session_state.bc_alt}cm | {st.session_state.bc_pes}kg")
    jugador_base_actual_bc = calcular_stats_base_jugador(st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
    stats_con_skills_bc = aplicar_mejoras_habilidad(jugador_base_actual_bc, df_skill_trees_global, st.session_state.bc_unlocked_nodes, st.session_state.bc_alt)
    
    col_bc_info, col_bc_radar_display = st.columns([1,1]) # Renombrada para evitar conflicto
    with col_bc_info: # Info del build actual
        st.metric("Puntos Restantes:", f"{st.session_state.bc_points_remaining} / {TOTAL_SKILL_POINTS}")
        st.metric("Pierna Mala (WF):", "⭐" * stats_con_skills_bc.get('PIERNA_MALA', BASE_WF))
        st.metric("Filigranas (SM):", "⭐" * stats_con_skills_bc.get('FILIGRANAS', BASE_SM))
        st.metric("Estilo de Carrera:", stats_con_skills_bc.get('AcceleRATE', "N/A"))
        st.metric("IGS (con Habilidades):", f"{stats_con_skills_bc.get('IGS', 0)}")


    with col_bc_radar_display: # Radar del build actual
        radar_attrs_bc = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']
        valid_radar_attrs_bc = [attr for attr in radar_attrs_bc if attr in stats_con_skills_bc.index]
        radar_values_bc = stats_con_skills_bc[valid_radar_attrs_bc].values.flatten().tolist()
        if len(valid_radar_attrs_bc) >=3:
            fig_radar_bc = go.Figure()
            fig_radar_bc.add_trace(go.Scatterpolar(r=radar_values_bc, theta=valid_radar_attrs_bc, fill='toself', name="Con Habilidades", line_color=PLAYER_COLORS[0]))
            fig_radar_bc.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Perfil Actual del Build")
            st.plotly_chart(fig_radar_bc, use_container_width=True)
    
    with st.expander("Ver todas las estadísticas del Build Actual (con habilidades)", expanded=False):
        # Para evitar error Arrow, convertir Series a string para display si contiene AcceleRATE
        st.dataframe(stats_con_skills_bc.astype(str).rename("Valor Build"))
    
    st.divider()
    st.subheader("Árboles de Habilidad para Personalizar")
    col_tree_selector_bc, col_tree_summary_bc = st.columns([1,2])
    with col_tree_selector_bc:
        arbol_seleccionado_bc = st.selectbox("Selecciona un Árbol:", options=["Todos"] + sorted(df_skill_trees_global['Arbol'].unique()), key="skill_tree_select_bc_v26")
        if st.button("Resetear Puntos de Habilidad", key="reset_skills_btn_bc_v26"):
            st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
            st.rerun()

    with col_tree_summary_bc: # Visualización dinámica de stats del árbol seleccionado
        if arbol_seleccionado_bc != "Todos":
            st.markdown(f"**Impacto del Árbol '{arbol_seleccionado_bc}' en Stats Base:**")
            stats_afectadas_por_arbol = {}
            for node_id in st.session_state.bc_unlocked_nodes:
                if node_id not in df_skill_trees_global['ID_Nodo'].values: continue
                node_data = df_skill_trees_global[df_skill_trees_global['ID_Nodo'] == node_id].iloc[0]
                if node_data['Arbol'] == arbol_seleccionado_bc:
                    for col_stat_name in stat_cols_order: # Iterar sobre las stats conocidas
                        if col_stat_name in node_data.index and node_data[col_stat_name] > 0:
                            stats_afectadas_por_arbol[col_stat_name] = stats_afectadas_por_arbol.get(col_stat_name, 0) + node_data[col_stat_name]
            
            if stats_afectadas_por_arbol:
                data_summary = []
                for stat_name, total_boost in stats_afectadas_por_arbol.items():
                    base_val = jugador_base_actual_bc.get(stat_name, 0)
                    final_val = min(base_val + total_boost, MAX_STAT_VAL) # Aplicar cap de 99
                    data_summary.append({"Stat": stat_name, "Base": base_val, f"Boost ({arbol_seleccionado_bc})": f"+{total_boost}", "Parcial": final_val})
                df_summary = pd.DataFrame(data_summary).set_index("Stat")
                st.dataframe(df_summary)
            else:
                st.caption(f"No hay nodos desbloqueados de '{arbol_seleccionado_bc}' que mejoren estadísticas directamente.")
    
    # ... (lógica de mostrar y desbloquear nodos igual que v2.5, con la mejora visual de tarjeta) ...
    if arbol_seleccionado_bc:
        nodos_a_mostrar = df_skill_trees_global
        if arbol_seleccionado_bc != "Todos":
            nodos_a_mostrar = df_skill_trees_global[df_skill_trees_global['Arbol'] == arbol_seleccionado_bc]
        st.markdown(f"**Nodos para '{arbol_seleccionado_bc}':** ({len(nodos_a_mostrar)} nodos)")

        for _, nodo in nodos_a_mostrar.iterrows():
            node_id = nodo['ID_Nodo']
            is_unlocked = node_id in st.session_state.bc_unlocked_nodes
            can_be_unlocked = check_prerequisites(node_id, df_skill_trees_global, st.session_state.bc_unlocked_nodes)
            
            beneficios_str_list = []
            cols_para_beneficios = stat_cols_order + ['PIERNA_MALA', 'FILIGRANAS']
            for col_stat in cols_para_beneficios: 
                if col_stat in nodo.index and pd.notna(nodo[col_stat]) and nodo[col_stat] > 0:
                    if col_stat == 'PIERNA_MALA': beneficios_str_list.append(f"+{int(nodo[col_stat])}⭐ WF")
                    elif col_stat == 'FILIGRANAS': beneficios_str_list.append(f"+{int(nodo[col_stat])}⭐ SM")
                    else: beneficios_str_list.append(f"+{int(nodo[col_stat])} {col_stat}")
            if 'PlayStyle' in nodo.index and pd.notna(nodo['PlayStyle']) and nodo['PlayStyle'] != '':
                beneficios_str_list.append(f"PlayStyle: {nodo['PlayStyle']}")
            beneficios_str = ", ".join(beneficios_str_list) if beneficios_str_list else "Sin bonus directo"

            # Tarjeta para el nodo
            with st.container():
                st.markdown(f"**{nodo['Nombre_Visible']}** (Costo: {nodo['Costo']})")
                st.caption(f"ID: {node_id} | Beneficios: {beneficios_str}")
                # No mostrar Prerrequisito crudo aquí
                # if pd.notna(nodo['Prerrequisito']) and nodo['Prerrequisito'] != '':
                # st.caption(f"Req: {nodo['Prerrequisito']}") # Ocultado por feedback de usuario

                if is_unlocked:
                    st.button("✅ Desbloqueado", key=f"skill_node_{node_id}_v26", disabled=True, help="Ya desbloqueado.")
                elif can_be_unlocked and st.session_state.bc_points_remaining >= nodo['Costo']:
                    if st.button("🔓 Desbloquear", key=f"skill_node_{node_id}_v26"):
                        st.session_state.bc_unlocked_nodes.add(node_id)
                        st.session_state.bc_points_remaining -= nodo['Costo']
                        st.rerun()
                else:
                    help_text = "Prerreq. no cumplidos" if not can_be_unlocked else f"Puntos insuficientes ({nodo['Costo']})"
                    st.button("🔒 Bloqueado", key=f"skill_node_{node_id}_v26", disabled=True, help=help_text)
                st.markdown("---")


with tab_best_combo:
    # ... (igual que v2.4, con botón "Enviar a Build Craft" y usando stat_cols_order) ...
    st.header("Búsqueda de Mejor Combinación por Atributos (Priorizados) - Stats Base")
    if not all_stats_df_base.empty:
        queryable_stats_ordered_bc = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE']]
        col1_bc_opt, col2_bc_opt, col3_bc_opt = st.columns(3)
        with col1_bc_opt: attr_pri_opt = st.selectbox("1er Atributo:", options=queryable_stats_ordered_bc, key="attr_pri_v26_opt")
        with col2_bc_opt: attr_sec_opt = st.selectbox("2do Atributo:", options=["(Ninguno)"] + queryable_stats_ordered_bc, key="attr_sec_v26_opt")
        with col3_bc_opt: attr_ter_opt = st.selectbox("3er Atributo:", options=["(Ninguno)"] + queryable_stats_ordered_bc, key="attr_ter_v26_opt")

        if st.button("Buscar Mejor Combinación Priorizada", key="btn_multi_attr_find_v26_opt"):
            temp_df_opt = all_stats_df_base.copy(); sort_by_attrs_opt = [attr_pri_opt]
            if attr_sec_opt != "(Ninguno)": sort_by_attrs_opt.append(attr_sec_opt)
            if attr_ter_opt != "(Ninguno)": sort_by_attrs_opt.append(attr_ter_opt)
            temp_df_opt = temp_df_opt.sort_values(by=sort_by_attrs_opt, ascending=[False]*len(sort_by_attrs_opt))
            if not temp_df_opt.empty:
                best_player_opt = temp_df_opt.iloc[0]
                st.subheader(f"Mejor Jugador (Stats Base): {best_player_opt['Posicion']} | {best_player_opt['Altura']}cm | {best_player_opt['Peso']}kg")
                if st.button(f"🛠️ Editar Habilidades para este Mejor Build", key=f"send_to_bc_best_opt_v26"):
                    st.session_state.bc_pos, st.session_state.bc_alt, st.session_state.bc_pes = best_player_opt['Posicion'], best_player_opt['Altura'], best_player_opt['Peso']
                    st.session_state.bc_unlocked_nodes, st.session_state.bc_points_remaining = set(), TOTAL_SKILL_POINTS
                    st.success(f"Perfil de Mejor Build enviado a Build Craft. Ajusta selectores en sidebar y ve a la pestaña '🛠️ Build Craft'.")
                accele_rate_best_opt = determinar_estilo_carrera(best_player_opt['Altura'],best_player_opt.get('AGI',0),best_player_opt.get('STR',0),best_player_opt.get('Acc',0))
                st.metric(label="Estilo de Carrera (AcceleRATE Base)", value=accele_rate_best_opt); st.divider()
                # ... (lógica de st.metric y expander como antes) ...
                cols_card_1, cols_card_2 = st.columns(2)
                key_metrics = {'IGS': best_player_opt.get('IGS', 'N/A'), attr_pri_opt: best_player_opt.get(attr_pri_opt, 'N/A')}
                if attr_sec_opt != "(Ninguno)": key_metrics[attr_sec_opt] = best_player_opt.get(attr_sec_opt, 'N/A')
                if attr_ter_opt != "(Ninguno)": key_metrics[attr_ter_opt] = best_player_opt.get(attr_ter_opt, 'N/A')
                valid_key_metrics = {k: v for k, v in key_metrics.items() if v != 'N/A' and k is not None}
                metrics_list = list(valid_key_metrics.items())
                with cols_card_1:
                    if len(metrics_list) > 0: st.metric(label=str(metrics_list[0][0]), value=str(metrics_list[0][1]))
                    if len(metrics_list) > 2: st.metric(label=str(metrics_list[2][0]), value=str(metrics_list[2][1]))
                with cols_card_2:
                    if len(metrics_list) > 1: st.metric(label=str(metrics_list[1][0]), value=str(metrics_list[1][1]))
                    if len(metrics_list) > 3: st.metric(label=str(metrics_list[3][0]), value=str(metrics_list[3][1]))
                with st.expander("Ver todos los atributos base del mejor jugador"):
                    st.json(best_player_opt.drop(['Posicion', 'Altura', 'Peso']).to_dict())

            else: st.info("No se encontraron combinaciones.")
    else: st.warning("Datos para las búsquedas no disponibles.")


with tab_filters:
    # ... (igual que v2.4, usando stat_cols_order para options, y actualizando keys) ...
    st.header("Filtros Múltiples Avanzados (Stats Base)")
    if not all_stats_df_base.empty:
        queryable_stats_filter_ordered_tf = [col for col in stat_cols_order if col in all_stats_df_base.columns and col not in ['Posicion', 'Altura', 'Peso', 'AcceleRATE', 'IGS']]
        def add_filter_cb_v26():
            fid = st.session_state.next_filter_id
            st.session_state.filters.append({'id': fid, 'attribute': queryable_stats_filter_ordered_tf[0], 'condition': '>=', 'value': 70})
            st.session_state.next_filter_id += 1
        def remove_filter_cb_v26(fid_to_remove):
            st.session_state.filters = [f for f in st.session_state.filters if f['id'] != fid_to_remove]
        st.button("➕ Añadir Criterio", on_click=add_filter_cb_v26, key="add_filter_btn_v26_tab")
        # ... (resto de la lógica de filtros igual, solo actualizando keys) ...
        filter_cont = st.container()
        with filter_cont:
            for item in st.session_state.filters:
                fid = item['id']; cols_f = st.columns([5,3,2,1])
                try: current_attr_idx = queryable_stats_filter_ordered_tf.index(item['attribute'])
                except ValueError: current_attr_idx = 0 
                item['attribute'] = cols_f[0].selectbox("Atributo:", options=queryable_stats_filter_ordered_tf, index=current_attr_idx, key=f"f_attr_{fid}_v26")
                cond_opts = ['>=', '<=', '==', '>', '<']
                try: current_cond_idx = cond_opts.index(item['condition'])
                except ValueError: current_cond_idx = 0
                item['condition'] = cols_f[1].selectbox("Condición:", options=cond_opts, index=current_cond_idx, key=f"f_cond_{fid}_v26")
                item['value'] = cols_f[2].number_input("Valor:", value=int(item['value']), step=1, key=f"f_val_{fid}_v26")
                if cols_f[3].button("➖", key=f"f_rem_{fid}_v26", help="Eliminar"): remove_filter_cb_v26(fid); st.rerun() 
        
        if st.button("Aplicar Filtros (Stats Base)", key="btn_apply_adv_f_v26_tab"):
            if not st.session_state.filters: st.info("No hay criterios definidos.")
            else:
                if 'AcceleRATE' not in all_stats_df_base.columns:
                    all_stats_df_base['AcceleRATE'] = all_stats_df_base.apply(lambda row: determinar_estilo_carrera(row['Altura'], row.get('AGI',0),row.get('STR',0),row.get('Acc',0)), axis=1)
                df_adv_f = all_stats_df_base.copy(); valid_q = True; act_f_attrs = []
                for f_app in st.session_state.filters:
                    attr, cond, val = f_app['attribute'], f_app['condition'], f_app['value']
                    act_f_attrs.append(attr)
                    if attr not in df_adv_f.columns: st.error(f"Atributo '{attr}' no encontrado."); valid_q = False; break
                    if cond == '>=': df_adv_f = df_adv_f[df_adv_f[attr] >= val]
                    elif cond == '<=': df_adv_f = df_adv_f[df_adv_f[attr] <= val]
                    # ... (otras condiciones)
                    elif cond == '==': df_adv_f = df_adv_f[df_adv_f[attr] == val]
                    elif cond == '>': df_adv_f = df_adv_f[df_adv_f[attr] > val]
                    elif cond == '<': df_adv_f = df_adv_f[df_adv_f[attr] < val]
                if valid_q:
                    if not df_adv_f.empty:
                        st.write(f"Combinaciones ({len(df_adv_f)}):")
                        cols_disp = ['Posicion', 'Altura', 'Peso'] + [col for col in queryable_stats_filter_ordered_tf if col in act_f_attrs] + ['AcceleRATE', 'IGS']
                        final_cols_disp = []; [final_cols_disp.append(col) for col in cols_disp if col not in final_cols_disp]
                        final_cols_disp_existing = [col for col in final_cols_disp if col in df_adv_f.columns]
                        st.dataframe(df_adv_f[final_cols_disp_existing])
                    else: st.info("Ninguna combinación cumple criterios.")
    else: st.warning("Datos para filtros no disponibles.")