import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Calculadora Stats FC25 v2.3", layout="wide")

# --- Carga y Preparación de Datos (Función Principal) ---
@st.cache_data
def cargar_y_preparar_datos():
    file_altura = "Datos FC25 - ALTURA.csv"
    file_peso = "Datos FC25 - PESO.csv"
    file_posiciones = "Datos FC25 - POSICIONES.csv"
    try:
        df_altura_raw = pd.read_csv(file_altura)
        df_peso_raw = pd.read_csv(file_peso)
        df_posiciones_raw = pd.read_csv(file_posiciones)
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos CSV: {e}.")
        return None, None, None, None, None, None
    except Exception as e_gen:
        st.error(f"Error inesperado al cargar CSVs: {e_gen}")
        return None, None, None, None, None, None
    try:
        df_altura = df_altura_raw.set_index('Altura')
        # IMPORTANTE: Se eliminó la línea df_peso.rename(). 
        # Asegúrate que tus CSVs (especialmente PESO.csv) ya tengan los nombres de columna consistentes
        # con POSICIONES.csv (ej. 'DRI' para categoría, 'REG' para habilidad individual si así lo definiste).
        df_peso = df_peso_raw.set_index('Peso')
        df_posiciones = df_posiciones_raw.set_index('Posicion')

        expected_cols = df_posiciones.columns.tolist()
        
        for df, name in [(df_altura, "ALTURA"), (df_peso, "PESO")]:
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Advertencia: Columnas faltantes en {name}.csv (se rellenarán con 0): {missing_cols}.")
        df_altura = df_altura.reindex(columns=expected_cols).fillna(0).astype(int)
        df_peso = df_peso.reindex(columns=expected_cols).fillna(0).astype(int)
        df_posiciones = df_posiciones[expected_cols].fillna(0).astype(int) # Asegurar que posiciones también sea int

        altura_base_val, peso_base_val, posicion_base_val = 162, 45, 'LB/RB'
        if posicion_base_val not in df_posiciones.index: st.error(f"Posición base '{posicion_base_val}' no encontrada."); return None, None, None, None, None, None
        stats_base_lb_rb = df_posiciones.loc[posicion_base_val]
        if altura_base_val not in df_altura.index: st.error(f"Altura base '{altura_base_val}cm' no encontrada."); return None, None, None, None, None, None
        stats_altura_base_ref = df_altura.loc[altura_base_val]
        if peso_base_val not in df_peso.index: st.error(f"Peso base '{peso_base_val}kg' no encontrado."); return None, None, None, None, None, None
        stats_peso_base_ref = df_peso.loc[peso_base_val]

        modificadores_altura = df_altura.subtract(stats_altura_base_ref, axis=1)
        modificadores_peso = df_peso.subtract(stats_peso_base_ref, axis=1)
        diferenciales_posicion = df_posiciones.subtract(stats_base_lb_rb, axis=1)
        
        return stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, df_posiciones.index.tolist(), expected_cols
    except KeyError as ke: st.error(f"Error de clave (KeyError): '{ke}'. Verifica nombres de columna en CSVs."); return None, None, None, None, None, None
    except Exception as e_proc: st.error(f"Error al procesar DataFrames: {e_proc}"); return None, None, None, None, None, None

# --- Función de Cálculo de Estadísticas Individuales ---
def calcular_stats_finales(posicion_sel, altura_sel, peso_sel,
                           stats_base_ref, mod_altura_df, mod_peso_df, diff_pos_df):
    # ... (igual que antes) ...
    diferencial_pos = diff_pos_df.loc[posicion_sel]
    modificador_alt = mod_altura_df.loc[altura_sel]
    modificador_pes = mod_peso_df.loc[peso_sel]
    stats_finales = stats_base_ref.add(diferencial_pos).add(modificador_alt).add(modificador_pes)
    return stats_finales.round().astype(int)


# --- Pre-cálculo de todas las combinaciones ---
@st.cache_data
def precompute_all_stats(_stats_base_lb_rb, _mod_altura_df, _mod_peso_df, _diff_pos_df, _stat_cols_order):
    # ... (igual que antes, asegurando que los df no sean None) ...
    all_data = []
    if not all([isinstance(df, pd.DataFrame) for df in [_mod_altura_df, _mod_peso_df, _diff_pos_df]]):
         st.error("Error: datos base para precompute_all_stats no son DataFrames válidos.")
         return pd.DataFrame()
    if not isinstance(_stats_base_lb_rb, pd.Series):
        st.error("Error: stats_base_lb_rb no es una Series válida.")
        return pd.DataFrame()

    for pos in _diff_pos_df.index:
        for alt in _mod_altura_df.index:
            for pes in _mod_peso_df.index:
                stats = calcular_stats_finales(pos, alt, pes, _stats_base_lb_rb, _mod_altura_df, _mod_peso_df, _diff_pos_df)
                entry = {'Posicion': pos, 'Altura': alt, 'Peso': pes, **stats.to_dict()}
                all_data.append(entry)
    df = pd.DataFrame(all_data)
    if not df.empty:
        valid_cols = [col for col in _stat_cols_order if col in df.columns]
        df = df[['Posicion', 'Altura', 'Peso'] + valid_cols]
    return df

# --- Carga de Datos Principal ---
APP_VERSION = "v2.3" 
# ... (lógica de carga y session_state igual que antes) ...
if 'app_version' not in st.session_state or st.session_state.app_version != APP_VERSION:
    st.session_state.clear(); st.session_state.app_version = APP_VERSION

carga_exitosa = False
datos_cargados = cargar_y_preparar_datos()
if datos_cargados and all(d is not None for d in datos_cargados):
    stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, lista_posiciones, stat_cols_order = datos_cargados
    all_stats_df = precompute_all_stats(stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion, stat_cols_order)
    if not all_stats_df.empty: carga_exitosa = True
    else: st.error("No se pudieron pre-calcular todas las estadísticas.")
else: st.error("Fallo en la carga o preparación de datos iniciales.")

if carga_exitosa and 'filters' not in st.session_state: st.session_state.filters = []
if carga_exitosa and 'next_filter_id' not in st.session_state: st.session_state.next_filter_id = 0


# --- Interfaz de Usuario con Streamlit ---
st.title(f"Calculadora Avanzada de Estadísticas FC25 ({APP_VERSION})")
if not carga_exitosa: st.warning("Aplicación no inicializada correctamente."); st.stop()

# Definir colores para los jugadores
PLAYER_COLORS = ['rgba(0,100,255,0.6)', 'rgba(255,0,0,0.6)', 'rgba(0,180,0,0.6)'] # Azul, Rojo, Verde con opacidad

# Función para resaltar el máximo en una fila de la tabla comparativa
def highlight_max_in_row(row):
    # Comprobar si todos los valores son NaN o si la fila está vacía (no debería pasar con int stats)
    if row.isnull().all():
        return ['' for _ in row]
    
    max_val = row.max()
    # Asegurarse de que solo se aplica el estilo si el valor no es NaN (aunque nuestras stats son int)
    return ['background-color: #d4edda; color: #155724; font-weight: bold;' if (v == max_val and pd.notna(v)) else '' for v in row]
    # Este es un verde pastel más suave (#d4edda) con texto verde oscuro (#155724)
    # Alternativa: 'background-color: palegreen; color: black; font-weight: bold;'

tab_calc, tab_best_combo, tab_filters = st.tabs(["🧮 Calculadora y Comparador", "🔍 Búsqueda Óptima", "📊 Filtros Múltiples"])

with tab_calc:
    st.header("Calculadora de Estadísticas y Comparador de Jugadores")
    
    num_players_to_compare = st.radio("Número de jugadores a definir/comparar:", (1, 2, 3), index=0, horizontal=True, key="num_players_radio")

    cols_selectors = st.columns(num_players_to_compare)
    player_stats_list = []
    player_names = []

    for i in range(num_players_to_compare):
        with cols_selectors[i]:
            st.subheader(f"Jugador {chr(65+i)}") # Jugador A, B, C
            pos = st.selectbox(f"Posición ({chr(65+i)}):", options=sorted(lista_posiciones), key=f"pos_p{i}")
            alt = st.selectbox(f"Altura ({chr(65+i)}):", options=sorted(modificadores_altura.index.unique().tolist()), key=f"alt_p{i}")
            pes = st.selectbox(f"Peso ({chr(65+i)}):", options=sorted(modificadores_peso.index.unique().tolist()), key=f"pes_p{i}")
            
            if pos and alt is not None and pes is not None:
                stats = calcular_stats_finales(pos, alt, pes, stats_base_lb_rb, modificadores_altura, modificadores_peso, diferenciales_posicion)
                player_stats_list.append(stats)
                player_names.append(f"Jugador {chr(65+i)} ({pos}, {alt}cm, {pes}kg)")
            else:
                player_stats_list.append(None) # Placeholder si no se seleccionó todo
                player_names.append(f"Jugador {chr(65+i)}")


    st.divider()
    st.subheader("Perfiles y Estadísticas")

    if any(ps is not None for ps in player_stats_list): # Si al menos un jugador está definido
        # Gráfico de Radar
        fig_radar_comp = go.Figure()
        radar_attrs = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'] # Asegúrate que estos existan
        
        for i, stats in enumerate(player_stats_list):
            if stats is not None:
                valid_radar_attrs = [attr for attr in radar_attrs if attr in stats.index]
                radar_values = stats[valid_radar_attrs].values.flatten().tolist()
                if len(valid_radar_attrs) >= 3:
                    fig_radar_comp.add_trace(go.Scatterpolar(
                        r=radar_values,
                        theta=valid_radar_attrs,
                        fill='toself',
                        name=player_names[i],
                        line_color=PLAYER_COLORS[i % len(PLAYER_COLORS)], # Ciclar colores
                        fillcolor=PLAYER_COLORS[i % len(PLAYER_COLORS)].replace('0.6', '0.3') # Opacidad menor para relleno
                    ))
        
        if fig_radar_comp.data: # Si se añadió al menos una traza
            fig_radar_comp.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Comparativa de Perfiles Principales",
                legend_title_text='Jugadores'
            )
            st.plotly_chart(fig_radar_comp, use_container_width=True)
        else:
            st.info("Define al menos un jugador para ver su perfil de radar.")

        # Tablas de Estadísticas Detalladas y Comparación
        if num_players_to_compare > 1 and all(ps is not None for ps in player_stats_list):
            # Crear DataFrame para comparación
            compare_dict = {player_names[i]: player_stats_list[i] for i in range(num_players_to_compare) if player_stats_list[i] is not None}
            if len(compare_dict) > 1 : # Solo mostrar si hay al menos dos para comparar
                df_compare = pd.DataFrame(compare_dict)
                df_compare_styled = df_compare.style.apply(highlight_max_in_row, axis=1)
                st.subheader("Tabla Comparativa de Estadísticas")
                st.dataframe(df_compare_styled)
        elif player_stats_list[0] is not None: # Si solo hay un jugador, mostrar su tabla
            st.subheader(f"Estadísticas Detalladas: {player_names[0]}")
            st.dataframe(player_stats_list[0].rename("Valor"))
        
        # Mover Top 5 IGS aquí
        st.divider()
        st.header("🏆 Top 5 Combinaciones por IGS (Global)")
        if not all_stats_df.empty and 'IGS' in all_stats_df.columns:
            top_5_igs = all_stats_df.sort_values(by='IGS', ascending=False).head(5)
            st.dataframe(top_5_igs[['Posicion', 'Altura', 'Peso', 'IGS']])
        else:
            st.warning("Datos para el Top 5 no disponibles o columna 'IGS' no encontrada.")

    else:
        st.info("Define los parámetros de al menos un jugador para ver sus estadísticas y perfil.")


with tab_best_combo:
    st.header("Búsqueda de Mejor Combinación por Atributos (Priorizados)")
    if not all_stats_df.empty:
        sortable_stats = [col for col in all_stats_df.columns if col not in ['Posicion', 'Altura', 'Peso']]
        
        col1_bc, col2_bc, col3_bc = st.columns(3)
        with col1_bc: attr_pri = st.selectbox("1er Atributo:", options=stat_cols_order, key="attr_pri_v2.3") # Usar orden original
        with col2_bc: attr_sec = st.selectbox("2do Atributo:", options=["(Ninguno)"] + stat_cols_order, key="attr_sec_v2.3")
        with col3_bc: attr_ter = st.selectbox("3er Atributo:", options=["(Ninguno)"] + stat_cols_order, key="attr_ter_v2.3")

        if st.button("Buscar Mejor Combinación Priorizada", key="btn_multi_attr_find_v2.3"):
            # ... (lógica de búsqueda igual que antes, pero el display cambiará) ...
            temp_df = all_stats_df.copy(); sort_by_attrs = [attr_pri]
            if attr_sec != "(Ninguno)": sort_by_attrs.append(attr_sec)
            if attr_ter != "(Ninguno)": sort_by_attrs.append(attr_ter)
            temp_df = temp_df.sort_values(by=sort_by_attrs, ascending=[False]*len(sort_by_attrs))
            if not temp_df.empty:
                best_player = temp_df.iloc[0]
                st.subheader(f"Mejor Jugador Encontrado: {best_player['Posicion']} | {best_player['Altura']}cm | {best_player['Peso']}kg")
                
                cols_card = st.columns(3) # Para mostrar métricas principales
                key_metrics = {'IGS': best_player.get('IGS', 'N/A'), 
                               attr_pri: best_player.get(attr_pri, 'N/A')}
                if attr_sec != "(Ninguno)": key_metrics[attr_sec] = best_player.get(attr_sec, 'N/A')
                if attr_ter != "(Ninguno)": key_metrics[attr_ter] = best_player.get(attr_ter, 'N/A')

                i = 0
                for k, v in key_metrics.items():
                    cols_card[i % 3].metric(label=str(k), value=str(v))
                    i += 1
                
                with st.expander("Ver todos los atributos del mejor jugador"):
                    st.json(best_player.to_dict())
            else: st.info("No se encontraron combinaciones.")
    else: st.warning("Datos para las búsquedas no disponibles.")


with tab_filters:
    st.header("Filtros Múltiples Avanzados")
    if not all_stats_df.empty:
        # Usar stat_cols_order para mantener el orden original en el selector
        queryable_stats_filter_ordered = [col for col in stat_cols_order if col not in ['Posicion', 'Altura', 'Peso', 'IGS']] # IGS es usualmente un resultado
        
        def add_filter_cb():
            fid = st.session_state.next_filter_id
            st.session_state.filters.append({'id': fid, 'attribute': queryable_stats_filter_ordered[0], 'condition': '>=', 'value': 70})
            st.session_state.next_filter_id += 1
        def remove_filter_cb(fid_to_remove):
            st.session_state.filters = [f for f in st.session_state.filters if f['id'] != fid_to_remove]

        st.button("➕ Añadir Criterio", on_click=add_filter_cb, key="add_filter_btn_v2.3")
        
        filter_cont = st.container()
        with filter_cont:
            for item in st.session_state.filters:
                fid = item['id']; cols_f = st.columns([5,3,2,1]) # Ajustar ratios
                
                try: current_attr_idx = queryable_stats_filter_ordered.index(item['attribute'])
                except ValueError: current_attr_idx = 0 
                item['attribute'] = cols_f[0].selectbox("Atributo:", options=queryable_stats_filter_ordered, index=current_attr_idx, key=f"f_attr_{fid}")
                
                cond_opts = ['>=', '<=', '==', '>', '<']
                try: current_cond_idx = cond_opts.index(item['condition'])
                except ValueError: current_cond_idx = 0
                item['condition'] = cols_f[1].selectbox("Condición:", options=cond_opts, index=current_cond_idx, key=f"f_cond_{fid}")
                item['value'] = cols_f[2].number_input("Valor:", value=int(item['value']), step=1, key=f"f_val_{fid}")
                if cols_f[3].button("➖", key=f"f_rem_{fid}", help="Eliminar"): remove_filter_cb(fid); st.rerun() 
        
        if st.button("Aplicar Filtros", key="btn_apply_adv_f_v2.3"):
            # ... (lógica de aplicar filtros igual que antes, usando queryable_stats_filter_ordered para las columnas a mostrar si es necesario) ...
            if not st.session_state.filters: st.info("No hay criterios definidos.")
            else:
                df_adv_f = all_stats_df.copy(); valid_q = True; act_f_attrs = []
                for f_app in st.session_state.filters:
                    attr, cond, val = f_app['attribute'], f_app['condition'], f_app['value']
                    act_f_attrs.append(attr)
                    if attr not in df_adv_f.columns: st.error(f"Atributo '{attr}' no encontrado."); valid_q = False; break
                    if cond == '>=': df_adv_f = df_adv_f[df_adv_f[attr] >= val]
                    elif cond == '<=': df_adv_f = df_adv_f[df_adv_f[attr] <= val]
                    elif cond == '==': df_adv_f = df_adv_f[df_adv_f[attr] == val]
                    elif cond == '>': df_adv_f = df_adv_f[df_adv_f[attr] > val]
                    elif cond == '<': df_adv_f = df_adv_f[df_adv_f[attr] < val]
                if valid_q:
                    if not df_adv_f.empty:
                        st.write(f"Combinaciones ({len(df_adv_f)}):")
                        cols_disp = ['Posicion', 'Altura', 'Peso'] + [attr for attr in queryable_stats_filter_ordered if attr in act_f_attrs] + ['IGS']
                        final_cols_disp = []; [final_cols_disp.append(col) for col in cols_disp if col not in final_cols_disp]
                        # Asegurar que las columnas existan antes de mostrarlas
                        final_cols_disp_existing = [col for col in final_cols_disp if col in df_adv_f.columns]
                        st.dataframe(df_adv_f[final_cols_disp_existing])

                    else: st.info("Ninguna combinación cumple criterios.")
    else: st.warning("Datos para filtros no disponibles.")