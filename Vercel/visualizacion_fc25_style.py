# Visualización del árbol de habilidades estilo FC25
import streamlit as st
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go

def create_hexagonal_tree_layout(df_trees, tree_name, unlocked_nodes):
    """
    Crea un layout hexagonal similar al juego FC25
    """
    tree_data = df_trees[df_trees['Arbol'] == tree_name].copy()
    
    if tree_data.empty:
        st.info(f"No hay nodos en el árbol {tree_name}")
        return
    
    # Agrupar nodos por ramas y niveles
    branches = organize_nodes_by_branches(tree_data)
    
    # Mostrar estadísticas del árbol en métricas compactas
    show_tree_metrics(tree_data, unlocked_nodes, tree_name)
    
    # Crear el layout hexagonal
    create_hexagonal_grid(branches, tree_data, unlocked_nodes)

def organize_nodes_by_branches(tree_data):
    """
    Organiza los nodos por Sub_Arbol y Nivel usando las nuevas columnas del CSV
    """
    branches = defaultdict(lambda: defaultdict(list))
    
    for _, node in tree_data.iterrows():
        node_id = node['ID_Nodo']
        
        # Usar Sub_Arbol como clave principal
        sub_arbol = str(node.get('Sub_Arbol', ''))
        if sub_arbol and sub_arbol != '' and sub_arbol != 'nan':
            branch_key = sub_arbol
        else:
            # Fallback: usar nomenclatura original (A, B, C)
            parts = node_id.split('_')
            if len(parts) >= 2:
                branch_key = parts[1][0]  # A, B, C, etc.
            else:
                branch_key = "General"
        
        # Usar Nivel directamente del CSV
        level = int(node.get('Nivel', 1)) - 1  # Convertir a base 0
        
        branches[branch_key][level].append(node)
    
    return branches

def show_tree_metrics(tree_data, unlocked_nodes, tree_name):
    """
    Muestra métricas del árbol de forma compacta como en FC25
    """
    total_nodes = len(tree_data)
    unlocked_count = len([n for n in tree_data['ID_Nodo'] if n in unlocked_nodes])
    total_cost = tree_data['Costo'].sum()
    unlocked_cost = tree_data[tree_data['ID_Nodo'].isin(unlocked_nodes)]['Costo'].sum()
    progress = unlocked_count / total_nodes if total_nodes > 0 else 0
    
    # Estilo similar a FC25 con métricas horizontales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Progreso", f"{unlocked_count}/{total_nodes}", f"{progress:.0%}")
    with col2:
        st.metric("⚡ Puntos", f"{unlocked_cost}/{total_cost}")
    with col3:
        st.metric("🌳 Ramas", len(set([node['ID_Nodo'].split('_')[1][0] for _, node in tree_data.iterrows()])))
    with col4:
        remaining_points = st.session_state.get('bc_points_remaining', 0)
        st.metric("💎 Disponibles", f"{remaining_points}")
    
    # Barra de progreso estilo FC25
    st.progress(progress, f"🎯 {tree_name}")

def create_hexagonal_grid(branches, tree_data, unlocked_nodes):
    """
    Crea un grid hexagonal similar al diseño de FC25 con mejor diferenciación de sub-árboles
    """    # Obtener el número máximo de niveles
    max_level = 0
    for branch_nodes in branches.values():
        if branch_nodes:
            max_level = max(max_level, *branch_nodes.keys())
      # Definir colores y estilos por sub-árbol para mejor diferenciación visual - MEJORADO
    subtree_styles = {
        'A': {'color': '🔴', 'name': 'RAMA ALFA', 'bg_color': '#ff4b4b15', 'border_color': '#ff4b4b80'},
        'B': {'color': '🔵', 'name': 'RAMA BETA', 'bg_color': '#1f77b415', 'border_color': '#1f77b480'},  
        'C': {'color': '🟢', 'name': 'RAMA GAMMA', 'bg_color': '#00cc8815', 'border_color': '#00cc8880'},
        'D': {'color': '🟡', 'name': 'RAMA DELTA', 'bg_color': '#ffeb3b15', 'border_color': '#ffeb3b80'},
        'E': {'color': '🟣', 'name': 'RAMA EPSILON', 'bg_color': '#9c27b015', 'border_color': '#9c27b080'},
        'F': {'color': '🟠', 'name': 'RAMA ZETA', 'bg_color': '#ff970015', 'border_color': '#ff970080'},
        'G': {'color': '⚫', 'name': 'RAMA OMEGA', 'bg_color': '#42424215', 'border_color': '#42424280'},
        # Mapeo para nombres específicos de sub-árboles - MEJORADO
        'REMATE': {'color': '⚽', 'name': 'ESPECIALIDAD REMATE', 'bg_color': '#ff4b4b15', 'border_color': '#ff4b4b'},
        'POTENCIA': {'color': '💥', 'name': 'ESPECIALIDAD POTENCIA', 'bg_color': '#ff970015', 'border_color': '#ff9700'},
        'TIROS_LIBRES': {'color': '🎯', 'name': 'ESPECIALIDAD TIROS LIBRES', 'bg_color': '#1f77b415', 'border_color': '#1f77b4'},
        'PENALES': {'color': '⚡', 'name': 'ESPECIALIDAD PENALES', 'bg_color': '#9c27b015', 'border_color': '#9c27b0'},
        'PASE': {'color': '🎱', 'name': 'ESPECIALIDAD PASE', 'bg_color': '#00cc8815', 'border_color': '#00cc88'},
        'DEFENSA': {'color': '🛡️', 'name': 'ESPECIALIDAD DEFENSA', 'bg_color': '#42424215', 'border_color': '#424242'},
        'FISICO': {'color': '💪', 'name': 'ESPECIALIDAD FÍSICO', 'bg_color': '#ffeb3b15', 'border_color': '#ffeb3b'},
        'RITMO': {'color': '💨', 'name': 'RITMO', 'bg_color': '#00bcd415', 'border_color': '#00bcd4'},
        'DRIBBLING': {'color': '✨', 'name': 'DRIBBLING', 'bg_color': '#e91e6315', 'border_color': '#e91e63'},
        'GENERAL': {'color': '⭐', 'name': 'HABILIDADES GENERALES', 'bg_color': '#60606015', 'border_color': '#606060'}
    }
    
    # Organizar por niveles (como en las imágenes del juego)
    for level in range(max_level + 1):
        # Título del nivel con estilo FC25
        if level == 0:
            st.markdown("### 🔰 **NIVEL BASE** ###")
        else:
            st.markdown(f"### ⚡ **NIVEL {level + 1}** ###")
        
        # Recopilar todos los nodos de este nivel organizados por sub-árbol
        level_by_subtree = {}
        for branch_key in sorted(branches.keys()):
            if level in branches[branch_key]:
                level_by_subtree[branch_key] = branches[branch_key][level]
        
        if level_by_subtree:
            # Crear columnas para los sub-árboles del nivel
            subtree_cols = st.columns(len(level_by_subtree))
            
            for idx, (subtree_key, subtree_nodes) in enumerate(sorted(level_by_subtree.items())):
                with subtree_cols[idx]:
                    if subtree_nodes:                        # Obtener estilo del sub-árbol
                        style = subtree_styles.get(subtree_key, {
                            'color': '⚪', 
                            'name': f'SUB-ÁRBOL {subtree_key}', 
                            'bg_color': '#f0f0f015',
                            'border_color': '#f0f0f080'
                        })
                        
                        # Container con fondo de color para distinguir visualmente - MEJORADO
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {style['bg_color']}, {style['bg_color'].replace('15', '10')}); 
                            border-radius: 15px; 
                            padding: 20px; 
                            margin: 8px;
                            border: 3px solid {style['border_color']};
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            transition: all 0.3s ease;
                        ">
                        """, unsafe_allow_html=True)
                        
                        # Identificador visual del sub-árbol con mejor formato
                        st.markdown(f"""
                        <h4 style="
                            text-align: center; 
                            margin: 0 0 15px 0; 
                            color: {style['border_color']};
                            font-weight: bold;
                            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                        ">
                            {style['color']} {style['name']}
                        </h4>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar cada nodo del sub-árbol
                        for node in subtree_nodes:
                            show_hexagonal_skill_node(node, unlocked_nodes, tree_data, subtree_style=style)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Separador entre niveles
        if level < max_level:
            st.markdown("")
            st.markdown("---")
            st.markdown("")

def show_hexagonal_skill_node(node, unlocked_nodes, tree_data, subtree_style=None):
    """
    Muestra un nodo de habilidad con estilo hexagonal como FC25
    """
    node_id = node['ID_Nodo']
    is_unlocked = node_id in unlocked_nodes
    can_unlock = can_unlock_node(node_id, unlocked_nodes, tree_data)
      # Determinar el estilo del nodo
    if is_unlocked:
        icon = "✅"
        disabled = True
        help_text = "✅ Nodo completado"
    elif can_unlock:
        icon = "🔓" 
        disabled = False
        help_text = f"💰 Costo: {node['Costo']} puntos - ¡Haz clic para desbloquear!"
    else:
        icon = "🔒"
        disabled = True
        prereqs = get_prerequisite_names(node, tree_data)
        help_text = f"🔒 Prerequisitos: {prereqs}" if prereqs else "🔒 Bloqueado"
    
    # Crear container hexagonal estilo FC25
    with st.container():
        # Título del nodo con icono y estilo del sub-árbol
        if subtree_style:
            node_title = f"{subtree_style['color']} {icon} **{node['Nombre_Visible']}**"
        else:
            node_title = f"{icon} **{node['Nombre_Visible']}**"
        
        # Botón principal con estilo hexagonal más compacto
        button_style = "primary" if is_unlocked else ("secondary" if disabled else "primary")
        
        if st.button(
            node_title,
            key=f"hex_node_{node_id}",
            disabled=disabled,
            help=help_text,
            use_container_width=True,
            type=button_style
        ):
            if can_unlock and not is_unlocked:
                handle_node_unlock(node_id, node['Costo'])
        
        # Botón de devolver para nodos desbloqueados (más compacto)
        if is_unlocked:
            if st.button(
                "↩️ Devolver",
                key=f"return_hex_node_{node_id}",
                help="Devolver puntos y desbloquear este nodo",
                use_container_width=True,
                type="secondary"
            ):
                handle_node_return(node_id, node['Costo'], tree_data)
        
        # Información compacta del nodo
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.caption(f"💎 {node['Costo']} pts")
        
        # Eliminar el nombre de la sub-rama en la ficha del nodo (antes estaba aquí):
        # with col_info2:
        #     if subtree_style:
        #         st.caption(f"{subtree_style['color']} {subtreeStyle['name'][:8]}...")
        #     else:
        #         branch = node_id.split('_')[1][0] if '_' in node_id else "?"
        #         st.caption(f"🌿 Rama {branch}")
        
        # Beneficios del nodo (compacto)
        benefits = get_node_benefits_compact(node)
        if benefits:
            st.caption(f"⚡ {benefits}")
        
        # PlayStyle si existe
        if 'PlayStyle' in node and pd.notna(node['PlayStyle']) and node['PlayStyle'] != '':
            st.caption(f"🎭 **{node['PlayStyle']}**")
        
        # Beneficios del nodo (compacto)
        benefits = get_node_benefits_compact(node)
        if benefits:
            st.caption(f"⚡ {benefits}")
        
        # PlayStyle si existe
        if 'PlayStyle' in node and pd.notna(node['PlayStyle']) and node['PlayStyle'] != '':
            st.caption(f"🎭 **{node['PlayStyle']}**")

def get_node_benefits_compact(node):
    """
    Obtiene los beneficios de un nodo de forma compacta
    """
    benefits = []
    # Mostrar todos los atributos que suma el nodo, no solo los "importantes"
    for col in node.index:
        if col in ['ID_Nodo', 'Arbol', 'Sub_Arbol', 'Nivel', 'Nombre_Visible', 'Costo', 'Prerrequisito', 'PlayStyle', 'Es_Arquetipo', 'Notas', 'Puntos_Req_Arbol']:
            continue
        if pd.notna(node[col]) and isinstance(node[col], (int, float)) and node[col] != 0:
            if col == 'PIERNA_MALA':
                benefits.append(f"+{int(node[col])}⭐ WF")
            elif col == 'FILIGRANAS':
                benefits.append(f"+{int(node[col])}⭐ SM")
            else:
                benefits.append(f"+{int(node[col])} {col}")
    # Limitar a los 3 beneficios más importantes (opcional, puedes quitar el [:3] si quieres mostrar todos)
    return " | ".join(benefits) if benefits else ""

def get_node_position_in_tree(node_id, tree_data):
    """
    Determina la posición de un nodo en el árbol basándose en las nuevas columnas
    Sub_Arbol y Nivel del CSV actualizado
    """
    node = tree_data[tree_data['ID_Nodo'] == node_id].iloc[0]
    
    # Usar la columna Nivel directamente (ya viene definida en el CSV)
    level = int(node.get('Nivel', 1)) - 1  # Convertir a base 0 (nivel 1 -> posición 0)
    
    # Posición horizontal basada en Sub_Arbol
    sub_arbol = str(node.get('Sub_Arbol', ''))
    
    if sub_arbol and sub_arbol != '' and sub_arbol != 'nan':
        # Si tiene sub_arbol definido, usar eso para posición horizontal
        # Crear un mapeo de sub_arboles conocidos a posiciones
        sub_arbol_positions = {
            'REMATE': 0,
            'POTENCIA': 1,
            'TIROS_LIBRES': 2,
            'PENALES': 3,
            # Añadir más según aparezcan en el CSV
        }
        branch_order = sub_arbol_positions.get(sub_arbol, 0)
    else:
        # Fallback: usar la nomenclatura original (A, B, C)
        node_name = node_id.split('_')
        if len(node_name) >= 2:
            branch = node_name[1][0]  # A, B, C, etc.
            branch_order = ord(branch) - ord('A')
        else:
            branch_order = 0
    
    return level, branch_order

def create_fc25_tree_layout(df_trees, selected_tree, unlocked_nodes):
    """
    Crea un layout estilo FC25 del árbol de habilidades
    """
    if selected_tree == "Todos":
        tree_data = df_trees
        trees = df_trees['Arbol'].unique()
    else:
        tree_data = df_trees[df_trees['Arbol'] == selected_tree]
        trees = [selected_tree]
    
    st.markdown(f"### 🌟 Árbol de Habilidades: {selected_tree}")
    
    # Crear contenedores para cada árbol
    for tree in trees:
        if selected_tree == "Todos":
            with st.expander(f"🎯 {tree}", expanded=False):
                show_tree_grid(df_trees, tree, unlocked_nodes)
        else:
            show_tree_grid(tree_data, tree, unlocked_nodes)

def show_tree_grid(df_trees, tree_name, unlocked_nodes):
    """
    Muestra el árbol en formato de grilla estilo FC25
    """
    tree_data = df_trees[df_trees['Arbol'] == tree_name].copy()
    
    if tree_data.empty:
        st.info(f"No hay nodos en el árbol {tree_name}")
        return
    
    # Agrupar nodos por ramas (A, B, C, etc.)
    branches = defaultdict(list)
    
    for _, node in tree_data.iterrows():
        node_id = node['ID_Nodo']
        parts = node_id.split('_')
        if len(parts) >= 2:
            branch_key = parts[1][0]  # A, B, C, etc.
            level, _ = get_node_position_in_tree(node_id, tree_data)
            branches[branch_key].append((level, node))
    
    # Ordenar ramas y nodos dentro de cada rama
    sorted_branches = sorted(branches.keys())
    
    # Mostrar estadísticas del árbol
    total_nodes = len(tree_data)
    unlocked_count = len([n for n in tree_data['ID_Nodo'] if n in unlocked_nodes])
    total_cost = tree_data['Costo'].sum()
    unlocked_cost = tree_data[tree_data['ID_Nodo'].isin(unlocked_nodes)]['Costo'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodos Desbloqueados", f"{unlocked_count}/{total_nodes}")
    with col2:
        st.metric("Puntos Gastados", f"{unlocked_cost}/{total_cost}")
    with col3:
        progress = unlocked_count / total_nodes if total_nodes > 0 else 0
        st.metric("Progreso", f"{progress:.1%}")
    with col4:
        st.metric("Ramas", len(sorted_branches))
    
    # Mostrar progreso visual
    if total_nodes > 0:
        st.progress(progress, f"Progreso del árbol {tree_name}")
    
    st.divider()
    
    # Crear grid de habilidades estilo FC25
    max_level = max([max([level for level, _ in nodes]) for nodes in branches.values()]) if branches else 0
    
    # Crear layout de grid
    for level in range(max_level + 1):
        st.markdown(f"**Nivel {level + 1}**")
        level_nodes = []
        
        for branch in sorted_branches:
            branch_nodes = [node for lv, node in branches[branch] if lv == level]
            level_nodes.extend(branch_nodes)
        
        if level_nodes:
            # Mostrar nodos de este nivel en columnas
            cols = st.columns(min(len(level_nodes), 4))
            
            for i, node in enumerate(level_nodes):
                with cols[i % len(cols)]:
                    show_skill_node_fc25(node, unlocked_nodes, tree_data)
        
        if level < max_level:
            st.markdown("---")

def show_skill_node_fc25(node, unlocked_nodes, tree_data):
    """
    Muestra un nodo de habilidad con estilo FC25
    """
    node_id = node['ID_Nodo']
    is_unlocked = node_id in unlocked_nodes
    
    # Verificar si se puede desbloquear
    can_unlock = can_unlock_node(node_id, unlocked_nodes, tree_data)
    
    # Determinar estilo del botón
    if is_unlocked:
        button_style = "🟢"
        button_text = f"{button_style} {node['Nombre_Visible']}"
        disabled = True
        help_text = "Nodo desbloqueado"
    elif can_unlock:
        button_style = "🔵"
        button_text = f"{button_style} {node['Nombre_Visible']}"
        disabled = False
        help_text = f"Costo: {node['Costo']} puntos"
    else:
        button_style = "🔒"
        button_text = f"{button_style} {node['Nombre_Visible']}"
        disabled = True
        prereqs = get_prerequisite_names(node, tree_data)
        help_text = f"Prerrequisitos: {prereqs}" if prereqs else "Bloqueado"
    
    # Obtener información de beneficios
    benefits = get_node_benefits(node)
    
    # Crear container estilo FC25
    container = st.container()
    with container:
        # Botón principal
        clicked = st.button(
            button_text,
            key=f"fc25_node_{node_id}",
            disabled=disabled,
            help=help_text,
            use_container_width=True
        )
        
        # Información adicional
        if benefits:
            st.caption(f"✨ {benefits}")
        
        st.caption(f"💰 Costo: {node['Costo']} pts")
        
        # PlayStyle si existe
        if 'PlayStyle' in node and pd.notna(node['PlayStyle']) and node['PlayStyle'] != '':
            st.caption(f"🎭 {node['PlayStyle']}")
        
        # Manejar clic del botón
        if clicked and can_unlock and not is_unlocked:
            handle_node_unlock(node_id, node['Costo'])

def can_unlock_node(node_id, unlocked_nodes, tree_data):
    """
    Verifica si un nodo puede ser desbloqueado usando lógica OR (solo necesita UNO de los prerrequisitos)
    """
    node = tree_data[tree_data['ID_Nodo'] == node_id].iloc[0]
    prereqs = str(node.get('Prerrequisito', ''))
    
    # Si no hay prerequisitos, se puede desbloquear
    if not prereqs or prereqs == 'nan' or prereqs == '':
        return True
    
    # Verificar prerequisitos con lógica OR (solo necesita uno)
    prereq_list = [p.strip() for p in prereqs.split(',') if p.strip()]
    if not prereq_list:
        return True
    
    # CAMBIO: OR en lugar de AND - solo necesita UNO de los prerrequisitos
    for prereq in prereq_list:
        if prereq in unlocked_nodes:
            return True  # Si encuentra AL MENOS UNO desbloqueado, puede desbloquearse
    
    return False  # Si NO encuentra NINGUNO desbloqueado, no puede desbloquearse

def get_prerequisite_names(node, tree_data):
    """
    Obtiene los nombres de los prerequisitos
    """
    prereqs = str(node.get('Prerrequisito', ''))
    if not prereqs or prereqs == 'nan' or prereqs == '':
        return ""
    
    prereq_list = [p.strip() for p in prereqs.split(',') if p.strip()]
    names = []
    
    for prereq_id in prereq_list:
        prereq_node = tree_data[tree_data['ID_Nodo'] == prereq_id]
        if not prereq_node.empty:
            names.append(prereq_node.iloc[0]['Nombre_Visible'])
    
    return ", ".join(names)

def get_node_benefits(node):
    """
    Obtiene los beneficios de un nodo
    """
    benefits = []
    
    # Lista de columnas de stats
    stat_columns = ['Acc', 'Spr', 'Fin', 'FKA', 'HAcc', 'SPow', 'Lon S', 'Vol', 'Pen', 
                   'Vis', 'Cros', 'LP', 'SP', 'Cur', 'AGI', 'BAL', 'APOS', 'BCON', 
                   'REG', 'INT', 'AWA', 'STAN', 'SLID', 'JUMP', 'STA', 'STR', 'REA', 
                   'AGGR', 'COMP']
    
    for col in stat_columns:
        if col in node.index and pd.notna(node[col]) and node[col] > 0:
            benefits.append(f"+{int(node[col])} {col}")
    
    # Habilidades especiales
    if 'PIERNA_MALA' in node.index and pd.notna(node['PIERNA_MALA']) and node['PIERNA_MALA'] > 0:
        benefits.append(f"+{int(node['PIERNA_MALA'])}⭐ Pierna Mala")
    
    if 'FILIGRANAS' in node.index and pd.notna(node['FILIGRANAS']) and node['FILIGRANAS'] > 0:
        benefits.append(f"+{int(node['FILIGRANAS'])}⭐ Filigranas")
    
    return " | ".join(benefits)

def handle_node_unlock(node_id, cost):
    """
    Maneja el desbloqueo de un nodo
    """
    # Verificar si hay puntos suficientes usando la clave correcta del session_state
    remaining_points = st.session_state.get('bc_points_remaining', 0)
    if remaining_points >= cost:
        # Inicializar conjunto si no existe
        if 'bc_unlocked_nodes' not in st.session_state:
            st.session_state.bc_unlocked_nodes = set()
        
        st.session_state.bc_unlocked_nodes.add(node_id)
        st.session_state.bc_points_remaining = remaining_points - cost
        st.rerun()
    else:
        st.error(f"No tienes suficientes puntos. Necesitas {cost}, tienes {remaining_points}")

def handle_node_return(node_id, cost, tree_data):
    """
    Maneja la devolución de un nodo (devolver puntos y desbloquear) - Versión mejorada con confirmación
    """
    # Verificar dependencias antes de devolver
    dependent_nodes = []
    unlocked_nodes = st.session_state.get('bc_unlocked_nodes', set())
    
    if unlocked_nodes:
        for unlocked_node_id in unlocked_nodes:
            if unlocked_node_id == node_id: 
                continue
            
            # Buscar este nodo en tree_data
            node_data_series = tree_data[tree_data['ID_Nodo'] == unlocked_node_id]
            if not node_data_series.empty:
                node_data = node_data_series.iloc[0]
                prereqs_str = str(node_data.get('Prerrequisito', '')).strip()
                if prereqs_str:
                    prereq_list = [p.strip() for p in prereqs_str.split(',') if p.strip()]
                    if node_id in prereq_list:
                        dependent_nodes.append(node_data.get('Nombre_Visible', unlocked_node_id))
    
    # Si no hay dependencias, devolver el nodo
    if not dependent_nodes:
        if 'bc_unlocked_nodes' not in st.session_state:
            st.session_state.bc_unlocked_nodes = set()
        
        st.session_state.bc_unlocked_nodes.discard(node_id)
        remaining_points = st.session_state.get('bc_points_remaining', 0)
        st.session_state.bc_points_remaining = remaining_points + cost
        
        # Mostrar confirmación de devolución
        st.success(f"✅ Devueltos {cost} puntos. Total disponible: {remaining_points + cost}")
        st.rerun()
    else:
        nombres_dependencias = ", ".join(list(set(dependent_nodes)))
        st.error(f"❌ No se puede devolver. Nodos dependientes: {nombres_dependencias}")
        st.info("💡 Consejo: Devuelve primero los nodos que dependen de este.")

def create_interactive_path_view(df_trees, selected_tree, unlocked_nodes):
    """
    Crea una vista interactiva de rutas de progresión
    """
    st.markdown("### 🛤️ Rutas de Progresión Recomendadas")
    
    tree_data = df_trees[df_trees['Arbol'] == selected_tree] if selected_tree != "Todos" else df_trees
    
    # Crear diferentes rutas estratégicas
    routes = {
        "🏃 Enfoque Velocidad": ["RITMO_A1", "RITMO_B1", "RITMO_A2", "RITMO_B2", "RITMO_C1"],
        "⚽ Enfoque Tiro": ["TIRO_E1", "TIRO_E2", "TIRO_G1", "TIRO_G2", "TIRO_F1"],
        "🎯 Precisión": ["TIRO_A1", "TIRO_A2", "TIRO_C1", "TIRO_C2"],
        "💪 Físico": ["RITMO_A1", "RITMO_B1", "RITMO_C1"]
    }
    
    # Seleccionar ruta
    selected_route = st.selectbox("Selecciona una ruta recomendada:", list(routes.keys()))
    
    if selected_route:
        route_nodes = routes[selected_route]
        route_cost = 0
        route_progress = 0
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Ruta: {selected_route}**")
            # Puedes dejar la lógica aquí o simplemente poner pass si no la usas
            pass
        with col2:
            pass

def mostrar_visualizacion_fc25(df_skill_trees_global, arbol_sel, unlocked_nodes, df_instalaciones=None, unlocked_facilities=None, stats_base=None):
    """
    Función principal para mostrar la visualización estilo FC25 mejorada.
    """
    st.subheader("🎮 Árbol de Habilidades - Estilo FC25")
    # Eliminar llamada y lógica de la tabla de atributos aquí
    if arbol_sel == "Todos":
        available_trees = df_skill_trees_global['Arbol'].unique()
        if len(available_trees) > 1:
            tabs = st.tabs([f"🌳 {tree}" for tree in available_trees])
            for i, tree in enumerate(available_trees):
                with tabs[i]:
                    create_hexagonal_tree_layout(df_skill_trees_global, tree, unlocked_nodes)
        else:
            create_hexagonal_tree_layout(df_skill_trees_global, available_trees[0], unlocked_nodes)
    else:
        create_hexagonal_tree_layout(df_skill_trees_global, arbol_sel, unlocked_nodes)
