# ClubsStatsFC
Calculate the best combination of stats in Pro Clubs


# Change History

v1.x (Fase Inicial - No Streamlit):

    Carga de datos base de Altura, Peso y Posiciones desde CSVs.
    Cálculo de estadísticas base y modificadores/diferenciales.
    Lógica para combinar estos factores y obtener estadísticas finales.

v2.0 - v2.4 (Primeras Versiones Streamlit - Fundamentos):

    Introducción de la interfaz web con Streamlit.
    Calculadora individual interactiva (selección de Pos, Alt, Pes).
    Gráfico de radar para el perfil del jugador.
    Pre-cálculo de todas las combinaciones base para búsquedas.
    Pestañas para organizar: "Calculadora", "Top 5 IGS", "Búsqueda Óptima", "Filtros Múltiples".
    Funcionalidad "Enviar a Editor" desde diferentes secciones.
    Mejoras en la robustez de la carga de datos y manejo de errores.

v2.5 - v2.8 (Introducción y Refinamiento del "Build Craft"):

    Creación de la Pestaña "🛠️ Build Craft" (antes "Editor de Habilidades"):
        Carga del archivo CSV ARBOL DE HABILIDAD - ARBOL.csv con todos los nodos de habilidad.
        Sistema de puntos de habilidad (184 puntos).
        Lógica para desbloquear nodos, verificar prerrequisitos (condiciones OR) y restar costos.
        Cálculo y visualización de Pierna Mala y Filigranas (base + incrementos del árbol, tope 5 estrellas).
        Cálculo y visualización del estilo de carrera AcceleRATE.
        Tope de Atributos en 99.
        Recálculo de Atributos Generales (PAC, SHO, etc.): Ahora se calculan como promedio (redondeado hacia arriba) de sus sub-stats correspondientes después de aplicar los bonus de habilidad a las sub-stats.
        Recálculo de IGS: Ahora se calcula como el IGS base (de Pos/Alt/Pes) + la suma de todos los bonus directos que los nodos desbloqueados aportan a las sub-stats individuales.
        Visualización Dinámica de Stats por Árbol: Tabla que muestra el impacto de los nodos desbloqueados de un árbol específico sobre las stats base.
        Mejoras Visuales: Coloreado condicional para métricas de atributos generales, presentación de nodos en "tarjetas", agrupación de nodos por "tiers".
        Corrección de errores de consola (ValueError, PyArrow).

v3.0 (Versión Actual - Consolidación y Resumen):

    Simplificación de Tabla "Impacto del Árbol": Se eliminó la columna "Total Build" para mayor claridad, mostrando ahora Stat | Base | Boost (del árbol) | Resultado (con este árbol). El bonus se resalta en verde.
    Implementación de "Resumen de Texto Detallado" en "Build Craft":
        Botón "📋 Generar Resumen del Build para Copiar".
        Genera un texto formateado con: Perfil Base, Puntos de Habilidad, Características Clave (WF, SM, AcceleRATE), Atributos Generales Finales, IGS Final, y una lista de todos los Nodos Desbloqueados (agrupados por árbol, con su costo) y PlayStyles (si la columna PlayStyle existe y está poblada en ARBOL DE HABILIDAD - ARBOL.csv).
        El resumen se muestra en un st.text_area para fácil copiado.
    Confirmación de que todos los cálculos (IGS, Categorías Generales, WF/SM) y la interfaz funcionan como se esperaba.
