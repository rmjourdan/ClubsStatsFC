# Changelog

## [3.3] - 2024-06-06

### Nuevas características

- Integración de la visualización de árbol de habilidades estilo EA FC25 en la pestaña Build Craft.
- Opción de selector para elegir entre visualización FC25, lista de nodos clásica y grafo interactivo (si está disponible).

### Mejoras

- Refactor de imports para evitar errores si faltan módulos de visualización.
- Mejor experiencia de usuario en la pestaña Build Craft.

### Notas

- Se creó un backup de la versión 3.3 como `analizador_fc_v33_backup.py` antes de aplicar estos cambios.

## [v3.3.1] - 2024-06-07

### 🆕 Nueva Funcionalidad

- **Tabla de Impacto de Árbol e Instalaciones en Build Craft**
  - Se agregó una tabla dinámica que muestra, para el árbol de habilidades seleccionado, el impacto de los nodos desbloqueados y de las instalaciones activas sobre las estadísticas base del jugador.
  - La tabla aparece en la pestaña "🛠️ Build Craft", junto al selector de árbol, solo cuando se selecciona un árbol específico.

### 🏗️ Características Técnicas

- **Estructura de la Tabla:**
  - Columnas: Stat, Base, Boost (Árbol), Boost (Instalaciones), Final.
  - El boost del árbol se muestra en verde y el de instalaciones en azul si son positivos.
  - El valor final está limitado a 99.
- **Lógica:**
  - Determina las sub-stats relevantes según el árbol seleccionado (por mapeo global o por análisis dinámico).
  - Calcula los boosts de instalaciones solo si el toggle correspondiente está activo.
  - Suma los boosts de los nodos desbloqueados solo del árbol seleccionado.
  - Renderiza la tabla con formato HTML para resaltar los boosts.
- **Comportamiento:**
  - Se actualiza en vivo al cambiar árbol, desbloquear nodos o instalaciones, o modificar el perfil base.

### 🔧 Correcciones

- Se corrigieron problemas de indentación y duplicidad de widgets en la pestaña Build Craft.
- Se reorganizó la lógica para evitar claves duplicadas en Streamlit.
- Se eliminó la tabla de atributos del visualizador de árbol para evitar redundancia.

### 🎯 Beneficios

- Permite al usuario visualizar de forma clara y rápida el impacto real de sus decisiones de build sobre las estadísticas.
- Facilita la comparación y optimización de builds.
- Mejora la experiencia de usuario y la transparencia del sistema.

### 📝 Notas Técnicas

- La tabla utiliza `st.markdown(..., unsafe_allow_html=True)` para permitir el formateo de colores.
- El cálculo de boosts de instalaciones y árbol es independiente y seguro ante cambios de estado.
- Compatible con builds existentes y no afecta otras funcionalidades.
- El código es fácilmente extensible para futuras mejoras o nuevos tipos de boosts.

---
