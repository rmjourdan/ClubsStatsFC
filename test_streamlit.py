import pandas as pd

# Definir un diccionario para los tipos de datos esperados para cada columna de mejora de stats
# Esto es para asegurar que se lean como números y los NaNs se manejen bien.
# Lista de todas las posibles columnas de stats que podrían tener mejoras.
# Deberías ajustar esta lista para que coincida EXACTAMENTE con los nombres de columna
# que usas en tu CSV para las mejoras de atributos (ej. 'Acc', 'Spr', 'Fin', etc.)
# y también 'PIERNA_MALA', 'FILIGRANAS'.
expected_stat_cols = [
    'Acc', 'Spr', 'Fin', 'FKA', 'HAcc', 'SPow', 'Lon S', 'Vol', 'Pen', 
    'Vis', 'Cros', 'LP', 'SP', 'Cur', 'AGI', 'BAL', 'APOS', 'BCON', 
    'REG', 'INT', 'AWA', 'STAN', 'SLID', 'JUMP', 'STA', 'STR', 'REA', 
    'AGGR', 'COMP', 'PIERNA_MALA', 'FILIGRANAS'
    # Añade aquí cualquier otra columna de estadística que estés usando para las mejoras.
]
# Crear el diccionario de tipos de datos: todas las stats como float64 para permitir NaNs
# y luego las convertiremos a int donde sea apropiado (ej. después de fillna(0)).
# Costo sí puede ser int directamente si nunca está vacío.
dtype_spec = {col: 'float64' for col in expected_stat_cols}
dtype_spec['Costo'] = 'int64' # Asumiendo que Costo siempre es un entero y no está vacío
# Las demás columnas (ID_Nodo, Arbol, Nombre_Visible, Prerrequisito, PlayStyle, etc.) se leerán como object (string).

try:
    df_skill_trees = pd.read_csv("ARBOL DE HABILIDAD - ARBOL.csv", dtype=dtype_spec)
    print("Archivo CSV 'ARBOL DE HABILIDAD - ARBOL.csv' cargado exitosamente.")
    print(f"Total de nodos transcritos: {len(df_skill_trees)}")
    
    print("\nPrimeras 5 filas del DataFrame:")
    print(df_skill_trees.head())
    
    print("\nNombres de las columnas:")
    print(df_skill_trees.columns.tolist())
    
    print("\nInformación del DataFrame (tipos de datos, nulos):")
    df_skill_trees.info()
    
    if 'Arbol' in df_skill_trees.columns:
        print("\nCategorías de 'Arbol' encontradas y conteo de nodos por árbol:")
        print(df_skill_trees['Arbol'].value_counts(dropna=False)) # dropna=False para ver si hay NaNs en 'Arbol'
    else:
        print("\nADVERTENCIA CRÍTICA: La columna 'Arbol' no se encontró. Es ESENCIAL para distinguir entre diferentes árboles de habilidad.")

    # Verificar si las columnas de Pierna Mala y Filigranas contienen principalmente 0s, 1s o NaNs
    if 'PIERNA_MALA' in df_skill_trees.columns:
        print("\nValores únicos en 'PIERNA_MALA' (para verificar si son incrementos de +1):")
        print(df_skill_trees['PIERNA_MALA'].value_counts(dropna=False))
    if 'FILIGRANAS' in df_skill_trees.columns:
        print("\nValores únicos en 'FILIGRANAS' (para verificar si son incrementos de +1):")
        print(df_skill_trees['FILIGRANAS'].value_counts(dropna=False))

except FileNotFoundError:
    print("ERROR: No se encontró el archivo 'ARBOL DE HABILIDAD - ARBOL.csv'.")
    df_skill_trees = None
except Exception as e:
    print(f"Se produjo un error al cargar o procesar el CSV: {e}")
    df_skill_trees = None