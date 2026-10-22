import re
import pandas as pd

PATRONES = {
    ("auto", "Antiguo"):  re.compile(r"^[A-Z]{3}\d{3}$"),
    ("auto", "Mercosur"): re.compile(r"^[A-Z]{4}\d{3}$"),
    ("moto", "Antiguo"):  re.compile(r"^\d{3}[A-Z]{3}$"),
    ("moto", "Mercosur"): re.compile(r"^\d{3}[A-Z]{4}$"),
}

def clasificar_matricula(texto):
    t = texto.strip().upper().replace(" ", "").replace("-", "")
    for (categoria, formato), patron in PATRONES.items():
        if patron.match(t):
            return categoria, formato
    return None, None  # no coincide con ningún formato conocido

def devolver_datos_clasificados(texto):
    categoria, formato = clasificar_matricula(texto)

    return {
            "texto_matricula": texto,
            "categoria": categoria,
            "tipo_formato": formato
        }
   

def crear_csv_clasificado(entrada_csv, salida_csv):
    df = pd.read_csv(entrada_csv)
    resultados = []
    
    for _, row in df.iterrows():
        datos_clasificados = devolver_datos_clasificados(row['texto_matricula'])
        resultados.append(datos_clasificados)
    
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(salida_csv, index=False)
    
    
crear_csv_clasificado("datos_matriculas.csv", "datos_matriculas_clasificadas.csv")   
    
