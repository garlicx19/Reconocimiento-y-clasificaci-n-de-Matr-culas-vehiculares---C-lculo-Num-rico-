import random
import string
import re
import pandas as pd,random,os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

#Generación de Base de Datos

LETRAS_ESTANDAR = string.ascii_uppercase.replace('Ñ', '')

def generar_matricula_paraguay(tipo='mercosur_auto'):
    digitos = string.digits
    
    if tipo == 'mercosur_auto':
        # Formato ABCD 123
        l = ''.join(random.choices(LETRAS_ESTANDAR, k=4))
        d = ''.join(random.choices(digitos, k=3))
        return f"{l} {d}"
        
    elif tipo == 'mercosur_moto':
        # Formato 123 ABCD
        d = ''.join(random.choices(digitos, k=3))
        l = ''.join(random.choices(LETRAS_ESTANDAR, k=4))
        return f"{d} {l}"
        
    elif tipo == 'antigua_auto':
        # Formato ABC 123
        l = ''.join(random.choices(LETRAS_ESTANDAR, k=3))
        d = ''.join(random.choices(digitos, k=3))
        return f"{l} {d}"
        
    elif tipo == 'antigua_moto':
        # Formato 123 ABC
        d = ''.join(random.choices(digitos, k=3))
        l = ''.join(random.choices(LETRAS_ESTANDAR, k=3))
        return f"{d} {l}"

def crear_dataset_paraguay(muestras_por_tipo=100, incluir_sin_espacio=True):
    tipos = ['mercosur_auto', 'mercosur_moto', 'antigua_auto', 'antigua_moto']
    registros = []
    
    for t in tipos:
        for _ in range(muestras_por_tipo):
            placa = generar_matricula_paraguay(t)
            
            if incluir_sin_espacio and random.random() > 0.5:
                placa_procesada = placa.replace(" ", "")
            else:
                placa_procesada = placa
                
            registros.append({
                'texto_matricula': placa_procesada,
                'categoria': t,
                'tipo_formato': 'Mercosur' if 'mercosur' in t else 'Antiguo'
            })
            
    df = pd.DataFrame(registros)
    return df.sample(frac=1).reset_index(drop=True)

# Generamos 10k pruebas
dataset_py = crear_dataset_paraguay(muestras_por_tipo=2500)
dataset_py.to_csv("dataset_matriculas_paraguay.csv", index=False)

print(f"Dataset generado exitosamente con {len(dataset_py)} filas.")
print(dataset_py.head(10))

df = pd.read_csv("dataset_matriculas_paraguay.csv")

# Ajusta tamaños y colores al diseño oficial (verifícalo en una fuente oficial)
ESTILOS = {
    "mercosur_auto": {"size": (440, 140), "franja": (0, 51, 153), "fondo": (255, 255, 255)},
    "mercosur_moto": {"size": (300, 200), "franja": (0, 51, 153), "fondo": (255, 255, 255)},
    "antigua_auto":  {"size": (440, 140), "franja": None,         "fondo": (255, 255, 255)},
    "antigua_moto":  {"size": (300, 200), "franja": None,         "fondo": (255, 255, 255)},
}

def cargar_fuente(tam):
    rutas = ["fuente_placa.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    for r in rutas:
        try:
            return ImageFont.truetype(r, tam)
        except OSError:
            continue
    return ImageFont.load_default(size=tam)   # requiere Pillow >= 10.1

def ajustar_fuente(texto, ancho_max, alto_max):
    medidor = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for tam in range(120, 10, -2):
        f = cargar_fuente(tam)
        x0, y0, x1, y1 = medidor.textbbox((0, 0), texto, font=f)
        if x1 - x0 <= ancho_max and y1 - y0 <= alto_max:
            return f
    return f

def con_espacio(texto):
    # inserta espacio en el límite letra/número si no lo tiene
    return re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", texto.replace(" ", ""))

def dibujar_placa(texto, categoria):
    est = ESTILOS[categoria]
    w, h = est["size"]
    img = Image.new("RGB", (w, h), est["fondo"])
    d = ImageDraw.Draw(img)

    alto_franja = 0
    if est["franja"]:
        alto_franja = int(h * 0.22)
        d.rectangle([0, 0, w, alto_franja], fill=est["franja"])

    texto = con_espacio(texto)
    f = ajustar_fuente(texto, w * 0.88, (h - alto_franja) * 0.6)
    x0, y0, x1, y1 = d.textbbox((0, 0), texto, font=f)
    cx = w / 2 - (x1 - x0) / 2 - x0
    cy = alto_franja + (h - alto_franja) / 2 - (y1 - y0) / 2 - y0
    d.text((cx, cy), texto, font=f, fill=(20, 20, 20))
    d.rectangle([2, 2, w - 3, h - 3], outline=(20, 20, 20), width=3)   # borde

    # variaciones para que parezca real
    img = img.rotate(random.uniform(-6, 6), expand=True, fillcolor=(120, 120, 120))
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.5)))
    return img

def elegir_split():
    r = random.random()
    return "train" if r < 0.8 else "val" if r < 0.9 else "test"

for i, fila in df.iterrows():
    cat = fila["categoria"]
    carpeta = f"dataset/{elegir_split()}/{cat}"
    os.makedirs(carpeta, exist_ok=True)
    dibujar_placa(fila["texto_matricula"], cat).save(f"{carpeta}/{i}.jpg")

print("Imágenes generadas en dataset/")
