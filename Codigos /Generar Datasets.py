import random
import string
import re
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------
MODO_PRUEBA = False          # True: solo 5 imagenes en prueba/ | False: dataset completo
MUESTRAS_POR_TIPO = 500    # 2500 x 4 categorias = 10.000 filas
FRANJA_AZUL = True          # True: Mercosur con bloque azul liso | False: solo fondo, letras y marco

LETRAS_ESTANDAR = string.ascii_uppercase.replace("Ñ", "")
TIPOS = ["mercosur_auto", "mercosur_moto", "antigua_auto", "antigua_moto"]
AZUL = (0, 51, 153)
ROJO = (200, 30, 45)   # marco y caracteres de las placas nacionales (antiguas)

# ---------------------------------------------------------------
# GENERACION DE TEXTOS (CSV)
# ---------------------------------------------------------------
def generar_matricula_paraguay(tipo="mercosur_auto"):
    digitos = string.digits
    if tipo == "mercosur_auto":      # ABCD 123
        l = "".join(random.choices(LETRAS_ESTANDAR, k=4))
        d = "".join(random.choices(digitos, k=3))
        return f"{l} {d}"
    elif tipo == "mercosur_moto":    # 123 ABCD (dos lineas en la placa)
        d = "".join(random.choices(digitos, k=3))
        l = "".join(random.choices(LETRAS_ESTANDAR, k=4))
        return f"{d} {l}"
    elif tipo == "antigua_auto":     # ABC 123
        l = "".join(random.choices(LETRAS_ESTANDAR, k=3))
        d = "".join(random.choices(digitos, k=3))
        return f"{l} {d}"
    elif tipo == "antigua_moto":     # 123 ABC (una linea)
        d = "".join(random.choices(digitos, k=3))
        l = "".join(random.choices(LETRAS_ESTANDAR, k=3))
        return f"{d} {l}"


def crear_dataset_paraguay(muestras_por_tipo=100, incluir_sin_espacio=True):
    registros = []
    for t in TIPOS:
        for _ in range(muestras_por_tipo):
            placa = generar_matricula_paraguay(t)
            if incluir_sin_espacio and random.random() > 0.5:
                placa = placa.replace(" ", "")
            registros.append({
                "texto_matricula": placa,
                "categoria": t,
                "tipo_formato": "Mercosur" if "mercosur" in t else "Antiguo",
            })
    return pd.DataFrame(registros).sample(frac=1).reset_index(drop=True)


dataset_py = crear_dataset_paraguay(muestras_por_tipo=MUESTRAS_POR_TIPO)
dataset_py.to_csv("dataset_matriculas_paraguay.csv", index=False)
print(f"Dataset generado con {len(dataset_py)} filas.")

df = pd.read_csv("dataset_matriculas_paraguay.csv")

# ---------------------------------------------------------------
# DISENO DE LAS PLACAS (ajusta con una referencia oficial)
# ---------------------------------------------------------------
ESTILOS = {
    "mercosur_auto": {"size": (440, 140), "franja": True,  "lineas": 1},
    "mercosur_moto": {"size": (300, 260), "franja": True,  "lineas": 2},
    "antigua_auto":  {"size": (440, 140), "franja": False, "lineas": 1},
    "antigua_moto":  {"size": (300, 110), "franja": False, "lineas": 1},
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


def con_espacio(texto):
    # inserta espacio en el limite letra/numero si no lo tiene
    return re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", texto.replace(" ", ""))


def ajustar_fuente(lineas, ancho_max, alto_max):
    medidor = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    f = cargar_fuente(10)
    for tam in range(160, 10, -2):
        f = cargar_fuente(tam)
        cabe = True
        for t in lineas:
            x0, y0, x1, y1 = medidor.textbbox((0, 0), t, font=f)
            if x1 - x0 > ancho_max or y1 - y0 > alto_max:
                cabe = False
                break
        if cabe:
            return f
    return f


def dibujar_franja(d, w, alto_franja, m):
    # solo el bloque de color, sin textos ni bandera
    d.rectangle([m, m, w - m, alto_franja], fill=AZUL)


def dibujar_placa(texto, categoria):
    est = ESTILOS[categoria]
    w, h = est["size"]
    antigua = not est["franja"]
    m = 16 if antigua else 4
    color_texto = ROJO if antigua else (20, 20, 20)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)

    alto_franja = m
    if est["franja"] and FRANJA_AZUL:
        alto_franja = int(h * 0.20)
        dibujar_franja(d, w, alto_franja, m)

    partes = con_espacio(texto).split(" ")
    lineas = [" ".join(partes)] if est["lineas"] == 1 else partes

    slot = (h - alto_franja - m) / len(lineas)
    f = ajustar_fuente(lineas, w * 0.85, slot * 0.75)
    for k, t in enumerate(lineas):
        cy = alto_franja + slot * (k + 0.5)
        x0, y0, x1, y1 = d.textbbox((0, 0), t, font=f)
        d.text((w / 2 - (x1 - x0) / 2 - x0, cy - (y1 - y0) / 2 - y0),
               t, font=f, fill=color_texto)

    if antigua:
        # borde exterior blanco/gris y linea roja interior
        d.rounded_rectangle([1, 1, w - 2, h - 2], radius=14, outline=(190, 190, 190), width=2)
        d.rounded_rectangle([9, 9, w - 10, h - 10], radius=10, outline=ROJO, width=5)
    else:
        d.rounded_rectangle([1, 1, w - 2, h - 2], radius=12, outline=(20, 20, 20), width=4)

    # variaciones para que parezca real
    img = img.rotate(random.uniform(-6, 6), expand=True, fillcolor=(120, 120, 120))
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.5)))
    return img


# ---------------------------------------------------------------
# GENERACION DE IMAGENES
# ---------------------------------------------------------------
if MODO_PRUEBA:
    os.makedirs("prueba", exist_ok=True)
    muestra = pd.concat([df.groupby("categoria").sample(1), df.sample(1)]).reset_index(drop=True)
    for i, fila in muestra.iterrows():
        img = dibujar_placa(fila["texto_matricula"], fila["categoria"])
        img.save(f"prueba/{i}_{fila['categoria']}.jpg")
        print("Guardada:", fila["texto_matricula"], "->", fila["categoria"])
    print("5 imagenes de prueba en la carpeta prueba/")
else:
    # Todo en train: val y test deben ser fotos REALES que pondras a mano
    for i, fila in df.iterrows():
        cat = fila["categoria"]
        carpeta = f"dataset/train/{cat}"
        os.makedirs(carpeta, exist_ok=True)
        dibujar_placa(fila["texto_matricula"], cat).save(f"{carpeta}/{i}.jpg")
    print("Imagenes generadas en dataset/train/")
