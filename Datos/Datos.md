# Datos

Datos utilizados en el proyecto: matrículas obtenidas desde las imagenes y el dataset organizado para entrenamiento y prueba.

## Contenido

| Carpeta | Descripción |
|---|---|
| [Matriculas obtenidas](Matriculas%20obtenidas/) | Valores leidos por desde las imagenes |
| [dataset](dataset/) | Imágenes organizadas en conjuntos `train` y `test` por tipo de matrícula para entrenar el modelo de reconocimiento de imagenes |

## Matriculas obtenidas

Carpeta con los datos recolectados.

| Archivo | Descripción |
|---|---|
| [datos_matriculas.csv](Matriculas%20obtenidas/datos_matriculas.csv) | *Guarda los datos de la imagenes que lee el algoritmo* |
| [datos_clasificados.csv](Matriculas%20obtenidas/datos_clasificados.csv) | *Se separan las matriculas segun categoria (auto/moto) y formato (Antiguo / Mercosur)* |

## dataset

Imágenes divididas en dos conjuntos, `train` y `test`, y dentro de cada uno en carpetas por tipo de matrícula y de vehículo.

```
dataset/
├── test/
│   ├── antigua_auto/
│   ├── antigua_moto/
│   ├── mercosur_auto/
│   └── mercosur_moto/
└── train/
    ├── antiguo_auto/
    ├── antiguo_moto/
    ├── mercosur_auto/
    └── ...
```

### Clases

| Clase | Descripción |
|---|---|
| `antigua` / `antiguo` | Matrículas del formato anterior |
| `mercosur` | Matrículas del formato Mercosur |
| `auto` | Matrículas de automóviles |
| `moto` | Matrículas de motocicletas |


---

[← Volver al README](../README.md)

