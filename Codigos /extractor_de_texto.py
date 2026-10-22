"""
OCR de matrículas paraguayas con clasificador softmax hecho desde cero (numpy).
Entrena con el dataset que genera proyecto.py (dataset/train/<categoria>/<i>.jpg
+ dataset_matriculas_paraguay.csv, donde la fila i es la etiqueta de <i>.jpg).

Uso:
  python ocr_softmax.py train [--dir dataset/train] [--csv dataset_matriculas_paraguay.csv] [--limit N]
  python ocr_softmax.py eval  carpeta_con_imagenes     # el nombre del archivo es el texto real: DCO333.jpg
  python ocr_softmax.py predict foto1.jpg foto2.jpg
"""
import os
import sys
import argparse
from collections import deque
from multiprocessing import Pool

import numpy as np
from PIL import Image

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list(range(0, 10))
LETTERS = list(range(10, 36))
TEMPLATES = {6: ["LLLDDD", "DDDLLL"], 7: ["LLLLDDD", "DDDLLLL"]}
GRID, TARGET_H = 24, 20
MODEL_PATH = "model_softmax.npz"


# ----------------------------------------------------------------------------
# Features de un glifo
# ----------------------------------------------------------------------------
def glyph_features(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = min(TARGET_H / h, GRID / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((GRID, GRID), np.float32)
    y0, x0 = (GRID - nh) // 2, (GRID - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = np.asarray(im, np.float32) / 255.0
    return canvas.ravel()


# ----------------------------------------------------------------------------
# Segmentación
# ----------------------------------------------------------------------------
def color_masks(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (r > 130) & (r - g > 60) & (r - b > 60)
    blue = (b > 90) & (b - r > 50)
    dark = (r < 100) & (g < 100) & (b < 100) & ~blue
    return red, blue, dark


def deskew(mask):
    """Busca el ángulo que maximiza la varianza del perfil horizontal (en baja resolución)."""
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    small = im.resize((max(1, im.size[0] // 2), max(1, im.size[1] // 2)), Image.BILINEAR)

    def score(a):
        return (np.asarray(small.rotate(a, resample=Image.BILINEAR)) > 127).sum(1).astype(np.float32).var()

    best_a = max(np.arange(-15, 15.5, 1.0), key=score)
    best_a = max(np.arange(best_a - 0.75, best_a + 0.76, 0.25), key=score)
    return np.asarray(im.rotate(best_a, resample=Image.BILINEAR, expand=True)) > 127


def connected_components(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x]:
            continue
        n = len(comps) + 1
        q = deque([(y, x)])
        lab[y, x] = n
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        q.append((ny, nx))
        ys, xs = np.array(pts).T
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), bool)
        sub[ys - y0, xs - x0] = True
        comps.append(dict(x0=x0, x1=x1, y0=y0, y1=y1, cy=(y0 + y1) / 2, mask=sub))
    return comps


def extract_glyphs(path):
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    red, blue, dark = color_masks(rgb)
    ink = dark if blue.mean() > 0.03 else red   # Mercosur: texto negro | antigua: texto rojo
    ink = deskew(ink)
    H, W = ink.shape
    comps = connected_components(ink)
    comps = [c for c in comps
             if c["y1"] - c["y0"] < 0.7 * H and c["x1"] - c["x0"] < 0.7 * W
             and c["x0"] > 0 and c["y0"] > 0 and c["x1"] < W - 1 and c["y1"] < H - 1]
    if not comps:
        return []
    maxh = max(c["y1"] - c["y0"] + 1 for c in comps)
    comps = [c for c in comps if (c["y1"] - c["y0"] + 1) >= 0.5 * maxh]
    cys = sorted(c["cy"] for c in comps)
    if cys[-1] - cys[0] > 0.6 * maxh:           # dos renglones (moto Mercosur)
        comps.sort(key=lambda c: c["cy"])
        gaps = [comps[i + 1]["cy"] - comps[i]["cy"] for i in range(len(comps) - 1)]
        k = int(np.argmax(gaps)) + 1
        comps = sorted(comps[:k], key=lambda c: c["x0"]) + sorted(comps[k:], key=lambda c: c["x0"])
    else:
        comps.sort(key=lambda c: c["x0"])
    return [c["mask"] for c in comps]


def _job(args):
    """Procesa una imagen de entrenamiento -> lista de (features, clase) o None si no segmenta bien."""
    path, label = args
    try:
        masks = extract_glyphs(path)
    except Exception:
        return None
    if len(masks) != len(label):
        return None
    out = []
    for m, ch in zip(masks, label):
        f = glyph_features(m)
        if f is None:
            return None
        out.append((f, CLASSES.index(ch)))
    return out


# ----------------------------------------------------------------------------
# Softmax
# ----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_softmax(X, y, K=36, epochs=40, lr=0.05, bs=128, l2=1e-4, seed=0, Xv=None, yv=None):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    W = np.zeros((D, K), np.float32)
    b = np.zeros(K, np.float32)
    vW, vb = np.zeros_like(W), np.zeros_like(b)
    for ep in range(epochs):
        idx = rng.permutation(N)
        loss = 0.0
        for i in range(0, N, bs):
            j = idx[i:i + bs]
            P = softmax(X[j] @ W + b)
            loss += -np.log(P[np.arange(len(j)), y[j]] + 1e-9).sum()
            P[np.arange(len(j)), y[j]] -= 1
            P /= len(j)
            vW = 0.9 * vW - lr * (X[j].T @ P + l2 * W)
            vb = 0.9 * vb - lr * P.sum(0)
            W += vW
            b += vb
        if ep % 10 == 0 or ep == epochs - 1:
            msg = f"epoch {ep:3d}  loss {loss / N:.4f}  acc_train {(softmax(X @ W + b).argmax(1) == y).mean():.4f}"
            if Xv is not None:
                msg += f"  acc_val {(softmax(Xv @ W + b).argmax(1) == yv).mean():.4f}"
            print(msg, flush=True)
    return W, b


# ----------------------------------------------------------------------------
# Decodificación con formato (softmax restringido a letras/dígitos por posición)
# ----------------------------------------------------------------------------
def decode(P):
    n = len(P)
    if n not in TEMPLATES:
        return "".join(CLASSES[i] for i in P.argmax(1))
    best, best_s = None, -1e18
    for t in TEMPLATES[n]:
        s, out = 0.0, ""
        for i, kind in enumerate(t):
            allowed = DIGITS if kind == "D" else LETTERS
            k = allowed[int(np.argmax(P[i, allowed]))]
            s += np.log(P[i, k] + 1e-9)
            out += CLASSES[k]
        if s > best_s:
            best, best_s = out, s
    return best


def read_plate(path, W, b):
    feats = [glyph_features(m) for m in extract_glyphs(path)]
    feats = [f for f in feats if f is not None]
    if not feats:
        return ""
    return decode(softmax(np.array(feats) @ W + b))


# ----------------------------------------------------------------------------
# Comandos
# ----------------------------------------------------------------------------
def cmd_train(a):
    import pandas as pd
    df = pd.read_csv(a.csv)
    jobs = []
    for cat in sorted(os.listdir(a.dir)):
        d = os.path.join(a.dir, cat)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in (".jpg", ".jpeg", ".png") or not stem.isdigit():
                continue
            label = str(df.loc[int(stem), "texto_matricula"]).replace(" ", "").upper()
            jobs.append((os.path.join(d, f), label))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} imágenes; segmentando con {os.cpu_count()} procesos...", flush=True)
    with Pool() as p:
        res = p.map(_job, jobs, chunksize=20)
    ok = [r for r in res if r]
    print(f"segmentadas correctamente: {len(ok)}/{len(jobs)} ({100 * len(ok) / max(1, len(jobs)):.1f}%)")
    pairs = [x for r in ok for x in r]
    X = np.array([f for f, _ in pairs], np.float32)
    y = np.array([c for _, c in pairs])
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    nv = len(X) // 10
    W, b = train_softmax(X[nv:], y[nv:], epochs=a.epochs, Xv=X[:nv], yv=y[:nv])
    np.savez(MODEL_PATH, W=W, b=b)
    print("guardado", MODEL_PATH)


def cmd_eval(a):
    m = np.load(MODEL_PATH)
    tot = ok = 0
    for f in sorted(os.listdir(a.carpeta)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        real = "".join(c for c in stem.split("_")[0].upper() if c.isalnum())
        pred = read_plate(os.path.join(a.carpeta, f), m["W"], m["b"])
        tot += 1
        ok += pred == real
        print(f"{'OK ' if pred == real else 'MAL'} real={real:8s} pred={pred}")
    print(f"exactitud por matrícula: {ok}/{tot}")


def cmd_predict(a):
    m = np.load(MODEL_PATH)
    for p in a.imagenes:
        print(p, "->", read_plate(p, m["W"], m["b"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--dir", default="dataset/train")
    t.add_argument("--csv", default="dataset_matriculas_paraguay.csv")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--epochs", type=int, default=40)
    t.set_defaults(fn=cmd_train)
    e = sp.add_parser("eval")
    e.add_argument("carpeta")
    e.set_defaults(fn=cmd_eval)
    p = sp.add_parser("predict")
    p.add_argument("imagenes", nargs="+")
    p.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)
