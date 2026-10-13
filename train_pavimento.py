#!/usr/bin/env python3
"""
Entrena el detector de fallas de pavimento (Smart Road) con Ultralytics YOLO y guarda los
pesos (best.pt / last.pt) en la MISMA carpeta raíz donde está este script.

Uso rápido:
    # Descarga el dataset de Kaggle (uzbtrust/smartroad-yolo), lo descomprime y entrena
    python train_pavimento.py --kaggle

    # Dataset ya descargado/descomprimido (archivo data.yaml o carpeta que lo contenga)
    python train_pavimento.py --data ./datasets/smartroad-yolo

    # Otro modelo / resolución / presupuesto de tiempo
    python train_pavimento.py --kaggle --model yolo11m.pt --imgsz 1024 --batch 16 --name pavimento_y11m
    python train_pavimento.py --kaggle --hours 5

Kaggle CLI necesita credenciales: ~/.kaggle/kaggle.json  o  variables KAGGLE_USERNAME y KAGGLE_KEY.
(pip install kaggle)

Salida (en la carpeta raíz del script):
    <name>_best.pt      <- mejores pesos (úsalos en app_qt.py)
    <name>_last.pt      <- últimos pesos
    runs/<name>/        <- gráficas, métricas y checkpoints completos de Ultralytics
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # carpeta raíz: aquí se guardan los pesos
KAGGLE_SLUG = "uzbtrust/smartroad-yolo"


# --------------------------------------------------------------------------- dataset
def descargar_kaggle(slug: str) -> Path:
    """Equivale a `kaggle datasets download <slug>`, pero lo descomprime en ./datasets/<nombre>."""
    destino = ROOT / "datasets" / slug.split("/")[-1]
    if destino.exists() and any(destino.rglob("data.yaml")):
        print(f"Dataset ya descargado en {destino}, se reutiliza.")
        return destino
    destino.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", slug, "-p", str(destino), "--unzip"]
    print("Ejecutando:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("No se encontró el comando 'kaggle'. Instala con: pip install kaggle")
    except subprocess.CalledProcessError:
        sys.exit("Falló la descarga de Kaggle. Revisa kaggle.json / KAGGLE_USERNAME / KAGGLE_KEY.")
    return destino


def encontrar_yaml(ruta: Path) -> Path:
    """Acepta un data.yaml o una carpeta; si es carpeta busca el data.yaml (menos profundo primero)."""
    if ruta.is_file():
        return ruta
    candidatos = sorted(ruta.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not candidatos:
        sys.exit(f"No se encontró data.yaml dentro de {ruta}")
    return candidatos[0]


def _resolver(base: Path, valor, alternativas=()):
    """Devuelve la primera ruta existente entre `valor` (relativo a base o absoluto) y alternativas."""
    opciones = []
    if valor:
        for v in (valor if isinstance(valor, list) else [valor]):
            p = Path(v)
            opciones.append(p if p.is_absolute() else (base / p).resolve())
    opciones += [base / a for a in alternativas]
    for p in opciones:
        if p.exists():
            return p
    return None


def preparar_yaml(data_yaml: Path) -> Path:
    """
    Escribe un data_local.yaml con rutas absolutas válidas en ESTA máquina (Colab, Kaggle o local).
    Las clases (names/nc) se toman del data.yaml original: no se asume ninguna lista fija.
    """
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = Path(cfg["path"]) if cfg.get("path") and Path(cfg["path"]).exists() else data_yaml.parent
    base = base.resolve()

    names = cfg.get("names")
    if not names:
        sys.exit("El data.yaml no define 'names'.")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    train = _resolver(base, cfg.get("train"), ("train/images", "images/train"))
    val = _resolver(base, cfg.get("val"), ("valid/images", "val/images", "images/val", "images/valid"))
    test = _resolver(base, cfg.get("test"), ("test/images", "images/test"))
    if train is None or val is None:
        sys.exit(f"No pude ubicar train/val a partir de {data_yaml}. Revisa las rutas del yaml.")

    lineas = [f"path: {base.as_posix()}", f"train: {train.as_posix()}", f"val: {val.as_posix()}"]
    if test is not None:
        lineas.append(f"test: {test.as_posix()}")
    lineas.append(f"nc: {len(names)}")
    lineas.append("names:")
    lineas += [f"  {i}: {n}" for i, n in names.items()]

    # El dataset puede estar en una carpeta de solo lectura (p. ej. /kaggle/input)
    salida = data_yaml.parent / "data_local.yaml"
    try:
        salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    except OSError:
        salida = ROOT / "data_local.yaml"
        salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    print(f"\ndata.yaml -> {salida}\n" + "\n".join(lineas) + "\n")
    return salida


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Entrenamiento YOLO - fallas de pavimento")
    fuente = p.add_mutually_exclusive_group()
    fuente.add_argument("--kaggle", nargs="?", const=KAGGLE_SLUG, metavar="SLUG",
                        help=f"Descargar dataset de Kaggle (por defecto {KAGGLE_SLUG})")
    fuente.add_argument("--data", type=Path, help="data.yaml o carpeta del dataset ya descargado")
    p.add_argument("--model", default="yolov8s.pt", help="Pesos base (yolov8s.pt, yolo11m.pt, ...)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hours", type=float, default=None,
                   help="Presupuesto de tiempo en horas (Ultralytics ajusta el schedule; reemplaza a --epochs)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32, help="-1 = automático")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="0, 0,1, cpu ... (por defecto: GPU si hay, si no CPU)")
    p.add_argument("--name", default="DetectorFallasPavimento", help="Nombre del experimento")
    p.add_argument("--no-cache", action="store_true", help="No cachear imágenes en RAM")
    p.add_argument("--resume", action="store_true", help="Reanudar el último entrenamiento de --name")
    a = p.parse_args()

    runs_dir = ROOT / "runs"
    last_ckpt = runs_dir / a.name / "weights" / "last.pt"

    from ultralytics import YOLO

    if a.resume:
        if not last_ckpt.exists():
            sys.exit(f"No hay checkpoint para reanudar: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        if a.kaggle:
            origen = descargar_kaggle(a.kaggle)
        elif a.data:
            origen = a.data
        else:
            sys.exit("Indica --kaggle o --data ruta/al/dataset")
        if not origen.exists():
            sys.exit(f"No existe {origen}")
        data_yaml = preparar_yaml(encontrar_yaml(origen))

        kwargs = dict(
            data=str(data_yaml),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            patience=a.patience,
            workers=a.workers,
            device=a.device,
            cache=not a.no_cache,
            project=str(runs_dir),  # ruta absoluta: evita el anidado runs/detect/runs/train/...
            name=a.name,
            exist_ok=True,
        )
        if a.hours:
            kwargs["time"] = a.hours
        model = YOLO(a.model)
        model.train(**kwargs)

    # Ultralytics guarda en <save_dir>/weights; copiamos a la raíz
    weights_dir = Path(model.trainer.save_dir) / "weights"
    copiados = []
    for tipo in ("best", "last"):
        src = weights_dir / f"{tipo}.pt"
        if src.exists():
            dst = ROOT / f"{a.name}_{tipo}.pt"
            shutil.copy2(src, dst)
            copiados.append(dst)

    print("\n=== Entrenamiento terminado ===")
    for d in copiados:
        print(f"Pesos guardados en: {d}")
    print(f"Métricas y gráficas en: {weights_dir.parent}")


if __name__ == "__main__":
    main()
