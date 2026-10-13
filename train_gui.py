#!/usr/bin/env python3
"""
Interfaz gráfica (PySide6) para entrenar el detector de fallas de pavimento.

Lanza `train_pavimento.py` (misma carpeta) como subproceso, muestra el log en vivo, una barra de
progreso por época y permite detener el entrenamiento. Los pesos quedan en la carpeta raíz:
<nombre>_best.pt y <nombre>_last.pt.

Uso:  python train_gui.py
Requiere: pip install PySide6 ultralytics kaggle
"""
import os
import re
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

FROZEN = getattr(sys, "frozen", False)  # True cuando corre como .exe (PyInstaller)
# En el .exe, __file__ apunta a una carpeta temporal: usamos la carpeta del ejecutable.
ROOT = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent
SCRIPT = ROOT / "train_pavimento.py"
# En el .exe, sys.executable es el propio .exe: hay que usar el Python del sistema para entrenar.
PYTHON = (shutil.which("python") or shutil.which("py") or sys.executable) if FROZEN else sys.executable
KAGGLE_SLUG = "uzbtrust/smartroad-yolo"
MODELOS = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
           "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

# Línea de progreso de Ultralytics:  "       1/100      4.2G  1.23 ..."
RE_EPOCA = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G")
RE_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class VentanaEntrenamiento(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Road — Entrenamiento YOLO")
        self.resize(900, 720)
        self.proc: QProcess | None = None
        self._buffer = ""
        self.settings = QSettings("SmartRoad", "TrainGUI")

        central = QWidget()
        self.setCentralWidget(central)
        raiz = QVBoxLayout(central)

        raiz.addWidget(self._grupo_dataset())
        raiz.addWidget(self._grupo_modelo())

        fila = QHBoxLayout()
        self.btn_iniciar = QPushButton("▶  Iniciar entrenamiento")
        self.btn_iniciar.setMinimumHeight(36)
        self.btn_detener = QPushButton("■  Detener")
        self.btn_detener.setMinimumHeight(36)
        self.btn_detener.setEnabled(False)
        self.btn_carpeta = QPushButton("Abrir carpeta de pesos")
        self.btn_carpeta.setMinimumHeight(36)
        fila.addWidget(self.btn_iniciar, 2)
        fila.addWidget(self.btn_detener, 1)
        fila.addWidget(self.btn_carpeta, 1)
        raiz.addLayout(fila)

        self.barra = QProgressBar()
        self.barra.setFormat("Época %v / %m")
        self.barra.setRange(0, 1)
        self.barra.setValue(0)
        raiz.addWidget(self.barra)

        self.estado = QLabel("Listo.")
        self.estado.setTextInteractionFlags(Qt.TextSelectableByMouse)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.estado.setFont(mono)
        raiz.addWidget(self.estado)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(mono)
        self.log.setMaximumBlockCount(5000)
        raiz.addWidget(self.log, 1)

        self.btn_iniciar.clicked.connect(self.iniciar)
        self.btn_detener.clicked.connect(self.detener)
        self.btn_carpeta.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT))))
        self._cargar_ajustes()
        self._actualizar_fuente()

    # ------------------------------------------------------------------ UI
    def _grupo_dataset(self):
        g = QGroupBox("Dataset")
        lay = QVBoxLayout(g)

        self.rb_kaggle = QRadioButton("Descargar de Kaggle")
        self.ed_slug = QLineEdit(KAGGLE_SLUG)
        f1 = QHBoxLayout()
        f1.addWidget(self.rb_kaggle)
        f1.addWidget(self.ed_slug, 1)
        lay.addLayout(f1)

        self.rb_local = QRadioButton("Usar dataset local")
        self.ed_local = QLineEdit()
        self.ed_local.setPlaceholderText("data.yaml o carpeta del dataset")
        b_archivo = QPushButton("Archivo…")
        b_carpeta = QPushButton("Carpeta…")
        f2 = QHBoxLayout()
        f2.addWidget(self.rb_local)
        f2.addWidget(self.ed_local, 1)
        f2.addWidget(b_archivo)
        f2.addWidget(b_carpeta)
        lay.addLayout(f2)

        self.rb_kaggle.setChecked(True)
        self.rb_kaggle.toggled.connect(self._actualizar_fuente)
        b_archivo.clicked.connect(self._elegir_yaml)
        b_carpeta.clicked.connect(self._elegir_carpeta)
        self._b_local = (b_archivo, b_carpeta)
        return g

    def _grupo_modelo(self):
        g = QGroupBox("Entrenamiento")
        form = QFormLayout(g)

        self.cb_modelo = QComboBox()
        self.cb_modelo.setEditable(True)  # permite escribir otro .pt o la ruta a pesos propios
        self.cb_modelo.addItems(MODELOS)
        self.cb_modelo.setCurrentText("yolo12s.pt")

        self.sp_epocas = QSpinBox(); self.sp_epocas.setRange(1, 1000); self.sp_epocas.setValue(300)
        self.sp_horas = QDoubleSpinBox(); self.sp_horas.setRange(0, 48); self.sp_horas.setDecimals(1)
        self.sp_horas.setSingleStep(0.5)
        self.sp_horas.setSpecialValueText("sin límite")
        self.sp_horas.setSuffix(" h")
        self.sp_imgsz = QComboBox(); self.sp_imgsz.addItems(["320", "416", "512", "640", "800", "1024", "1280"])
        self.sp_imgsz.setCurrentText("640")
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(-1, 256); self.sp_batch.setValue(32)
        self.sp_batch.setToolTip("-1 = automático")
        self.sp_patience = QSpinBox(); self.sp_patience.setRange(0, 200); self.sp_patience.setValue(15)
        self.sp_workers = QSpinBox(); self.sp_workers.setRange(0, 32); self.sp_workers.setValue(4)
        self.cb_device = QComboBox(); self.cb_device.addItems(["auto", "cpu", "0", "0,1"])
        self.ed_nombre = QLineEdit("DetectorFallasPavimento")
        self.chk_nocache = QCheckBox("No cachear imágenes en RAM")
        self.chk_resume = QCheckBox("Reanudar último entrenamiento de este nombre")

        form.addRow("Modelo base", self.cb_modelo)
        form.addRow("Épocas", self.sp_epocas)
        form.addRow("Presupuesto de tiempo", self.sp_horas)
        form.addRow("Tamaño de imagen (imgsz)", self.sp_imgsz)
        form.addRow("Batch", self.sp_batch)
        form.addRow("Patience", self.sp_patience)
        form.addRow("Workers", self.sp_workers)
        form.addRow("Dispositivo", self.cb_device)
        form.addRow("Nombre del experimento", self.ed_nombre)
        form.addRow("", self.chk_nocache)
        form.addRow("", self.chk_resume)
        return g

    def _actualizar_fuente(self):
        kaggle = self.rb_kaggle.isChecked()
        self.ed_slug.setEnabled(kaggle)
        self.ed_local.setEnabled(not kaggle)
        for b in self._b_local:
            b.setEnabled(not kaggle)

    def _elegir_yaml(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Elegir data.yaml", str(ROOT), "YAML (*.yaml *.yml)")
        if ruta:
            self.ed_local.setText(ruta)

    def _elegir_carpeta(self):
        ruta = QFileDialog.getExistingDirectory(self, "Elegir carpeta del dataset", str(ROOT))
        if ruta:
            self.ed_local.setText(ruta)

    # ------------------------------------------------------------------ ajustes
    _CAMPOS = ("ed_slug", "ed_local", "ed_nombre")

    def _cargar_ajustes(self):
        s = self.settings
        for n in self._CAMPOS:
            v = s.value(n)
            if v:
                getattr(self, n).setText(str(v))
        if s.value("modelo"):
            self.cb_modelo.setCurrentText(str(s.value("modelo")))
        for n, w in (("epocas", self.sp_epocas), ("batch", self.sp_batch), ("patience", self.sp_patience),
                     ("workers", self.sp_workers)):
            v = s.value(n)
            if v is not None:
                w.setValue(int(v))
        if s.value("horas") is not None:
            self.sp_horas.setValue(float(s.value("horas")))
        if s.value("imgsz"):
            self.sp_imgsz.setCurrentText(str(s.value("imgsz")))
        if s.value("device"):
            self.cb_device.setCurrentText(str(s.value("device")))
        if s.value("fuente") == "local":
            self.rb_local.setChecked(True)

    def _guardar_ajustes(self):
        s = self.settings
        for n in self._CAMPOS:
            s.setValue(n, getattr(self, n).text())
        s.setValue("modelo", self.cb_modelo.currentText())
        s.setValue("epocas", self.sp_epocas.value())
        s.setValue("batch", self.sp_batch.value())
        s.setValue("patience", self.sp_patience.value())
        s.setValue("workers", self.sp_workers.value())
        s.setValue("horas", self.sp_horas.value())
        s.setValue("imgsz", self.sp_imgsz.currentText())
        s.setValue("device", self.cb_device.currentText())
        s.setValue("fuente", "kaggle" if self.rb_kaggle.isChecked() else "local")

    # ------------------------------------------------------------------ proceso
    def construir_argumentos(self) -> list[str] | None:
        nombre = self.ed_nombre.text().strip()
        if not nombre or re.search(r'[\\/:*?"<>|\s]', nombre):
            QMessageBox.warning(self, "Nombre inválido", "Usa un nombre sin espacios ni caracteres especiales.")
            return None

        args = ["-u", str(SCRIPT)]
        if not self.chk_resume.isChecked():
            if self.rb_kaggle.isChecked():
                slug = self.ed_slug.text().strip() or KAGGLE_SLUG
                args += ["--kaggle", slug]
            else:
                ruta = self.ed_local.text().strip()
                if not ruta or not Path(ruta).exists():
                    QMessageBox.warning(self, "Dataset", "Elige un data.yaml o una carpeta de dataset que exista.")
                    return None
                args += ["--data", ruta]
        args += [
            "--model", self.cb_modelo.currentText().strip(),
            "--epochs", str(self.sp_epocas.value()),
            "--imgsz", self.sp_imgsz.currentText(),
            "--batch", str(self.sp_batch.value()),
            "--patience", str(self.sp_patience.value()),
            "--workers", str(self.sp_workers.value()),
            "--name", nombre,
        ]
        if self.sp_horas.value() > 0:
            args += ["--hours", str(self.sp_horas.value())]
        if self.cb_device.currentText() != "auto":
            args += ["--device", self.cb_device.currentText()]
        if self.chk_nocache.isChecked():
            args.append("--no-cache")
        if self.chk_resume.isChecked():
            args.append("--resume")
        return args

    def iniciar(self):
        if self.proc is not None:
            return
        if not SCRIPT.exists():
            QMessageBox.critical(self, "Falta el script", f"No encuentro {SCRIPT.name} junto a esta GUI.")
            return
        args = self.construir_argumentos()
        if args is None:
            return
        self._guardar_ajustes()

        self.log.clear()
        self._buffer = ""
        self.barra.setRange(0, 1)
        self.barra.setValue(0)
        self.log.appendPlainText(f"$ {PYTHON} " + " ".join(args) + "\n")

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._leer)
        self.proc.finished.connect(self._terminado)
        self.proc.errorOccurred.connect(self._error)
        self.proc.start(PYTHON, args)
        self._set_corriendo(True)
        self.estado.setText("Entrenando…")

    def detener(self):
        if self.proc is None:
            return
        self.estado.setText("Deteniendo…")
        self.proc.kill()

    def _set_corriendo(self, corriendo: bool):
        self.btn_iniciar.setEnabled(not corriendo)
        self.btn_detener.setEnabled(corriendo)

    def _leer(self):
        datos = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        datos = RE_ANSI.sub("", datos)
        self._buffer += datos
        # Las barras tqdm se reescriben con \r: solo las líneas terminadas en \n van al log;
        # el último tramo con \r se muestra como estado "en vivo".
        *completas, resto = self._buffer.split("\n")
        self._buffer = resto
        for linea in completas:
            ultimo = linea.split("\r")[-1].rstrip()
            if ultimo:
                self.log.appendPlainText(ultimo)
                self._progreso(ultimo)
        vivo = self._buffer.split("\r")[-1].strip()
        if vivo:
            self.estado.setText(vivo[:160])
            self._progreso(vivo)

    def _progreso(self, linea: str):
        m = RE_EPOCA.match(linea)
        if m:
            actual, total = int(m.group(1)), int(m.group(2))
            self.barra.setRange(0, total)
            self.barra.setValue(actual)

    def _terminado(self, codigo: int, _estado):
        if self._buffer.strip():
            self.log.appendPlainText(self._buffer.split("\r")[-1].strip())
            self._buffer = ""
        ok = codigo == 0
        self.estado.setText("Entrenamiento terminado. Pesos guardados en la carpeta raíz." if ok
                            else f"El proceso terminó con código {codigo}.")
        self.log.appendPlainText(f"\n=== Proceso finalizado (código {codigo}) ===")
        self.proc = None
        self._set_corriendo(False)
        if ok:
            self.barra.setValue(self.barra.maximum())

    def _error(self, _err):
        if self.proc is not None and self.proc.state() == QProcess.NotRunning:
            self.log.appendPlainText(f"No se pudo iniciar Python: {self.proc.errorString()}")

    def closeEvent(self, ev):
        if self.proc is not None:
            r = QMessageBox.question(self, "Entrenamiento en curso",
                                     "Hay un entrenamiento en curso. ¿Detenerlo y salir?")
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            self.proc.kill()
            self.proc.waitForFinished(3000)
        self._guardar_ajustes()
        ev.accept()


def main():
    app = QApplication(sys.argv)
    w = VentanaEntrenamiento()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
