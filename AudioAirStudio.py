# -*- coding: utf-8 -*-
"""
Audio Air Studio - phan mem upsample + EQ "air" cho giong noi.

Tinh nang:
  - Mo WAV/FLAC/MP3/OGG, keo-tha file vao cua so.
  - Xem waveform toan file + pho tan (truoc / sau xu ly).
  - Mau chinh san (preset), chinh high-shelf gain / tan so / Q.
  - Nghe thu: goc, sau xu ly, A/B tu dong (5 giay doi mot).
  - Xuat WAV 16/24/32-bit, chon 44.1 / 48 / 96 kHz.
  - Xuat ban A/B 5 giay de kiem tra tren Premiere/DaVinci.

Chay:  python AudioAirStudio.py
Can:    PyQt5, matplotlib, soundfile, scipy, numpy
"""

import itertools
import os
import sys
import tempfile
import time
import traceback
from math import gcd
from pathlib import Path

# Che do kiem tra ban dong goi: chay khong can cua so
if "--self-test" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from scipy import signal

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

APP_NAME = "Audio Air Studio"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".w64"}

PRESETS = [
    ("Nguyên bản (không EQ)", 0.0, 8000, 0.707, False),
    ("Tự nhiên  ·  +1 dB @ 8 kHz", 1.0, 8000, 0.707, True),
    ("Air  ·  +2 dB @ 8 kHz", 2.0, 8000, 0.707, True),
    ("Sáng  ·  +3 dB @ 7 kHz", 3.0, 7000, 0.707, True),
    ("Chi tiết  ·  +4 dB @ 10 kHz", 4.0, 10000, 0.707, True),
    ("Giọng nữ  ·  +3.5 dB @ 9 kHz", 3.5, 9000, 0.80, True),
    ("Giọng nam  ·  +2.5 dB @ 6.5 kHz", 2.5, 6500, 0.707, True),
    ("Tùy chỉnh", 0.0, 8000, 0.707, True),
]
CUSTOM_INDEX = len(PRESETS) - 1

BG = "#0f131b"
PANEL = "#171e29"
PANEL_2 = "#1d2634"
TEXT = "#e6edf7"
MUTED = "#8b97ab"
ACCENT = "#38d6f5"
ACCENT_DIM = "#1c7f92"
WARM = "#ffb454"


# ----------------------------------------------------------------------------
# Xu ly tin hieu
# ----------------------------------------------------------------------------
def high_shelf_sos(fs, f0, gain_db, q=0.707):
    """High-shelf biquad (RBJ cookbook)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)

    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def resample_audio(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    y = signal.resample_poly(x, up, down, axis=0)
    return np.ascontiguousarray(y, dtype=np.float32)


def process_audio(x, sr_in, sr_out, eq_gain_db, eq_freq, eq_q,
                  normalize=False, eq_enabled=True):
    """Resample + high-shelf + chong clipping. Tra ve (audio, gain_da_ap)."""
    y = resample_audio(x, sr_in, sr_out)

    # high-shelf phai nam duoi Nyquist, neu khong filter se khong on dinh
    if eq_enabled and abs(eq_gain_db) > 1e-9 and eq_freq < 0.49 * sr_out:
        sos = high_shelf_sos(sr_out, eq_freq, eq_gain_db, eq_q)
        y = signal.sosfiltfilt(sos, y, axis=0).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if normalize and peak > 1e-9:
        gain = 0.891 / peak          # -1 dBFS
    elif peak > 0.999:
        gain = 0.999 / peak
    else:
        gain = 1.0

    if gain != 1.0:
        y = (y * gain).astype(np.float32, copy=False)
    return y, gain


def mono_mix(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def waveform_envelope(x, sr, width=1800):
    """Tra ve (t, min, max) da downsample de ve waveform nhanh."""
    m = mono_mix(x)
    n = len(m)
    if n == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    if n <= width:
        t = np.arange(n) / sr
        return t, m, m
    bins = n // width
    trimmed = m[: bins * width].reshape(width, bins)
    lo, hi = trimmed.min(axis=1), trimmed.max(axis=1)
    t = (np.arange(width) + 0.5) * bins / sr
    return t, lo, hi


def spectrum_db(x, sr, nfft=8192):
    m = mono_mix(x).astype(np.float64)
    nfft = int(min(nfft, len(m)))
    if nfft < 256:
        return np.array([20.0]), np.array([-120.0])
    f, p = signal.welch(m, fs=sr, nperseg=nfft, noverlap=nfft // 2)
    return f, 10.0 * np.log10(p + 1e-16)


def ab_combine(a, b, sr, block_s=5.0, fade_ms=15.0):
    """Tron 2 ban giong nhau: a, b, a, b... moi block_s giay (co fade)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    blk = max(1, int(block_s * sr))
    fade = max(1, int(fade_ms / 1000.0 * sr))
    parts = []
    i = 0
    k = 0
    while i < n:
        seg = (a if k % 2 == 0 else b)[i:min(i + blk, n)].copy()
        f = min(fade, len(seg) // 2)
        if f > 1:
            w = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
            seg[:f] *= w
            seg[-f:] *= w[::-1]
        parts.append(seg)
        i += blk
        k += 1
    return np.concatenate(parts) if parts else np.zeros((0, a.shape[1]), np.float32)


def fmt_time(sec):
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def fmt_hz(hz):
    return f"{hz / 1000:.1f} kHz".replace(".0 ", " ")


# ----------------------------------------------------------------------------
# Giao dien
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self.setMinimumSize(1000, 680)
        self.setAcceptDrops(True)

        self.raw = None              # audio goc, float32, shape (n, ch)
        self.sr = 0
        self.path = None
        self.duration = 0.0
        self.env = None              # cache waveform
        self.prev_orig = None        # preview truoc EQ (da resample)
        self.prev_proc = None        # preview sau xu ly
        self.prev_sr = 48000

        self._applying = False
        self._tmpdir = Path(tempfile.mkdtemp(prefix="audioair_"))
        self._tmp_count = itertools.count()
        self._play_label = ""

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(320)
        self._debounce.timeout.connect(self._refresh_preview)

        self._build_ui()
        self._apply_style()
        self._enable_controls(False)
        self._update_summary()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Mở một file âm thanh để bắt đầu (Ctrl+O, hoặc kéo-thả file vào cửa sổ)")

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        # ---- panel trai -------------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(430)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 14, 14, 14)
        lv.setSpacing(12)

        title = QLabel(APP_NAME)
        title.setObjectName("AppTitle")
        sub = QLabel("Upsample 24 → 48 kHz  ·  EQ độ thoáng cho giọng nói")
        sub.setObjectName("AppSub")
        sub.setWordWrap(True)
        lv.addWidget(title)
        lv.addWidget(sub)

        # --- file
        g_file = QGroupBox("Tệp âm thanh")
        fv = QVBoxLayout(g_file)
        self.btn_open = QPushButton("📂  Mở file…")
        self.btn_open.setObjectName("Primary")
        self.btn_open.clicked.connect(self.open_dialog)
        self.lbl_file = QLabel("Chưa có file")
        self.lbl_file.setObjectName("FileName")
        self.lbl_file.setWordWrap(True)
        self.lbl_info = QLabel("Kéo-thả WAV / FLAC / MP3 vào đây")
        self.lbl_info.setObjectName("Muted")
        self.lbl_info.setWordWrap(True)
        fv.addWidget(self.btn_open)
        fv.addWidget(self.lbl_file)
        fv.addWidget(self.lbl_info)
        lv.addWidget(g_file)

        # --- preset
        g_pre = QGroupBox("Mẫu chỉnh sẵn")
        pv = QVBoxLayout(g_pre)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems([p[0] for p in PRESETS])
        self.cb_preset.setCurrentIndex(2)   # Air +2 dB (khop voi slider mac dinh)
        self.cb_preset.currentIndexChanged.connect(self._on_preset)
        self.lbl_summary = QLabel()
        self.lbl_summary.setObjectName("Muted")
        self.lbl_summary.setWordWrap(True)
        pv.addWidget(self.cb_preset)
        pv.addWidget(self.lbl_summary)
        lv.addWidget(g_pre)

        # --- EQ
        g_eq = QGroupBox("EQ độ thoáng (high-shelf)")
        ev = QVBoxLayout(g_eq)
        self.chk_eq = QCheckBox("Bật EQ")
        self.chk_eq.setChecked(True)
        self.chk_eq.toggled.connect(self._on_param_changed)

        self.sl_gain, row_gain, self.lbl_gain = self._slider_row(
            -60, 60, 20, "Mức nâng", lambda v: f"{v / 10:+.1f} dB")
        self.sl_freq, row_freq, self.lbl_freq = self._slider_row(
            4000, 14000, 8000, "Từ tần số", lambda v: fmt_hz(v), step=100)
        self.sl_q, row_q, self.lbl_q = self._slider_row(
            40, 160, 71, "Độ dốc Q", lambda v: f"{v / 100:.2f}")

        for w in (self.sl_gain, self.sl_freq, self.sl_q):
            w.valueChanged.connect(self._on_param_changed)

        ev.addWidget(self.chk_eq)
        ev.addLayout(row_gain)
        ev.addLayout(row_freq)
        ev.addLayout(row_q)
        tip = QLabel("Gợi ý: 1–3 dB là đủ. Cao hơn dễ bị chói âm 's'.")
        tip.setObjectName("Muted")
        tip.setWordWrap(True)
        ev.addWidget(tip)
        lv.addWidget(g_eq)

        # --- nghe thu
        g_prev = QGroupBox("Đoạn nghe thử")
        pv2 = QGridLayout(g_prev)
        self.sp_start = QDoubleSpinBox()
        self.sp_start.setRange(0, 36000)
        self.sp_start.setDecimals(2)
        self.sp_start.setSuffix(" s")
        self.sp_start.setSingleStep(1.0)
        self.sp_len = QDoubleSpinBox()
        self.sp_len.setRange(1, 3600)
        self.sp_len.setDecimals(2)
        self.sp_len.setSuffix(" s")
        self.sp_len.setSingleStep(1.0)
        self.sp_len.setValue(20.0)
        self.sp_start.valueChanged.connect(self._on_segment_changed)
        self.sp_len.valueChanged.connect(self._on_segment_changed)
        self.btn_all = QPushButton("Toàn bộ")
        self.btn_all.clicked.connect(self._select_all)
        pv2.addWidget(QLabel("Bắt đầu"), 0, 0)
        pv2.addWidget(self.sp_start, 0, 1)
        pv2.addWidget(QLabel("Dài"), 1, 0)
        pv2.addWidget(self.sp_len, 1, 1)
        pv2.addWidget(self.btn_all, 2, 0, 1, 2)
        hint = QLabel("Mẹo: bấm vào waveform để đặt điểm bắt đầu.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        pv2.addWidget(hint, 3, 0, 1, 2)
        lv.addWidget(g_prev)

        # --- xuat
        g_out = QGroupBox("Xuất file")
        ov = QGridLayout(g_out)
        self.cb_sr = QComboBox()
        self.cb_sr.addItems(["Giữ nguyên", "44.1 kHz", "48 kHz", "96 kHz"])
        self.cb_sr.setCurrentIndex(2)
        self.cb_depth = QComboBox()
        self.cb_depth.addItems(["16-bit", "24-bit", "32-bit float"])
        self.cb_depth.setCurrentIndex(1)
        self.chk_norm = QCheckBox("Chuẩn hóa đỉnh −1 dBFS")
        self.chk_norm.toggled.connect(self._on_param_changed)
        ov.addWidget(QLabel("Sample rate"), 0, 0)
        ov.addWidget(self.cb_sr, 0, 1)
        ov.addWidget(QLabel("Độ sâu bit"), 1, 0)
        ov.addWidget(self.cb_depth, 1, 1)
        ov.addWidget(self.chk_norm, 2, 0, 1, 2)

        self.btn_export = QPushButton("💾  Xuất WAV…")
        self.btn_export.setObjectName("Primary")
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_export_ab = QPushButton("Xuất bản A/B 5 giây…")
        self.btn_export_ab.clicked.connect(self.export_ab_dialog)
        ov.addWidget(self.btn_export, 3, 0, 1, 2)
        ov.addWidget(self.btn_export_ab, 4, 0, 1, 2)
        lv.addWidget(g_out)

        lv.addStretch(1)
        version = QLabel("audio engine: polyphase + RBJ high-shelf")
        version.setObjectName("Muted")
        version.setAlignment(Qt.AlignCenter)
        lv.addWidget(version)

        scroll.setWidget(left)
        split.addWidget(scroll)

        # ---- panel phai --------------------------------------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 12, 14, 10)
        rv.setSpacing(10)

        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor(BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        rv.addWidget(self.canvas, 1)

        transport = QFrame()
        transport.setObjectName("Transport")
        tv = QHBoxLayout(transport)
        tv.setContentsMargins(14, 10, 14, 10)
        tv.setSpacing(8)

        self.btn_play_orig = QPushButton("▶  Nghe gốc")
        self.btn_play_proc = QPushButton("▶  Nghe xử lý")
        self.btn_play_ab = QPushButton("▶  A/B 5s")
        self.btn_stop = QPushButton("■  Dừng")
        for b in (self.btn_play_orig, self.btn_play_proc):
            b.setObjectName("Play")
        self.btn_stop.setObjectName("Stop")
        self.btn_play_orig.clicked.connect(self.play_original)
        self.btn_play_proc.clicked.connect(self.play_processed)
        self.btn_play_ab.clicked.connect(self.play_ab)
        self.btn_stop.clicked.connect(self._stop)

        tv.addWidget(self.btn_play_orig)
        tv.addWidget(self.btn_play_proc)
        tv.addWidget(self.btn_play_ab)
        tv.addWidget(self.btn_stop)
        tv.addStretch(1)
        self.lbl_playing = QLabel("")
        self.lbl_playing.setObjectName("Playing")
        tv.addWidget(self.lbl_playing)
        tv.addWidget(QLabel("🔊"))
        self.sl_vol = QSlider(Qt.Horizontal)
        self.sl_vol.setRange(0, 100)
        self.sl_vol.setValue(85)
        self.sl_vol.setFixedWidth(110)
        self.sl_vol.valueChanged.connect(
            lambda v: self.player.setVolume(v) if self.player is not None else None)
        tv.addWidget(self.sl_vol)
        rv.addWidget(transport)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        # ---- player ------------------------------------------------------
        self.player = None
        self._init_player()
        self._draw_placeholder()

    @staticmethod
    def _slider_row(lo, hi, value, label, fmt, step=1):
        row = QHBoxLayout()
        name = QLabel(label)
        name.setMinimumWidth(96)
        val = QLabel(fmt(value))
        val.setObjectName("Value")
        val.setMinimumWidth(74)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(value)
        sl.setSingleStep(step)
        sl.setPageStep(step * 5)
        row.addWidget(name)
        row.addWidget(sl, 1)
        row.addWidget(val)
        # cap nhat nhan ngay ca truoc khi connect ben ngoai
        sl.valueChanged.connect(lambda v, l=val, f=fmt: l.setText(f(v)))
        return sl, row, val

    def _init_player(self):
        try:
            from PyQt5.QtMultimedia import QMediaPlayer
            self.player = QMediaPlayer(self)
            self.player.setVolume(self.sl_vol.value())
            self.player.error.connect(self._on_player_error)
            self.player.stateChanged.connect(self._on_player_state)
        except Exception:
            self.player = None

    def _on_player_state(self, state):
        from PyQt5.QtMultimedia import QMediaPlayer
        if state == QMediaPlayer.StoppedState and self.lbl_playing.text():
            self.lbl_playing.setText("")

    def _on_player_error(self, *_):
        # QtMultimedia loi -> fallback winsound (chi Windows)
        if os.name == "nt" and self._play_label and getattr(self, "_last_wav", None):
            try:
                import winsound
                winsound.PlaySound(self._last_wav,
                                   winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass

    # --------------------------------------------------------------- style
    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT};
                font-family:'Segoe UI'; font-size:13px; }}
            QScrollArea {{ background:{BG}; }}
            QGroupBox {{ background:{PANEL}; border:1px solid #263142;
                border-radius:10px; margin-top:18px; padding:12px 12px 10px 12px;
                font-weight:600; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:12px;
                padding:0 6px; color:{MUTED}; font-size:11px;
                letter-spacing:1px; text-transform:uppercase; }}
            QPushButton {{ background:{PANEL_2}; border:1px solid #2b3749;
                border-radius:7px; padding:7px 12px; color:{TEXT}; }}
            QPushButton:hover {{ background:#26324a; border-color:#3a4a63; }}
            QPushButton:pressed {{ background:#1a2332; }}
            QPushButton:disabled {{ color:#5b6577; border-color:#222b39; }}
            QPushButton#Primary {{ background:{ACCENT}; color:#06222a;
                border:none; font-weight:700; }}
            QPushButton#Primary:hover {{ background:#5ee4ff; }}
            QPushButton#Primary:disabled {{ background:#1d3d46; color:#5a7a83; }}
            QPushButton#Play {{ background:#173b45; border:1px solid #1f5a69;
                color:{ACCENT}; font-weight:600; }}
            QPushButton#Play:hover {{ background:#1c4b58; }}
            QPushButton#Stop {{ background:#2a2028; border:1px solid #4a2b38;
                color:#ff8fa3; }}
            QPushButton#Stop:hover {{ background:#3a2731; }}
            QLabel#AppTitle {{ font-size:22px; font-weight:800;
                color:{TEXT}; letter-spacing:.4px; }}
            QLabel#AppSub {{ color:{MUTED}; font-size:12px; }}
            QLabel#FileName {{ font-weight:600; color:{ACCENT}; }}
            QLabel#Muted {{ color:{MUTED}; font-size:11.5px; }}
            QLabel#Value {{ color:{ACCENT}; font-weight:700;
                font-family:'Consolas'; font-size:12px; }}
            QLabel#Playing {{ color:{WARM}; font-size:12px; }}
            QComboBox {{ background:{PANEL_2}; border:1px solid #2b3749;
                border-radius:7px; padding:6px 10px; min-height:20px; }}
            QComboBox:hover {{ border-color:#3a4a63; }}
            QComboBox QAbstractItemView {{ background:{PANEL_2};
                selection-background-color:{ACCENT_DIM}; outline:none;
                border:1px solid #2b3749; }}
            QSpinBox, QDoubleSpinBox {{ background:{PANEL_2};
                border:1px solid #2b3749; border-radius:7px;
                padding:5px 8px; }}
            QCheckBox {{ spacing:8px; }}
            QCheckBox::indicator {{ width:16px; height:16px; border-radius:4px;
                border:1px solid #3a4a63; background:{PANEL_2}; }}
            QCheckBox::indicator:checked {{ background:{ACCENT};
                border-color:{ACCENT}; image:none; }}
            QSlider::groove:horizontal {{ height:5px; background:#2a3547;
                border-radius:2px; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT};
                border-radius:2px; }}
            QSlider::handle:horizontal {{ width:15px; height:15px;
                margin:-6px 0; border-radius:8px; background:#eafcff;
                border:2px solid {ACCENT}; }}
            QSlider::handle:horizontal:hover {{ background:#ffffff; }}
            QScrollBar:vertical {{ background:transparent; width:10px;
                margin:0; }}
            QScrollBar::handle:vertical {{ background:#2c3748;
                border-radius:5px; min-height:28px; }}
            QScrollBar::handle:vertical:hover {{ background:#3a4a63; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
            QFrame#Transport {{ background:{PANEL}; border:1px solid #263142;
                border-radius:10px; }}
            QStatusBar {{ background:{PANEL}; color:{MUTED};
                border-top:1px solid #263142; }}
            QToolTip {{ background:{PANEL_2}; color:{TEXT};
                border:1px solid #3a4a63; padding:6px; }}
        """)

    # --------------------------------------------------------------- state
    def _enable_controls(self, on):
        for w in (self.cb_preset, self.chk_eq, self.sl_gain, self.sl_freq,
                  self.sl_q, self.sp_start, self.sp_len, self.btn_all,
                  self.cb_sr, self.cb_depth, self.chk_norm,
                  self.btn_export, self.btn_export_ab, self.btn_play_orig,
                  self.btn_play_proc, self.btn_play_ab, self.btn_stop):
            w.setEnabled(on)
        self._update_eq_enabled()

    def _update_eq_enabled(self):
        on = self.chk_eq.isChecked() and self.raw is not None
        for w in (self.sl_gain, self.sl_freq, self.sl_q):
            w.setEnabled(on)

    def _on_preset(self, idx):
        if self._applying or idx < 0 or idx >= CUSTOM_INDEX:
            return
        _, gain, freq, q, enabled = PRESETS[idx]
        self._applying = True
        self.chk_eq.setChecked(enabled)
        self.sl_gain.setValue(int(round(gain * 10)))
        self.sl_freq.setValue(int(round(freq)))
        self.sl_q.setValue(int(round(q * 100)))
        self._applying = False
        self._update_eq_enabled()
        self._update_summary()
        self._schedule_preview()

    def _on_param_changed(self, *_):
        if self._applying:
            return
        self._applying = True
        self.cb_preset.setCurrentIndex(CUSTOM_INDEX)
        self._applying = False
        self._update_eq_enabled()
        self._update_summary()
        self._schedule_preview()

    def _on_segment_changed(self, *_):
        if self._applying:
            return
        self._schedule_preview()

    def _select_all(self):
        if self.raw is None:
            return
        self._applying = True
        self.sp_start.setValue(0.0)
        self.sp_len.setValue(max(1.0, self.duration))
        self._applying = False
        self._schedule_preview()

    def _update_summary(self):
        if self.chk_eq.isChecked() and self.sl_gain.value() != 0:
            txt = (f"High-shelf <b>{self.sl_gain.value() / 10:+.1f} dB</b> "
                   f"từ <b>{fmt_hz(self.sl_freq.value())}</b>, Q {self.sl_q.value() / 100:.2f}")
        else:
            txt = "Không đổi EQ — chỉ upsample / đổi sample rate."
        self.lbl_summary.setText(txt)

    def _schedule_preview(self):
        if self.raw is not None:
            self._debounce.start()

    def _target_sr(self):
        return {0: self.sr, 1: 44100, 2: 48000, 3: 96000}.get(
            self.cb_sr.currentIndex(), 48000)

    def _subtype(self):
        return {0: "PCM_16", 1: "PCM_24", 2: "FLOAT"}[self.cb_depth.currentIndex()]

    # ---------------------------------------------------------------- file
    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở file âm thanh", "",
            "Âm thanh (*.wav *.flac *.mp3 *.ogg *.aiff *.aif *.w64);;Tất cả (*)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            x, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception as e:
            QMessageBox.critical(self, "Không đọc được file", f"{path}\n\n{e}")
            return
        if x.size == 0:
            QMessageBox.warning(self, "File rỗng", "File không có dữ liệu âm thanh.")
            return

        self._stop()
        self.raw, self.sr, self.path = x, sr, path
        self.duration = len(x) / sr
        self.env = waveform_envelope(x, sr)

        name = Path(path).name
        ch = "mono" if x.shape[1] == 1 else f"{x.shape[1]} kênh"
        self.lbl_file.setText(name)
        self.lbl_file.setToolTip(path)
        self.lbl_info.setText(f"{sr:,} Hz · {ch} · {fmt_time(self.duration)}")

        self._applying = True
        self.sp_start.setRange(0, max(0.0, self.duration))
        self.sp_start.setValue(0.0)
        self.sp_len.setRange(1.0, max(1.0, self.duration))
        self.sp_len.setValue(min(20.0, max(1.0, self.duration)))
        self._applying = False

        self._enable_controls(True)
        # de xu ly preview ngay (khong doi debounce)
        self._refresh_preview()
        self.statusBar().showMessage(
            f"Đã mở {name} — {sr:,} Hz, {ch}, {fmt_time(self.duration)}")

    # -------------------------------------------------------------- preview
    def _preview_segment(self):
        if self.raw is None:
            return None
        sr = self.sr
        start = min(float(self.sp_start.value()), max(0.0, self.duration - 0.05))
        length = max(0.05, float(self.sp_len.value()))
        i0 = int(round(start * sr))
        i1 = min(len(self.raw), int(round((start + length) * sr)))
        return self.raw[i0:i1], start, (i1 - i0) / sr

    def _refresh_preview(self):
        seg = self._preview_segment()
        if seg is None:
            return
        x, _, _ = seg
        sr_out = self._target_sr()
        t0 = time.perf_counter()
        try:
            proc, gain = process_audio(
                x, self.sr, sr_out,
                self.sl_gain.value() / 10.0, self.sl_freq.value(),
                self.sl_q.value() / 100.0,
                normalize=self.chk_norm.isChecked(),
                eq_enabled=self.chk_eq.isChecked())
            orig = resample_audio(x, self.sr, sr_out)
            if gain != 1.0:
                orig = np.clip(orig * gain, -1.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("Lỗi khi xử lý preview — xem console.")
            return

        self.prev_proc = proc
        self.prev_orig = orig
        self.prev_sr = sr_out
        self._draw()
        dt = (time.perf_counter() - t0) * 1000
        self.statusBar().showMessage(
            f"Preview {len(proc) / sr_out:.1f} s @ {sr_out / 1000:g} kHz — xử lý {dt:.0f} ms")

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, "Mở file âm thanh để xem waveform và phổ tần",
                ha="center", va="center", color=MUTED, fontsize=12,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        self.canvas.draw_idle()

    def _draw(self):
        if self.raw is None or self.prev_proc is None:
            self._draw_placeholder()
            return
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42,
                                   left=0.075, right=0.985, top=0.94, bottom=0.09)
        ax_w = self.fig.add_subplot(gs[0])
        ax_s = self.fig.add_subplot(gs[1])

        # --- waveform toan file
        t, lo, hi = self.env
        ax_w.set_facecolor(BG)
        ax_w.fill_between(t, lo, hi, color=ACCENT, alpha=0.55, linewidth=0)
        start = float(self.sp_start.value())
        end = min(self.duration, start + float(self.sp_len.value()))
        ax_w.axvspan(start, end, color=WARM, alpha=0.13, linewidth=0)
        ax_w.axvline(start, color=WARM, alpha=0.65, linewidth=1)
        ax_w.set_xlim(0, max(0.1, self.duration))
        ax_w.set_title(f"Waveform  ·  {Path(self.path).name}",
                       color=TEXT, fontsize=11, loc="left", pad=8)
        ax_w.set_ylabel("biên độ", color=MUTED, fontsize=9)
        self._style_axes(ax_w, grid_axis="y")
        ax_w.set_xlabel("giây", color=MUTED, fontsize=9)

        # --- pho tan
        f1, p1 = spectrum_db(self.prev_orig, self.prev_sr)
        f2, p2 = spectrum_db(self.prev_proc, self.prev_sr)
        ax_s.set_facecolor(BG)
        ax_s.plot(f1, p1, color=MUTED, linewidth=1.0, alpha=0.85,
                  label="Trước xử lý")
        if self.chk_eq.isChecked() and self.sl_gain.value() != 0:
            ax_s.plot(f2, p2, color=ACCENT, linewidth=1.5, label="Sau xử lý")
            ax_s.legend(loc="upper right", frameon=False, fontsize=9,
                        labelcolor=TEXT)
        ax_s.set_xscale("log")
        fmax = min(20000.0, self.prev_sr / 2.0)
        ax_s.set_xlim(30, fmax)
        top = float(np.max(p2)) + 6
        ax_s.set_ylim(top - 85, top)
        ax_s.set_title("Phổ tần  ·  so sánh trước / sau",
                       color=TEXT, fontsize=11, loc="left", pad=8)
        ax_s.set_ylabel("dB", color=MUTED, fontsize=9)
        ax_s.set_xlabel("Hz", color=MUTED, fontsize=9)
        ax_s.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax_s.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        self._style_axes(ax_s, grid_axis="both")

        self.canvas.draw_idle()

    @staticmethod
    def _style_axes(ax, grid_axis="both"):
        ax.grid(True, axis=grid_axis, color="#2a3547", alpha=0.5, linewidth=0.7)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#2a3547")

    def _on_canvas_click(self, event):
        if self.raw is None or event.inaxes is None or not self.fig.axes:
            return
        # chi xu ly khi bam vao subplot waveform (subplot dau tien)
        if event.inaxes is not self.fig.axes[0]:
            return
        if event.xdata is None:
            return
        start = max(0.0, min(event.xdata, max(0.0, self.duration - 0.1)))
        self._applying = True
        self.sp_start.setValue(round(start, 2))
        self._applying = False
        self._schedule_preview()

    # ------------------------------------------------------------- playback
    def play_original(self):
        if self.prev_orig is None:
            return
        self._play_array(self.prev_orig, "Gốc (đã resample)")

    def play_processed(self):
        if self.prev_proc is None:
            return
        self._play_array(self.prev_proc, "Sau xử lý")

    def play_ab(self):
        if self.prev_proc is None or self.prev_orig is None:
            return
        # goc 5s -> xu ly 5s -> lap lai
        ab = ab_combine(self.prev_orig, self.prev_proc, self.prev_sr, 5.0)
        self._play_array(ab, "A/B: gốc ↔ xử lý (5 s)")

    def _play_array(self, y, label):
        path = self._tmpdir / f"play_{next(self._tmp_count):03d}.wav"
        try:
            sf.write(path, np.clip(y, -1.0, 1.0), self.prev_sr, subtype="PCM_16")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi phát", str(e))
            return

        self._stop()
        self._last_wav = str(path)
        self._play_label = label
        self.lbl_playing.setText("● " + label)

        if self.player is not None:
            from PyQt5.QtMultimedia import QMediaContent
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(path))))
            self.player.setVolume(self.sl_vol.value())
            self.player.play()
        elif os.name == "nt":
            import winsound
            winsound.PlaySound(str(path),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)

    def _stop(self):
        if self.player is not None:
            self.player.stop()
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self.lbl_playing.setText("")

    # --------------------------------------------------------------- export
    def _default_name(self, suffix=""):
        stem = Path(self.path).stem if self.path else "audio"
        sr = self._target_sr()
        return f"{stem}_{sr / 1000:g}k_air{suffix}.wav"

    def _do_export(self, path, y, sr):
        sf.write(path, np.clip(y, -1.0, 1.0), sr, subtype=self._subtype())
        return Path(path).stat().st_size / (1024 * 1024)

    def export_dialog(self):
        if self.raw is None:
            return
        self._stop()
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất WAV", str(Path.home() / "Music" / self._default_name()),
            "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"

        self._set_busy(True, "Đang xử lý toàn bộ file…")
        QApplication.processEvents()
        t0 = time.perf_counter()
        try:
            y, _ = process_audio(
                self.raw, self.sr, self._target_sr(),
                self.sl_gain.value() / 10.0, self.sl_freq.value(),
                self.sl_q.value() / 100.0,
                normalize=self.chk_norm.isChecked(),
                eq_enabled=self.chk_eq.isChecked())
            mb = self._do_export(path, y, self._target_sr())
        except Exception as e:
            self._set_busy(False)
            traceback.print_exc()
            QMessageBox.critical(self, "Xuất thất bại", str(e))
            return
        self._set_busy(False)
        dt = time.perf_counter() - t0
        self.statusBar().showMessage(f"Đã xuất {path} ({mb:.1f} MB, {dt:.1f} s)")
        QMessageBox.information(
            self, "Xuất xong",
            f"Đã lưu:\n{path}\n\n"
            f"{self._target_sr():,} Hz · {self.cb_depth.currentText()} · "
            f"{self.duration:.1f} s · {mb:.1f} MB\n"
            f"Thời gian xử lý: {dt:.1f} s")

    def export_ab_dialog(self):
        if self.raw is None:
            return
        self._stop()
        path, _ = QFileDialog.getSaveFileName(
            self, "Xuất bản A/B 5 giây",
            str(Path.home() / "Music" / self._default_name("_AB5s")),
            "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"

        seg = self._preview_segment()
        if seg is None:
            return
        x, _, _ = seg
        self._set_busy(True, "Đang tạo bản A/B…")
        QApplication.processEvents()
        try:
            proc, gain = process_audio(
                x, self.sr, self._target_sr(),
                self.sl_gain.value() / 10.0, self.sl_freq.value(),
                self.sl_q.value() / 100.0,
                normalize=self.chk_norm.isChecked(),
                eq_enabled=self.chk_eq.isChecked())
            orig = resample_audio(x, self.sr, self._target_sr())
            if gain != 1.0:
                orig = np.clip(orig * gain, -1.0, 1.0).astype(np.float32, copy=False)
            ab = ab_combine(orig, proc, self._target_sr(), 5.0)
            mb = self._do_export(path, ab, self._target_sr())
        except Exception as e:
            self._set_busy(False)
            traceback.print_exc()
            QMessageBox.critical(self, "Xuất thất bại", str(e))
            return
        self._set_busy(False)
        self.statusBar().showMessage(f"Đã xuất bản A/B: {path} ({mb:.1f} MB)")
        QMessageBox.information(
            self, "Xuất xong",
            f"Đã lưu bản A/B:\n{path}\n\n"
            "Cứ 5 giây đổi giữa gốc và sau xử lý (bắt đầu bằng gốc).")

    def _set_busy(self, busy, msg=""):
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage(msg)
            self.setEnabled(False)
            QApplication.processEvents()
        else:
            self.setEnabled(True)
            QApplication.restoreOverrideCursor()

    # --------------------------------------------------------------- events
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTS:
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if Path(p).suffix.lower() in AUDIO_EXTS:
                self.load_file(p)
                e.acceptProposedAction()
                return
        e.ignore()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.play_processed()
        elif e.key() == Qt.Key_P:
            self.play_original()
        elif e.key() == Qt.Key_Escape:
            self._stop()
        elif e.key() == Qt.Key_O and e.modifiers() & Qt.ControlModifier:
            self.open_dialog()
        elif e.key() == Qt.Key_E and e.modifiers() & Qt.ControlModifier:
            self.export_dialog()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self._stop()
        try:
            if self.player is not None:
                from PyQt5.QtMultimedia import QMediaContent
                self.player.setMedia(QMediaContent())
        except Exception:
            pass
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().closeEvent(e)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    w = MainWindow()

    # Kiem tra nhanh ban dong goi: xu ly 1 file va ghi ket qua ra temp
    if "--self-test" in sys.argv:
        src = next((a for a in sys.argv[1:] if not a.startswith("--")), "")
        out = Path(tempfile.gettempdir()) / "audioair_selftest.txt"
        try:
            if not src or not Path(src).exists():
                raise FileNotFoundError(f"khong thay file test: {src!r}")
            w.load_file(src)
            w._refresh_preview()
            seg = w._preview_segment()
            y, _ = process_audio(seg[0], w.sr, 48000, 2.0, 8000, 0.707)
            info = f"OK {len(y)} samples | {w.raw.shape} @ {w.sr} Hz | preview {w.prev_proc.shape}"
            out.write_text(info, encoding="utf-8")
            print(info)
            return 0
        except Exception:
            tb = traceback.format_exc()
            out.write_text("FAIL\n" + tb, encoding="utf-8")
            print(tb, file=sys.stderr)
            return 1

    w.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        w.load_file(sys.argv[1])
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
