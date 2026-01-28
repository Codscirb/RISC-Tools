"""Image Reviewer v1.2 (Windows)
- Recursively scans a root folder for PNG/JPG/JPEG/EXR images
- Left panel: folder tree + image list + text filter + status/tag filters + file type filters
- Viewer: fit-to-window, zoom wheel, pan drag
- Tagging/approval persisted to SQLite (in %APPDATA%/ImageReviewer/reviews.db)
- Export reviewed data to CSV/JSON

EXR support:
- Uses OpenImageIO + numpy if installed
- Adds exposure slider for EXR preview
- Adds "layer" dropdown based on EXR channel groups

Install:
  python -m pip install PySide6
  # For EXR:
  python -m pip install OpenImageIO numpy

Run:
  python image_reviewer.py

Package to .exe (later):
  pyinstaller --noconsole --onefile image_reviewer.py
"""

from __future__ import annotations

import os
import sys
import csv
import json
import sqlite3
import pathlib
from typing import Dict, List, Optional, Set, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# Optional EXR support
try:
    import OpenImageIO as oiio  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    oiio = None  # type: ignore
    np = None  # type: ignore


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".exr"}
APP_NAME = "ImageReviewer"

STANDARD_TAGS: List[str] = ["Pass", "Fail", "Lighting", "Material", "Mesh"]
STATUS_CHOICES: List[str] = ["", "Approved", "Rejected", "NeedsWork"]

# NEW: file type filter buckets (JPG includes .jpg + .jpeg)
FILETYPE_BUCKETS: Dict[str, Set[str]] = {
    "PNG": {".png"},
    "JPG": {".jpg", ".jpeg"},
    "EXR": {".exr"},
}


def appdata_dir() -> pathlib.Path:
    base = os.environ.get("APPDATA") or str(pathlib.Path.home())
    p = pathlib.Path(base) / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonicalize_tags(tags: Set[str]) -> str:
    canon = []
    std_map = {t.lower(): t for t in STANDARD_TAGS}
    for t in sorted({x.strip() for x in tags if x.strip()}, key=lambda s: s.lower()):
        key = t.lower()
        canon.append(std_map.get(key, t))
    return ", ".join(canon)


def parse_tags(csv_text: str) -> Set[str]:
    if not csv_text:
        return set()
    parts = [p.strip() for p in csv_text.split(",")]
    return {p for p in parts if p}


# ---------------- EXR helpers ----------------

def _srgb_preview_from_linear(rgb: "np.ndarray", exposure_stops: float = 0.0) -> "np.ndarray":
    """Linear float RGB -> uint8 sRGB preview.

    Simple pipeline for review:
      exposure -> Reinhard tonemap -> gamma
    """
    if np is None:
        raise RuntimeError("numpy not available")

    rgb = rgb * (2.0 ** float(exposure_stops))
    rgb = rgb / (1.0 + rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _channel_role(name: str) -> str:
    """Return channel semantic: r/g/b/a or other."""
    n = name.lower()
    if n in ("r", "red"):
        return "r"
    if n in ("g", "green"):
        return "g"
    if n in ("b", "blue"):
        return "b"
    if n in ("a", "alpha"):
        return "a"
    return n


def exr_layers(path: str) -> Dict[str, Dict[str, int]]:
    """Return mapping: layer_name -> {role: channel_index}.

    Layer name is the channel prefix before the last '.', e.g. 'beauty.R' -> layer 'beauty'.
    Channels with no prefix end up in layer 'default'.
    """
    if oiio is None:
        return {}

    inp = oiio.ImageInput.open(path)
    if inp is None:
        return {}

    try:
        spec = inp.spec()
        names = list(getattr(spec, "channelnames", []) or [])
        out: Dict[str, Dict[str, int]] = {}
        for idx, ch in enumerate(names):
            if "." in ch:
                layer, leaf = ch.rsplit(".", 1)
            else:
                layer, leaf = "default", ch
            role = _channel_role(leaf)
            out.setdefault(layer, {})[role] = idx
        return out
    finally:
        inp.close()


def load_image_qimage(path: str, exposure_stops: float = 0.0, layer: str = "default") -> QtGui.QImage:
    """Load PNG/JPG via Qt, EXR via OpenImageIO (if installed)."""
    ext = os.path.splitext(path)[1].lower()
    if ext != ".exr":
        return QtGui.QImage(path)

    if oiio is None or np is None:
        return QtGui.QImage()

    inp = oiio.ImageInput.open(path)
    if inp is None:
        return QtGui.QImage()

    try:
        spec = inp.spec()
        w, h, nch = int(spec.width), int(spec.height), int(spec.nchannels)
        pixels = inp.read_image(format=oiio.FLOAT)
        if pixels is None:
            return QtGui.QImage()

        arr = np.array(pixels, dtype=np.float32).reshape((h, w, nch))

        # Build layer selection mapping
        layers = exr_layers(path)
        sel = layers.get(layer) or layers.get("default") or {}

        def channel(idx: int) -> "np.ndarray":
            return arr[:, :, idx]

        # Prefer RGB if present
        if all(k in sel for k in ("r", "g", "b")):
            rgb = np.stack([channel(sel["r"]), channel(sel["g"]), channel(sel["b"])], axis=2)
        else:
            # Fall back to first channel as grayscale
            rgb = np.repeat(arr[:, :, 0:1], 3, axis=2)

        rgb8 = _srgb_preview_from_linear(rgb, exposure_stops=exposure_stops)

        # Alpha if present
        if "a" in sel:
            a = np.clip(channel(sel["a"]), 0.0, 1.0)
            a8 = (a * 255.0 + 0.5).astype(np.uint8)
            rgba = np.dstack([rgb8, a8])
            qimg = QtGui.QImage(
                rgba.data, w, h, int(rgba.strides[0]), QtGui.QImage.Format.Format_RGBA8888
            )
            return qimg.copy()

        qimg = QtGui.QImage(rgb8.data, w, h, int(rgb8.strides[0]), QtGui.QImage.Format.Format_RGB888)
        return qimg.copy()
    finally:
        inp.close()


# ---------------- DB ----------------


class ReviewDB:
    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
              path TEXT PRIMARY KEY,
              status TEXT NOT NULL DEFAULT '',
              tags TEXT NOT NULL DEFAULT '',
              notes TEXT NOT NULL DEFAULT '',
              updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );
            """
        )
        self.conn.commit()

    def get(self, path: str) -> Tuple[str, str, str, int]:
        cur = self.conn.execute(
            "SELECT status, tags, notes, updated_at FROM reviews WHERE path=?", (path,)
        )
        row = cur.fetchone()
        if not row:
            return "", "", "", 0
        return row[0] or "", row[1] or "", row[2] or "", int(row[3] or 0)

    def get_many(self, paths: List[str]) -> Dict[str, Tuple[str, str, str, int]]:
        out: Dict[str, Tuple[str, str, str, int]] = {}
        if not paths:
            return out
        chunk = 800
        for i in range(0, len(paths), chunk):
            batch = paths[i : i + chunk]
            placeholders = ",".join(["?"] * len(batch))
            q = (
                f"SELECT path, status, tags, notes, updated_at FROM reviews WHERE path IN ({placeholders})"
            )
            for row in self.conn.execute(q, batch):
                out[row[0]] = (row[1] or "", row[2] or "", row[3] or "", int(row[4] or 0))
        return out

    def upsert(self, path: str, status: str, tags: str, notes: str) -> None:
        self.conn.execute(
            """
            INSERT INTO reviews(path, status, tags, notes, updated_at)
            VALUES(?,?,?,?,strftime('%s','now'))
            ON CONFLICT(path) DO UPDATE SET
              status=excluded.status,
              tags=excluded.tags,
              notes=excluded.notes,
              updated_at=strftime('%s','now');
            """,
            (path, status, tags, notes),
        )
        self.conn.commit()

    def export_rows(self, root_dir: str) -> List[Tuple[str, str, str, str, int]]:
        like = os.path.join(root_dir, "") + "%"
        cur = self.conn.execute(
            "SELECT path, status, tags, notes, updated_at FROM reviews WHERE path LIKE ? ORDER BY path ASC",
            (like,),
        )
        return [(r[0], r[1] or "", r[2] or "", r[3] or "", int(r[4] or 0)) for r in cur.fetchall()]


# ---------------- Viewer ----------------


class ZoomPanGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.5 if delta > 0 else 0.67
        self.scale(factor, factor)

    def fitInViewSmart(self) -> None:
        items = self.scene().items()
        if not items:
            return
        rect = items[0].boundingRect()
        if rect.isNull():
            return
        self.resetTransform()
        self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)


# ---------------- App ----------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Reviewer")
        self.resize(1500, 900)

        self.db = ReviewDB(appdata_dir() / "reviews.db")

        self.root_dir: Optional[str] = None
        self.all_images: List[str] = []
        self.filtered_images: List[str] = []
        self.current_index: int = -1

        self.review_cache: Dict[str, Tuple[str, str, str, int]] = {}

        # EXR UI state
        self.exposure_stops: float = 0.0
        self.current_layer: str = "default"

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        self.action_open = QtGui.QAction("Open Folder", self)
        self.action_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        tb.addAction(self.action_open)

        self.action_export_csv = QtGui.QAction("Export CSV", self)
        tb.addAction(self.action_export_csv)

        self.action_export_json = QtGui.QAction("Export JSON", self)
        tb.addAction(self.action_export_json)

        tb.addSeparator()

        self.btn_prev = QtWidgets.QToolButton(text="◀ Prev")
        self.btn_next = QtWidgets.QToolButton(text="Next ▶")
        tb.addWidget(self.btn_prev)
        tb.addWidget(self.btn_next)

        tb.addSeparator()

        self.btn_fit = QtWidgets.QToolButton(text="Fit")
        self.btn_actual = QtWidgets.QToolButton(text="100%")
        tb.addWidget(self.btn_fit)
        tb.addWidget(self.btn_actual)

        tb.addSeparator()

        self.status_label = QtWidgets.QLabel("Status: -")
        tb.addWidget(self.status_label)

        splitter = QtWidgets.QSplitter(orientation=QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ---------- Left panel ----------
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText(
            "Text filter (filename/path). Tip: type tag:Lighting or status:Approved"
        )
        left_layout.addWidget(self.filter_edit)

        filters_row = QtWidgets.QHBoxLayout()

        self.status_filter = QtWidgets.QComboBox()
        self.status_filter.addItems(["All Status", "Approved", "Rejected", "NeedsWork", "Unreviewed"])
        filters_row.addWidget(self.status_filter)

        # NEW: file type filter group
        self.filetype_filter_box = QtWidgets.QGroupBox("File types")
        ft_row = QtWidgets.QHBoxLayout(self.filetype_filter_box)
        ft_row.setContentsMargins(8, 8, 8, 8)
        self.filetype_filter_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for label in ("PNG", "JPG", "EXR"):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)  # default: show everything
            self.filetype_filter_checks[label] = cb
            ft_row.addWidget(cb)
        ft_row.addStretch(1)
        filters_row.addWidget(self.filetype_filter_box, 0)

        self.tag_filter_box = QtWidgets.QGroupBox("Tag filter")
        tag_row = QtWidgets.QHBoxLayout(self.tag_filter_box)
        tag_row.setContentsMargins(8, 8, 8, 8)
        self.tag_filter_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for t in STANDARD_TAGS:
            cb = QtWidgets.QCheckBox(t)
            self.tag_filter_checks[t] = cb
            tag_row.addWidget(cb)
        tag_row.addStretch(1)
        filters_row.addWidget(self.tag_filter_box, 1)

        left_layout.addLayout(filters_row)

        self.dir_model = QtWidgets.QFileSystemModel()
        self.dir_model.setFilter(QtCore.QDir.Filter.AllDirs | QtCore.QDir.Filter.NoDotAndDotDot)

        self.tree = QtWidgets.QTreeView(model=self.dir_model)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        left_layout.addWidget(self.tree, 2)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        left_layout.addWidget(self.list, 3)

        splitter.addWidget(left)

        # ---------- Right panel ----------
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.viewer_scene = QtWidgets.QGraphicsScene()
        self.viewer = ZoomPanGraphicsView()
        self.viewer.setScene(self.viewer_scene)
        right_layout.addWidget(self.viewer, 10)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.path_label = QtWidgets.QLabel("-")
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        form.addRow("Path:", self.path_label)

        # EXR controls row
        self.exr_controls = QtWidgets.QWidget()
        exr_row = QtWidgets.QHBoxLayout(self.exr_controls)
        exr_row.setContentsMargins(0, 0, 0, 0)
        exr_row.setSpacing(10)

        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.setMinimumWidth(220)
        exr_row.addWidget(QtWidgets.QLabel("Layer"))
        exr_row.addWidget(self.layer_combo)

        self.exposure_value = QtWidgets.QLabel("0.0")
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.exposure_slider.setMinimum(-60)   # -6.0 stops
        self.exposure_slider.setMaximum(60)    # +6.0 stops
        self.exposure_slider.setValue(0)
        self.exposure_slider.setSingleStep(1)  # 0.1 stop
        self.exposure_slider.setPageStep(5)
        exr_row.addWidget(QtWidgets.QLabel("Exposure"))
        exr_row.addWidget(self.exposure_slider, 1)
        exr_row.addWidget(self.exposure_value)

        form.addRow("EXR:", self.exr_controls)

        # Standard tags
        self.assign_tag_box = QtWidgets.QGroupBox("Tags")
        assign_row = QtWidgets.QHBoxLayout(self.assign_tag_box)
        assign_row.setContentsMargins(8, 8, 8, 8)
        self.assign_tag_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for t in STANDARD_TAGS:
            cb = QtWidgets.QCheckBox(t)
            self.assign_tag_checks[t] = cb
            assign_row.addWidget(cb)
        assign_row.addStretch(1)
        form.addRow("Standard tags:", self.assign_tag_box)

        self.extra_tags_edit = QtWidgets.QLineEdit()
        self.extra_tags_edit.setPlaceholderText("extra tags (comma-separated), optional")
        form.addRow("Extra tags:", self.extra_tags_edit)

        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Notes...")
        self.notes_edit.setFixedHeight(110)
        form.addRow("Notes:", self.notes_edit)

        right_layout.addLayout(form, 0)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, activated=self.prev_image)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, activated=self.next_image)
        QtGui.QShortcut(QtGui.QKeySequence("A"), self, activated=lambda: self.set_status("Approved"))
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=lambda: self.set_status("Rejected"))
        QtGui.QShortcut(QtGui.QKeySequence("N"), self, activated=lambda: self.set_status("NeedsWork"))
        QtGui.QShortcut(QtGui.QKeySequence("0"), self, activated=lambda: self.set_status(""))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self, activated=self.filter_edit.setFocus)
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, activated=self.extra_tags_edit.setFocus)

        menu = self.menuBar().addMenu("File")
        menu.addAction(self.action_open)
        menu.addSeparator()
        menu.addAction(self.action_export_csv)
        menu.addAction(self.action_export_json)
        menu.addSeparator()
        menu.addAction("Exit", self.close)

        # Default EXR controls hidden unless EXR is selected
        self.exr_controls.setVisible(False)

    def _connect_signals(self) -> None:
        self.action_open.triggered.connect(self.open_folder)
        self.action_export_csv.triggered.connect(self.export_csv)
        self.action_export_json.triggered.connect(self.export_json)

        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_fit.clicked.connect(self.viewer.fitInViewSmart)
        self.btn_actual.clicked.connect(self.viewer.resetTransform)

        self.filter_edit.textChanged.connect(self.apply_filter)
        self.status_filter.currentIndexChanged.connect(self.apply_filter)
        for cb in self.tag_filter_checks.values():
            cb.stateChanged.connect(self.apply_filter)

        # NEW: file type filter signals
        for cb in self.filetype_filter_checks.values():
            cb.stateChanged.connect(self.apply_filter)

        self.list.currentRowChanged.connect(self._on_list_row_changed)
        self.tree.clicked.connect(self._on_tree_clicked)

        # Review edits
        self.extra_tags_edit.editingFinished.connect(self.save_current_review)
        for cb in self.assign_tag_checks.values():
            cb.stateChanged.connect(self.save_current_review)

        self.notes_edit.textChanged.connect(self._debounced_save)
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_current_review)

        # EXR controls
        self.exposure_slider.valueChanged.connect(self._on_exposure_changed)
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)

    # ---------- Folder scan ----------
    def open_folder(self) -> None:
        start_dir = self.root_dir or str(pathlib.Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select root folder", start_dir)
        if not folder:
            return

        self.root_dir = folder
        self.statusBar().showMessage(f"Scanning: {folder}")

        self.dir_model.setRootPath(folder)
        self.tree.setRootIndex(self.dir_model.index(folder))
        self.tree.expandToDepth(1)

        self.all_images = self._scan_images(folder)
        self.review_cache = self.db.get_many(self.all_images)
        self.statusBar().showMessage(f"Found {len(self.all_images)} images", 4000)

        self.apply_filter()
        if self.filtered_images:
            self.set_current_by_index(0)

    def _scan_images(self, root: str) -> List[str]:
        out: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMAGE_EXTS:
                    out.append(os.path.join(dirpath, fn))
        out.sort(key=lambda p: p.lower())
        return out

    # ---------- Filtering ----------
    def _filter_tokens(self, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        status = None
        tag = None
        parts = [p.strip() for p in (text or "").split() if p.strip()]
        keep: List[str] = []
        for p in parts:
            pl = p.lower()
            if pl.startswith("status:"):
                status = p.split(":", 1)[1].strip()
                continue
            if pl.startswith("tag:"):
                tag = p.split(":", 1)[1].strip()
                continue
            keep.append(p)
        return " ".join(keep), status, tag

    def _selected_filter_tags(self) -> Set[str]:
        return {t for t, cb in self.tag_filter_checks.items() if cb.isChecked()}

    # NEW: selected filetype labels (PNG/JPG/EXR)
    def _selected_filetypes(self) -> Set[str]:
        selected = {k for k, cb in self.filetype_filter_checks.items() if cb.isChecked()}
        # If user unchecks everything, treat it as "show all" (less confusing).
        return selected or set(self.filetype_filter_checks.keys())

    def apply_filter(self) -> None:
        if not self.all_images:
            self.filtered_images = []
            self._rebuild_list()
            return

        raw = (self.filter_edit.text() or "").strip()
        free_text, status_token, tag_token = self._filter_tokens(raw)
        free_text_l = free_text.lower()

        status_combo = self.status_filter.currentText()
        status_filter: Optional[str]
        if status_combo == "All Status":
            status_filter = None
        elif status_combo == "Unreviewed":
            status_filter = "__UNREVIEWED__"
        else:
            status_filter = status_combo
        if status_token:
            status_filter = status_token

        tag_filters = self._selected_filter_tags()
        if tag_token:
            tag_filters.add(tag_token)

        # NEW: allowed extensions derived from selected file types
        selected_types = self._selected_filetypes()
        allowed_exts: Set[str] = set()
        for t in selected_types:
            allowed_exts |= FILETYPE_BUCKETS.get(t, set())

        def matches(path: str) -> bool:
            # file type filter first (cheap)
            ext = os.path.splitext(path)[1].lower()
            if allowed_exts and ext not in allowed_exts:
                return False

            if free_text_l and free_text_l not in path.lower():
                return False

            status, tags_csv, _notes, _updated = self.review_cache.get(path, ("", "", "", 0))

            if status_filter:
                if status_filter == "__UNREVIEWED__":
                    if status:
                        return False
                else:
                    if status.lower() != status_filter.lower():
                        return False

            if tag_filters:
                tags = {t.strip().lower() for t in tags_csv.split(",") if t.strip()}
                for req in tag_filters:
                    if req.strip().lower() not in tags:
                        return False

            return True

        self.filtered_images = [p for p in self.all_images if matches(p)]
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.filtered_images:
            rel = os.path.relpath(p, self.root_dir) if self.root_dir else p
            status, tags_csv, _notes, _updated = self.review_cache.get(p, ("", "", "", 0))
            prefix = ""
            if status:
                prefix += f"[{status}] "
            if tags_csv:
                prefix += f"({tags_csv}) "
            self.list.addItem(QtWidgets.QListWidgetItem(prefix + rel))
        self.list.blockSignals(False)

        if self.filtered_images:
            if 0 <= self.current_index < len(self.filtered_images):
                self.list.setCurrentRow(self.current_index)
            else:
                self.list.setCurrentRow(0)

    def _on_list_row_changed(self, row: int) -> None:
        if 0 <= row < len(self.filtered_images):
            self.set_current_by_index(row)

    def _on_tree_clicked(self, index: QtCore.QModelIndex) -> None:
        folder_path = self.dir_model.filePath(index)
        if self.root_dir and folder_path and folder_path.startswith(self.root_dir):
            rel = os.path.relpath(folder_path, self.root_dir).replace("\\", "/")
            self.filter_edit.setText(rel)

    # ---------- Navigation / Display ----------
    def set_current_by_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.filtered_images):
            return
        self.save_current_review()
        self.current_index = idx
        path = self.filtered_images[idx]
        self._load_image(path)
        self._load_review(path)

        self.list.blockSignals(True)
        self.list.setCurrentRow(idx)
        self.list.blockSignals(False)

    def prev_image(self) -> None:
        if self.filtered_images:
            self.set_current_by_index(max(0, self.current_index - 1))

    def next_image(self) -> None:
        if self.filtered_images:
            self.set_current_by_index(min(len(self.filtered_images) - 1, self.current_index + 1))

    def _load_image(self, path: str) -> None:
        self.viewer_scene.clear()

        ext = os.path.splitext(path)[1].lower()
        is_exr = ext == ".exr"

        # Show/hide EXR controls
        self.exr_controls.setVisible(is_exr)

        if is_exr:
            if oiio is None or np is None:
                self.statusBar().showMessage(
                    "EXR support not installed. Run: python -m pip install OpenImageIO numpy", 9000
                )
                img = QtGui.QImage()
            else:
                # Populate layers
                layers = exr_layers(path)
                layer_names = sorted(layers.keys(), key=lambda s: s.lower()) or ["default"]

                self.layer_combo.blockSignals(True)
                self.layer_combo.clear()
                self.layer_combo.addItems(layer_names)

                # Keep current_layer if present
                if self.current_layer in layer_names:
                    self.layer_combo.setCurrentText(self.current_layer)
                else:
                    self.current_layer = layer_names[0]
                    self.layer_combo.setCurrentText(self.current_layer)
                self.layer_combo.blockSignals(False)

                img = load_image_qimage(path, exposure_stops=self.exposure_stops, layer=self.current_layer)
        else:
            img = load_image_qimage(path)

        if img.isNull():
            txt = self.viewer_scene.addText("Failed to load image")
            txt.setDefaultTextColor(QtGui.QColor("red"))
            self.path_label.setText(path)
            self.statusBar().showMessage(f"Failed to load: {path}", 5000)
            return

        pix = QtGui.QPixmap.fromImage(img)
        item = self.viewer_scene.addPixmap(pix)
        item.setTransformationMode(QtCore.Qt.TransformationMode.SmoothTransformation)

        self.path_label.setText(path)
        self.viewer.fitInViewSmart()

    # ---------- EXR controls ----------
    def _on_exposure_changed(self, value: int) -> None:
        self.exposure_stops = float(value) / 10.0
        self.exposure_value.setText(f"{self.exposure_stops:.1f}")
        self._rerender_if_exr()

    def _on_layer_changed(self, text: str) -> None:
        self.current_layer = text or "default"
        self._rerender_if_exr()

    def _rerender_if_exr(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.filtered_images):
            return
        path = self.filtered_images[self.current_index]
        if os.path.splitext(path)[1].lower() == ".exr":
            # Keep current zoom/pan? For now we re-fit (simple + reliable).
            self._load_image(path)

    # ---------- Review persistence ----------
    def _debounced_save(self) -> None:
        self._save_timer.start(600)

    def _current_status(self) -> str:
        txt = self.status_label.text().replace("Status:", "").strip()
        return "" if txt == "-" else txt

    def _current_tags_csv(self) -> str:
        tags: Set[str] = set()
        for t, cb in self.assign_tag_checks.items():
            if cb.isChecked():
                tags.add(t)
        tags |= parse_tags(self.extra_tags_edit.text() or "")
        return canonicalize_tags(tags)

    def _load_review(self, path: str) -> None:
        status, tags_csv, notes, _updated = self.review_cache.get(path) or self.db.get(path)

        self.status_label.setText(f"Status: {status or '-'}")

        tags_set = {t.strip() for t in tags_csv.split(",") if t.strip()}
        std_lower = {t.lower() for t in STANDARD_TAGS}
        tags_lower = {t.lower() for t in tags_set}

        for t, cb in self.assign_tag_checks.items():
            cb.blockSignals(True)
            cb.setChecked(t.lower() in tags_lower)
            cb.blockSignals(False)

        extras = sorted([t for t in tags_set if t.lower() not in std_lower], key=lambda s: s.lower())
        self.extra_tags_edit.blockSignals(True)
        self.extra_tags_edit.setText(", ".join(extras))
        self.extra_tags_edit.blockSignals(False)

        self.notes_edit.blockSignals(True)
        self.notes_edit.setPlainText(notes)
        self.notes_edit.blockSignals(False)

    def save_current_review(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.filtered_images):
            return
        path = self.filtered_images[self.current_index]
        status = self._current_status()
        tags_csv = self._current_tags_csv()
        notes = (self.notes_edit.toPlainText() or "").strip()

        self.db.upsert(path, status, tags_csv, notes)
        self.review_cache[path] = (status, tags_csv, notes, int(QtCore.QDateTime.currentSecsSinceEpoch()))
        self._refresh_current_list_item()

    def _refresh_current_list_item(self) -> None:
        if self.current_index < 0 or self.current_index >= self.list.count():
            return
        path = self.filtered_images[self.current_index]
        rel = os.path.relpath(path, self.root_dir) if self.root_dir else path
        status, tags_csv, _notes, _updated = self.review_cache.get(path, ("", "", "", 0))
        prefix = ""
        if status:
            prefix += f"[{status}] "
        if tags_csv:
            prefix += f"({tags_csv}) "
        self.list.item(self.current_index).setText(prefix + rel)

    def set_status(self, status: str) -> None:
        if self.current_index < 0:
            return
        self.status_label.setText(f"Status: {status or '-'}")
        self.save_current_review()

    # ---------- Export ----------
    def _ensure_root(self) -> bool:
        if not self.root_dir:
            QtWidgets.QMessageBox.information(self, "No folder", "Open a folder first.")
            return False
        return True

    def export_csv(self) -> None:
        if not self._ensure_root():
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export reviews to CSV",
            str(pathlib.Path(self.root_dir) / "reviews.csv"),
            "CSV Files (*.csv)",
        )
        if not out_path:
            return
        rows = self.db.export_rows(self.root_dir)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "status", "tags", "notes", "updated_at"])
            w.writerows(rows)
        self.statusBar().showMessage(f"Exported CSV: {out_path}", 5000)

    def export_json(self) -> None:
        if not self._ensure_root():
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export reviews to JSON",
            str(pathlib.Path(self.root_dir) / "reviews.json"),
            "JSON Files (*.json)",
        )
        if not out_path:
            return
        rows = self.db.export_rows(self.root_dir)
        payload = [
            {"path": r[0], "status": r[1], "tags": r[2], "notes": r[3], "updated_at": r[4]} for r in rows
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.statusBar().showMessage(f"Exported JSON: {out_path}", 5000)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
