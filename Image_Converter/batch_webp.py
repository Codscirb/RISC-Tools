#!/usr/bin/env python3
"""batch_webp.py

Batch-convert images (JPG/PNG/EXR) to WebP, recursively, preserving folder structure.
Non-destructive by default: writes to an output root directory.

Includes:
- CLI mode (scriptable)
- GUI mode (PySide6) with progress + log + cancel

EXR notes:
- WebP is 8-bit. EXR (often float/HDR) will be tone-mapped to 8-bit using exposure + gamma.
- Requires OpenImageIO for EXR support.

Install:
  python -m pip install pillow
  # For EXR:
  python -m pip install OpenImageIO numpy
  # For GUI:
  python -m pip install PySide6

Usage examples:
  # GUI:
  python batch_webp.py --gui

  # CLI:
  python batch_webp.py --in D:/renders --out D:/renders_webp --quality 85
  python batch_webp.py --in ./input --out ./out --quality 82 --method 6 --include-exr
  python batch_webp.py --in ./input --out ./out --lossless-png

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image


# -----------------------------
# Optional EXR support via OIIO
# -----------------------------
try:
    import OpenImageIO as oiio  # type: ignore
except Exception:
    oiio = None


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".exr"}


@dataclass
class ConvertOptions:
    quality: int = 85
    method: int = 6
    overwrite: bool = False
    include_exr: bool = True
    lossless_png: bool = False
    exr_exposure: float = 0.0
    exr_gamma: float = 2.2
    exr_clamp: Tuple[float, float] = (0.0, 1.0)
    exr_keep_alpha: bool = True


def iter_images(root: Path, include_exr: bool) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        if ext == ".exr" and not include_exr:
            continue
        yield p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def out_path_for(in_path: Path, in_root: Path, out_root: Path) -> Path:
    rel = in_path.relative_to(in_root)
    return (out_root / rel).with_suffix(".webp")


def save_pil_as_webp(img: Image.Image, dst: Path, opts: ConvertOptions, *, lossless: bool = False) -> None:
    ensure_parent(dst)

    save_kwargs = {
        "format": "WEBP",
        "quality": int(opts.quality),
        "method": int(opts.method),
    }

    # Preserve alpha when present
    if img.mode in ("RGBA", "LA"):
        pass
    else:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

    if lossless:
        save_kwargs["lossless"] = True

    img.save(dst, **save_kwargs)


def convert_jpg_png(src: Path, dst: Path, opts: ConvertOptions) -> None:
    try:
        with Image.open(src) as im:
            im.load()

            if im.mode == "P":
                im = im.convert("RGBA" if "transparency" in im.info else "RGB")
            elif im.mode in ("RGBA", "LA"):
                pass
            else:
                im = im.convert("RGB")

            lossless = opts.lossless_png and src.suffix.lower() == ".png"
            save_pil_as_webp(im, dst, opts, lossless=lossless)
    except Exception as e:
        raise RuntimeError(f"Failed converting {src}: {e}")


def read_exr_to_uint8_rgba(
    src: Path,
    exposure: float,
    gamma: float,
    clamp_min: float,
    clamp_max: float,
    keep_alpha: bool,
) -> Image.Image:
    if oiio is None:
        raise RuntimeError(
            "OpenImageIO is not installed. Install with: python -m pip install OpenImageIO numpy"
        )

    inp = oiio.ImageInput.open(str(src))
    if inp is None:
        raise RuntimeError("OIIO could not open file")

    try:
        spec = inp.spec()
        w, h, nchans = spec.width, spec.height, spec.nchannels

        pixels = inp.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError("OIIO failed to read image")

        arr = np.array(pixels, dtype=np.float32).reshape(h, w, nchans)

        rgb = arr[:, :, :3]

        if exposure != 0.0:
            rgb = rgb * (2.0 ** float(exposure))

        rgb = np.clip(rgb, float(clamp_min), float(clamp_max))

        if gamma and gamma != 1.0:
            rgb = np.power(rgb, 1.0 / float(gamma))

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb8 = (rgb * 255.0 + 0.5).astype(np.uint8)

        if keep_alpha and nchans >= 4:
            a = arr[:, :, 3]
            a = np.clip(a, 0.0, 1.0)
            a8 = (a * 255.0 + 0.5).astype(np.uint8)
            rgba8 = np.dstack([rgb8, a8])
            return Image.fromarray(rgba8, mode="RGBA")

        return Image.fromarray(rgb8, mode="RGB")

    finally:
        inp.close()


def convert_exr(src: Path, dst: Path, opts: ConvertOptions) -> None:
    im = read_exr_to_uint8_rgba(
        src,
        exposure=opts.exr_exposure,
        gamma=opts.exr_gamma,
        clamp_min=opts.exr_clamp[0],
        clamp_max=opts.exr_clamp[1],
        keep_alpha=opts.exr_keep_alpha,
    )
    save_pil_as_webp(im, dst, opts, lossless=False)


def convert_one(src: Path, dst: Path, opts: ConvertOptions) -> str:
    if dst.exists() and not opts.overwrite:
        return "skip"

    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg", ".png"):
        convert_jpg_png(src, dst, opts)
        return "ok"
    if ext == ".exr":
        convert_exr(src, dst, opts)
        return "ok"
    return "skip"


def run_batch(
    in_root: Path,
    out_root: Path,
    opts: ConvertOptions,
    *,
    progress_cb=None,
    log_cb=None,
    should_cancel=None,
) -> Tuple[int, int, int, int]:
    """Run the conversion.

    Returns: (total, ok, skipped, failed)
    """

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    files = list(iter_images(in_root, include_exr=opts.include_exr))
    total_files = len(files)

    for i, src in enumerate(files, start=1):
        if should_cancel and should_cancel():
            if log_cb:
                log_cb("Canceled.")
            break

        dst = out_path_for(src, in_root, out_root)

        try:
            status = convert_one(src, dst, opts)
            total += 1
            if status == "ok":
                ok += 1
                if log_cb:
                    log_cb(f"OK   {src}  ->  {dst}")
            else:
                skipped += 1
                if log_cb:
                    log_cb(f"SKIP {src} (exists)")
        except Exception as e:
            total += 1
            failed += 1
            if log_cb:
                log_cb(f"FAIL {src}: {e}")

        if progress_cb:
            progress_cb(i, total_files)

    return total, ok, skipped, failed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch convert images to WebP (recursive, preserves folders).")
    p.add_argument("--gui", action="store_true", help="Launch GUI (requires PySide6)")

    # CLI args (optional if --gui)
    p.add_argument("--in", dest="in_root", help="Input root folder")
    p.add_argument("--out", dest="out_root", help="Output root folder")

    p.add_argument("--quality", type=int, default=85, help="WebP quality (0-100). Typical 80-85.")
    p.add_argument("--method", type=int, default=6, help="WebP method (0-6). 6 = best compression.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .webp files")

    p.add_argument("--include-exr", action="store_true", help="Convert EXR too (requires OpenImageIO).")
    p.add_argument("--no-exr", dest="include_exr", action="store_false", help="Do not convert EXR")
    p.set_defaults(include_exr=True)

    p.add_argument("--lossless-png", action="store_true", help="Save PNGs as lossless WebP")

    p.add_argument("--exr-exposure", type=float, default=0.0, help="EXR exposure in stops (2^stops multiplier)")
    p.add_argument("--exr-gamma", type=float, default=2.2, help="EXR gamma for 8-bit encode")
    p.add_argument(
        "--exr-clamp",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Clamp EXR RGB before gamma (default 0 1)",
    )
    p.add_argument("--exr-no-alpha", dest="exr_keep_alpha", action="store_false", help="Drop EXR alpha")
    p.set_defaults(exr_keep_alpha=True)

    return p.parse_args()


def validate_paths(in_root: Path, out_root: Path, overwrite: bool) -> None:
    if not in_root.exists() or not in_root.is_dir():
        raise RuntimeError(f"Input root does not exist or is not a directory: {in_root}")

    # Foot-gun guardrail: if output is inside input and overwrite, you can spiral.
    try:
        out_root.relative_to(in_root)
        if overwrite:
            raise RuntimeError(
                "Refusing: output folder is inside input folder AND overwrite is enabled. "
                "Move output elsewhere or disable overwrite."
            )
    except ValueError:
        pass


# -----------------------------
# GUI (PySide6)
# -----------------------------

def launch_gui() -> int:
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except Exception:
        print("PySide6 is not installed. Install with: python -m pip install PySide6", file=sys.stderr)
        return 2

    # Some PySide6 builds don't expose a 'ProgressBar' name; fall back.
    try:
        from PySide6.QtWidgets import QProgressBar as ProgressBar  # type: ignore
    except Exception:
        from PySide6.QtWidgets import QProgressBar as ProgressBar  # noqa

    class Worker(QThread):
        progress = Signal(int, int)
        log = Signal(str)
        finished = Signal(int, int, int, int)
        failed = Signal(str)

        def __init__(self, in_root: Path, out_root: Path, opts: ConvertOptions):
            super().__init__()
            self.in_root = in_root
            self.out_root = out_root
            self.opts = opts
            self._cancel = False

        def cancel(self) -> None:
            self._cancel = True

        def run(self) -> None:
            try:
                total, ok, skipped, failed = run_batch(
                    self.in_root,
                    self.out_root,
                    self.opts,
                    progress_cb=lambda a, b: self.progress.emit(a, b),
                    log_cb=lambda s: self.log.emit(s),
                    should_cancel=lambda: self._cancel,
                )
                self.finished.emit(total, ok, skipped, failed)
            except Exception as e:
                self.failed.emit(str(e))

    class MainWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Batch WebP Converter")
            self.worker: Optional[Worker] = None

            root = QVBoxLayout(self)

            # Paths
            paths_box = QGroupBox("Folders")
            paths_layout = QFormLayout(paths_box)

            self.in_edit = QLineEdit()
            self.out_edit = QLineEdit()

            in_btn = QPushButton("Browse…")
            out_btn = QPushButton("Browse…")

            def pick_in():
                d = QFileDialog.getExistingDirectory(self, "Select input folder")
                if d:
                    self.in_edit.setText(d)

            def pick_out():
                d = QFileDialog.getExistingDirectory(self, "Select output folder")
                if d:
                    self.out_edit.setText(d)

            in_btn.clicked.connect(pick_in)
            out_btn.clicked.connect(pick_out)

            in_row = QHBoxLayout()
            in_row.addWidget(self.in_edit)
            in_row.addWidget(in_btn)
            out_row = QHBoxLayout()
            out_row.addWidget(self.out_edit)
            out_row.addWidget(out_btn)

            paths_layout.addRow("Input", self._wrap(in_row))
            paths_layout.addRow("Output", self._wrap(out_row))
            root.addWidget(paths_box)

            # Options
            opts_box = QGroupBox("Options")
            opts_layout = QFormLayout(opts_box)

            self.quality = QSpinBox()
            self.quality.setRange(0, 100)
            self.quality.setValue(85)

            self.method = QSpinBox()
            self.method.setRange(0, 6)
            self.method.setValue(6)

            self.overwrite = QCheckBox("Overwrite existing .webp")
            self.lossless_png = QCheckBox("Lossless for PNG (UI/line art)")
            self.include_exr = QCheckBox("Convert EXR (requires OpenImageIO)")
            self.include_exr.setChecked(True)

            self.exr_exposure = QDoubleSpinBox()
            self.exr_exposure.setRange(-20.0, 20.0)
            self.exr_exposure.setDecimals(2)
            self.exr_exposure.setValue(0.0)

            self.exr_gamma = QDoubleSpinBox()
            self.exr_gamma.setRange(0.1, 6.0)
            self.exr_gamma.setDecimals(2)
            self.exr_gamma.setValue(2.2)

            self.exr_clamp_min = QDoubleSpinBox()
            self.exr_clamp_min.setRange(-1000.0, 1000.0)
            self.exr_clamp_min.setDecimals(3)
            self.exr_clamp_min.setValue(0.0)

            self.exr_clamp_max = QDoubleSpinBox()
            self.exr_clamp_max.setRange(-1000.0, 1000.0)
            self.exr_clamp_max.setDecimals(3)
            self.exr_clamp_max.setValue(1.0)

            self.exr_keep_alpha = QCheckBox("Keep EXR alpha")
            self.exr_keep_alpha.setChecked(True)

            opts_layout.addRow("WebP quality", self.quality)
            opts_layout.addRow("WebP method", self.method)
            opts_layout.addRow("", self.overwrite)
            opts_layout.addRow("", self.lossless_png)
            opts_layout.addRow("", self.include_exr)
            opts_layout.addRow("EXR exposure (stops)", self.exr_exposure)
            opts_layout.addRow("EXR gamma", self.exr_gamma)
            clamp_row = QHBoxLayout()
            clamp_row.addWidget(QLabel("Min"))
            clamp_row.addWidget(self.exr_clamp_min)
            clamp_row.addSpacing(10)
            clamp_row.addWidget(QLabel("Max"))
            clamp_row.addWidget(self.exr_clamp_max)
            opts_layout.addRow("EXR clamp", self._wrap(clamp_row))
            opts_layout.addRow("", self.exr_keep_alpha)

            root.addWidget(opts_box)

            # Controls
            controls = QHBoxLayout()
            self.start_btn = QPushButton("Start")
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.setEnabled(False)
            controls.addWidget(self.start_btn)
            controls.addWidget(self.cancel_btn)
            controls.addStretch(1)

            root.addLayout(controls)

            # Progress + log
            self.progress = ProgressBar()
            self.progress.setValue(0)
            root.addWidget(self.progress)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMinimumHeight(220)
            root.addWidget(self.log)

            self.start_btn.clicked.connect(self.on_start)
            self.cancel_btn.clicked.connect(self.on_cancel)

            if oiio is None:
                self.log.appendPlainText(
                    "NOTE: OpenImageIO not found. EXR conversion will fail until you install: pip install OpenImageIO numpy"
                )

        def _wrap(self, layout: QHBoxLayout) -> QWidget:
            w = QWidget()
            w.setLayout(layout)
            return w

        def append_log(self, text: str) -> None:
            self.log.appendPlainText(text)

        def set_busy(self, busy: bool) -> None:
            self.start_btn.setEnabled(not busy)
            self.cancel_btn.setEnabled(busy)

        def on_start(self) -> None:
            in_txt = self.in_edit.text().strip().strip('"')
            out_txt = self.out_edit.text().strip().strip('"')

            if not in_txt or not out_txt:
                QMessageBox.warning(self, "Missing folders", "Select both input and output folders.")
                return

            in_root = Path(in_txt).expanduser().resolve()
            out_root = Path(out_txt).expanduser().resolve()

            opts = ConvertOptions(
                quality=int(self.quality.value()),
                method=int(self.method.value()),
                overwrite=bool(self.overwrite.isChecked()),
                include_exr=bool(self.include_exr.isChecked()),
                lossless_png=bool(self.lossless_png.isChecked()),
                exr_exposure=float(self.exr_exposure.value()),
                exr_gamma=float(self.exr_gamma.value()),
                exr_clamp=(float(self.exr_clamp_min.value()), float(self.exr_clamp_max.value())),
                exr_keep_alpha=bool(self.exr_keep_alpha.isChecked()),
            )

            try:
                validate_paths(in_root, out_root, overwrite=opts.overwrite)
            except Exception as e:
                QMessageBox.critical(self, "Invalid paths", str(e))
                return

            self.log.clear()
            self.progress.setValue(0)
            self.set_busy(True)

            self.worker = Worker(in_root, out_root, opts)
            self.worker.progress.connect(self.on_progress)
            self.worker.log.connect(self.append_log)
            self.worker.failed.connect(self.on_failed)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()

        def on_cancel(self) -> None:
            if self.worker:
                self.worker.cancel()
                self.append_log("Cancel requested…")

        def on_progress(self, current: int, total: int) -> None:
            if total <= 0:
                self.progress.setValue(0)
                return
            pct = int((current / total) * 100)
            self.progress.setValue(pct)

        def on_failed(self, msg: str) -> None:
            self.set_busy(False)
            QMessageBox.critical(self, "Error", msg)

        def on_finished(self, total: int, ok: int, skipped: int, failed: int) -> None:
            self.set_busy(False)
            self.append_log("\n---")
            self.append_log(f"Total: {total} | OK: {ok} | Skipped: {skipped} | Failed: {failed}")
            QMessageBox.information(
                self,
                "Done",
                f"OK: {ok}\nSkipped: {skipped}\nFailed: {failed}",
            )


    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 650)
    w.show()
    return int(app.exec())


def cli_main(args: argparse.Namespace) -> int:
    if not args.in_root or not args.out_root:
        print("CLI requires --in and --out (or run with --gui).", file=sys.stderr)
        return 2

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    try:
        validate_paths(in_root, out_root, overwrite=bool(args.overwrite))
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    opts = ConvertOptions(
        quality=max(0, min(100, int(args.quality))),
        method=max(0, min(6, int(args.method))),
        overwrite=bool(args.overwrite),
        include_exr=bool(args.include_exr),
        lossless_png=bool(args.lossless_png),
        exr_exposure=float(args.exr_exposure),
        exr_gamma=float(args.exr_gamma),
        exr_clamp=(float(args.exr_clamp[0]), float(args.exr_clamp[1])),
        exr_keep_alpha=bool(args.exr_keep_alpha),
    )

    if opts.include_exr and oiio is None:
        print(
            "NOTE: EXR conversion requested but OpenImageIO isn't installed. "
            "EXR files will fail unless you install: python -m pip install OpenImageIO numpy",
            file=sys.stderr,
        )

    def progress_cb(cur: int, total: int) -> None:
        if total > 0:
            pct = int((cur / total) * 100)
            print(f"[{pct:3d}%] {cur}/{total}")

    total, ok, skipped, failed = run_batch(
        in_root,
        out_root,
        opts,
        progress_cb=progress_cb,
        log_cb=lambda s: print(s),
        should_cancel=None,
    )

    print("---")
    print(f"Input : {in_root}")
    print(f"Output: {out_root}")
    print(f"Total: {total} | OK: {ok} | Skipped: {skipped} | Failed: {failed}")

    return 1 if failed else 0


def main() -> int:
    args = parse_args()

    # If --gui is passed, always launch GUI.
    if args.gui:
        return launch_gui()

    # If no CLI paths provided, default to GUI (friendlier).
    if not args.in_root and not args.out_root:
        return launch_gui()

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
