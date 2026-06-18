#!/usr/bin/env python
"""Generate MLX Master Trainer icons: the emerald 'layers' brand mark (matches the app's sidebar logo).
  - app icon  -> a 1024px master + the macOS icon.icns (+ the PNG sizes Tauri references)
  - menu-bar  -> tray-*-Template.png (monochrome black-on-transparent; macOS recolors for light/dark bars)
Pure Pillow + macOS iconutil — no SVG rasterizer needed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "desktop" / "src-tauri" / "icons"
ICONS.mkdir(parents=True, exist_ok=True)

# brand: 3 stacked emerald layers (back→front, brightest in front), dark rounded-square bg
BACK, MID, FRONT = (21, 121, 92), (31, 174, 132), (54, 227, 166)
DARK_TOP, DARK_BOT = (37, 41, 47), (20, 22, 26)


def rhombus(cx, cy, hw, hh):
    return [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]


def app_master(size=1024):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # vertical dark gradient
    grad = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / size
        grad.putpixel((0, y), tuple(int(DARK_TOP[i] * (1 - t) + DARK_BOT[i] * t) for i in range(3)))
    grad = grad.resize((size, size)).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    cx, cy = size // 2, int(size * 0.55)
    hw, hh, off = int(size * 0.30), int(size * 0.15), int(size * 0.135)
    for color, dy in ((BACK, off), (MID, 0), (FRONT, -off)):
        d.polygon(rhombus(cx, cy + dy, hw, hh), fill=color)
    return img


def tray_template(px=44):
    s = px * 4                                   # supersample for crisp edges, then downscale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = s // 2, int(s * 0.52)
    hw, hh, off = int(s * 0.34), int(s * 0.16), int(s * 0.17)
    for dy in (off, 0, -off):                    # solid black; macOS template = alpha mask, recolored
        d.polygon(rhombus(cx, cy + dy, hw, hh), fill=(0, 0, 0, 255))
    return img.resize((px, px), Image.LANCZOS)


def main():
    master = app_master(1024)
    master.save("/tmp/mmt_icon_1024.png")
    # PNG sizes Tauri's macOS bundle references
    for n in (32, 64, 128, 256, 512, 1024):
        master.resize((n, n), Image.LANCZOS).save(ICONS / f"{n}x{n}.png")
    master.resize((256, 256), Image.LANCZOS).save(ICONS / "128x128@2x.png")
    master.save(ICONS / "icon.png")

    # icon.icns via an .iconset + iconutil (macOS native)
    iconset = Path("/tmp/mmt.iconset")
    iconset.mkdir(exist_ok=True)
    for base in (16, 32, 128, 256, 512):
        master.resize((base, base), Image.LANCZOS).save(iconset / f"icon_{base}x{base}.png")
        master.resize((base * 2, base * 2), Image.LANCZOS).save(iconset / f"icon_{base}x{base}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", "-o", str(ICONS / "icon.icns"), str(iconset)], check=True)

    # menu-bar template (all tray states share the mark; lib.rs uses idle)
    tray = tray_template(44)
    for name in ("tray-idle", "tray-active", "tray-ai", "tray-alert"):
        tray.save(ICONS / f"{name}-Template.png")

    print("icons written to", ICONS)
    print("icon.icns:", (ICONS / "icon.icns").stat().st_size, "bytes")
    print("tray-idle-Template.png:", Image.open(ICONS / "tray-idle-Template.png").size)


if __name__ == "__main__":
    main()
