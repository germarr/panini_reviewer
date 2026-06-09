"""
Diagnostic: SPACE to capture. Crops to the pill ROI and shows exactly what
Tesseract receives. Run: uv run python debug_scan.py
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

OUT_DIR = Path("/tmp/panini_debug")
OUT_DIR.mkdir(exist_ok=True)

CODE_PATTERN = re.compile(r"[A-Z]{2,4}\s?[0-9SIOZB]{1,2}")
_DIGIT_SUBS = str.maketrans("SIOZB", "91028")

def normalize_code(raw: str) -> str:
    raw = raw.replace(" ", "")
    m = re.match(r"^([A-Z]{2,4})([0-9SIOZB]{1,2})$", raw)
    if not m:
        return raw
    prefix, suffix = m.groups()
    return prefix + suffix.translate(_DIGIT_SUBS)

ROI_CX, ROI_CY = 0.50, 0.42
ROI_W,  ROI_H  = 0.32, 0.14

CONFIGS = {
    "psm7+wl":   "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "psm8+wl":   "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "psm7":      "--psm 7",
    "psm6+wl":   "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
}


def roi_rect(frame):
    fh, fw = frame.shape[:2]
    cx, cy = int(fw * ROI_CX), int(fh * ROI_CY)
    hw, hh = int(fw * ROI_W / 2), int(fh * ROI_H / 2)
    return cx - hw, cy - hh, cx + hw, cy + hh


def diagnose(frame: np.ndarray, n: int) -> None:
    x1, y1, x2, y2 = roi_rect(frame)
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    inverted = cv2.bitwise_not(big)
    _, otsu = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )

    def crop_to_badge_blob(img):
        ih, iw = img.shape[:2]
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            ratio = bw / bh if bh else 0
            if ratio > 1.5 and area > iw * ih * 0.05:
                if area > best_area:
                    best_area = area
                    best = (bx, by, bw, bh)
        if best is None:
            return img
        bx, by, bw, bh = best
        pad = 15
        return img[max(0, by-pad):min(ih, by+bh+pad), max(0, bx-pad):min(iw, bx+bw+pad)]

    otsu_blob    = crop_to_badge_blob(otsu)
    adaptive_blob = crop_to_badge_blob(adaptive)

    cv2.imwrite(str(OUT_DIR / f"{n:02d}_roi_color.png"), roi)
    cv2.imwrite(str(OUT_DIR / f"{n:02d}_inverted.png"), inverted)
    cv2.imwrite(str(OUT_DIR / f"{n:02d}_otsu.png"), otsu)
    cv2.imwrite(str(OUT_DIR / f"{n:02d}_otsu_blob.png"), otsu_blob)
    cv2.imwrite(str(OUT_DIR / f"{n:02d}_adaptive_blob.png"), adaptive_blob)

    print(f"\n{'='*60}")
    print(f"Capture #{n}  —  ROI {x2-x1}x{y2-y1}px  (scaled to {big.shape[1]}x{big.shape[0]})")
    print(f"  Badge blob: {otsu_blob.shape[1]}x{otsu_blob.shape[0]}px")

    for img_name, img in [("otsu_blob", otsu_blob), ("adaptive_blob", adaptive_blob), ("gray", inverted)]:
        for cfg_name, config in CONFIGS.items():
            text = pytesseract.image_to_string(img, config=config)
            text_clean = text.strip().replace("\n", " | ")
            hits = [normalize_code(r) for r in CODE_PATTERN.findall(text)]
            if text_clean or hits:
                print(f"  [{img_name}/{cfg_name}]  raw={repr(text_clean[:60])}  hits={hits}")

    print(f"{'='*60}")


def draw_guide(frame, scanning=False):
    x1, y1, x2, y2 = roi_rect(frame)
    h_half = (y2 - y1) // 2
    cxl, cxr, cy = x1 + h_half, x2 - h_half, (y1 + y2) // 2
    color = (0, 255, 0) if scanning else (0, 200, 255)
    cv2.line(frame, (cxl, y1), (cxr, y1), color, 2)
    cv2.line(frame, (cxl, y2), (cxr, y2), color, 2)
    cv2.ellipse(frame, (cxl, cy), (h_half, h_half), 0, 90,  270, color, 2)
    cv2.ellipse(frame, (cxr, cy), (h_half, h_half), 0, 270, 90,  color, 2)
    cv2.putText(frame, "Alinea el codigo aqui", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main():
    print(f"Tesseract: {pytesseract.get_tesseract_version()}")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    n = 0
    print("Align the code badge inside the pill guide. SPACE to scan, Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        draw_guide(frame)
        cv2.putText(frame, "SPACE: diagnose | Q: quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow("Panini Debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            n += 1
            diagnose(frame, n)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
