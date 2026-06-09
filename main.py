import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).parent / "data" / "stamps.csv"
CODE_PATTERN = re.compile(r"[A-Z]{2,4}\s?[0-9SIOZB]{1,2}")
_DIGIT_SUBS = str.maketrans("SIOZB", "91028")

# ROI: pill guide where the user aligns the badge.
ROI_CX, ROI_CY = 0.50, 0.42
ROI_W,  ROI_H  = 0.32, 0.14


def roi_rect(frame: np.ndarray) -> tuple[int, int, int, int]:
    fh, fw = frame.shape[:2]
    cx, cy = int(fw * ROI_CX), int(fh * ROI_CY)
    hw, hh = int(fw * ROI_W / 2), int(fh * ROI_H / 2)
    return cx - hw, cy - hh, cx + hw, cy + hh


def normalize_code(raw: str) -> str:
    raw = raw.replace(" ", "")
    m = re.match(r"^([A-Z]{2,4})([0-9SIOZB]{1,2})$", raw)
    if not m:
        return raw
    prefix, suffix = m.groups()
    return prefix + suffix.translate(_DIGIT_SUBS)


def load_stamps(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {
        row["STICKER_CODE"]: {
            "owned": str(row["LATENGO"]).upper() == "TRUE",
            "team": row["TEAM"],
            "group": row["GROUP"],
        }
        for _, row in df.iterrows()
    }


def detect_code(frame: np.ndarray, reader: easyocr.Reader, stamps: dict) -> str | None:
    x1, y1, x2, y2 = roi_rect(frame)
    roi = frame[y1:y2, x1:x2]

    # EasyOCR expects RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    results = reader.readtext(roi_rgb, detail=1, paragraph=False)

    candidates: list[str] = []
    for (_bbox, text, conf) in results:
        if conf < 0.2:
            continue
        for raw in CODE_PATTERN.findall(text.upper()):
            candidates.append(normalize_code(raw))

    for code in candidates:
        if code in stamps:
            return code
    return candidates[0] if candidates else None


def draw_guide(frame: np.ndarray, scanning: bool = False) -> None:
    x1, y1, x2, y2 = roi_rect(frame)
    h_half = (y2 - y1) // 2
    cxl, cxr, cy = x1 + h_half, x2 - h_half, (y1 + y2) // 2
    color = (0, 255, 0) if scanning else (200, 200, 200)
    cv2.line(frame, (cxl, y1), (cxr, y1), color, 2)
    cv2.line(frame, (cxl, y2), (cxr, y2), color, 2)
    cv2.ellipse(frame, (cxl, cy), (h_half, h_half), 0, 90,  270, color, 2)
    cv2.ellipse(frame, (cxr, cy), (h_half, h_half), 0, 270, 90,  color, 2)
    label = "Escaneando..." if scanning else "Alinea el codigo aqui"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def draw_hint(frame: np.ndarray) -> None:
    cv2.putText(frame, "ESPACIO: escanear  |  Q: salir", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)


def draw_result(frame: np.ndarray, code: str | None, stamps: dict) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    banner_top = h - 110

    if code is None:
        color = (0, 140, 255)
        lines = ["No se reconocio ningun codigo"]
    elif code not in stamps:
        color = (0, 140, 255)
        lines = [f"{code}  —  codigo no encontrado en el album"]
    else:
        info = stamps[code]
        color = (30, 160, 30) if info["owned"] else (0, 0, 200)
        status = "YA LO TIENES  v" if info["owned"] else "TE FALTA  !"
        lines = [f"{code}  |  {info['team']}  |  {info['group']}", status]

    cv2.rectangle(overlay, (0, banner_top), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, banner_top + 38 + i * 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    stamps = load_stamps(CSV_PATH)

    print("Cargando modelo EasyOCR (primera vez puede tardar un momento)...")
    reader = easyocr.Reader(["en"], gpu=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se puede abrir la camara.")
        return

    last_code: str | None = None
    has_result = False
    scanning = False

    print(f"Panini Scanner listo. {len(stamps)} stickers cargados.")
    print("Alinea el codigo dentro del ovalo. ESPACIO para escanear, Q para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_guide(frame, scanning=scanning)
        draw_hint(frame)
        if has_result:
            draw_result(frame, last_code, stamps)

        cv2.imshow("Panini Scanner", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            scanning = True
            draw_guide(frame, scanning=True)
            cv2.imshow("Panini Scanner", frame)
            cv2.waitKey(1)

            last_code = detect_code(frame, reader, stamps)
            has_result = True
            scanning = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
