#!/usr/bin/env python3
"""
anpr.py — Multi-stream ANPR for NVIDIA Jetson
- Runs N camera streams concurrently (one thread per stream)
- Lock-on tracking: confirm plate → OCR off → YOLO presence only
- Posts arrival/departure events to 3Netra API
- departure_time calculated when vehicle absent for `absence_timeout` seconds

Install:
    pip install easyocr ultralytics opencv-python-headless pyyaml requests

Run:
    python3 anpr.py --config anpr_config.yml
"""

import os
import re
import sys
import uuid
import time
import json
import base64
import logging
import argparse
import threading
import requests
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARN] pip install ultralytics")

try:
    import easyocr
    EASYOCR_OK = True
except ImportError:
    EASYOCR_OK = False
    print("[WARN] pip install easyocr")


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = "") -> logging.Logger:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("ANPR")


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def cfg(c: dict, *keys, default=None):
    v = c
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
    return v


# ─────────────────────────────────────────────────────────────
# GPU check
# ─────────────────────────────────────────────────────────────

def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# UAE plate validation + OCR fix
# ─────────────────────────────────────────────────────────────

# GCC plate patterns — UAE, Oman, Saudi, Kuwait, Qatar, Bahrain
# UAE:    1234 | A1234 | 12A345
# Oman:   A12345 | AB1234
# Saudi:  123ABC | 1234AB
# Kuwait: 12345 | A12345
# Qatar:  1234 | A1234
# Bahrain: 12345 | A1234
_PLATE_RE = re.compile(
    r'^[0-9]{3,6}$'                  # UAE/Kuwait/Qatar/Bahrain numeric
    r'|^[A-Z][0-9]{3,6}$'            # UAE/Oman/Kuwait letter+numbers
    r'|^[0-9]{1,2}[A-Z][0-9]{3,5}$' # UAE mixed
    r'|^[A-Z]{2}[0-9]{3,6}$'        # Oman two-letter prefix
    r'|^[0-9]{3,5}[A-Z]{2,3}$'      # Saudi numbers+letters
    r'|^[0-9]{4}[A-Z]{3}$'          # Saudi standard
    r'|^[A-Z][0-9]{4,6}$'           # Generic GCC
)

def is_valid_uae_plate(text: str) -> bool:
    return bool(_PLATE_RE.match(text))

def fix_ocr(raw: str) -> str:
    clean = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if clean.isdigit() and clean.startswith('3'):
        clean = '1' + clean[1:]
    return clean


# ─────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────

class APIClient:
    def __init__(self, base_url: str, email: str, password: str,
                 domain: str, log: logging.Logger,
                 bay_status_url: str = ""):
        self.base_url       = base_url.rstrip('/')
        self.email          = email
        self.password       = password
        self.domain         = domain
        self.log            = log
        self._token: str    = ""
        self._token_expiry: float = 0.0
        self._lock          = threading.Lock()

        # Bay occupancy status endpoint (e.g. https://…/v1/parking-bays)
        self._bay_status_url: str = bay_status_url.rstrip("/") if bay_status_url else ""
        # Cache: bay_id → last pushed status, to suppress duplicate PUT calls
        self._last_bay_status: dict[str, str] = {}

    def _login(self) -> bool:
        try:
            r = requests.post(
                f"{self.base_url}/auth/login",
                json={"email": self.email, "password": self.password,
                      "domain_name": self.domain},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            self._token = data.get("access_token") or data.get("token", "")
            # Token expires in ~10 days per JWT; refresh every 23h to be safe
            self._token_expiry = time.monotonic() + 23 * 3600
            self.log.info("API login successful")
            return True
        except Exception as e:
            self.log.error(f"API login failed: {e}")
            return False

    def _get_token(self) -> str:
        with self._lock:
            if not self._token or time.monotonic() > self._token_expiry:
                self._login()
            return self._token

    def post_event(self, payload: dict) -> dict:
        """Returns response dict on success, empty dict on failure."""
        token = self._get_token()
        if not token:
            return {}
        try:
            r = requests.post(
                f"{self.base_url}/v1/parking-events/create/",
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            r.raise_for_status()
            self.log.debug(f"API event posted: {r.status_code}")
            return r.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                self._token = ""
                token = self._get_token()
                if token:
                    return self.post_event(payload)
            self.log.error(f"API post failed: {e}")
            return {}
        except Exception as e:
            self.log.error(f"API post error: {e}")
            return {}

    # ── Bay occupancy status ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_bay_status(status: str) -> str:
        """Normalises status string to 'Occupied' or 'Available'."""
        return {"occupied": "Occupied", "available": "Available"}.get(
            status.strip().lower(), status.strip()
        )

    def update_bay_status(self, bay_id: str, status: str,
                          force: bool = False) -> bool:
        """
        PUT /v1/parking-bays/{bay_id}  {"bay_status": "Occupied"|"Available"}

        Parameters
        ----------
        bay_id : str   — UUID / identifier of the parking bay
        status : str   — "Occupied" or "Available" (case-insensitive)
        force  : bool  — if True, skip the dedup cache check
        """
        if not bay_id:
            self.log.warning("Bay status update skipped — bay_id is empty.")
            return False
        if not self._bay_status_url:
            self.log.warning("Bay status update skipped — parking_api.status_url not configured.")
            return False

        status = self._normalize_bay_status(status)

        # Dedup: skip if we already pushed the same status for this bay (thread-safe)
        with self._lock:
            if not force and self._last_bay_status.get(bay_id) == status:
                self.log.debug(
                    "Bay %s already synced as %s — skipping duplicate PUT.",
                    bay_id, status,
                )
                return True

        token = self._get_token()
        if not token:
            self.log.warning("Cannot update bay status — login failed.")
            return False

        url     = f"{self._bay_status_url}/{bay_id}"
        payload = {"bay_status": status}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "accept":        "application/json",
        }

        self.log.info("Bay status PUT  %s  → %s", url, status)

        try:
            resp = requests.put(url, json=payload, headers=headers, timeout=15)

            # Transparent token-refresh on 401
            if resp.status_code == 401:
                self.log.info("Bay status: token expired — re-login.")
                with self._lock:
                    self._token = ""
                token = self._get_token()
                if not token:
                    return False
                headers["Authorization"] = f"Bearer {token}"
                resp = requests.put(url, json=payload, headers=headers, timeout=15)

            # Accept any 2xx as success (API may return 200, 201, or 204)
            if 200 <= resp.status_code < 300:
                with self._lock:
                    self._last_bay_status[bay_id] = status
                self.log.info(
                    "Bay %s → %s  (HTTP %d OK)", bay_id, status, resp.status_code
                )
                return True
            else:
                self.log.error(
                    "Bay status update FAILED: HTTP %d — %s",
                    resp.status_code, resp.text[:300],
                )
                return False

        except Exception as exc:
            self.log.error("Bay status update error: %s", exc)
            return False

    def set_bay_occupied(self, bay_id: str) -> bool:
        """Mark a bay as Occupied (vehicle present)."""
        return self.update_bay_status(bay_id, "Occupied")

    def set_bay_available(self, bay_id: str) -> bool:
        """Mark a bay as Available (vehicle departed)."""
        return self.update_bay_status(bay_id, "Available")

    def get_cached_bay_status(self, bay_id: str) -> str:
        """Return the last successfully pushed status for bay_id, or '' if unknown."""
        with self._lock:
            return self._last_bay_status.get(bay_id, "")

    def init_bays_available(self, bay_ids: list[str]) -> None:
        """
        Called once at startup: force-mark every configured bay as Available.
        Retries up to 3 times per bay (login may be slow on first attempt).
        """
        for bay_id in bay_ids:
            if not bay_id:
                continue
            for attempt in range(1, 4):
                self.log.info(
                    "Startup: bay %s → Available  (attempt %d/3)", bay_id, attempt
                )
                if self.update_bay_status(bay_id, "Available", force=True):
                    break
                if attempt < 3:
                    time.sleep(1.5)
            else:
                self.log.warning(
                    "Startup: failed to reset bay %s to Available after 3 attempts."
                    " Watchdog will retry.", bay_id
                )

    def build_arrival_payload(self, plate: str, entry_time: str,
                               ocr_conf: float, crop_b64: str,
                               cam_cfg: dict, bay_id: str = "") -> dict:
        return {
            "vehicle_image":  crop_b64,
            "site_id":        cam_cfg.get("site_id", ""),
            "sys_id":         cam_cfg.get("sys_id", ""),
            "bay_id":         bay_id or cam_cfg.get("bay_id", ""),
            "plate_text":     plate,
            "emirate":        cam_cfg.get("emirate", ""),
            "vehicle_color":  "",
            "body_type":      "",
            "vehicle_make":   "",
            "vehicle_type":   "",
            "vehicle_id":     str(uuid.uuid4()),
            "entry_time":     entry_time,
            "ocr_confidence": round(ocr_conf, 4),
            "duration_secs":  0,
        }

    def update_departure(self, event_id: str, departure_time: str,
                          duration_secs: int) -> bool:
        """PUT /v1/parking-events/update/ — updates exit_time."""
        token = self._get_token()
        if not token:
            return False
        try:
            r = requests.put(
                f"{self.base_url}/v1/parking-events/update/",
                json={"event_id": event_id, "exit_time": departure_time},
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                timeout=10,
            )
            r.raise_for_status()
            self.log.debug(f"Departure updated for event {event_id}")
            return True
        except Exception as e:
            self.log.error(f"API departure update failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────
# Locked slot
# ─────────────────────────────────────────────────────────────

class LockedSlot:
    def __init__(self, plate: str, bbox: list, arrived_at: str,
                 absence_timeout: float, ocr_conf: float,
                 api_event_id: str = ""):
        self.plate           = plate
        self.bbox            = bbox
        self.arrived_at      = arrived_at
        self.absence_timeout = absence_timeout
        self.ocr_conf        = ocr_conf
        self.api_event_id    = api_event_id   # returned by API on arrival POST
        self.last_seen       = time.monotonic()
        self.session_uuid    = str(uuid.uuid4())
        self._arrived_mono   = time.monotonic()

    def seen(self):
        self.last_seen = time.monotonic()

    @property
    def is_departed(self) -> bool:
        return time.monotonic() - self.last_seen >= self.absence_timeout

    @property
    def duration_secs(self) -> int:
        return int(time.monotonic() - self._arrived_mono)

    def bbox_overlap(self, other: list, min_iou: float = 0.2) -> bool:
        ax1,ay1,ax2,ay2 = self.bbox
        bx1,by1,bx2,by2 = other
        ix1=max(ax1,bx1); iy1=max(ay1,by1)
        ix2=min(ax2,bx2); iy2=min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1:
            return False
        inter = (ix2-ix1)*(iy2-iy1)
        ua    = (ax2-ax1)*(ay2-ay1)
        ub    = (bx2-bx1)*(by2-by1)
        union = ua+ub-inter
        return (inter/union) >= min_iou if union > 0 else False


# ─────────────────────────────────────────────────────────────
# Vote buffer
# ─────────────────────────────────────────────────────────────

class VoteBuffer:
    def __init__(self, window: float = 3.0, min_votes: int = 5,
                 px_tol: int = 120):
        self.window    = window
        self.min_votes = min_votes
        self.px_tol    = px_tol
        self._buckets: dict[str, dict] = {}

    def _key(self, cx: int, cy: int) -> str:
        return f"{(cx//self.px_tol)*self.px_tol}_{(cy//self.px_tol)*self.px_tol}"

    def add(self, bbox, text, ocr_score, det_conf, timestamp, full_frame):
        x1,y1,x2,y2 = bbox
        key = self._key((x1+x2)//2, (y1+y2)//2)
        now = time.monotonic()
        if key not in self._buckets:
            bw0 = bbox[2] - bbox[0]
            bh0 = bbox[3] - bbox[1]
            self._buckets[key] = {
                "texts":[], "scores":[], "bbox":bbox,
                "first_ts": timestamp, "first_seen": now,
                "det_conf": det_conf, "best_frame": full_frame,
                "best_bbox": bbox,
                "best_area": bw0 * bh0,
            }
        b = self._buckets[key]
        b["det_conf"] = max(b["det_conf"], det_conf)
        if text:
            b["texts"].append(text)
            b["scores"].append(ocr_score)
        # Keep frame where plate bbox area is largest = plate closest/biggest in frame
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        plate_area = bw * bh
        if plate_area > b.get("best_area", 0):
            b["best_area"]  = plate_area
            b["best_frame"] = full_frame
            b["best_bbox"]  = bbox

    def flush(self):
        now, results, expired = time.monotonic(), [], []
        for key, b in self._buckets.items():
            age = now - b["first_seen"]
            if len(b["texts"]) >= self.min_votes or age >= self.window:
                if len(b["texts"]) >= self.min_votes:
                    winner, count = Counter(b["texts"]).most_common(1)[0]
                    win_s = [s for t,s in zip(b["texts"],b["scores"]) if t==winner]
                    # Reject if mean OCR confidence is too low
                    if not win_s or float(np.mean(win_s)) < 0.40:
                        expired.append(key)
                        continue
                    # Crop tightly around vehicle (5x plate bbox) + draw plate bbox + label
                    annotated = None
                    if b["best_frame"] is not None:
                        frame_h, frame_w = b["best_frame"].shape[:2]
                        x1,y1,x2,y2 = b["best_bbox"]
                        pw, ph = x2-x1, y2-y1
                        # Expand 5x around the plate to capture full vehicle
                        pad_x = pw * 4
                        pad_y = ph * 5
                        cx1 = max(0,      int(x1 - pad_x))
                        cy1 = max(0,      int(y1 - pad_y))
                        cx2 = min(frame_w, int(x2 + pad_x))
                        cy2 = min(frame_h, int(y2 + pad_y))
                        annotated = b["best_frame"][cy1:cy2, cx1:cx2].copy()
                        # Plate bbox coords relative to cropped region
                        rx1, ry1 = x1-cx1, y1-cy1
                        rx2, ry2 = x2-cx1, y2-cy1
                        cv2.rectangle(annotated,(rx1,ry1),(rx2,ry2),(0,220,80),2)
                        lbl = winner
                        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
                        cv2.rectangle(annotated,(rx1,max(0,ry1-th-10)),(rx1+tw+8,ry1),(0,220,80),-1)
                        cv2.putText(annotated,lbl,(rx1+4,max(th,ry1-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(10,10,10),2)
                    results.append({
                        "plate":      winner,
                        "det_conf":   b["det_conf"],
                        "ocr_conf":   float(np.mean(win_s)) if win_s else 0.0,
                        "bbox":       b["best_bbox"],
                        "arrived_at": b["first_ts"],
                        "vote_count": count,
                        "frame_img":  annotated,
                    })
                expired.append(key)
        for k in expired:
            del self._buckets[k]
        return results

    def clear_region(self, bbox):
        ax1,ay1,ax2,ay2 = bbox
        to_del = [k for k,b in self._buckets.items()
                  if max(ax1,b["bbox"][0])<min(ax2,b["bbox"][2])
                  and max(ay1,b["bbox"][1])<min(ay2,b["bbox"][3])]
        for k in to_del:
            del self._buckets[k]


# ─────────────────────────────────────────────────────────────
# EasyOCR wrapper (shared singleton)
# ─────────────────────────────────────────────────────────────

_ocr_instance = None
_ocr_lock     = threading.Lock()

def get_ocr(use_gpu: bool, lang: list) -> "PlateOCR":
    global _ocr_instance
    with _ocr_lock:
        if _ocr_instance is None:
            _ocr_instance = PlateOCR(use_gpu, lang)
    return _ocr_instance

class PlateOCR:
    def __init__(self, use_gpu: bool = True, lang: list = None):
        self._reader = easyocr.Reader(lang or ["en"], gpu=use_gpu, verbose=False)
        self._lock   = threading.Lock()

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        try:
            with self._lock:
                results = self._reader.readtext(crop, detail=1, paragraph=False)
            if not results:
                return "", 0.0
            texts = [t for _,t,_ in results]
            scores= [float(s) for _,_,s in results]
            clean = fix_ocr("".join(texts))
            return clean, float(np.mean(scores))
        except Exception:
            return "", 0.0


# ─────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────

def preprocess(crop: np.ndarray, upscale: int = 3) -> np.ndarray:
    h, w = crop.shape[:2]
    if w < 150:
        crop = cv2.resize(crop, None, fx=150/w, fy=150/w,
                          interpolation=cv2.INTER_CUBIC)
    if upscale > 1:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l),a,b]), cv2.COLOR_LAB2BGR)

def crop_to_b64(crop: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


# ─────────────────────────────────────────────────────────────
# Per-camera worker
# ─────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────
# Local event log (JSONL) — for troubleshooting
# ─────────────────────────────────────────────────────────────

class EventLog:
    """
    Writes one JSONL record per event to a local file.
    Each record: { event_type, camera, bay_id, vehicle_plate,
                   arrived_at, departed_at, duration_secs,
                   det_confidence, ocr_confidence, api_event_id,
                   api_status }
    """
    def __init__(self, path: str):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh   = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def arrival(self, plate: str, bay_id: str, arrived_at: str,
                det_conf: float, ocr_conf: float,
                api_event_id: str, api_ok: bool, cam_name: str):
        self._write({
            "event_type":     "ARRIVAL",
            "camera":         cam_name,
            "bay_id":         bay_id,
            "vehicle_plate":  plate,
            "arrived_at":     arrived_at,
            "departed_at":    None,
            "duration_secs":  None,
            "det_confidence": round(det_conf, 4),
            "ocr_confidence": round(ocr_conf, 4),
            "api_event_id":   api_event_id,
            "api_status":     "OK" if api_ok else "FAIL",
        })

    def departure(self, plate: str, bay_id: str, arrived_at: str,
                  departed_at: str, duration_secs: int,
                  api_event_id: str, api_ok: bool, cam_name: str):
        self._write({
            "event_type":     "DEPARTURE",
            "camera":         cam_name,
            "bay_id":         bay_id,
            "vehicle_plate":  plate,
            "arrived_at":     arrived_at,
            "departed_at":    departed_at,
            "duration_secs":  duration_secs,
            "det_confidence": None,
            "ocr_confidence": None,
            "api_event_id":   api_event_id,
            "api_status":     "OK" if api_ok else "FAIL",
        })

    def still_present(self, plate: str, bay_id: str, arrived_at: str,
                      duration_secs: int, cam_name: str):
        self._write({
            "event_type":     "STILL_PRESENT",
            "camera":         cam_name,
            "bay_id":         bay_id,
            "vehicle_plate":  plate,
            "arrived_at":     arrived_at,
            "departed_at":    None,
            "duration_secs":  duration_secs,
            "det_confidence": None,
            "ocr_confidence": None,
            "api_event_id":   None,
            "api_status":     None,
        })

    def _write(self, record: dict):
        record["logged_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._fh.write(json.dumps(record) + "\n")

    def close(self):
        self._fh.close()


def resolve_bay_id(bbox: list, frame_width: int, cam_cfg: dict) -> str:
    """
    Resolve bay_id based on x-position in frame.
    Single-bay camera: always returns bay_id.
    Dual-bay camera:   if bay_id_right is set, detections in the right half
                       get bay_id_right, left half get bay_id.
    Config:
        bay_id:       "bay-2"   # left side
        bay_id_right: "bay-3"   # right side (optional)
    """
    bay_id_right = cam_cfg.get("bay_id_right", "")
    if not bay_id_right:
        return cam_cfg.get("bay_id", "")
    cx = (bbox[0] + bbox[2]) // 2
    return bay_id_right if cx > frame_width // 2 else cam_cfg.get("bay_id", "")

def camera_worker(cam_cfg: dict, global_cfg: dict,
                  model: YOLO, ocr: PlateOCR,
                  api: APIClient, log: logging.Logger,
                  stop_event: threading.Event,
                  global_active_plates: set = None,
                  global_plates_lock: threading.Lock = None):
    """Runs in its own thread. One thread per camera."""

    name      = cam_cfg.get("name", cam_cfg.get("rtsp", cam_cfg.get("file", "cam")))
    tag       = f"[{cam_cfg.get('name','cam')}]"

    det_cfg   = global_cfg.get("detector", {})
    conf_t    = float(det_cfg.get("confidence",   0.10))
    iou_t     = float(det_cfg.get("iou",          0.45))
    cls_id    = det_cfg.get("plate_class_id",      0)
    skip      = int(det_cfg.get("frame_skip",      5))
    crop_pad  = int(det_cfg.get("crop_padding",    20))
    upscale   = int(det_cfg.get("upscale",          3))
    yolo_dev  = det_cfg.get("device", 0)

    ocr_cfg   = global_cfg.get("ocr", {})
    min_chars = int(ocr_cfg.get("min_chars", 3))

    out_cfg      = global_cfg.get("output", {})
    absence_tout = float(out_cfg.get("absence_timeout", 30.0))
    save_crops   = out_cfg.get("save_crops", True)
    crops_dir    = Path(out_cfg.get("dir", "anpr_output")) / out_cfg.get("crops_dir","crops")
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Open source
    source = cam_cfg.get("rtsp") or cam_cfg.get("file") or str(cam_cfg.get("camera_index",0))
    if cam_cfg.get("rtsp"):
        transport = cam_cfg.get("rtsp_transport", "tcp")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        cap = cv2.VideoCapture(cam_cfg["rtsp"], cv2.CAP_FFMPEG)
    elif cam_cfg.get("file"):
        cap = cv2.VideoCapture(cam_cfg["file"])
    else:
        cap = cv2.VideoCapture(int(cam_cfg.get("camera_index", 0)))

    w = cam_cfg.get("width",  1280)
    h_res = cam_cfg.get("height", 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_res)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        log.error(f"{tag} Cannot open source: {source}")
        return

    log.info(f"{tag} Started  source={source}")

    # Local event log for this camera
    event_log_path = Path(out_cfg.get("dir", "anpr_output")) / f"events_{cam_cfg.get('name','cam')}.jsonl"
    event_log = EventLog(str(event_log_path))

    votes:  VoteBuffer           = VoteBuffer(window=5.0, min_votes=5, px_tol=120)
    locked: dict[str, LockedSlot] = {}

    # ── Bay watchdog ─────────────────────────────────────────────────────────
    # All bay IDs monitored by this camera
    cam_bays: list[str] = [
        b for b in [
            cam_cfg.get("bay_id", ""),
            cam_cfg.get("bay_id_right", ""),
        ] if b
    ]
    # Initialise last-locked timestamps far in the past so the watchdog fires
    # on the very first cycle if no vehicle shows up — catches stale Occupied
    # states left from a previous run or a failed startup init.
    bay_last_locked: dict[str, float] = {
        bid: time.monotonic() - absence_tout for bid in cam_bays
    }
    _watchdog_t: float = time.monotonic()   # next watchdog check timer
    _WATCHDOG_INTERVAL = 30.0               # check every 30 s

    reconnect = int(cam_cfg.get("reconnect_delay", 5))
    is_rtsp   = bool(cam_cfg.get("rtsp"))
    frame_id  = fps_count = 0
    fps_val   = 0.0
    fps_t     = time.perf_counter()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_rtsp:
                log.warning(f"{tag} Stream lost — retry in {reconnect}s")
                cap.release()
                time.sleep(reconnect)
                cap = cv2.VideoCapture(cam_cfg["rtsp"], cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            else:
                log.info(f"{tag} End of source")
                break

        frame_id  += 1
        fps_count += 1
        now_t = time.perf_counter()
        if now_t - fps_t >= 10.0:
            fps_val   = fps_count / (now_t - fps_t)
            fps_t     = now_t
            fps_count = 0
            log.info(f"{tag} FPS {fps_val:.1f}  frame {frame_id}  "
                     f"locked={list(locked.keys()) or 'none'}")

        fh, fw = frame.shape[:2]
        now_ts = datetime.now(timezone.utc).isoformat()

        # ── Resize frame for YOLO detection (keeps full res for crops) ──
        # Target max dimension for detection — plates are more visible at 1280px wide
        det_max_w = 1280
        if fw > det_max_w:
            scale = det_max_w / fw
            det_frame = cv2.resize(frame, (det_max_w, int(fh * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            det_frame = frame

        # ── YOLO every frame when slots locked, else every skip ──
        run_yolo = locked or (frame_id % (skip + 1) == 0)
        all_dets = []
        if run_yolo:
            results = model.predict(
                det_frame, conf=conf_t, iou=iou_t,
                classes=[cls_id], verbose=False, device=yolo_dev,
            )
            for r in results:
                for box in r.boxes:
                    # Scale bbox back to original full resolution
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    x1 = int(x1 / scale); y1 = int(y1 / scale)
                    x2 = int(x2 / scale); y2 = int(y2 / scale)
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(fw,x2), min(fh,y2)
                    bw,bh = x2-x1, y2-y1
                    if bh==0 or not (1.5<=bw/bh<=6.0) or bw<40 or bh<10:
                        continue
                    # Exclude bottom 5% of frame — camera timestamp OSD
                    if y2 > fh * 0.95:
                        log.debug(f"{tag} OSD excluded y2={y2}")
                        continue
                    log.debug(f"{tag} Detection conf={float(box.conf[0]):.3f} bbox=[{x1},{y1},{x2},{y2}] ratio={bw/bh:.2f}")
                    all_dets.append([x1,y1,x2,y2,float(box.conf[0])])

        # ── Refresh locked slots ─────────────────────────────────
        # Update bay_last_locked for every currently-active plate
        for plate, slot in locked.items():
            bid = resolve_bay_id(slot.bbox, fw, cam_cfg)
            bay_last_locked[bid] = time.monotonic()

        for plate, slot in list(locked.items()):
            seen = any(slot.bbox_overlap(d[:4]) for d in all_dets)
            if seen:
                slot.seen()

            if slot.is_departed:
                departed_at   = now_ts
                duration_secs = slot.duration_secs
                bay_id_dep    = resolve_bay_id(slot.bbox, fw, cam_cfg)
                log.info(
                    f"{tag} DEPARTED  {plate:<12}  "
                    f"arrived={slot.arrived_at}  "
                    f"departed={departed_at}  "
                    f"duration={duration_secs}s"
                )
                # Update API with departure
                dep_ok = False
                if slot.api_event_id:
                    dep_ok = api.update_departure(slot.api_event_id, departed_at,
                                                   duration_secs)
                # ── Bay occupancy: mark Available ──────────────────────────
                api.set_bay_available(bay_id_dep)

                event_log.departure(
                    plate=plate, bay_id=bay_id_dep,
                    arrived_at=slot.arrived_at, departed_at=departed_at,
                    duration_secs=duration_secs,
                    api_event_id=slot.api_event_id, api_ok=dep_ok,
                    cam_name=cam_cfg.get("name", "cam"),
                )
                # Release from global active set
                if global_active_plates is not None and global_plates_lock is not None:
                    with global_plates_lock:
                        global_active_plates.discard(plate)
                del locked[plate]

        # ── Bay watchdog ─────────────────────────────────────────
        # Periodically: for each bay this camera owns, if no plate has been
        # locked to it for >= absence_tout seconds AND the API status is not
        # already 'Available', force it Available.  This recovers:
        #   • Stale Occupied from a previous crash / failed departure call
        #   • Failed startup init (cache status is '' which is also != 'Available')
        now_mono = time.monotonic()
        if now_mono - _watchdog_t >= _WATCHDOG_INTERVAL:
            _watchdog_t = now_mono
            active_bays = {
                resolve_bay_id(s.bbox, fw, cam_cfg) for s in locked.values()
            }
            for bid in cam_bays:
                if bid in active_bays:
                    continue               # vehicle is still locked — skip
                age = now_mono - bay_last_locked[bid]
                if age >= absence_tout:
                    cached = api.get_cached_bay_status(bid)
                    if cached != "Available":
                        log.warning(
                            "%s Watchdog: bay %s status=%r, no vehicle for %.0fs "
                            "— forcing Available",
                            tag, bid, cached or "<unknown>", age,
                        )
                        api.set_bay_available(bid)

        # ── OCR on unlocked detections ───────────────────────────
        if frame_id % (skip + 1) == 0:
            for det in all_dets:
                x1,y1,x2,y2,det_conf = det
                bbox = [x1,y1,x2,y2]

                if any(s.bbox_overlap(bbox) for s in locked.values()):
                    continue

                crop = frame[
                    max(0,y1-crop_pad):min(fh,y2+crop_pad),
                    max(0,x1-crop_pad):min(fw,x2+crop_pad)
                ]
                # Use only top 60% of crop — UAE plates have number on top,
                # Arabic/English text on bottom which confuses OCR
                ch = crop.shape[0]
                crop = crop[:int(ch * 0.60), :]
                plate, oconf = ocr.read(preprocess(crop, upscale))
                log.debug(f"{tag} OCR raw={plate!r} conf={oconf:.2f} valid={is_valid_uae_plate(plate)}")

                if len(plate)<min_chars or not is_valid_uae_plate(plate):
                    continue
                if plate in locked:
                    locked[plate].seen()
                    continue

                votes.add(bbox, plate, oconf, det_conf, now_ts, frame)

        # ── Flush votes ──────────────────────────────────────────
        for v in votes.flush():
            plate = v["plate"]
            if not is_valid_uae_plate(plate) or plate in locked:
                continue

            # Global dedup — skip if another camera already has this plate active
            if global_active_plates is not None and global_plates_lock is not None:
                with global_plates_lock:
                    if plate in global_active_plates:
                        log.debug(f"{tag} Skipping {plate} — already active on another camera")
                        continue
                    global_active_plates.add(plate)

            # Encode full annotated frame for API
            frame_b64 = crop_to_b64(v["frame_img"]) if v.get("frame_img") is not None else ""

            # Resolve bay_id — split-frame for dual-bay cameras
            bay_id = resolve_bay_id(v["bbox"], fw, cam_cfg)

            # POST arrival to API
            payload    = api.build_arrival_payload(
                plate, v["arrived_at"], v["ocr_conf"], frame_b64, cam_cfg,
                bay_id=bay_id
            )
            api_resp   = api.post_event(payload)
            api_evt_id = api_resp.get("event_id", "") if api_resp else ""
            api_ok     = bool(api_evt_id)

            # ── Bay occupancy: mark Occupied ───────────────────────────────
            api.set_bay_occupied(bay_id)

            slot = LockedSlot(
                plate=plate, bbox=v["bbox"],
                arrived_at=v["arrived_at"],
                absence_timeout=absence_tout,
                ocr_conf=v["ocr_conf"],
                api_event_id=api_evt_id,
            )
            locked[plate] = slot
            votes.clear_region(v["bbox"])

            log.info(
                f"{tag} ARRIVED   {plate:<12}  bay={bay_id}  "
                f"det={v['det_conf']:.0%}  ocr={v['ocr_conf']:.0%}  "
                f"votes={v['vote_count']}  api={'OK' if api_ok else 'FAIL'}"
            )
            event_log.arrival(
                plate=plate, bay_id=bay_id, arrived_at=v["arrived_at"],
                det_conf=v["det_conf"], ocr_conf=v["ocr_conf"],
                api_event_id=api_evt_id, api_ok=api_ok,
                cam_name=cam_cfg.get("name", "cam"),
            )

            if save_crops and v.get("frame_img") is not None:
                safe = re.sub(r'[^A-Z0-9]', '_', plate)
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(crops_dir / f"{safe}_{cam_cfg.get('name','cam')}_{ts}.jpg"),
                            v["frame_img"])

    # Flush remaining locked slots on exit
    for plate, slot in locked.items():
        log.info(f"{tag} STILL PRESENT  {plate}  (no departure recorded)")
        if slot.api_event_id:
            api.update_departure(slot.api_event_id,
                                  datetime.now(timezone.utc).isoformat(),
                                  slot.duration_secs)
        event_log.still_present(
            plate=plate, bay_id=resolve_bay_id(slot.bbox, fw, cam_cfg),
            arrived_at=slot.arrived_at, duration_secs=slot.duration_secs,
            cam_name=cam_cfg.get("name", "cam"),
        )
        if global_active_plates is not None and global_plates_lock is not None:
            with global_plates_lock:
                global_active_plates.discard(plate)
    cap.release()
    event_log.close()
    log.info(f"{tag} Worker stopped")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-stream ANPR — YOLO8 + EasyOCR")
    ap.add_argument("--config", default="anpr_config.yml")
    args = ap.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}"); sys.exit(1)

    config = load_config(args.config)
    lc     = config.get("logging", {})
    log    = setup_logging(lc.get("level", "INFO"), lc.get("log_file", ""))

    if not YOLO_OK:
        log.error("ultralytics missing — pip install ultralytics"); sys.exit(1)
    if not EASYOCR_OK:
        log.error("easyocr missing — pip install easyocr"); sys.exit(1)

    # Output dir
    out_dir = Path(cfg(config, "output", "dir", default="anpr_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared YOLO model (thread-safe predict)
    det_cfg = config.get("detector", {})
    yolo_dev = det_cfg.get("device", 0)
    if yolo_dev != "cpu" and not cuda_available():
        log.warning("CUDA unavailable — using CPU")
        yolo_dev = "cpu"
    log.info(f"Loading YOLO: {det_cfg.get('model_path','best.pt')}")
    model = YOLO(det_cfg.get("model_path", "best.pt"))

    # Shared EasyOCR (serialised via internal lock)
    ocr_cfg = config.get("ocr", {})
    use_gpu = ocr_cfg.get("use_gpu", True) and cuda_available()
    lang    = ocr_cfg.get("lang", ["en"])
    if isinstance(lang, str):
        lang = [lang]
    log.info(f"Loading EasyOCR  lang={lang}  gpu={use_gpu}")
    ocr = PlateOCR(use_gpu=use_gpu, lang=lang)

    # API client
    api_cfg = config.get("api", {})
    # Bay status URL: prefer parking_api.status_url, fall back to api.bay_status_url
    parking_api_cfg = config.get("parking_api", {})
    bay_status_url  = (
        parking_api_cfg.get("status_url", "")
        or api_cfg.get("bay_status_url", "")
    )
    api = APIClient(
        base_url       = api_cfg.get("base_url", "https://3netraapi.koushik.cc"),
        email          = api_cfg.get("email",    "sunny@blackrockitsolutions.com"),
        password       = api_cfg.get("password", "testpassword"),
        domain         = api_cfg.get("domain",   "du"),
        log            = log,
        bay_status_url = bay_status_url,
    )
    if bay_status_url:
        log.info(f"Bay status API: {bay_status_url}")
    else:
        log.warning("parking_api.status_url not set — bay occupancy updates will be skipped.")

    # Cameras list
    cameras = config.get("cameras", [])
    if not cameras:
        log.error("No cameras defined in config under 'cameras:'")
        sys.exit(1)

    # ── Startup: initialise all configured bays to Available ────────────────
    # Collect every unique bay_id / bay_id_right across all cameras
    if bay_status_url:
        startup_bays: list[str] = []
        for cam in cameras:
            for key in ("bay_id", "bay_id_right"):
                bid = cam.get(key, "")
                if bid and bid not in startup_bays:
                    startup_bays.append(bid)
        if startup_bays:
            log.info("Initialising %d bay(s) to Available on startup: %s",
                     len(startup_bays), startup_bays)
            api.init_bays_available(startup_bays)

    log.info(f"Starting {len(cameras)} camera worker(s)")
    stop_event = threading.Event()
    threads    = []

    # Shared set of currently active plates across ALL cameras
    # Prevents duplicate API events when two cameras see the same vehicle
    global_active_plates: set = set()
    global_plates_lock = threading.Lock()

    for cam in cameras:
        t = threading.Thread(
            target=camera_worker,
            args=(cam, config, model, ocr, api, log, stop_event,
                  global_active_plates, global_plates_lock),
            daemon=True,
            name=cam.get("name", "cam"),
        )
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Stopping all workers...")
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

    log.info("All workers stopped")


if __name__ == "__main__":
    main()
