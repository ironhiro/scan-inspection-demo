"""Reference backend for the inspection service.

- Source images are read from the existing system through `SourceImages`
  (read-only). The demo implementation reads a folder; a production one would
  query the existing admin DB / object storage. Nothing in the existing system
  is modified.
- Analysis results, inspector reviews and status live in this service's own
  tables (SQLite here; same schema works on PostgreSQL/MSSQL).
- Analysis uses ml/pipeline.py and the exported ONNX model, i.e. the same
  logic the browser demo runs.

Run:  uvicorn server.app:app --reload   (from the repository root)
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from pipeline import (MODEL_H, MODEL_W, components, detect_card, grade, load_rules,  # noqa: E402
                      measure_margins, rectify)

DB_PATH = os.environ.get("INSPECT_DB", str(ROOT / "server" / "inspect.db"))
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", ROOT / "web" / "public" / "samples"))
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "web" / "public" / "models" / "damage_unet.onnx"))
MODEL_VERSION = "damage-unet / synth-v1"

STATUSES = ("new", "analyzed", "reviewing", "done")


# ----------------------------------------------------------------------------
# existing system (read-only)
# ----------------------------------------------------------------------------

class SourceImages:
    """Adapter for images registered in the existing admin system."""

    def list_ids(self) -> List[str]:
        return sorted(p.name for p in SOURCE_DIR.glob("*.jpg"))

    def read(self, source_id: str) -> bytes:
        p = SOURCE_DIR / source_id
        if not p.is_file() or p.parent != SOURCE_DIR:
            raise KeyError(source_id)
        return p.read_bytes()


source = SourceImages()


# ----------------------------------------------------------------------------
# own storage
# ----------------------------------------------------------------------------

SCHEMA = """
create table if not exists inspection (
  id            text primary key,
  source_id     text not null unique,     -- key in the existing system
  status        text not null default 'new',
  created_at    text not null,
  completed_at  text,
  ai_json       text,                     -- model output (immutable per run)
  review_json   text,                     -- inspector edits
  final_grade   integer
);
create table if not exists inspection_log (
  id            integer primary key autoincrement,
  inspection_id text not null references inspection(id),
  at            text not null,
  text          text not null
);
"""


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(con, iid, text):
    con.execute("insert into inspection_log(inspection_id, at, text) values (?,?,?)", (iid, now(), text))


def sync_from_source():
    """Register images that exist in the source system but not here yet."""
    with db() as con:
        known = {r["source_id"] for r in con.execute("select source_id from inspection")}
        n = con.execute("select count(*) from inspection").fetchone()[0]
        for sid in source.list_ids():
            if sid in known:
                continue
            n += 1
            iid = f"SC-{n:04d}"
            con.execute("insert into inspection(id, source_id, created_at) values (?,?,?)", (iid, sid, now()))
            log(con, iid, "기존 DB에서 스캔 이미지 불러옴")


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

_session: Optional[ort.InferenceSession] = None


def session():
    global _session
    if _session is None:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # errors only
        _session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def analyze_image(data: bytes) -> Dict:
    scan = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if scan is None:
        raise ValueError("이미지를 읽을 수 없습니다")
    corners = detect_card(scan)
    rect = rectify(scan, corners)
    rh, rw = rect.shape[:2]
    margins = measure_margins(rect)
    small = rectify(scan, corners, (MODEL_W, MODEL_H))
    x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - 0.5
    cls, conf = session().run(None, {"image": x.transpose(2, 0, 1)[None]})
    sx, sy = rw / MODEL_W, rh / MODEL_H
    damages = []
    for i, d in enumerate(components(cls[0], conf[0]), 1):
        x0, y0, bw, bh = d["bbox"]
        damages.append({"id": f"ai-{i}", "type": d["type"], "bbox": [x0 * sx, y0 * sy, bw * sx, bh * sy],
                        "area": d["area"] * sx * sy, "conf": d["conf"], "source": "ai"})
    return {"model_version": MODEL_VERSION, "corners": corners.tolist(), "rect_w": rw, "rect_h": rh,
            "margins": {k: margins[k] for k in ("left", "right", "top", "bottom")}, "damages": damages}


def current_grade(ai: Dict, review: Optional[Dict]) -> int:
    eff = review or ai
    if review and review.get("grade_override"):
        return int(review["grade_override"])
    return grade(eff["damages"], eff["margins"], ai["rect_w"], load_rules())["grade"]


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

app = FastAPI(title="Scan inspection API (reference)")


@app.on_event("startup")
def startup():
    with db() as con:
        con.executescript(SCHEMA)
    sync_from_source()


def row_out(r, with_detail=False):
    out = {k: r[k] for k in ("id", "source_id", "status", "created_at", "completed_at", "final_grade")}
    ai = json.loads(r["ai_json"]) if r["ai_json"] else None
    out["ai_damage_count"] = len(ai["damages"]) if ai else None
    if with_detail:
        out["ai"] = ai
        out["review"] = json.loads(r["review_json"]) if r["review_json"] else None
    return out


def get_row(con, iid):
    r = con.execute("select * from inspection where id=?", (iid,)).fetchone()
    if not r:
        raise HTTPException(404, "검사 건을 찾을 수 없습니다")
    return r


@app.post("/sync")
def sync():
    sync_from_source()
    return {"ok": True}


@app.get("/inspections")
def list_inspections(status: Optional[str] = None):
    with db() as con:
        q, args = "select * from inspection", ()
        if status:
            if status not in STATUSES:
                raise HTTPException(400, "unknown status")
            q, args = q + " where status=?", (status,)
        return [row_out(r) for r in con.execute(q + " order by id desc", args)]


@app.get("/inspections/{iid}")
def get_inspection(iid: str):
    with db() as con:
        r = get_row(con, iid)
        out = row_out(r, True)
        out["log"] = [dict(x) for x in con.execute(
            "select at, text from inspection_log where inspection_id=? order by id", (iid,))]
        return out


@app.get("/inspections/{iid}/image")
def get_image(iid: str):
    with db() as con:
        r = get_row(con, iid)
    return Response(source.read(r["source_id"]), media_type="image/jpeg")


@app.post("/inspections/{iid}/analyze")
def analyze(iid: str):
    with db() as con:
        r = get_row(con, iid)
        if r["status"] == "done":
            raise HTTPException(409, "완료된 건은 다시 열어야 재분석할 수 있습니다")
        try:
            ai = analyze_image(source.read(r["source_id"]))
        except ValueError as e:
            raise HTTPException(422, str(e))
        con.execute("update inspection set ai_json=?, review_json=null, status='analyzed', final_grade=null where id=?",
                    (json.dumps(ai), iid))
        log(con, iid, f"AI 분석 완료 · 손상 {len(ai['damages'])}건")
    return get_inspection(iid)


class Review(BaseModel):
    damages: List[Dict]
    margins: Dict
    grade_override: Optional[int] = None
    note: str = ""
    change: str = ""   # human-readable description for the log


@app.put("/inspections/{iid}/review")
def put_review(iid: str, body: Review):
    with db() as con:
        r = get_row(con, iid)
        if not r["ai_json"]:
            raise HTTPException(409, "AI 분석 전입니다")
        if r["status"] == "done":
            raise HTTPException(409, "완료된 건입니다")
        if r["status"] == "analyzed":
            log(con, iid, "검토 시작")
        con.execute("update inspection set review_json=?, status='reviewing' where id=?",
                    (body.model_dump_json(exclude={"change"}), iid))
        if body.change:
            log(con, iid, body.change)
    return get_inspection(iid)


@app.post("/inspections/{iid}/complete")
def complete(iid: str):
    with db() as con:
        r = get_row(con, iid)
        if not r["ai_json"]:
            raise HTTPException(409, "AI 분석 전입니다")
        ai = json.loads(r["ai_json"])
        review = json.loads(r["review_json"]) if r["review_json"] else None
        g = current_grade(ai, review)
        ai_g = current_grade(ai, None)
        con.execute("update inspection set status='done', completed_at=?, final_grade=? where id=?", (now(), g, iid))
        log(con, iid, f"검사 완료 · 확정 등급 {g} (AI {ai_g})")
    return get_inspection(iid)


@app.post("/inspections/{iid}/reopen")
def reopen(iid: str):
    with db() as con:
        get_row(con, iid)
        con.execute("update inspection set status='reviewing', completed_at=null where id=?", (iid,))
        log(con, iid, "검사 다시 열기")
    return get_inspection(iid)
