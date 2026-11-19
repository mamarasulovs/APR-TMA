"""Per-dataset rules that turn a folder of face crops into manifest rows.

The crops are produced by ``scripts/prepare_faces.py`` (``<video path without extension>/<frame>.png``).
Rules use the original dataset naming and are intentionally small and explicit; for any dataset
whose layout differs, use the ``generic`` adapter with ``--live-regex`` / ``--spoof-regex``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

OULU_ATTACK = {"2": "print1", "3": "print2", "4": "replay1", "5": "replay2"}


@dataclass
class Row:
    label: int
    video_id: str
    attack: str = ""
    phone: str = ""
    session: str = ""
    subset: str = ""


def oulu(rel: Path, subset: str = "") -> Optional[Row]:
    """OULU-NPU file name Phone_Session_User_File: File 1 live, 2/3 print, 4/5 replay."""
    m = re.match(r"^(\d+)_(\d+)_(\d+)_(\d+)$", rel.parts[-2])
    if not m:
        return None
    phone, session, user, fid = m.groups()
    return Row(0 if fid == "1" else 1, rel.parts[-2], OULU_ATTACK.get(fid, ""), str(int(phone)), str(int(session)), subset)


def casia(rel: Path, subset: str = "") -> Optional[Row]:
    """CASIA-MFSD: clips 1, 2 and HR_1 are live; the other clips are attacks."""
    vid = rel.parts[-2]
    return Row(0 if vid in {"1", "2", "HR_1"} else 1, "/".join(rel.parts[:-1]), subset=subset)


def replay(rel: Path, subset: str = "") -> Optional[Row]:
    """Idiap Replay-Attack: folders named real / attack."""
    parts = [p.lower() for p in rel.parts]
    if "real" in parts:
        return Row(0, "/".join(rel.parts[:-1]), subset=subset)
    if "attack" in parts:
        return Row(1, "/".join(rel.parts[:-1]), subset=subset)
    return None


def msu(rel: Path, subset: str = "") -> Optional[Row]:
    """MSU-MFSD: file names start with real_ or attack_."""
    name = rel.parts[-2].lower()
    if name.startswith("real"):
        return Row(0, rel.parts[-2], subset=subset)
    if name.startswith("attack"):
        return Row(1, rel.parts[-2], subset=subset)
    return None


def siw(rel: Path, subset: str = "") -> Optional[Row]:
    """SiW / SiW-Mv2: a path component named Live or Spoof."""
    parts = [p.lower() for p in rel.parts]
    if "live" in parts:
        return Row(0, "/".join(rel.parts[:-1]), subset=subset)
    if "spoof" in parts:
        return Row(1, "/".join(rel.parts[:-1]), subset=subset)
    return None


def generic(live_regex: str, spoof_regex: str, attack_regex: Optional[str] = None
            ) -> Callable[[Path, str], Optional[Row]]:
    """Regex rules on the POSIX relative path. ``attack_regex`` (first capture group) fills the
    ``attack`` column, which the WMCA leave-one-attack-out protocol needs."""
    live, spoof = re.compile(live_regex), re.compile(spoof_regex)
    attack = re.compile(attack_regex) if attack_regex else None

    def rule(rel: Path, subset: str = "") -> Optional[Row]:
        s = rel.as_posix()
        vid = "/".join(rel.parts[:-1]) or rel.stem
        if live.search(s):
            return Row(0, vid, subset=subset)
        if spoof.search(s):
            m = attack.search(s) if attack else None
            return Row(1, vid, attack=m.group(1) if m else "", subset=subset)
        return None
    return rule


ADAPTERS = {"oulu": oulu, "casia": casia, "replay": replay, "msu": msu, "siw": siw}
DOMAIN_TAG = {"oulu": "O", "casia": "C", "replay": "I", "msu": "M", "siw": "SW"}
