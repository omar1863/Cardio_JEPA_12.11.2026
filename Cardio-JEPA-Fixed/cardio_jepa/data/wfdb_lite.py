"""Minimal dependency-free WFDB reader (headers, signals, annotations).

The default Kaggle image does not ship the ``wfdb`` package and Kaggle notebooks
are often run with internet disabled, so we cannot rely on ``pip install wfdb``.
This module implements just enough of the format to read the databases the
proposal needs:

  * ``.hea`` headers (single-segment)
  * signal formats 212 (MIT-BIH, LTAFDB), 16, 61, 80 and 32
  * ``.atr`` / ``.qrs`` annotation files, including AUX rhythm strings such as
    ``(AFIB`` -- these are what the AF-onset anticipation benchmark is built from

``read_record`` / ``read_annotation`` transparently delegate to the real ``wfdb``
package when it is importable, and fall back to this implementation otherwise.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # pragma: no cover - environment dependent
    import wfdb as _wfdb
except Exception:                       # pragma: no cover
    _wfdb = None


# MIT annotation code -> symbol (WFDB Programmer's Guide, Table 'ecgcodes')
ANN_SYMBOLS: Dict[int, str] = {
    0: " ", 1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A",
    9: "S", 10: "E", 11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s",
    19: "T", 20: "*", 21: "D", 22: '"', 23: "=", 24: "p", 25: "B", 26: "^",
    27: "t", 28: "+", 29: "u", 30: "?", 31: "!", 32: "[", 33: "]", 34: "e",
    35: "n", 36: "@", 37: "x", 38: "f", 39: "(", 40: ")", 41: "r",
}

# Beat labels that count as a "normal" beat vs. an ectopic/arrhythmic beat.
NORMAL_BEATS = frozenset("NLRej")
VENTRICULAR_BEATS = frozenset("VEF")
SUPRAVENTRICULAR_BEATS = frozenset("AaJS")
BEAT_SYMBOLS = frozenset("NLRejVEFAaJS/fQ")


@dataclass
class SignalSpec:
    file_name: str
    fmt: str
    samps_per_frame: int = 1
    byte_offset: int = 0
    adc_gain: float = 200.0
    baseline: int = 0
    units: str = "mV"
    adc_res: int = 12
    adc_zero: int = 0
    init_value: int = 0
    description: str = ""


@dataclass
class Header:
    record_name: str
    n_sig: int
    fs: float
    n_samp: int
    signals: List[SignalSpec] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    @property
    def sig_names(self) -> List[str]:
        return [s.description for s in self.signals]

    @property
    def duration_s(self) -> float:
        return self.n_samp / self.fs if self.fs else 0.0


@dataclass
class Record:
    p_signal: np.ndarray          # (n_samp, n_sig) physical units
    fs: float
    sig_name: List[str]
    record_name: str = ""


@dataclass
class Annotation:
    sample: np.ndarray            # int64 sample indices
    symbol: List[str]
    aux_note: List[str]
    fs: float
    subtype: Optional[np.ndarray] = None


# --------------------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------------------
def parse_header(path: str) -> Header:
    with open(path, "r", errors="replace") as fh:
        raw_lines = fh.read().splitlines()

    comments = [ln[1:].strip() for ln in raw_lines if ln.startswith("#")]
    lines = [ln.strip() for ln in raw_lines if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"empty WFDB header: {path}")

    head = lines[0].split()
    record_name = head[0].split("/")[0]
    n_sig = int(head[1])
    fs = float(head[2].split("/")[0]) if len(head) > 2 else 250.0
    n_samp = int(head[3]) if len(head) > 3 else 0

    signals: List[SignalSpec] = []
    for ln in lines[1:1 + n_sig]:
        parts = ln.split()
        file_name = parts[0]

        fmt_field = parts[1]
        samps_per_frame, byte_offset = 1, 0
        if "+" in fmt_field:
            fmt_field, off = fmt_field.split("+", 1)
            byte_offset = int(off)
        if ":" in fmt_field:
            fmt_field, _skew = fmt_field.split(":", 1)
        if "x" in fmt_field:
            fmt_field, spf = fmt_field.split("x", 1)
            samps_per_frame = int(spf)
        fmt = fmt_field

        adc_gain, baseline, units = 200.0, None, "mV"
        if len(parts) > 2:
            gain_field = parts[2]
            if "/" in gain_field:
                gain_field, units = gain_field.split("/", 1)
            if "(" in gain_field:
                gain_field, base = gain_field.split("(", 1)
                baseline = int(float(base.rstrip(")")))
            adc_gain = float(gain_field) if gain_field else 200.0
            if adc_gain == 0:
                adc_gain = 200.0

        adc_res = int(parts[3]) if len(parts) > 3 and parts[3] else 12
        adc_zero = int(parts[4]) if len(parts) > 4 and parts[4] else 0
        init_value = int(float(parts[5])) if len(parts) > 5 and parts[5] else 0
        desc = " ".join(parts[8:]) if len(parts) > 8 else f"ch{len(signals)}"

        signals.append(SignalSpec(
            file_name=file_name, fmt=fmt, samps_per_frame=samps_per_frame,
            byte_offset=byte_offset, adc_gain=adc_gain,
            baseline=adc_zero if baseline is None else baseline,
            units=units, adc_res=adc_res, adc_zero=adc_zero,
            init_value=init_value, description=desc))

    return Header(record_name=record_name, n_sig=n_sig, fs=fs, n_samp=n_samp,
                  signals=signals, comments=comments)


# --------------------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------------------
def _decode_212(buf: np.ndarray, n_samples: int) -> np.ndarray:
    """Format 212: two 12-bit samples packed into three bytes."""
    n_triplets = len(buf) // 3
    b = buf[:n_triplets * 3].reshape(-1, 3).astype(np.int32)
    s0 = b[:, 0] + ((b[:, 1] & 0x0F) << 8)
    s1 = b[:, 2] + ((b[:, 1] >> 4) << 8)
    s0 = np.where(s0 > 2047, s0 - 4096, s0)
    s1 = np.where(s1 > 2047, s1 - 4096, s1)
    out = np.empty(n_triplets * 2, dtype=np.int32)
    out[0::2] = s0
    out[1::2] = s1
    return out[:n_samples]


def _decode_16(buf: np.ndarray, n_samples: int) -> np.ndarray:
    return buf.view(np.int16)[:n_samples].astype(np.int32)


def _decode_80(buf: np.ndarray, n_samples: int) -> np.ndarray:
    return buf[:n_samples].astype(np.int32) - 128


def _decode_32(buf: np.ndarray, n_samples: int) -> np.ndarray:
    return buf.view(np.int32)[:n_samples].astype(np.int32)


def _decode_61(buf: np.ndarray, n_samples: int) -> np.ndarray:
    """Format 61: 16-bit, big-endian."""
    return buf.view(">i2")[:n_samples].astype(np.int32)


def read_signal_lite(dirname: str, header: Header, sampfrom: int = 0,
                     sampto: Optional[int] = None) -> np.ndarray:
    """Return (n, n_sig) physical-unit signal.

    Assumes all channels live in a single interleaved data file (true for every
    database used here). Partial reads seek into the file so a 24-hour LTAFDB
    record never has to be materialised in full.
    """
    spec0 = header.signals[0]
    fmt = spec0.fmt
    n_sig = header.n_sig
    total = header.n_samp
    sampto = total if sampto is None else min(sampto, total)
    sampfrom = max(0, min(sampfrom, sampto))
    n_frames = sampto - sampfrom
    if n_frames <= 0:
        return np.zeros((0, n_sig), dtype=np.float32)

    path = os.path.join(dirname, spec0.file_name)

    # PhysioNet Challenge 2020/2021 mirrors (CPSC2018, Chapman, Ningbo) ship the
    # samples in a MATLAB `.mat` file holding a single (n_sig, n_samp) `val` array,
    # with the `.hea` still describing gains and baselines.
    if spec0.file_name.lower().endswith(".mat"):
        from scipy.io import loadmat
        val = loadmat(path)["val"]
        sig = np.asarray(val, dtype=np.float32).T          # -> (n_samp, n_sig)
        sig = sig[sampfrom:sampto]
        for i, spec in enumerate(header.signals[:sig.shape[1]]):
            sig[:, i] = (sig[:, i] - spec.baseline) / (spec.adc_gain or 200.0)
        return sig

    if fmt == "212":
        # 2 samples per 3 bytes over the interleaved stream
        start_sample = sampfrom * n_sig
        aligned_start = (start_sample // 2) * 2            # 3-byte boundary
        lead_in = start_sample - aligned_start
        n_needed = n_frames * n_sig + lead_in
        byte_start = spec0.byte_offset + (aligned_start // 2) * 3
        n_bytes = int(math.ceil((n_needed + 1) / 2) * 3)
        with open(path, "rb") as fh:
            fh.seek(byte_start)
            buf = np.frombuffer(fh.read(n_bytes), dtype=np.uint8)
        flat = _decode_212(buf, n_needed)[lead_in:lead_in + n_frames * n_sig]
    else:
        itemsize = {"16": 2, "61": 2, "80": 1, "32": 4, "8": 1}.get(fmt)
        if itemsize is None:
            raise NotImplementedError(
                f"WFDB format {fmt!r} not supported by wfdb_lite; install the `wfdb` "
                f"package to read this record")
        byte_start = spec0.byte_offset + sampfrom * n_sig * itemsize
        n_bytes = n_frames * n_sig * itemsize
        with open(path, "rb") as fh:
            fh.seek(byte_start)
            buf = np.frombuffer(fh.read(n_bytes), dtype=np.uint8)
        decoder = {"16": _decode_16, "61": _decode_61, "80": _decode_80,
                   "8": _decode_80, "32": _decode_32}[fmt]
        flat = decoder(buf, n_frames * n_sig)

    got = len(flat) // n_sig
    sig = flat[:got * n_sig].reshape(got, n_sig).astype(np.float32)

    for i, spec in enumerate(header.signals):
        sig[:, i] = (sig[:, i] - spec.baseline) / (spec.adc_gain or 200.0)
    return sig


# --------------------------------------------------------------------------------------
# annotations
# --------------------------------------------------------------------------------------
def read_annotation_lite(path: str, fs: float = 250.0) -> Annotation:
    """Parse a WFDB annotation file (``.atr``, ``.qrs``, ...).

    Follows the layout in the WFDB Programmer's Guide: each annotation is a
    16-bit word whose top 6 bits are the type and bottom 10 bits the interval
    since the previous annotation, optionally followed by SUB/CHN/NUM/AUX
    modifier words.
    """
    with open(path, "rb") as fh:
        raw = np.frombuffer(fh.read(), dtype=np.uint8)
    if len(raw) % 2:
        raw = raw[:-1]
    fb = raw.reshape(-1, 2)                       # [:, 0] = low byte, [:, 1] = high byte

    samples: List[int] = []
    symbols: List[str] = []
    auxes: List[str] = []
    subtypes: List[int] = []

    pos, cur = 0, 0
    n = len(fb)
    while pos < n:
        code = int(fb[pos, 1]) >> 2
        if code == 0 and int(fb[pos, 0]) == 0:
            break                                  # end of file
        if code == 59:                             # SKIP: 32-bit interval follows
            if pos + 3 >= n:
                break
            diff = (65536 * int(fb[pos + 1, 0]) + 16777216 * int(fb[pos + 1, 1])
                    + 256 * int(fb[pos + 2, 0]) + int(fb[pos + 2, 1]))
            if diff > 2147483647:
                diff -= 4294967296
            code = int(fb[pos + 3, 1]) >> 2
            pos += 4
        else:
            diff = int(fb[pos, 0]) + 256 * (int(fb[pos, 1]) & 0b11)
            pos += 1

        cur += diff
        aux, sub = "", 0
        # modifier words
        while pos < n:
            mcode = int(fb[pos, 1]) >> 2
            if mcode == 61:                        # SUB
                sub = int(fb[pos, 0])
                pos += 1
            elif mcode in (60, 62):                # NUM, CHN
                pos += 1
            elif mcode == 63:                      # AUX
                alen = int(fb[pos, 0]) + 256 * (int(fb[pos, 1]) & 0b11)
                nwords = int(math.ceil(alen / 2))
                chunk = fb[pos + 1:pos + 1 + nwords].reshape(-1)
                aux = bytes(chunk.tolist())[:alen].decode("ascii", errors="replace")
                aux = aux.rstrip("\x00").strip()
                pos += 1 + nwords
            else:
                break

        samples.append(cur)
        symbols.append(ANN_SYMBOLS.get(code, "?"))
        auxes.append(aux)
        subtypes.append(sub)

    return Annotation(sample=np.asarray(samples, dtype=np.int64), symbol=symbols,
                      aux_note=auxes, fs=fs, subtype=np.asarray(subtypes, dtype=np.int16))


# --------------------------------------------------------------------------------------
# public API (delegates to `wfdb` when available)
# --------------------------------------------------------------------------------------
def read_header(record_path: str) -> Header:
    return parse_header(record_path + ".hea")


def read_record(record_path: str, sampfrom: int = 0, sampto: Optional[int] = None,
                channels: Optional[Sequence[int]] = None, prefer_wfdb: bool = True
                ) -> Record:
    """Read a WFDB record. ``record_path`` is the path *without* extension."""
    if prefer_wfdb and _wfdb is not None:
        try:
            kw = {"sampfrom": sampfrom}
            if sampto is not None:
                kw["sampto"] = sampto
            if channels is not None:
                kw["channels"] = list(channels)
            r = _wfdb.rdrecord(record_path, **kw)
            sig = np.asarray(r.p_signal, dtype=np.float32)
            return Record(p_signal=sig, fs=float(r.fs), sig_name=list(r.sig_name),
                          record_name=os.path.basename(record_path))
        except Exception:
            pass                                   # fall through to the lite reader

    hdr = parse_header(record_path + ".hea")
    sig = read_signal_lite(os.path.dirname(record_path), hdr, sampfrom, sampto)
    names = hdr.sig_names
    if channels is not None:
        sig = sig[:, list(channels)]
        names = [names[i] for i in channels]
    return Record(p_signal=sig, fs=hdr.fs, sig_name=names,
                  record_name=os.path.basename(record_path))


def read_annotation(record_path: str, extension: str = "atr",
                    prefer_wfdb: bool = True) -> Annotation:
    fs = 250.0
    try:
        fs = parse_header(record_path + ".hea").fs
    except Exception:
        pass

    if prefer_wfdb and _wfdb is not None:
        try:
            a = _wfdb.rdann(record_path, extension)
            aux = [("" if x is None else str(x).rstrip("\x00").strip())
                   for x in (a.aux_note or [""] * len(a.sample))]
            return Annotation(sample=np.asarray(a.sample, dtype=np.int64),
                              symbol=list(a.symbol), aux_note=aux,
                              fs=float(a.fs or fs))
        except Exception:
            pass

    return read_annotation_lite(record_path + "." + extension, fs=fs)


def list_records(root: str, extensions: Sequence[str] = ("hea",),
                 limit: Optional[int] = None,
                 exclude_dirs: Sequence[str] = ()) -> List[str]:
    """Find WFDB records (paths without extension) under ``root``.

    ``exclude_dirs`` prunes sub-trees that belong to a different corpus -- real
    Kaggle mounts nest one database inside another (a PTB-XL mount that also ships
    the MIT-BIH Noise Stress database), and without pruning those records leak into
    the parent corpus's index.
    """
    out: List[str] = []
    excluded = {os.path.abspath(p) for p in exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) in excluded:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if os.path.abspath(os.path.join(dirpath, d)) not in excluded]
        for fn in sorted(filenames):
            for ext in extensions:
                if fn.endswith("." + ext):
                    out.append(os.path.join(dirpath, fn[: -(len(ext) + 1)]))
                    break
        if limit is not None and len(out) >= limit:
            return sorted(out)[:limit]
    out = sorted(set(out))
    return out[:limit] if limit is not None else out
