"""Small file helpers shared by every protocol."""
import json
from datetime import datetime
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]


def session_dir(output, prefix):
    """<output>/<prefix>_YYYYmmdd_HHMMSS (created; _2, _3, ... if started twice in one second)."""
    base = Path(output).expanduser().resolve()/f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    d, k = base, 1
    while True:
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            k += 1;d = base.with_name(f'{base.name}_{k}')
