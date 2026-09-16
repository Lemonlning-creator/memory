from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-v4-flash"
JUDGE = "gpt-4o-mini"
CANONICAL = "5927bbff03fda74eebaeb99e0c57203a644cfd74"
PREDICTIONS_SHA = "ba3941f9fd2088f7d6877409c0ed1f468002ded304e782560e1475da3a9bad81"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def rows(path: Path) -> list[dict]:
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["result_id"] for row in result]
    if not result or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate result IDs: {path}")
    return result


def verify_source() -> dict:
    expected = read(ROOT / "provenance/canonical_source.json")["files"]
    bad = [name for name, value in expected.items() if not (ROOT / name).is_file() or sha(ROOT / name) != value]
    if bad:
        raise ValueError(f"Frozen source differs from {CANONICAL}: {bad}")
    return {"canonical_commit": CANONICAL, "byte_identical_files": len(expected)}


def bind_output(output: Path, identity: dict) -> None:
    binding = output / "release_binding.json"
    if binding.exists():
        if read(binding) != identity:
            raise ValueError("Output belongs to a different input, prompt, model, or release; use a new directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to adopt a populated directory without a release binding")
    else:
        write(binding, identity)
