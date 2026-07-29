from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiment_protocol" / "core_prompt_baseline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    for relative_path, expected_hash in manifest["protected_files"].items():
        path = ROOT / relative_path
        if not path.is_file():
            mismatches.append(f"MISSING  {relative_path}")
            continue

        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            mismatches.append(
                f"CHANGED  {relative_path}\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}"
            )

    if mismatches:
        print("Core prompt baseline verification failed:")
        print("\n".join(mismatches))
        return 1

    print(
        "Core prompt baseline verified at "
        f"{manifest['baseline_commit']}: {len(manifest['protected_files'])} files unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
