"""Export readable prompts/schemas from the frozen Python constants, not hand copies."""
from pathlib import Path
from .common import ROOT, write, sha, verify_source
from src.experiments import realtalk_ours as ours
from src.experiments import realtalk_gpt_judge as judge


def export(output: Path = ROOT / "contracts") -> dict:
    verify_source()
    output.mkdir(parents=True, exist_ok=True)
    names = [name for name in vars(ours) if name.endswith(("_PROMPT", "_TEMPLATE")) and isinstance(getattr(ours, name), str)]
    for name in names:
        (output / f"{name.lower()}.txt").write_text(getattr(ours, name), encoding="utf-8", newline="\n")
    for name in ("REFLECTIVENESS_PROMPT", "GROUNDING_PROMPT", "EMPATHY_PROMPT"):
        (output / f"judge_{name.lower()}.txt").write_text(getattr(judge, name), encoding="utf-8", newline="\n")
    for name in ("SELF_DOMAIN_SCHEMA", "USER_DOMAIN_SCHEMA", "ALIGNMENT_SCHEMA"):
        write(output / f"{name.lower()}.json", getattr(ours, name))
    hashes = {f.name: sha(f) for f in sorted(output.iterdir()) if f.is_file() and f.name != "index.json"}
    write(output / "index.json", {"source": "frozen Python constants", "files_sha256": hashes, "canonical_prompt_hashes": ours._prompt_hashes()})
    return hashes


if __name__ == "__main__":
    print(export())
