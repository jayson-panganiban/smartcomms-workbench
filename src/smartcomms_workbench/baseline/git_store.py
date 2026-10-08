"""Atomic directory-backed baselines suitable for manual Git versioning.

No Git commands are executed; consumers own approvals and version control.
"""

import os
import re
import tempfile
from pathlib import Path

from smartcomms_workbench.diff.pdf import open_pdf


class GitBaselineStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name.endswith("."):
            raise ValueError("Baseline name must be a simple alphanumeric identifier")
        stem = name.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}:
            raise ValueError("Baseline name cannot be a reserved Windows device name")
        path = self.root / f"{name}.pdf"
        if path.resolve().parent != self.root:
            raise ValueError("Baseline path escapes the store")
        return path

    def resolve(self, name: str) -> bytes | None:
        try:
            return self._path(name).read_bytes()
        except FileNotFoundError:
            return None

    def approve(self, name: str, pdf: bytes) -> None:
        path = self._path(name)
        with open_pdf(pdf):
            pass
        self.root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(pdf)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
