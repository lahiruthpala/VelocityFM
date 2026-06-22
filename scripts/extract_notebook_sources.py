#!/usr/bin/env python3
"""Export notebook cells to searchable Jupytext-style Python files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_notebook(source: Path, destination: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    lines = [
        "# Auto-exported notebook reference.",
        "# Notebook magics are commented and the result may require manual cleanup.",
        "",
    ]
    for index, cell in enumerate(notebook.get("cells", [])):
        source_text = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            lines.append(f"# %% [markdown] cell {index}")
            lines.extend("# " + line for line in source_text.splitlines())
        elif cell.get("cell_type") == "code":
            lines.append(f"# %% cell {index}")
            for line in source_text.splitlines():
                stripped = line.lstrip()
                indent = line[: len(line) - len(stripped)]
                if stripped.startswith(("!", "%")):
                    lines.append(indent + "# NOTEBOOK_MAGIC: " + stripped)
                else:
                    lines.append(line)
        lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
