#!/usr/bin/env python3
"""Generate a valid Jupyter notebook (.ipynb, nbformat 4.5) from a Python spec file.

Usage:
    python src/scripts/gen_notebook.py <spec.py> <output.ipynb>

The spec file must define a top-level `CELLS` list of dicts:
    {"type": "md" | "code", "source": "cell text (\\n for newlines)"}

Markdown cells become `cell_type: markdown`, code cells become `cell_type: code`
with empty outputs and execution_count None. Metadata/kernelspec is pinned to
match the existing notebooks in this repo (kernelspec "magus" / python3).
"""
import importlib.util
import json
import os
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python src/scripts/gen_notebook.py <spec.py> <output.ipynb>")

    spec_path, out_path = sys.argv[1], sys.argv[2]

    spec = importlib.util.spec_from_file_location("nb_spec", spec_path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load spec: {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cells: list[dict] = []
    for c in mod.CELLS:
        src = c["source"]
        if not src.endswith("\n"):
            src += "\n"
        if c["type"] == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src,
                }
            )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "magus", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.7",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
