import json
import sys


def clean(notebook: dict) -> dict:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)

    notebook.get("metadata", {}).pop("widgets", None)
    return notebook


def main():
    notebook = json.load(sys.stdin)
    json.dump(clean(notebook), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


def demo():
    dirty = {
        "cells": [
            {"cell_type": "code", "execution_count": 7, "outputs": [{"text": "hi"}], "metadata": {}},
            {"cell_type": "markdown", "source": "# title", "metadata": {}},
        ],
        "metadata": {"widgets": {"junk": 1}},
    }
    cleaned = clean(dirty)

    assert cleaned["cells"][0]["execution_count"] is None
    assert cleaned["cells"][0]["outputs"] == []
    assert "source" in cleaned["cells"][1], "markdown cells must survive untouched"
    assert "widgets" not in cleaned["metadata"]
    print("nbclean demo OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
