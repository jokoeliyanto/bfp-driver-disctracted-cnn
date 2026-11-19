"""Render notebooks/original_bangkit_capstone.ipynb as HTML for the Notebook page.

The 2020 notebook has no markdown cells, so section headings are inserted before the cells they describe (display
only; the .ipynb file is left unchanged). It is converted with nbconvert's "basic" template and cached until the file
changes.
"""
import re
import warnings
from functools import lru_cache

from django.conf import settings

HEADING = re.compile(r'<h([1-3]) id="([^"]+)"[^>]*>(.*?)</h\1>', re.S)
TAG = re.compile(r"<[^>]+>")

# cell index -> (heading, note)
SECTIONS = {
    0: ("1. Setup", "Imports and PyDrive, used to fetch the dataset zip from Google Drive in Colab."),
    2: ("2. Download the data", "The State Farm images were stored on the team's Google Drive and unzipped in Colab."),
    5: ("3. Data generators", "224 x 224 images, `validation_split=0.2` (a random 80/20 split per class). "
                              "The class order comes from the folder names."),
    8: ("4. Baseline: a CNN from scratch", "Four convolution blocks, trained for up to 10 epochs with early stopping."),
    10: ("5. Baseline training curves", ""),
    12: ("6. Transfer learning with MobileNet", "ImageNet MobileNet with frozen weights, a new 1024-filter "
                                                "convolution, dropout and a 10-way softmax."),
    16: ("7. Transfer-learning curves", "Note: this cell plots `history` (the baseline run) instead of `history1`, "
                                        "so the figure below repeats the baseline curves. The Evaluation page "
                                        "re-plots both runs from the training logs."),
    18: ("8. Export for the Android app", "Convert the best checkpoint to TensorFlow Lite."),
}


@lru_cache(maxsize=2)
def _render(path, mtime):
    import nbformat
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    with warnings.catch_warnings():        # the 2020 notebook lacks cell ids; harmless here
        warnings.simplefilter("ignore")
        nb = nbformat.read(path, as_version=4)
        n_cells, n_code = len(nb.cells), sum(c.cell_type == "code" for c in nb.cells)
        cells = []
        for i, cell in enumerate(nb.cells):
            if i in SECTIONS:
                title, note = SECTIONS[i]
                cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{note}".strip()))
            cells.append(cell)
        nb.cells = cells
        body, _ = exporter.from_notebook_node(nb)
    toc = [{"level": int(level), "id": anchor, "title": TAG.sub("", title).replace("¶", "").strip()}
           for level, anchor, title in HEADING.findall(body)]
    return body, toc, n_cells, n_code


def render():
    path = settings.NOTEBOOK_PATH
    body, toc, cells, code = _render(str(path), path.stat().st_mtime)
    return {"html": body, "toc": toc, "cells": cells, "code": code, "filename": path.name}


@lru_cache(maxsize=1)
def highlight_css():
    from pygments.formatters import HtmlFormatter
    return HtmlFormatter(style="friendly").get_style_defs(".nbv .highlight")
