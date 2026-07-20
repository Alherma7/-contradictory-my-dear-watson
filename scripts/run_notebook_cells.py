"""Execute a subset of watson-notebook.ipynb's cells, in order, in one live kernel
session, and save the notebook in place with the real outputs/execution_counts.

Use this instead of `jupyter nbconvert --execute` on the full notebook when you only
need a subset of cells re-run (e.g. prerequisite setup cells + a few new cells) --
avoids re-running expensive sections unnecessarily.

Usage:
    .venv/Scripts/python.exe scripts/run_notebook_cells.py 0 1 2 4 6 8 ... 100 101
"""
import sys

import nbformat
from nbclient import NotebookClient

# Windows' default stdout encoding (cp1252 here, even though the console codepage is
# 850) can't represent characters that legitimately appear in cell outputs (e.g. the
# bold/box-drawing formatting in transformers' "LOAD REPORT" table). Without this, a
# UnicodeEncodeError while echoing a cell's output kills the whole script mid-run --
# discovered when cell 28 (AutoModel.from_pretrained load report) crashed the runner.
# This only affects what we echo to the console/log for our own visibility; the actual
# cell outputs written into the notebook JSON by nbclient are unaffected either way.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

NOTEBOOK_PATH = 'watson-notebook.ipynb'
KERNEL_NAME = 'watson-nli'


def main(cell_indices):
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(nb, kernel_name=KERNEL_NAME, timeout=1800)
    with client.setup_kernel():
        for idx in cell_indices:
            cell = nb.cells[idx]
            if cell.cell_type != 'code':
                continue
            print(f'--- executing cell {idx} ---', flush=True)
            client.execute_cell(cell, idx)
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream':
                    print(output['text'], end='', flush=True)
                elif output.get('output_type') == 'error':
                    print(f"ERROR in cell {idx}: {output['ename']}: {output['evalue']}", flush=True)
    nbformat.write(nb, NOTEBOOK_PATH)
    print('Notebook saved.', flush=True)


if __name__ == '__main__':
    indices = [int(x) for x in sys.argv[1:]]
    main(indices)
