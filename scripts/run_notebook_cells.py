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
            try:
                client.execute_cell(cell, idx)
            except Exception as exc:
                # nbclient's default allow_errors=False means a failing cell raises
                # CellExecutionError (already rendering the cell's own traceback in
                # str(exc)) instead of producing an 'error'-type output we could catch
                # in the loop below -- so this is the only place a failure is visible.
                # Save whatever nbclient already recorded for this cell (it mutates
                # `cell` with outputs/execution_count before raising) so the failure
                # itself isn't lost, then propagate so the caller sees a hard failure.
                print(f'--- cell {idx} FAILED ---', flush=True)
                print(f'{type(exc).__name__}: {exc}', flush=True)
                nbformat.write(nb, NOTEBOOK_PATH)
                print(f'Notebook saved with progress through cell {idx} (inclusive of its failure state).', flush=True)
                raise
            # Persist after every successful cell, not just at the end of the run.
            # A Windows NVIDIA driver TDR reset (see CLAUDE.md) can kill this process
            # with no catchable Python exception at all, so a single write-at-the-end
            # (or even a try/finally around the whole loop) would still lose all
            # progress in that scenario -- only writing per-cell survives it.
            nbformat.write(nb, NOTEBOOK_PATH)
            # No 'error' output_type branch here: with nbclient's default
            # allow_errors=False, a failing cell always raises CellExecutionError out
            # of execute_cell() above (handled in the except block) rather than
            # completing and leaving an 'error' output for this loop to find -- an
            # elif for it here would be dead code that never runs.
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream':
                    print(output['text'], end='', flush=True)
    print('Notebook saved.', flush=True)


if __name__ == '__main__':
    indices = [int(x) for x in sys.argv[1:]]
    try:
        main(indices)
    except Exception:
        sys.exit(1)
