"""Standalone plotting worker — spawned as a fresh subprocess by dispatch_debug_plot.

Because this is a new Python process, 'import debug_plot' always reads the
.py file from disk, so any edits made while training are picked up immediately.

Usage (internal):
    python _plot_worker.py <pickle_path>
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import debug_plot  # fresh read from disk every time this process starts


def main():
    import traceback

    tmp_path = sys.argv[1]

    try:
        with open(tmp_path, 'rb') as f:
            p = pickle.load(f)
        os.unlink(tmp_path)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    out_dir  = p.get('out_dir', 'debug')
    epoch    = p.get('epoch', 0)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f'plot.log')
    log_f    = open(log_path, 'w')
    sys.stdout = sys.stderr = log_f

    try:
        print(f'zero_emb={p.get("zero_emb")}  zero_dec={p.get("zero_dec")}  '
              f'out_dir={p.get("out_dir")}  epoch={epoch}')
        debug_plot.debug_plot(**p)
        print('done')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        log_f.close()


if __name__ == '__main__':
    main()
