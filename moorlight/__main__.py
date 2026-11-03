"""moorlight [files or TorchRef result dirs...] — start the viewer server and open the browser.

    python -m moorlight refined.pdb refined.mtz
    python -m moorlight path/to/torchref/run/
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import webbrowser

os.environ.setdefault("TORCHREF_DEVICE", "cpu")


def _free_port(preferred, wait=5.0):
    """The preferred port, waiting briefly for a previous server to release it; else the next free one."""
    import time

    def free(p):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return True
            except OSError:
                return False

    t0 = time.time()
    while time.time() - t0 < wait:
        if free(preferred):
            return preferred
        time.sleep(0.25)
    for p in range(preferred + 1, preferred + 50):
        if free(p):
            print(f"port {preferred} is busy, using {p}")
            return p
    raise RuntimeError("no free port")


def main():
    ap = argparse.ArgumentParser(prog="moorlight", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", help="PDB/mmCIF models, MTZ files, CCP4 maps, restraint CIFs or TorchRef dirs")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--workdir", default=None, help="directory for chat / relative paths (default: first file's dir)")
    ap.add_argument("--sample-rate", type=float, default=3.0, help="map grid sampling (Coot-like 3 ≈ d/6 spacing)")
    ap.add_argument("--allow-bash", action="store_true",
                    help="let the built-in Claude chat run shell commands (e.g. TorchRef) without asking")
    ap.add_argument("--model", default=None, help="Claude model for the built-in chat")
    a = ap.parse_args()

    from aiohttp import web
    from .server import App, write_state_file
    from .session import Session

    files = [os.path.abspath(f) for f in a.files]
    workdir = a.workdir or (files[0] if files and os.path.isdir(files[0]) else
                            os.path.dirname(files[0]) if files else os.getcwd())
    port = _free_port(a.port)
    s = Session(workdir=os.path.abspath(workdir), sample_rate=a.sample_rate)
    app = App(s, chat_opts={"allow_bash": a.allow_bash, "model": a.model, "port": port})

    for f in files:
        low = f.lower()
        try:
            if os.path.isdir(f):
                s.load_torchref(f)
            elif low.endswith((".pdb", ".ent")) or (low.endswith((".cif", ".mmcif")) and _is_model(f)):
                s.load_model(f)
            elif low.endswith(".cif"):
                s.add_dictionary(f)
            elif low.endswith(".mtz"):
                s.load_mtz(f)
            elif low.endswith((".map", ".ccp4", ".mrc")):
                s.load_map(f)
            else:
                print(f"skipping {f}: unknown type")
        except Exception as e:
            print(f"could not load {f}: {e}")

    from .rsr import warm_up
    threading.Thread(target=lambda: _quiet(warm_up), daemon=True).start()

    url = f"http://127.0.0.1:{port}/"
    write_state_file(url, s.workdir)
    print(f"Moorlight: {url}")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    web.run_app(app.make_app(), host="127.0.0.1", port=port, print=None)


def _quiet(fn):
    try:
        fn()
    except Exception as e:
        print(f"TorchRef warm-up failed: {e}")


def _is_model(path):
    import gemmi
    try:
        doc = gemmi.cif.read(path)
        return any(b.find_value("_atom_site.id") is not None or b.find_loop("_atom_site.id") for b in doc)
    except Exception:
        return False


if __name__ == "__main__":
    main()
