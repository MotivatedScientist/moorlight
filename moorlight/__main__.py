"""moorlight [files or TorchRef result dirs...] — start the viewer server and open the browser.

    python -m moorlight refined.pdb refined.mtz
    python -m moorlight path/to/torchref/run/
"""
from __future__ import annotations

import argparse
import json
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
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # a restarted viewer can take over its old port (TIME_WAIT)
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
            return p
    raise RuntimeError("no free port")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cmd":     # client mode: drive a running viewer
        from .client import main as client_main
        sys.exit(client_main(sys.argv[2:]))
    ap = argparse.ArgumentParser(prog="moorlight", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="*", help="PDB/mmCIF models, MTZ files (file.mtz:label names the maps), CCP4 maps (file.map:label), "
                                             "restraint CIFs or TorchRef dirs.\n`moorlight cmd ...` drives a running viewer (see `moorlight cmd --help`)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--workdir", default=None, help="directory for chat / relative paths (default: first file's dir)")
    ap.add_argument("--sample-rate", type=float, default=3.0, help="map grid sampling (Coot-like 3 ≈ d/6 spacing)")
    ap.add_argument("--allow-bash", action="store_true",
                    help="let the built-in Claude chat run shell commands (e.g. TorchRef) without asking")
    ap.add_argument("--model", default=None, help="Claude model for the built-in chat")
    ap.add_argument("--install-skill", nargs="?", const="~/.claude/skills", metavar="DIR",
                    help="copy the bundled Claude Code skills (result tours) into DIR (default ~/.claude/skills) and exit")
    a = ap.parse_args()
    if a.install_skill:
        import shutil
        src = os.path.join(os.path.dirname(__file__), "skills")
        dst = os.path.expanduser(a.install_skill)
        for name in sorted(os.listdir(src)):
            if os.path.isdir(os.path.join(src, name)):
                shutil.copytree(os.path.join(src, name), os.path.join(dst, name), dirs_exist_ok=True)
                print(f"installed skill {name} -> {os.path.join(dst, name)}")
        return

    from aiohttp import web
    from .server import App, write_state_file
    from .session import Session

    files, labels = [], {}
    for f in a.files:     # file.mtz:label
        if not os.path.exists(f) and ":" in f and os.path.exists(f.rsplit(":", 1)[0]):
            f, lab = f.rsplit(":", 1)
            labels[os.path.abspath(f)] = lab
        files.append(os.path.abspath(f))
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
                s.load_mtz(f, label=labels.get(f))
            elif low.endswith((".map", ".ccp4", ".mrc")):
                s.load_map(f, label=labels.get(f))
            else:
                print(f"skipping {f}: unknown type")
        except Exception as e:
            print(f"could not load {f}: {e}")

    from .rsr import warm_up
    threading.Thread(target=lambda: _quiet(warm_up), daemon=True).start()

    url = f"http://127.0.0.1:{port}/"
    write_state_file(url, s.workdir)
    if port != a.port:
        print(f"port {a.port} is busy, using {port}", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    webapp = app.make_app()

    async def forget_state_file(_):
        from .server import STATE_FILE
        try:
            with open(STATE_FILE) as fh:
                if json.load(fh).get("pid") == os.getpid():
                    os.remove(STATE_FILE)
        except Exception:
            pass

    webapp.on_cleanup.append(forget_state_file)
    # printed once the port is open: an agent can wait for this line
    web.run_app(webapp, host="127.0.0.1", port=port, print=lambda _: print(f"Moorlight ready: {url}", flush=True))


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
