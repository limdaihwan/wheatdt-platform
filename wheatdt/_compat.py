"""Import SaQC in headless environments (SaQC imports tkinter for its optional GUI)."""
import os, sys, types, warnings
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")


def _stub_tk():
    try:
        import tkinter  # noqa: F401
        return
    except Exception:
        pass

    class _D:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return _D()
        def __call__(self, *a, **k): return _D()

    class _M(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return _D
    for m in ["tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.filedialog", "tkinter.simpledialog", "tkinter.font", "_tkinter"]:
        sys.modules.setdefault(m, _M(m))


def load_saqc():
    _stub_tk()
    try:
        from saqc import SaQC
        return SaQC
    except Exception:
        return None
