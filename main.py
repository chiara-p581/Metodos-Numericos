"""
Aplicación: Buscador de Raíces Numéricas
=========================================
Módulos:
  - Bisección       (biseccion.py  → BiseccionApp)
  - Punto Fijo      (punto_fijo.py → PuntoFijoApp)
  - Aitken Δ²       (aitken.py     → AitkenApp)
  - Newton-Raphson  (newton.py     → NewtonApp)

Uso:
    python main.py
"""

import tkinter as tk
from tkinter import ttk

from biseccion  import BiseccionApp
from punto_fijo import PuntoFijoApp
from aitken     import AitkenApp
from newton     import NewtonApp


# ══════════════════════════════════════
# PALETA — compartida con los módulos
# ══════════════════════════════════════
BG     = "#0d1117"
BG2    = "#161b22"
BG3    = "#1c2128"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
ACCENT = "#58a6ff"


# ══════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ══════════════════════════════════════
class RootFinderApp:

    APP_TITLE = "Métodos Numéricos — Búsqueda de Raíces"
    WIN_SIZE  = "1280x740"

    METHODS = [
        ("  Bisección  ",      BiseccionApp),
        ("  Punto Fijo  ",     PuntoFijoApp),
        ("  Aitken Δ²  ",      AitkenApp),
        ("  Newton-Raphson  ", NewtonApp),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.APP_TITLE)
        self.root.geometry(self.WIN_SIZE)
        self.root.resizable(True, True)
        self.root.configure(bg=BG)

        self._apply_theme()
        self._build_ui()

    # ──────────── TEMA ────────────
    def _apply_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        # notebook vacío (sin pestañas propias de ttk —
        # usamos nuestra tab bar personalizada)
        style.configure("Dark.TNotebook",
                        background=BG,
                        borderwidth=0,
                        tabmargins=0)
        style.configure("Dark.TNotebook.Tab",
                        background=BG2,
                        foreground=MUTED,
                        font=("Segoe UI", 10, "bold"),
                        padding=[20, 9],
                        borderwidth=0,
                        focuscolor=BG)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG3),
                               ("active",   BG3)],
                  foreground=[("selected", TEXT),
                               ("active",   TEXT)])
        style.layout("Dark.TNotebook.Tab", [
            ("Notebook.tab", {
                "sticky": "nswe",
                "children": [("Notebook.padding", {
                    "side": "top",
                    "sticky": "nswe",
                    "children": [("Notebook.label", {"sticky": ""})]
                })]
            })
        ])

    # ──────────── UI ────────────
    def _build_ui(self):
        self._topbar()
        self._notebook()

    def _topbar(self):
        bar = tk.Frame(self.root, bg=BG2, height=50)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        # borde inferior
        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill=tk.X)

        tk.Label(bar,
                 text="⚙  Métodos Numéricos — Búsqueda de Raíces",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG2, fg=TEXT).pack(side=tk.LEFT, padx=20)

        tk.Label(bar,
                 text="bisección  ·  punto fijo  ·  aitken  ·  newton",
                 font=("Segoe UI", 9),
                 bg=BG2, fg=MUTED).pack(side=tk.RIGHT, padx=20)

    def _notebook(self):
        nb = ttk.Notebook(self.root, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)

        for label, AppClass in self.METHODS:
            tab = tk.Frame(nb, bg=BG)
            nb.add(tab, text=label)

            # standalone=False → no reconfigura la ventana raíz
            instance = AppClass(tab, standalone=False)
            instance.pack(fill=tk.BOTH, expand=True)

    # ──────────── RUN ────────────
    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    app = RootFinderApp()
    app.run()