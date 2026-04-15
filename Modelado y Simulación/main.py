"""
Metodos Numericos — Modelado y Simulacion
==========================================
Tab bar custom con pill-style, esquinas redondeadas,
paleta suave/futurista y tipografia grande.

Ref: Caceres, Fundamentos de Modelado y Simulacion, 2 ed. 2026
"""

import tkinter as tk
from tkinter import font as tkfont

from biseccion           import BiseccionApp
from punto_fijo          import PuntoFijoApp
from aitken              import AitkenApp
from newton              import NewtonApp
from Lagrange            import LagrangeApp
from Diferencias_Finitas import DiferenciasFinitasApp
from integracion         import IntegracionApp
from montecarlo          import MontecarloApp
from runge_kutta         import RungeKuttaApp


# ══════════════════════════════════════════════════════
# PALETA  — dark suave / futurista
# ══════════════════════════════════════════════════════
BG        = "#0d1117"      # fondo principal
BG2       = "#161b22"      # superficies
BG3       = "#1c2128"      # inputs / cards
BORDER    = "#30363d"      # bordes sutiles

TEXT      = "#e6edf3"      # texto principal
MUTED     = "#8b949e"      # texto secundario

# pill activo — gradiente azul-violeta simulado con un color medio
PILL_BG   = "#2d4a7a"      # fondo pill activo (azul noche)
PILL_FG   = "#a5d4ff"      # texto pill activo (celeste suave)
PILL_IDLE = "#1c2128"      # fondo pill inactivo
PILL_IDLE_FG = "#6e7f96"  # texto pill inactivo

# acento suave para hover
HOVER_BG  = "#243044"
HOVER_FG  = "#c9dfff"

# indicador inferior pill activo
INDICATOR = "#58a6ff"

TOPBAR_BG = "#10161e"      # topbar ligeramente mas oscuro

# ══════════════════════════════════════════════════════
# FUENTES  — mas grandes, modernas
# ══════════════════════════════════════════════════════
FONT_TOP_TITLE = ("Segoe UI", 15, "bold")
FONT_TOP_SUB   = ("Segoe UI", 10)
FONT_TAB       = ("Segoe UI", 11, "bold")
FONT_BODY      = ("Segoe UI", 11)
FONT_MONO      = ("Consolas", 11)


# ══════════════════════════════════════════════════════
# HELPER — rounded rectangle en Canvas
# ══════════════════════════════════════════════════════
def _rounded_rect(canvas, x1, y1, x2, y2, r=10, **kwargs):
    """Dibuja un rectangulo con esquinas redondeadas."""
    pts = [
        x1+r, y1,
        x2-r, y1,
        x2,   y1,
        x2,   y1+r,
        x2,   y2-r,
        x2,   y2,
        x2-r, y2,
        x1+r, y2,
        x1,   y2,
        x1,   y2-r,
        x1,   y1+r,
        x1,   y1,
        x1+r, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


# ══════════════════════════════════════════════════════
# CUSTOM TAB BAR
# ══════════════════════════════════════════════════════
class ModernTabBar(tk.Frame):
    """
    Barra de pestanas custom:
    - Pills con esquinas redondeadas
    - Indicador inferior con color acento
    - Hover suave
    - Fuente grande
    """

    PAD_X   = 22    # padding horizontal dentro del pill
    PAD_Y   = 7     # padding vertical
    RADIUS  = 10    # radio de esquinas
    GAP     = 8     # espacio entre pills
    MARGIN  = 16    # margen izquierdo
    HEIGHT  = 52    # altura total de la barra

    def __init__(self, parent, tabs, on_select, **kwargs):
        super().__init__(parent, bg=TOPBAR_BG,
                         height=self.HEIGHT, **kwargs)
        self.pack_propagate(False)

        self._tabs      = tabs          # lista de strings
        self._on_select = on_select
        self._active    = 0
        self._hover     = -1

        self._canvas = tk.Canvas(self, bg=TOPBAR_BG,
                                 highlightthickness=0,
                                 height=self.HEIGHT)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        self._canvas.bind("<Motion>",    self._on_motion)
        self._canvas.bind("<Leave>",     self._on_leave)
        self._canvas.bind("<Button-1>",  self._on_click)
        self.bind("<Configure>",         lambda e: self._draw())

        self._draw()

    # ── COORDENADAS DE CADA PILL ──────────────────────
    def _pill_coords(self):
        """Devuelve lista de (x1,y1,x2,y2) por cada tab."""
        coords = []
        x = self.MARGIN
        h = self.HEIGHT

        # medir texto para cada tab
        f = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        for tab in self._tabs:
            tw = f.measure(tab)
            w  = tw + self.PAD_X * 2
            y1 = (h - (f.metrics("linespace") + self.PAD_Y * 2)) // 2
            y2 = h - y1
            coords.append((x, y1, x + w, y2))
            x += w + self.GAP

        return coords

    # ── DIBUJAR ───────────────────────────────────────
    def _draw(self):
        c = self._canvas
        c.delete("all")
        h = self.HEIGHT

        # separador inferior sutil
        c.create_line(0, h-1, c.winfo_width(), h-1,
                      fill=BORDER, width=1)

        coords = self._pill_coords()

        for i, (tab, (x1, y1, x2, y2)) in enumerate(zip(self._tabs, coords)):
            active = (i == self._active)
            hover  = (i == self._hover and not active)

            # ── fondo del pill
            if active:
                bg = PILL_BG
                fg = PILL_FG
            elif hover:
                bg = HOVER_BG
                fg = HOVER_FG
            else:
                bg = PILL_IDLE
                fg = PILL_IDLE_FG

            _rounded_rect(c, x1, y1, x2, y2,
                          r=self.RADIUS, fill=bg, outline="")

            # ── borde suave en hover/idle
            if not active:
                _rounded_rect(c, x1, y1, x2, y2,
                              r=self.RADIUS, fill="",
                              outline="#2a3548" if hover else "#232c3a",
                              width=1)

            # ── indicador inferior activo
            if active:
                mid_x = (x1 + x2) // 2
                iw = min(x2 - x1 - 16, 40)
                c.create_line(mid_x - iw//2, y2 + 2,
                               mid_x + iw//2, y2 + 2,
                               fill=INDICATOR, width=2,
                               capstyle="round")

            # ── texto
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            c.create_text(cx, cy, text=tab, fill=fg,
                          font=FONT_TAB, anchor="center",
                          tags=(f"tab_{i}",))

    # ── EVENTOS ───────────────────────────────────────
    def _tab_at(self, x, y):
        coords = self._pill_coords()
        for i, (x1, y1, x2, y2) in enumerate(coords):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return -1

    def _on_motion(self, event):
        idx = self._tab_at(event.x, event.y)
        if idx != self._hover:
            self._hover = idx
            self._draw()

    def _on_leave(self, event):
        if self._hover != -1:
            self._hover = -1
            self._draw()

    def _on_click(self, event):
        idx = self._tab_at(event.x, event.y)
        if idx != -1 and idx != self._active:
            self._active = idx
            self._draw()
            self._on_select(idx)

    def set_active(self, idx):
        if 0 <= idx < len(self._tabs):
            self._active = idx
            self._draw()


# ══════════════════════════════════════════════════════
# APLICACION PRINCIPAL
# ══════════════════════════════════════════════════════
class MetodosNumericos:

    WIN_TITLE = "Metodos Numericos — Modelado y Simulacion"
    WIN_SIZE  = "1400x800"
    WIN_MIN   = (1100, 640)

    METHODS = [
        ("Biseccion",        BiseccionApp),
        ("Punto Fijo",       PuntoFijoApp),
        ("Aitken",           AitkenApp),
        ("Newton-Raphson",   NewtonApp),
        ("Lagrange",         LagrangeApp),
        ("Dif. Finitas",     DiferenciasFinitasApp),
        ("Integracion",      IntegracionApp),
        ("Monte Carlo",      MontecarloApp),
        ("Runge-Kutta",      RungeKuttaApp),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.WIN_TITLE)
        self.root.geometry(self.WIN_SIZE)
        self.root.minsize(*self.WIN_MIN)
        self.root.configure(bg=BG)

        self._frames  = []
        self._current = 0

        self._build_topbar()
        self._build_tabbar()
        self._build_content()

        self._switch(0)

    # ── TOPBAR ────────────────────────────────────────
    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=TOPBAR_BG, height=68)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="◈", bg=TOPBAR_BG, fg=INDICATOR,
                 font=("Segoe UI", 22)).pack(side=tk.LEFT, padx=(18, 8))

        titles = tk.Frame(bar, bg=TOPBAR_BG)
        titles.pack(side=tk.LEFT, pady=10)
        tk.Label(titles, text="Metodos Numericos",
                 bg=TOPBAR_BG, fg=TEXT,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(titles,
                 text="Modelado y Simulacion",
                 bg=TOPBAR_BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w")

        tk.Frame(bar, bg=BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=20, pady=12)

        badge_frame = tk.Frame(bar, bg=TOPBAR_BG)
        badge_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._badge_labels = {}
        for i, (name, _) in enumerate(self.METHODS):
            lbl = tk.Label(badge_frame, text=name,
                           bg=TOPBAR_BG, fg=MUTED,
                           font=("Segoe UI", 10),
                           padx=0, pady=0)
            lbl.pack(side=tk.LEFT, padx=8, anchor="center")
            self._badge_labels[i] = lbl

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

    # ── TAB BAR ───────────────────────────────────────
    def _build_tabbar(self):
        tab_names = [name for name, _ in self.METHODS]
        self._tabbar = ModernTabBar(self.root, tab_names,
                                    on_select=self._switch)
        self._tabbar.pack(fill=tk.X)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

    # ── CONTENIDO ────────────────────────────────────
    def _build_content(self):
        self._content = tk.Frame(self.root, bg=BG)
        self._content.pack(fill=tk.BOTH, expand=True)

        for i, (name, AppClass) in enumerate(self.METHODS):
            frame = tk.Frame(self._content, bg=BG)
            app   = AppClass(frame, standalone=False)
            app.pack(fill=tk.BOTH, expand=True)
            self._frames.append(frame)

    # ── SWITCH ────────────────────────────────────────
    def _switch(self, idx):
        if self._frames:
            self._frames[self._current].pack_forget()

        self._current = idx
        self._frames[idx].pack(fill=tk.BOTH, expand=True)
        self._tabbar.set_active(idx)

        for i, lbl in self._badge_labels.items():
            if i == idx:
                lbl.config(fg=PILL_FG,
                            font=("Segoe UI", 10, "bold"))
            else:
                lbl.config(fg=MUTED,
                            font=("Segoe UI", 10))

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MetodosNumericos()
    app.run()