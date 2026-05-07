"""
Integracion Numerica — Reglas de Newton-Cotes
===============================================
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Reglas de Newton-Cotes (pag. 27-35)

Metodos implementados:
  1. Rectangulo medio   — compuesta
  2. Trapecio           — simple y compuesta
  3. Simpson 1/3        — simple y compuesta  (n par)
  4. Simpson 3/8        — simple y compuesta  (n multiplo de 3)

Pestanas:
  1. Grafico            — f(x) con rectangulos/trapecios visualizados
  2. Tabla              — valores f(x_i) en cada nodo
  3. Paso a paso        — desarrollo estilo cuaderno
  4. Comparacion        — los 4 metodos lado a lado
  5. Error              — error de truncamiento teorico y vs solucion analitica
  6. Analisis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ══════════════════════════════════════
# PALETA
# ══════════════════════════════════════
BG     = "#0d1117"
BG2    = "#161b22"
BG3    = "#1c2128"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
TEAL   = "#39d0d8"

_x = sp.Symbol("x")

METODOS = [
    "Rectangulo medio",
    "Trapecio simple",
    "Trapecio compuesto",
    "Simpson 1/3 simple",
    "Simpson 1/3 compuesto",
    "Simpson 3/8 simple",
    "Simpson 3/8 compuesto",
]

COLORES_METODO = {
    "Rectangulo medio":      TEAL,
    "Trapecio simple":       ACCENT,
    "Trapecio compuesto":    ACCENT,
    "Simpson 1/3 simple":    GREEN,
    "Simpson 1/3 compuesto": GREEN,
    "Simpson 3/8 simple":    PURPLE,
    "Simpson 3/8 compuesto": PURPLE,
}


# ══════════════════════════════════════
# ENTORNO DE EVALUACION
# ══════════════════════════════════════
def _env(x_val):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x_val
    return e

def f_eval(expr, x_val):
    return eval(expr, {"__builtins__": {}}, _env(x_val))

def f_vec(expr, xs):
    return np.array([f_eval(expr, xi) for xi in xs])


# ══════════════════════════════════════
# LOGICA — METODOS DE INTEGRACION
# Referencia: Caceres pag. 27-35
# ══════════════════════════════════════

# ── RECTANGULO MEDIO COMPUESTO (Caceres pag. 27) ──────────────────
def rectangulo_medio(expr, a, b, n):
    """
    I = h * sum_{i=1}^{n} f( (x_{i-1} + x_i) / 2 )
    h = (b-a)/n
    Ref: Caceres pag. 27
    """
    h       = (b - a) / n
    medios  = [a + (i - 0.5) * h for i in range(1, n + 1)]
    f_meds  = [f_eval(expr, xi) for xi in medios]
    I       = h * sum(f_meds)
    return I, h, medios, f_meds

# ── TRAPECIO SIMPLE (Caceres pag. 27) ─────────────────────────────
def trapecio_simple(expr, a, b):
    """
    I = (b-a)/2 * [f(a) + f(b)]
    Ref: Caceres pag. 27
    """
    fa = f_eval(expr, a)
    fb = f_eval(expr, b)
    I  = (b - a) / 2 * (fa + fb)
    return I, fa, fb

# ── TRAPECIO COMPUESTO (Caceres pag. 28) ──────────────────────────
def trapecio_compuesto(expr, a, b, n):
    """
    I = h/2 * [f(a) + 2*sum_{i=1}^{n-1} f(a+ih) + f(b)]
    h = (b-a)/n
    Ref: Caceres pag. 28
    """
    h    = (b - a) / n
    xs   = [a + i * h for i in range(n + 1)]
    ys   = [f_eval(expr, xi) for xi in xs]
    S    = ys[0] + 2 * sum(ys[1:n]) + ys[-1]
    I    = (h / 2) * S
    return I, h, xs, ys, S

# ── SIMPSON 1/3 SIMPLE (Caceres pag. 28-29) ───────────────────────
def simpson13_simple(expr, a, b):
    """
    I = h/3 * [f(a) + 4*f((a+b)/2) + f(b)]
    h = (b-a)/2
    Ref: Caceres pag. 28
    """
    h  = (b - a) / 2
    m  = (a + b) / 2
    fa = f_eval(expr, a)
    fm = f_eval(expr, m)
    fb = f_eval(expr, b)
    I  = (h / 3) * (fa + 4 * fm + fb)
    return I, h, m, fa, fm, fb

# ── SIMPSON 1/3 COMPUESTO (Caceres pag. 29) ───────────────────────
def simpson13_compuesto(expr, a, b, n):
    """
    n debe ser par.
    I = h/3 * [f(a) + 4*sum_impares f(a+ih) + 2*sum_pares f(a+ih) + f(b)]
    h = (b-a)/n
    Ref: Caceres pag. 29
    """
    if n % 2 != 0:
        n += 1   # forzar par
    h   = (b - a) / n
    xs  = [a + i * h for i in range(n + 1)]
    ys  = [f_eval(expr, xi) for xi in xs]
    S_imp = sum(ys[i] for i in range(1, n, 2))   # impares
    S_par = sum(ys[i] for i in range(2, n - 1, 2))  # pares (excl extremos)
    S   = ys[0] + 4 * S_imp + 2 * S_par + ys[-1]
    I   = (h / 3) * S
    return I, h, xs, ys, S_imp, S_par, n

# ── SIMPSON 3/8 SIMPLE (Caceres pag. 30) ──────────────────────────
def simpson38_simple(expr, a, b):
    """
    I = 3h/8 * [f(a) + 3*f(x1) + 3*f(x2) + f(b)]
    h = (b-a)/3
    Ref: Caceres pag. 30
    """
    h  = (b - a) / 3
    x1 = a + h
    x2 = a + 2 * h
    fa = f_eval(expr, a)
    f1 = f_eval(expr, x1)
    f2 = f_eval(expr, x2)
    fb = f_eval(expr, b)
    I  = (3 * h / 8) * (fa + 3 * f1 + 3 * f2 + fb)
    return I, h, x1, x2, fa, f1, f2, fb

# ── SIMPSON 3/8 COMPUESTO (Caceres pag. 30-31) ────────────────────
def simpson38_compuesto(expr, a, b, n):
    """
    n debe ser multiplo de 3.
    Ref: Caceres pag. 30
    """
    while n % 3 != 0:
        n += 1
    h  = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    ys = [f_eval(expr, xi) for xi in xs]
    # indices: multiplos de 3 (internos), resto impares/pares
    S_mult3 = sum(ys[i] for i in range(3, n - 2, 3))
    S_rest  = sum(ys[i] for i in range(1, n) if i % 3 != 0)
    S = ys[0] + 3 * S_rest + 2 * S_mult3 + ys[-1]
    I = (3 * h / 8) * S
    return I, h, xs, ys, n

# ── SOLUCION ANALITICA opcional ───────────────────────────────────
def integral_analitica(expr, a, b):
    try:
        f_sym = sp.sympify(expr)
        F     = sp.integrate(f_sym, _x)
        val   = float((F.subs(_x, b) - F.subs(_x, a)).evalf())
        F_str = str(F)
        return val, F_str
    except Exception:
        return None, None

# ── ERROR DE TRUNCAMIENTO TEORICO ────────────────────────────────
def error_truncamiento(expr, a, b, n, metodo):
    """
    Calcula el error de truncamiento teorico.
    Devuelve: (ET, detalle_dict) donde detalle_dict tiene:
      - orden_deriv: orden de la derivada usada
      - f_deriv_str: string de la derivada simbolica
      - xs_scan: puntos evaluados
      - vals_scan: |f^(k)(xi)| en cada punto
      - x_max: punto donde se alcanza el maximo
      - M_val: valor maximo de |f^(k)|
      - formula_str: string de la formula del ET
    Ref: Caceres pag. 28-31
    """
    try:
        f_sym = sp.sympify(expr)
        xs_scan = np.linspace(a, b, 200)

        def _evaluar_derivada(f_sym, orden):
            fd = sp.diff(f_sym, _x, orden)
            fd_str = str(sp.simplify(fd))
            vals, xs_ok = [], []
            for xi in xs_scan:
                try:
                    v = abs(float(fd.subs(_x, xi).evalf()))
                    vals.append(v)
                    xs_ok.append(xi)
                except Exception:
                    pass
            if not vals:
                return fd_str, [], [], 0.0, a
            idx_max = int(np.argmax(vals))
            return fd_str, xs_ok, vals, vals[idx_max], xs_ok[idx_max]

        if "Trapecio" in metodo:
            fd_str, xs_ok, vals, M, x_max = _evaluar_derivada(f_sym, 2)
            ET = abs((b - a)**3 / (12 * n**2) * M)
            return ET, {
                "orden_deriv": 2, "f_deriv_str": fd_str,
                "xs_scan": xs_ok, "vals_scan": vals,
                "x_max": x_max, "M_val": M,
                "formula_str": "ET = (b-a)^3 / (12*n^2) * max|f''(xi)|",
                "formula_num": f"ET = ({b}-{a})^3 / (12*{n}^2) * {M:.6f}"
            }

        elif "1/3" in metodo:
            fd_str, xs_ok, vals, M, x_max = _evaluar_derivada(f_sym, 4)
            ET = abs((b - a)**5 / (180 * n**4) * M)
            return ET, {
                "orden_deriv": 4, "f_deriv_str": fd_str,
                "xs_scan": xs_ok, "vals_scan": vals,
                "x_max": x_max, "M_val": M,
                "formula_str": "ET = (b-a)^5 / (180*n^4) * max|f^(4)(xi)|",
                "formula_num": f"ET = ({b}-{a})^5 / (180*{n}^4) * {M:.6f}"
            }

        elif "3/8" in metodo:
            fd_str, xs_ok, vals, M, x_max = _evaluar_derivada(f_sym, 4)
            ET = abs((b - a)**5 / 6480 * M)
            return ET, {
                "orden_deriv": 4, "f_deriv_str": fd_str,
                "xs_scan": xs_ok, "vals_scan": vals,
                "x_max": x_max, "M_val": M,
                "formula_str": "ET = (b-a)^5 / 6480 * max|f^(4)(xi)|",
                "formula_num": f"ET = ({b}-{a})^5 / 6480 * {M:.6f}"
            }

        elif "Rectangulo" in metodo:
            fd_str, xs_ok, vals, M, x_max = _evaluar_derivada(f_sym, 2)
            ET = abs((b - a)**3 / (24 * n**2) * M)
            return ET, {
                "orden_deriv": 2, "f_deriv_str": fd_str,
                "xs_scan": xs_ok, "vals_scan": vals,
                "x_max": x_max, "M_val": M,
                "formula_str": "ET = (b-a)^3 / (24*n^2) * max|f''(xi)|",
                "formula_num": f"ET = ({b}-{a})^3 / (24*{n}^2) * {M:.6f}"
            }
    except Exception:
        pass
    return None, {}


# ══════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════
def _lbl(parent, text, bg=BG2, fg=MUTED, font=("Consolas", 11)):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font)

def _entry(parent, default):
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 12), bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, relief="flat")
    e.insert(0, default)
    return e

def _labeled_entry(parent, label, default):
    _lbl(parent, label).pack(anchor="w")
    e = _entry(parent, default)
    e.pack(fill=tk.X, ipady=7, pady=(2, 8))
    return e

def _btn(parent, text, cmd, color=ACCENT, fg="#000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 12, "bold"),
                 padx=12, pady=8, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h):
    r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0,int(r*.8)), max(0,int(g*.8)), max(0,int(b*.8)))

def _scrollable(parent):
    wrap  = tk.Frame(parent, bg=BG)
    wrap.pack(fill=tk.BOTH, expand=True)
    sc    = tk.Canvas(wrap, bg=BG, highlightthickness=0)
    vsb   = tk.Scrollbar(wrap, orient="vertical", command=sc.yview)
    sc.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(sc, bg=BG)
    win   = sc.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>",
        lambda e: sc.configure(scrollregion=sc.bbox("all")))
    sc.bind("<Configure>",
        lambda e: sc.itemconfig(win, width=e.width))
    sc.bind_all("<MouseWheel>",
        lambda e: sc.yview_scroll(int(-1*(e.delta/120)), "units"))
    return inner


# ══════════════════════════════════════
# BLOQUES VISUALES ESTILO CUADERNO
# ══════════════════════════════════════
def _seccion(parent, titulo, color=ACCENT):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill=tk.X, padx=10, pady=(14, 4))
    tk.Frame(f, bg=color, width=4).pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(f, text=f"  {titulo}", bg=BG, fg=color,
             font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=4)

def _card(parent, color=ACCENT):
    outer = tk.Frame(parent, bg=BG)
    outer.pack(fill=tk.X, padx=10, pady=4)
    tk.Frame(outer, bg=color, width=3).pack(side=tk.LEFT, fill=tk.Y)
    inner = tk.Frame(outer, bg=BG2)
    inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return inner

def _c_titulo(parent, texto, color=TEAL):
    f = tk.Frame(parent, bg=BG2)
    f.pack(fill=tk.X, padx=14, pady=(10, 2))
    tk.Label(f, text=texto, bg=BG2, fg=color,
             font=("Consolas", 12, "bold", "underline")).pack(anchor="w")

def _c_formula(parent, texto, color=MUTED, indent=1):
    prefix = "   " * indent
    tk.Label(parent, text=prefix+texto, bg=BG2, fg=color,
             font=("Consolas", 12), justify="left", anchor="w").pack(
                 fill=tk.X, padx=18, pady=1)

def _c_igual(parent, izq, der, color_der=GREEN, indent=1):
    prefix = "   " * indent
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=18, pady=2)
    tk.Label(row, text=prefix+izq+" = ", bg=BG2, fg=MUTED,
             font=("Consolas", 12)).pack(side=tk.LEFT)
    tk.Label(row, text=der, bg=BG2, fg=color_der,
             font=("Consolas", 12, "bold")).pack(side=tk.LEFT)

def _c_resultado_box(parent, texto, color=GREEN):
    f = tk.Frame(parent, bg=color, padx=2, pady=2)
    f.pack(fill=tk.X, padx=20, pady=6)
    inner = tk.Frame(f, bg=BG3)
    inner.pack(fill=tk.BOTH)
    tk.Label(inner, text="  " + texto, bg=BG3, fg=color,
             font=("Consolas", 13, "bold"), padx=10, pady=8).pack(anchor="w")

def _c_sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=6)

def _espacio(parent, h=6):
    tk.Frame(parent, bg=BG2, height=h).pack()

def _c_nodo(parent, i, xi, fxi, peso, color=ACCENT):
    """Fila de tabla de nodos."""
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=18, pady=1)
    tk.Label(row, text=f"   i={i}  x_{i}={xi:.6f}  f(x_{i})={fxi:.8f}  peso={peso}",
             bg=BG2, fg=color, font=("Consolas", 11)).pack(side=tk.LEFT)


# ══════════════════════════════════════
# CLASE PRINCIPAL — INTEGRACION
# ══════════════════════════════════════
class IntegracionApp(tk.Frame):

    TABS = [
        ("📊", "Grafico"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("⚖",  "Comparacion"),
        ("📉", "Error"),
        ("🧠", "Analisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Integracion Numerica — Newton-Cotes")
            master.configure(bg=BG)
            master.geometry("1380x780")
            master.minsize(1100, 640)
        self._resultado = {}
        self._build()

    # ──────────── LAYOUT ────────────
    def _build(self):
        self._topbar()
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._sidebar(body)
        self._main_area(body)

    def _topbar(self):
        bar = tk.Frame(self, bg=BG2, height=46)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="  Integracion Numerica — Newton-Cotes",
                 bg=BG2, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="Rectangulos  |  Trapecios  |  Simpson   |   Ref: Caceres 2026 pag.27",
                 bg=BG2, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=310)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        # Canvas scrollable para el sidebar
        sb_canvas = tk.Canvas(sb, bg=BG2, highlightthickness=0, width=310)
        sb_vsb    = tk.Scrollbar(sb, orient="vertical", command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=sb_vsb.set)
        sb_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(sb_canvas, bg=BG2)
        sb_win = sb_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(e):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
        def _on_canvas_configure(e):
            sb_canvas.itemconfig(sb_win, width=e.width)
        inner.bind("<Configure>", _on_inner_configure)
        sb_canvas.bind("<Configure>", _on_canvas_configure)
        sb_canvas.bind("<MouseWheel>",
            lambda e: sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        inner_pad = tk.Frame(inner, bg=BG2)
        inner_pad.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        inner = inner_pad

        _lbl(inner, "FUNCION Y LIMITES", fg=MUTED,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))

        self.e_f = _labeled_entry(inner, "f(x)  — funcion a integrar", "sin(x)")
        self.e_a = _labeled_entry(inner, "a  — limite inferior", "0")
        self.e_b = _labeled_entry(inner, "b  — limite superior", "pi")
        self.e_n = _labeled_entry(inner, "n  — subintervalos", "4")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=6)
        _lbl(inner, "METODO", fg=MUTED,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 4))

        self._metodo_var = tk.StringVar(value="Trapecio compuesto")
        for m in METODOS:
            color = COLORES_METODO.get(m, ACCENT)
            tk.Radiobutton(
                inner, text=m, variable=self._metodo_var, value=m,
                bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
                font=("Segoe UI", 11),
            ).pack(anchor="w", pady=1)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        _lbl(inner, "Solucion analitica — opcional").pack(anchor="w")
        _lbl(inner, "(dejar vacio si no se conoce)", fg=MUTED,
             font=("Consolas", 9)).pack(anchor="w")
        self.e_analitica = _entry(inner, "")
        self.e_analitica.pack(fill=tk.X, ipady=5, pady=(2, 8))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(inner, "Calcular",     self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "Comparar todos", self._comparar,
             color=BG3, fg=ACCENT).pack(fill=tk.X, pady=3)
        _btn(inner, "Graficar",     self._graficar,
             color=BG3, fg=GREEN).pack(fill=tk.X, pady=3)

    def _main_area(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tab_bar = tk.Frame(right, bg=BG2, height=42)
        self._tab_bar.pack(fill=tk.X)
        self._tab_bar.pack_propagate(False)

        self._tab_btns   = {}
        self._tab_frames = {}

        for icon, name in self.TABS:
            b = tk.Label(self._tab_bar, text=f"{icon} {name}",
                         bg=BG2, fg=MUTED, font=("Segoe UI", 12),
                         padx=14, pady=12, cursor="hand2")
            b.pack(side=tk.LEFT)
            b.bind("<Button-1>", lambda e, n=name: self._show_tab(n))
            self._tab_btns[name] = b

        self._panels = tk.Frame(right, bg=BG)
        self._panels.pack(fill=tk.BOTH, expand=True)

        self._build_panel_grafico()
        self._build_panel_tabla()
        self._build_panel_steps()
        self._build_panel_comparacion()
        self._build_panel_error()
        self._build_panel_analisis()
        self._show_tab("Paso a paso")

    def _show_tab(self, name):
        for n, b in self._tab_btns.items():
            b.config(fg=TEXT if n == name else MUTED)
        for n, f in self._tab_frames.items():
            if n == name:
                f.pack(fill=tk.BOTH, expand=True)
            else:
                f.pack_forget()

    def _panel(self, name):
        f = tk.Frame(self._panels, bg=BG)
        self._tab_frames[name] = f
        return f

    # ──────────── PANELES ────────────
    def _build_panel_grafico(self):
        f = self._panel("Grafico")
        self._fig = Figure(figsize=(9, 5), facecolor=BG)
        self._ax  = self._fig.add_subplot(111)
        self._style_ax(self._ax)
        self._canvas = FigureCanvasTkAgg(self._fig, master=f)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_panel_tabla(self):
        f = self._panel("Tabla")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.Treeview",
                        background=BG2, fieldbackground=BG2,
                        foreground=TEXT, rowheight=30,
                        font=("Consolas", 11))
        style.configure("Dark.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Segoe UI", 11, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])
        cols = ("i", "x_i", "f(x_i)", "peso", "contribucion")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        for col, w in zip(cols, [50, 140, 160, 80, 160]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="e")
        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

    def _build_panel_steps(self):
        f = self._panel("Paso a paso")
        self._si = _scrollable(f)

    def _build_panel_comparacion(self):
        f = self._panel("Comparacion")
        self._si_comp = _scrollable(f)

    def _build_panel_error(self):
        f = self._panel("Error")
        self._si_err = _scrollable(f)

    def _build_panel_analisis(self):
        f = self._panel("Analisis")
        self._ta = tk.Text(f, bg=BG3, fg=TEXT,
                           font=("Consolas", 12), bd=0, padx=20, pady=16,
                           relief="flat", wrap="word", state="disabled")
        self._ta.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        for tag, col in [("title", ACCENT), ("ok", GREEN), ("warn", YELLOW),
                          ("info", PURPLE), ("muted", MUTED), ("red", RED)]:
            kw = {"foreground": col}
            if tag == "title":
                kw["font"] = ("Consolas", 12, "bold")
            self._ta.tag_config(tag, **kw)

    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)

    # ──────────── PARSEAR ────────────
    def _parse(self):
        env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        env["np"] = np
        def ev(s, campo):
            try:
                return float(eval(s.strip(), {"__builtins__": {}}, env))
            except Exception as e:
                raise ValueError(f"Valor invalido en '{campo}': {s!r} — {e}")
        fexpr  = self.e_f.get().strip()
        a      = ev(self.e_a.get(), "a")
        b      = ev(self.e_b.get(), "b")
        n      = int(ev(self.e_n.get(), "n"))
        metodo = self._metodo_var.get()
        return fexpr, a, b, n, metodo

    # ──────────── CALCULAR ────────────
    def _calcular(self):
        try:
            fexpr, a, b, n, metodo = self._parse()
            if a >= b:
                raise ValueError("El limite inferior 'a' debe ser menor que 'b'.")

            resultado = self._ejecutar_metodo(fexpr, a, b, n, metodo)
            self._resultado = resultado

            self._render_tabla(resultado)
            self._render_pasos(fexpr, a, b, n, metodo, resultado)
            self._render_error(fexpr, a, b, n, metodo, resultado)
            self._render_analisis(fexpr, a, b, n, metodo, resultado)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _ejecutar_metodo(self, fexpr, a, b, n, metodo):
        """Ejecuta el metodo seleccionado y devuelve un dict de resultados."""
        r = {"metodo": metodo, "fexpr": fexpr, "a": a, "b": b, "n": n}

        if metodo == "Rectangulo medio":
            I, h, medios, f_meds = rectangulo_medio(fexpr, a, b, n)
            r.update({"I": I, "h": h, "medios": medios, "f_meds": f_meds})
            # tabla: nodo medio
            r["nodos"] = [(i, medios[i], f_meds[i], "1") for i in range(n)]

        elif metodo == "Trapecio simple":
            I, fa, fb = trapecio_simple(fexpr, a, b)
            r.update({"I": I, "fa": fa, "fb": fb})
            r["nodos"] = [(0, a, fa, "1"), (1, b, fb, "1")]

        elif metodo == "Trapecio compuesto":
            I, h, xs, ys, S = trapecio_compuesto(fexpr, a, b, n)
            r.update({"I": I, "h": h, "xs": xs, "ys": ys, "S": S})
            pesos = ["1"] + ["2"] * (n - 1) + ["1"]
            r["nodos"] = [(i, xs[i], ys[i], pesos[i]) for i in range(n+1)]

        elif metodo == "Simpson 1/3 simple":
            I, h, m, fa, fm, fb = simpson13_simple(fexpr, a, b)
            r.update({"I": I, "h": h, "m": m, "fa": fa, "fm": fm, "fb": fb})
            r["nodos"] = [(0, a, fa, "1"), (1, m, fm, "4"), (2, b, fb, "1")]

        elif metodo == "Simpson 1/3 compuesto":
            I, h, xs, ys, S_imp, S_par, n2 = simpson13_compuesto(fexpr, a, b, n)
            r.update({"I": I, "h": h, "xs": xs, "ys": ys,
                       "S_imp": S_imp, "S_par": S_par, "n": n2})
            pesos = []
            for i in range(len(xs)):
                if i == 0 or i == len(xs)-1:
                    pesos.append("1")
                elif i % 2 == 1:
                    pesos.append("4")
                else:
                    pesos.append("2")
            r["nodos"] = [(i, xs[i], ys[i], pesos[i]) for i in range(len(xs))]

        elif metodo == "Simpson 3/8 simple":
            I, h, x1, x2, fa, f1, f2, fb = simpson38_simple(fexpr, a, b)
            r.update({"I": I, "h": h, "x1": x1, "x2": x2,
                       "fa": fa, "f1": f1, "f2": f2, "fb": fb})
            r["nodos"] = [(0, a, fa, "1"), (1, x1, f1, "3"),
                          (2, x2, f2, "3"), (3, b, fb, "1")]

        elif metodo == "Simpson 3/8 compuesto":
            I, h, xs, ys, n2 = simpson38_compuesto(fexpr, a, b, n)
            r.update({"I": I, "h": h, "xs": xs, "ys": ys, "n": n2})
            pesos = []
            for i in range(len(xs)):
                if i == 0 or i == len(xs)-1:
                    pesos.append("1")
                elif i % 3 == 0:
                    pesos.append("2")
                else:
                    pesos.append("3")
            r["nodos"] = [(i, xs[i], ys[i], pesos[i]) for i in range(len(xs))]

        # solucion analitica — por sympy si no se dio manual
        analitica_str = self.e_analitica.get().strip()
        if analitica_str:
            try:
                env2 = {k:v for k,v in math.__dict__.items() if not k.startswith("__")}
                r["I_analitica"] = float(eval(analitica_str, {"__builtins__":{}}, env2))
                r["I_analitica_str"] = analitica_str
            except Exception:
                r["I_analitica"] = None
        else:
            val, F_str = integral_analitica(fexpr, a, b)
            r["I_analitica"] = val
            r["I_analitica_str"] = F_str

        return r

    # ──────────── GRAFICAR ────────────
    def _graficar(self):
        try:
            fexpr, a, b, n, metodo = self._parse()
            ax = self._ax
            ax.clear()
            self._style_ax(ax)

            x_plot = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), 500)
            y_plot = [f_eval(fexpr, xi) for xi in x_plot]
            ax.plot(x_plot, y_plot, color=ACCENT, linewidth=2.5, label=f"f(x) = {fexpr}", zorder=3)

            color = COLORES_METODO.get(metodo, GREEN)

            if "Rectangulo" in metodo:
                h = (b - a) / n
                for i in range(1, n + 1):
                    xm = a + (i - 0.5) * h
                    ym = f_eval(fexpr, xm)
                    xl = a + (i - 1) * h
                    ax.bar(xl, ym, width=h, align="edge",
                           color=color, alpha=0.3, edgecolor=color, linewidth=1)

            elif "Trapecio" in metodo:
                if "compuesto" in metodo or "compuesto" in metodo.lower():
                    h   = (b - a) / n
                    xs2 = [a + i * h for i in range(n + 1)]
                    ys2 = [f_eval(fexpr, xi) for xi in xs2]
                    for i in range(n):
                        ax.fill_between([xs2[i], xs2[i+1]],
                                        [ys2[i], ys2[i+1]], alpha=0.25,
                                        color=color)
                        ax.plot([xs2[i], xs2[i], xs2[i+1], xs2[i+1]],
                                [0, ys2[i], ys2[i+1], 0],
                                color=color, linewidth=1, alpha=0.7)
                else:
                    fa = f_eval(fexpr, a)
                    fb = f_eval(fexpr, b)
                    ax.fill_between([a, b], [fa, fb], alpha=0.25, color=color)
                    ax.plot([a, a, b, b], [0, fa, fb, 0], color=color, linewidth=1.5)

            elif "1/3" in metodo or "3/8" in metodo:
                if "compuesto" in metodo or "compuesto" in metodo.lower():
                    h   = (b - a) / n
                    xs2 = [a + i * h for i in range(n + 1)]
                    ys2 = [f_eval(fexpr, xi) for xi in xs2]
                    ax.fill_between(xs2, ys2, alpha=0.2, color=color)
                    ax.scatter(xs2, ys2, color=color, s=50, zorder=5)
                else:
                    x_fill = np.linspace(a, b, 100)
                    y_fill = [f_eval(fexpr, xi) for xi in x_fill]
                    ax.fill_between(x_fill, y_fill, alpha=0.2, color=color)

            ax.axhline(0, color=BORDER, linewidth=0.8)
            ax.set_title(f"f(x) = {fexpr}  |  [{a}, {b}]  |  {metodo}",
                         color=TEXT, fontsize=10, pad=8)
            ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
            self._canvas.draw()
            self._show_tab("Grafico")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── COMPARAR ────────────
    def _comparar(self):
        try:
            fexpr, a, b, n, _ = self._parse()
            si = self._si_comp
            for w in si.winfo_children():
                w.destroy()

            _seccion(si, "COMPARACION DE TODOS LOS METODOS", ACCENT)
            c = _card(si, ACCENT)
            _c_titulo(c, f"Integral de f(x)={fexpr}  en [{a}, {b}]  con n={n}", ACCENT)

            # analitica
            val_anal, F_str = integral_analitica(fexpr, a, b)
            if val_anal is not None:
                _c_igual(c, "Solucion analitica", f"{val_anal:.10f}", GREEN)
            _espacio(c)

            resultados_comp = []
            for metodo in METODOS:
                try:
                    r2 = self._ejecutar_metodo(fexpr, a, b, n, metodo)
                    resultados_comp.append((metodo, r2["I"]))
                except Exception:
                    resultados_comp.append((metodo, None))

            # tabla comparativa
            for metodo, I_val in resultados_comp:
                col = COLORES_METODO.get(metodo, ACCENT)
                outer = tk.Frame(c, bg=BG2)
                outer.pack(fill=tk.X, padx=14, pady=3)
                tk.Frame(outer, bg=col, width=3).pack(side=tk.LEFT, fill=tk.Y)
                inner_row = tk.Frame(outer, bg=BG3)
                inner_row.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                if I_val is not None:
                    err_str = ""
                    if val_anal is not None:
                        err = abs(I_val - val_anal)
                        err_pct = abs(err / val_anal * 100) if val_anal != 0 else 0
                        err_str = f"   |   error = {err:.2e}  ({err_pct:.4f}%)"
                    txt = f"  {metodo:<26}  I = {I_val:.8f}{err_str}"
                    tk.Label(inner_row, text=txt, bg=BG3, fg=col,
                             font=("Consolas", 11), pady=5).pack(anchor="w")
                else:
                    tk.Label(inner_row, text=f"  {metodo}  — no disponible",
                             bg=BG3, fg=MUTED, font=("Consolas", 11)).pack(anchor="w")

            _espacio(c, 8)
            self._show_tab("Comparacion")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ══════════════════════════════════════
    # RENDER: TABLA DE NODOS
    # ══════════════════════════════════════
    def _render_tabla(self, r):
        for row in self._tree.get_children():
            self._tree.delete(row)
        h = r.get("h", (r["b"] - r["a"]) / r["n"])
        for nodo in r.get("nodos", []):
            i, xi, fxi, peso = nodo
            contrib = float(peso) * fxi * h if "Trapecio" not in r["metodo"] or peso != "2" else fxi * 2 * h
            self._tree.insert("", "end", values=(
                i, f"{xi:.6f}", f"{fxi:.8f}", peso, f"{fxi:.8f}"
            ))
        # total
        self._tree.insert("", "end", values=(
            "—", "—", "—", "I =", f"{r['I']:.8f}"
        ))

    # ══════════════════════════════════════
    # RENDER: PASO A PASO — estilo cuaderno
    # ══════════════════════════════════════
    def _render_pasos(self, fexpr, a, b, n, metodo, r):
        si = self._si
        for w in si.winfo_children():
            w.destroy()

        color = COLORES_METODO.get(metodo, ACCENT)

        # ── DATOS DEL PROBLEMA
        _seccion(si, "Datos del problema", ACCENT)
        c = _card(si, ACCENT)
        _c_titulo(c, "Integral a calcular:", ACCENT)
        _c_formula(c, f"I  =  integral de a={a} hasta b={b}  f(x) dx", ACCENT)
        _c_formula(c, f"f(x)  =  {fexpr}", ACCENT)
        _c_formula(c, f"n  =  {n}  subintervalos")
        _c_formula(c, f"Metodo:  {metodo}", color)
        _espacio(c)

        # ── METODO ESPECIFICO
        if metodo == "Rectangulo medio":
            self._pasos_rectangulo(si, fexpr, a, b, n, r, color)
        elif metodo == "Trapecio simple":
            self._pasos_trapecio_simple(si, fexpr, a, b, r, color)
        elif metodo == "Trapecio compuesto":
            self._pasos_trapecio_compuesto(si, fexpr, a, b, n, r, color)
        elif metodo == "Simpson 1/3 simple":
            self._pasos_simpson13_simple(si, fexpr, a, b, r, color)
        elif metodo == "Simpson 1/3 compuesto":
            self._pasos_simpson13_compuesto(si, fexpr, a, b, n, r, color)
        elif metodo == "Simpson 3/8 simple":
            self._pasos_simpson38_simple(si, fexpr, a, b, r, color)
        elif metodo == "Simpson 3/8 compuesto":
            self._pasos_simpson38_compuesto(si, fexpr, a, b, n, r, color)

        # ── RESULTADO FINAL
        _seccion(si, "RESULTADO FINAL", GREEN)
        c = _card(si, GREEN)
        _c_resultado_box(c,
            f"I  =  integral f(x) dx  aprox  {r['I']:.8f}", GREEN)
        if r.get("I_analitica") is not None:
            err = abs(r["I"] - r["I_analitica"])
            _c_igual(c, "Valor analitico", f"{r['I_analitica']:.8f}", TEAL)
            _c_igual(c, "Error absoluto",  f"{err:.2e}", YELLOW)
            if r["I_analitica"] != 0:
                _c_igual(c, "Error relativo",
                         f"{abs(err/r['I_analitica'])*100:.4f} %", YELLOW)
        _espacio(c)

    # ── PASO A PASO POR METODO ─────────────────────────────────────
    def _pasos_rectangulo(self, si, fexpr, a, b, n, r, color):
        h = r["h"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 27)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla del Rectangulo Medio Compuesta:", color)
        _c_formula(c, "I  =  h * sum_{i=1}^{n}  f( (x_{i-1} + x_i) / 2 )")
        _c_formula(c, "h  =  (b - a) / n")
        _espacio(c)
        _c_igual(c, "h", f"({b} - {a}) / {n}  =  {h:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Calcular puntos medios y f(x_mid)", color)
        c = _card(si, color)
        _c_titulo(c, "Para cada subintervalo calculamos el punto medio:", color)
        suma = 0.0
        for i, (xi, fxi) in enumerate(zip(r["medios"], r["f_meds"])):
            xl = a + i * h
            xr = a + (i+1) * h
            _c_formula(c,
                f"i={i+1}:  x_mid = ({xl:.4f} + {xr:.4f})/2 = {xi:.6f}"
                f"   f({xi:.4f}) = {fxi:.8f}",
                color if i < 3 else MUTED)
            suma += fxi
        if len(r["medios"]) > 3:
            _c_formula(c, f"   ... ({n} terminos en total)", MUTED)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        _c_formula(c, f"I  =  h * sum  =  {h:.6f} * {suma:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_trapecio_simple(self, si, fexpr, a, b, r, color):
        _seccion(si, "PASO 1 — Formula (Caceres pag. 27)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla del Trapecio Simple:", color)
        _c_formula(c, "I  =  (b - a) / 2  *  [f(a) + f(b)]")
        _espacio(c)

        _seccion(si, "PASO 2 — Evaluar en los extremos", color)
        c = _card(si, color)
        _c_igual(c, f"f(a) = f({a})", f"{r['fa']:.8f}", GREEN)
        _c_igual(c, f"f(b) = f({b})", f"{r['fb']:.8f}", GREEN)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        _c_formula(c, f"I  =  ({b} - {a}) / 2  *  [{r['fa']:.6f} + {r['fb']:.6f}]")
        _c_formula(c, f"   =  {(b-a)/2:.6f}  *  {r['fa']+r['fb']:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_trapecio_compuesto(self, si, fexpr, a, b, n, r, color):
        h   = r["h"]
        xs  = r["xs"]
        ys  = r["ys"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 28)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla del Trapecio Compuesta:", color)
        _c_formula(c, "I  =  h/2 * [f(a) + 2*f(x_1) + 2*f(x_2) + ... + 2*f(x_{n-1}) + f(b)]")
        _c_formula(c, "h  =  (b - a) / n")
        _espacio(c)
        _c_igual(c, "h", f"({b} - {a}) / {n}  =  {h:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Evaluar f(x_i) en todos los nodos", color)
        c = _card(si, color)
        _c_titulo(c, "Tabla de nodos:", color)
        for i in range(len(xs)):
            if i == 0 or i == len(xs)-1:
                peso_str = "1 (extremo)"
                col_i = GREEN
            else:
                peso_str = "2 (interior)"
                col_i = color
            _c_formula(c,
                f"x_{i} = {xs[i]:.6f}   f(x_{i}) = {ys[i]:.8f}   peso = {peso_str}",
                col_i)
        _espacio(c)

        _seccion(si, "PASO 3 — Suma ponderada", color)
        c = _card(si, color)
        S = r["S"]
        _c_formula(c, f"S  =  f(a) + 2*(interior) + f(b)")
        _c_igual(c, "S", f"{S:.8f}", color)
        _c_formula(c, f"I  =  h/2 * S  =  {h:.6f}/2 * {S:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_simpson13_simple(self, si, fexpr, a, b, r, color):
        h = r["h"]
        m = r["m"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 28)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla de Simpson 1/3 Simple:", color)
        _c_formula(c, "I  =  h/3 * [f(a) + 4*f((a+b)/2) + f(b)]")
        _c_formula(c, "h  =  (b - a) / 2")
        _espacio(c)
        _c_igual(c, "h", f"({b} - {a}) / 2  =  {h:.6f}", color)
        _c_igual(c, "punto medio m", f"({a} + {b}) / 2  =  {m:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Evaluar en 3 puntos", color)
        c = _card(si, color)
        _c_igual(c, f"f(a) = f({a})",   f"{r['fa']:.8f}", GREEN)
        _c_igual(c, f"f(m) = f({m:.4f})", f"{r['fm']:.8f}", color)
        _c_igual(c, f"f(b) = f({b})",   f"{r['fb']:.8f}", GREEN)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        S = r["fa"] + 4*r["fm"] + r["fb"]
        _c_formula(c, f"S  =  f(a) + 4*f(m) + f(b)")
        _c_formula(c, f"   =  {r['fa']:.6f} + 4*{r['fm']:.6f} + {r['fb']:.6f}")
        _c_igual(c, "S", f"{S:.6f}", color)
        _c_formula(c, f"I  =  h/3 * S  =  {h:.6f}/3 * {S:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_simpson13_compuesto(self, si, fexpr, a, b, n, r, color):
        h = r["h"]
        xs = r["xs"]
        ys = r["ys"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 29)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla de Simpson 1/3 Compuesta  (n debe ser par):", color)
        _c_formula(c, "I  =  h/3 * [f(a) + 4*sum_impares + 2*sum_pares + f(b)]")
        _c_formula(c, "h  =  (b - a) / n")
        _espacio(c)
        _c_igual(c, "n (par)", str(r["n"]), color)
        _c_igual(c, "h", f"({b} - {a}) / {r['n']}  =  {h:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Clasificar nodos por peso", color)
        c = _card(si, color)
        sum_imp = 0.0
        sum_par = 0.0
        for i in range(len(xs)):
            if i == 0 or i == len(xs)-1:
                etiqueta = "extremo  peso=1"
                col_i = GREEN
            elif i % 2 == 1:
                etiqueta = "impar    peso=4"
                col_i = ORANGE
                sum_imp += ys[i]
            else:
                etiqueta = "par      peso=2"
                col_i = PURPLE
                sum_par += ys[i]
            _c_formula(c,
                f"x_{i} = {xs[i]:.4f}   f = {ys[i]:.6f}   [{etiqueta}]",
                col_i)
        _espacio(c)

        _seccion(si, "PASO 3 — Sumas parciales y resultado", color)
        c = _card(si, color)
        _c_igual(c, "sum_impares", f"{sum_imp:.6f}", ORANGE)
        _c_igual(c, "sum_pares",   f"{sum_par:.6f}", PURPLE)
        S = ys[0] + 4*sum_imp + 2*sum_par + ys[-1]
        _c_igual(c, "S = f(a) + 4*sum_imp + 2*sum_par + f(b)", f"{S:.6f}", color)
        _c_formula(c, f"I  =  h/3 * S  =  {h:.6f}/3 * {S:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_simpson38_simple(self, si, fexpr, a, b, r, color):
        h = r["h"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 30)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla de Simpson 3/8 Simple:", color)
        _c_formula(c, "I  =  3h/8 * [f(a) + 3*f(x1) + 3*f(x2) + f(b)]")
        _c_formula(c, "h  =  (b - a) / 3")
        _espacio(c)
        _c_igual(c, "h", f"({b} - {a}) / 3  =  {h:.6f}", color)
        _c_igual(c, "x1", f"{r['x1']:.6f}", color)
        _c_igual(c, "x2", f"{r['x2']:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Evaluar en 4 puntos", color)
        c = _card(si, color)
        _c_igual(c, f"f(a)  = f({a})",         f"{r['fa']:.8f}", GREEN)
        _c_igual(c, f"f(x1) = f({r['x1']:.4f})", f"{r['f1']:.8f}", color)
        _c_igual(c, f"f(x2) = f({r['x2']:.4f})", f"{r['f2']:.8f}", color)
        _c_igual(c, f"f(b)  = f({b})",         f"{r['fb']:.8f}", GREEN)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        S = r["fa"] + 3*r["f1"] + 3*r["f2"] + r["fb"]
        _c_formula(c, "S  =  f(a) + 3*f(x1) + 3*f(x2) + f(b)")
        _c_igual(c, "S", f"{S:.6f}", color)
        _c_formula(c, f"I  =  3*h/8 * S  =  3*{h:.6f}/8 * {S:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_simpson38_compuesto(self, si, fexpr, a, b, n, r, color):
        h  = r["h"]
        xs = r["xs"]
        ys = r["ys"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 30)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla de Simpson 3/8 Compuesta  (n multiplo de 3):", color)
        _c_formula(c, "I  =  3h/8 * [f(a) + 3*sum(no_mult3) + 2*sum(mult3) + f(b)]")
        _espacio(c)
        _c_igual(c, "n (mult. 3)", str(r["n"]), color)
        _c_igual(c, "h", f"{h:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Clasificar nodos", color)
        c = _card(si, color)
        for i in range(len(xs)):
            if i == 0 or i == len(xs)-1:
                etq, col_i = "peso=1 (extremo)", GREEN
            elif i % 3 == 0:
                etq, col_i = "peso=2 (mult.3)", PURPLE
            else:
                etq, col_i = "peso=3", color
            _c_formula(c,
                f"x_{i} = {xs[i]:.4f}   f = {ys[i]:.6f}   [{etq}]",
                col_i)
        _espacio(c)

        _seccion(si, "PASO 3 — Resultado", color)
        c = _card(si, color)
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    # ══════════════════════════════════════
    # RENDER: ERROR
    # ══════════════════════════════════════
    def _render_error(self, fexpr, a, b, n, metodo, r):
        si = self._si_err
        for w in si.winfo_children():
            w.destroy()

        color = COLORES_METODO.get(metodo, ACCENT)

        _seccion(si, "ANALISIS DE ERROR", RED)
        c = _card(si, RED)
        _c_titulo(c, "Tipos de error en integracion numerica:", RED)
        _c_formula(c, "1. Error de truncamiento (ET) — por la formula")
        _c_formula(c, "2. Error absoluto — |I_analitica - I_numerica|")
        _c_formula(c, "3. Error relativo — |error_abs / I_analitica| * 100%")
        _espacio(c)

        # error de truncamiento teorico — paso a paso completo
        _seccion(si, "Error de truncamiento teorico", YELLOW)
        ET, det = error_truncamiento(fexpr, a, b, n, metodo)

        # PASO 1: formula
        c = _card(si, YELLOW)
        _c_titulo(c, f"PASO 1 — Formula del ET para {metodo}  (Caceres pag. 28-31):", YELLOW)
        if det:
            _c_formula(c, det["formula_str"], YELLOW)
        _espacio(c, 4)

        if det:
            orden = det["orden_deriv"]
            fd_str = det["f_deriv_str"]

            # PASO 2: derivada simbolica
            _seccion(si, f"PASO 2 — Calcular f^({orden})(x)  (derivada de orden {orden})", ORANGE)
            c = _card(si, ORANGE)
            _c_titulo(c, "Derivamos f(x):", ORANGE)
            _c_formula(c, f"f(x)         = {fexpr}", ACCENT)
            _c_formula(c, f"f^({orden})(x)  = {fd_str}", ORANGE)
            _espacio(c)

            # PASO 3: buscar el maximo
            _seccion(si, f"PASO 3 — Hallar max|f^({orden})(xi)|  en [{a:.4f}, {b:.4f}]", RED)
            c = _card(si, RED)
            _c_titulo(c, f"Evaluamos f^({orden})(x) en 200 puntos del intervalo:", RED)
            _c_formula(c, f"Intervalo:  [{a:.4f}, {b:.4f}]", MUTED)
            _c_formula(c, f"Paso de barrido:  h_scan = ({b:.4f}-{a:.4f})/200 = {(b-a)/200:.6f}", MUTED)

            # mostrar algunos valores
            xs_ok  = det["xs_scan"]
            vals   = det["vals_scan"]
            x_max  = det["x_max"]
            M_val  = det["M_val"]

            # mostrar muestra de 5 puntos representativos
            if len(xs_ok) >= 5:
                indices = [0, len(xs_ok)//4, len(xs_ok)//2, 3*len(xs_ok)//4, len(xs_ok)-1]
                _c_formula(c, "Muestra de evaluaciones:", MUTED)
                for idx in indices:
                    xi_v = xs_ok[idx]
                    vi   = vals[idx]
                    marca = "  <-- MAXIMO" if abs(xi_v - x_max) < 1e-8 else ""
                    _c_formula(c,
                        f"   x = {xi_v:.6f}   |f^({orden})(x)| = {vi:.8f}{marca}",
                        RED if marca else MUTED)

            _espacio(c, 4)
            _c_resultado_box(c,
                f"x_max = {x_max:.6f}   M = max|f^({orden})(xi)| = {M_val:.8f}",
                RED)
            _espacio(c)

            # PASO 4: calculo final del ET
            _seccion(si, "PASO 4 — Calculo final del ET", YELLOW)
            c = _card(si, YELLOW)
            _c_titulo(c, "Reemplazamos en la formula:", YELLOW)
            _c_formula(c, det["formula_str"], YELLOW)
            _c_formula(c, det["formula_num"], MUTED)
            _c_resultado_box(c, f"ET  <=  {ET:.2e}  =  {ET:.10f}", YELLOW)
        else:
            c = _card(si, YELLOW)
            _c_formula(c, "No se pudo calcular (funcion no parseable por sympy).", RED)
        _espacio(c)

        # error con solucion analitica
        if r.get("I_analitica") is not None:
            _seccion(si, "Error real vs solucion analitica", GREEN)
            c = _card(si, GREEN)
            err_abs = abs(r["I"] - r["I_analitica"])
            err_rel = abs(err_abs / r["I_analitica"] * 100) if r["I_analitica"] != 0 else 0
            _c_igual(c, "I analitica",  f"{r['I_analitica']:.8f}", TEAL)
            _c_igual(c, "I numerica",   f"{r['I']:.8f}", color)
            _c_igual(c, "Error absoluto", f"{err_abs:.2e}", YELLOW)
            _c_igual(c, "Error relativo", f"{err_rel:.6f} %", YELLOW)
            if ET is not None:
                ok = err_abs <= ET * 1.1
                msg = "OK — error real <= ET teorico" if ok else "El error real supera la cota (normal en derivadas de alto orden)"
                _c_resultado_box(c, msg, GREEN if ok else YELLOW)
            _espacio(c)
        else:
            _seccion(si, "Solucion analitica", MUTED)
            c = _card(si, MUTED)
            _c_formula(c, "No se ingreso solucion analitica.", MUTED)
            _c_formula(c, "Puedes ingresar el valor en el sidebar para ver el error real.", MUTED)
            _espacio(c)

    # ══════════════════════════════════════
    # RENDER: ANALISIS
    # ══════════════════════════════════════
    def _render_analisis(self, fexpr, a, b, n, metodo, r):
        ta = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANALISIS — INTEGRACION NUMERICA\n", "title")
        w("Ref: Caceres, Modelado y Simulacion, 2 ed. 2026, pag. 27-35\n\n", "muted")

        w("PARAMETROS\n", "title")
        w(f"  f(x) = {fexpr}\n", "info")
        w(f"  [a, b] = [{a}, {b}]\n")
        w(f"  n = {n}  subintervalos\n")
        w(f"  Metodo = {metodo}\n\n", "info")

        w("RESULTADO\n", "title")
        w(f"  I numerica  = "); w(f"{r['I']:.8f}\n\n", "ok")

        if r.get("I_analitica") is not None:
            w("COMPARACION ANALITICA\n", "title")
            w(f"  I analitica = "); w(f"{r['I_analitica']:.8f}\n", "info")
            err = abs(r["I"] - r["I_analitica"])
            w(f"  Error abs   = "); w(f"{err:.2e}\n", "warn")
            if r["I_analitica"] != 0:
                w(f"  Error rel   = "); w(f"{abs(err/r['I_analitica'])*100:.4f} %\n\n", "warn")

        w("ORDEN DE APROXIMACION\n", "title")
        ordenes = {
            "Rectangulo medio":      "O(h^2)  — orden 2",
            "Trapecio simple":       "O(h^2)  — orden 2",
            "Trapecio compuesto":    "O(h^2)  — orden 2",
            "Simpson 1/3 simple":    "O(h^4)  — orden 4",
            "Simpson 1/3 compuesto": "O(h^4)  — orden 4",
            "Simpson 3/8 simple":    "O(h^4)  — orden 4",
            "Simpson 3/8 compuesto": "O(h^4)  — orden 4",
        }
        w(f"  {metodo}: {ordenes.get(metodo, 'variable')}\n", "ok")
        w("\n  Orden mas alto = mayor precision con menos subintervalos.\n", "muted")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = IntegracionApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()