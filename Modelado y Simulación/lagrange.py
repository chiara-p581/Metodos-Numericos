"""
Polinomio Interpolante de Lagrange
===================================
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Polinomio Interpolante de Lagrange (pag. 20-23)

Pestanas:
  1. Grafico
  2. Tabla L_i
  3. Paso a paso    — construccion simbolica estilo cuaderno
  4. Error Local    — |f(x) - P(x)| en un punto dado
  5. Error Global   — cota teorica con hallar max omega via g'(x)=0
  6. Analisis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import re
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


# ══════════════════════════════════════
# ENTORNO DE EVALUACION
# ══════════════════════════════════════
def _math_env(x_val=None):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["np"] = np
    if x_val is not None:
        env["x"] = x_val
    return env

def evaluar_expr(expr, x_val):
    return eval(expr, {"__builtins__": {}}, _math_env(x_val))


# ══════════════════════════════════════
# LOGICA NUMERICA — LAGRANGE
# ══════════════════════════════════════
def base_lagrange(i, x_val, xs):
    li = 1.0
    for j in range(len(xs)):
        if j != i:
            li *= (x_val - xs[j]) / (xs[i] - xs[j])
    return li

def polinomio_lagrange(x_val, xs, ys):
    return sum(float(ys[i]) * base_lagrange(i, x_val, xs) for i in range(len(xs)))

def tabla_li_numerica(xs, ys, x_eval):
    rows  = []
    total = 0.0
    for i in range(len(xs)):
        li      = base_lagrange(i, x_eval, xs)
        contrib = float(ys[i]) * li
        total  += contrib
        rows.append({"i": i, "xi": float(xs[i]), "yi": float(ys[i]),
                     "Li": li, "yiLi": contrib})
    return rows, total


# ══════════════════════════════════════
# LOGICA SIMBOLICA — LAGRANGE
# ══════════════════════════════════════
def li_simbolico(i, xs_vals):
    """
    Devuelve:
      num_expr   — numerador simbolico en x (sin dividir)
      den_val    — denominador numerico (float)
      li_expr    — L_i(x) expandido como polinomio en x
      num_parts  — lista de strings "(x - xj)"
      den_parts  — lista de strings "(xi - xj)"
    """
    n = len(xs_vals)
    xi = xs_vals[i]
    num = sp.Integer(1)
    den_val = 1.0
    num_parts = []
    den_parts = []
    for j in range(n):
        if j != i:
            num      *= (_x - xs_vals[j])
            den_val  *= (xi - xs_vals[j])
            # formato legible
            xj_str = _fmt_num(xs_vals[j])
            xi_str = _fmt_num(xi)
            num_parts.append(f"(x - {xj_str})")
            den_parts.append(f"({xi_str} - {xj_str})")
    li_expr = sp.expand(num / den_val)
    return {
        "num_expr":  num,
        "den_val":   den_val,
        "li_expr":   li_expr,
        "num_expanded": sp.expand(num),
        "num_parts": num_parts,
        "den_parts": den_parts,
    }

def construir_Px_simbolico(xs_vals, ys_vals):
    """Construye P(x) como polinomio simbolico."""
    Px = sp.Integer(0)
    terminos = []
    for i in range(len(xs_vals)):
        r = li_simbolico(i, xs_vals)
        yi = ys_vals[i]
        # usar Rational si yi es entero/simple, float si no
        yi_sp = sp.Rational(yi).limit_denominator(1000) if abs(yi - round(yi)) < 1e-9 else float(yi)
        termino = yi_sp * r["li_expr"]
        terminos.append({
            **r,
            "i": i,
            "xi": xs_vals[i],
            "yi": yi,
            "yi_sp": yi_sp,
            "termino": termino,
            "termino_expanded": sp.expand(termino),
        })
        Px += termino
    Px_exp = sp.expand(Px)
    return terminos, Px_exp

def _fmt_num(v):
    """Formatea numero: entero si es entero, decimal si no."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}"

def _sympy_to_str(expr):
    """Convierte expresion sympy a string legible."""
    s = str(expr)
    s = s.replace("**", "^")
    return s

def _poly_to_str(expr):
    """Formatea polinomio sympy de forma elegante."""
    try:
        p = sp.Poly(expr, _x)
        coefs = p.all_coeffs()
        grado = p.degree()
        terms = []
        for k, c in enumerate(coefs):
            exp = grado - k
            cv  = float(c)
            if abs(cv) < 1e-10:
                continue
            if exp == 0:
                terms.append(f"{cv:+.4f}")
            elif exp == 1:
                terms.append(f"{cv:+.4f}x")
            else:
                terms.append(f"{cv:+.4f}x^{exp}")
        if not terms:
            return "0"
        s = "  ".join(terms)
        if s.startswith("+"):
            s = s[1:].strip()
        return s
    except Exception:
        return _sympy_to_str(sp.expand(expr))

def derivada_numerica_orden(expr, x_val, orden, h=1e-5):
    f = lambda v: evaluar_expr(expr, v)
    for _ in range(orden):
        fn = f
        f  = lambda v, fn=fn: (fn(v+h) - fn(v-h)) / (2*h)
    return f(x_val)

def max_derivada_intervalo(expr, orden, a, b, puntos=300):
    xs = np.linspace(a, b, puntos)
    vals = []
    for xi in xs:
        try:
            vals.append(abs(derivada_numerica_orden(expr, xi, orden)))
        except Exception:
            pass
    return max(vals) if vals else float("nan")

def hallar_max_omega(xs_vals, a, b):
    """
    Calcula:
      omega(x) = prod(x - x_i)
      omega'(x)
      criticos de omega en [a,b]
      maximo de |omega| en [a,b]
    Igual que en la foto del cuaderno.
    """
    omega = sp.Integer(1)
    for xi in xs_vals:
        xi_r = sp.Rational(xi).limit_denominator(1000) if abs(xi - round(xi)) < 1e-9 else xi
        omega *= (_x - xi_r)
    omega_exp = sp.expand(omega)
    domega    = sp.diff(omega_exp, _x)
    domega_s  = sp.simplify(domega)

    # raices de omega'
    try:
        crit_sym = sp.solve(domega, _x)
        criticos = []
        for c in crit_sym:
            try:
                cv = float(c.evalf())
                if a - 1e-9 <= cv <= b + 1e-9:
                    criticos.append(cv)
            except Exception:
                pass
    except Exception:
        criticos = []

    # evaluar |omega| en extremos + criticos
    puntos = [float(a), float(b)] + criticos
    evals  = []
    for pt in puntos:
        try:
            v = abs(float(omega_exp.subs(_x, pt).evalf()))
            evals.append((pt, v))
        except Exception:
            pass

    max_pt, max_val = max(evals, key=lambda e: e[1]) if evals else (a, 0.0)

    return {
        "omega_expr":  _poly_to_str(omega_exp),
        "domega_expr": _poly_to_str(domega_s),
        "omega_sym":   omega_exp,
        "domega_sym":  domega_s,
        "criticos":    criticos,
        "evals":       evals,
        "max_pt":      max_pt,
        "max_omega":   max_val,
    }


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

# ── estilo cuaderno ─────────────────
def _c_titulo(parent, texto, color=TEAL):
    f = tk.Frame(parent, bg=BG2)
    f.pack(fill=tk.X, padx=14, pady=(10, 2))
    tk.Label(f, text=texto, bg=BG2, fg=color,
             font=("Consolas", 12, "bold", "underline")).pack(anchor="w")

def _c_texto(parent, texto, color=MUTED, indent=0):
    prefix = "   " * indent
    tk.Label(parent, text=prefix+texto, bg=BG2, fg=color,
             font=("Consolas", 12), justify="left", anchor="w").pack(
                 fill=tk.X, padx=18, pady=1)

def _c_formula(parent, texto, color=TEXT, indent=1):
    prefix = "   " * indent
    tk.Label(parent, text=prefix+texto, bg=BG2, fg=color,
             font=("Consolas", 12), justify="left", anchor="w").pack(
                 fill=tk.X, padx=18, pady=1)

def _c_fraccion(parent, num_str, den_str, color_num=TEXT, color_den=MUTED, indent=2):
    """Dibuja una fraccion: numerador / linea / denominador."""
    prefix = "   " * indent
    ancho  = max(len(num_str), len(den_str)) + 4
    linea  = "─" * ancho
    for txt, col in [(prefix + num_str, color_num),
                     (prefix + linea,   BORDER),
                     (prefix + den_str, color_den)]:
        tk.Label(parent, text=txt, bg=BG2, fg=col,
                 font=("Consolas", 12), justify="left", anchor="w").pack(
                     fill=tk.X, padx=18, pady=0)

def _c_igual(parent, izq, der, color_der=GREEN, indent=2):
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
             font=("Consolas", 13, "bold"), padx=10, pady=6).pack(anchor="w")

def _c_sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=6)

def _espacio(parent, h=6):
    tk.Frame(parent, bg=BG2, height=h).pack()


# ══════════════════════════════════════
# CLASE PRINCIPAL — LAGRANGE
# ══════════════════════════════════════
class LagrangeApp(tk.Frame):

    TABS = [
        ("📊", "Grafico"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🔴", "Error Local"),
        ("🌐", "Error Global"),
        ("📈", "EG Grafico"),
        ("🧠", "Analisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Lagrange — Metodos Numericos")
            master.configure(bg=BG)
            master.geometry("1360x760")
            master.minsize(1080, 620)
        self._poly   = None
        self._xs     = np.array([])
        self._ys     = np.array([])
        self._x_eval = None          # None = no se evalua en un punto
        self._px     = None
        self._Px_sym = None          # P(x) simbolico
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
        tk.Label(bar, text="  Polinomio Interpolante de Lagrange",
                 bg=BG2, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="P(x) = sum  y_i * L_i(x)   |   Ref: Caceres 2026 pag. 20",
                 bg=BG2, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=360)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        _lbl(inner, "PUNTOS  (x_i, y_i)", fg=MUTED,
             font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 4))
        _lbl(inner, "Acepta: numeros, pi, e, sqrt(2), sin(pi/2)...").pack(anchor="w")

        self.e_xs = _labeled_entry(inner, "x_i  (separados por coma)", "0, 1, 2, 3, 4")
        self.e_ys = _labeled_entry(inner, "y_i  (separados por coma)", "1, 2, 0, 2, 3")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=4)
        _lbl(inner, "EVALUAR EN  (opcional)", fg=MUTED,
             font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4, 2))
        _lbl(inner, "Dejar vacio para solo construir P(x)").pack(anchor="w")
        self.e_xeval = _labeled_entry(inner, "x a evaluar  P(x)", "")

        _lbl(inner, "f(x) real  —  para calcular errores").pack(anchor="w")
        self.e_fx = _entry(inner, "")
        self.e_fx.pack(fill=tk.X, ipady=7, pady=(2, 8))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(inner, "▶  Calcular",   self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "📈  Graficar",  self._graficar,
             color=BG3, fg=ACCENT).pack(fill=tk.X, pady=3)

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
        self._build_panel_error_local()
        self._build_panel_error_global()
        self._build_panel_eg_grafico()
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
        self._fig = Figure(figsize=(8, 5), facecolor=BG)
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
                        foreground=TEXT, rowheight=32,
                        font=("Consolas", 11))
        style.configure("Dark.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Segoe UI", 11, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])
        cols = ("i", "x_i", "y_i", "L_i(x) en x*", "y_i*L_i")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        for col, w in zip(cols, [50, 120, 120, 160, 160]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="e")
        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

    def _build_panel_steps(self):
        f = self._panel("Paso a paso")
        self._si = _scrollable(f)

    def _build_panel_error_local(self):
        f = self._panel("Error Local")
        self._si_el = _scrollable(f)

    def _build_panel_error_global(self):
        f = self._panel("Error Global")
        self._si_eg = _scrollable(f)

    def _build_panel_eg_grafico(self):
        f = self._panel("EG Grafico")

        # ── parte superior: grafico matplotlib ────────────────────
        graf_top = tk.Frame(f, bg=BG, height=300)
        graf_top.pack(fill=tk.X)
        graf_top.pack_propagate(False)

        self._fig_eg = Figure(figsize=(8, 3.2), facecolor=BG)
        self._ax_eg  = self._fig_eg.add_subplot(111)
        self._style_ax(self._ax_eg)
        self._canvas_eg = FigureCanvasTkAgg(self._fig_eg, master=graf_top)
        self._canvas_eg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tk.Frame(f, bg=BORDER, height=2).pack(fill=tk.X)

        # ── parte inferior: tabla scrollable ──────────────────────
        bottom = tk.Frame(f, bg=BG)
        bottom.pack(fill=tk.BOTH, expand=True)

        # titulo tabla
        hdr = tk.Frame(bottom, bg=BG2)
        hdr.pack(fill=tk.X, padx=10, pady=(8, 0))
        tk.Label(hdr, text="  Tabla:  x_i   |   |f^(n+1)(x_i)|   |   es el maximo?",
                 bg=BG2, fg=MUTED, font=("Consolas", 11)).pack(anchor="w", pady=4)

        # treeview para la tabla
        style = ttk.Style()
        style.configure("EG.Treeview",
                        background=BG2, fieldbackground=BG2,
                        foreground=TEXT, rowheight=28,
                        font=("Consolas", 11))
        style.configure("EG.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Segoe UI", 11, "bold"), relief="flat")
        style.map("EG.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])

        cols_eg = ("i", "x_i", "|f^(n)(x_i)|", "maximo")
        self._tree_eg = ttk.Treeview(bottom, columns=cols_eg,
                                      show="headings", style="EG.Treeview")
        widths_eg = [50, 160, 200, 120]
        for col, w in zip(cols_eg, widths_eg):
            self._tree_eg.heading(col, text=col)
            self._tree_eg.column(col, width=w, anchor="e")
        sb_eg = ttk.Scrollbar(bottom, orient="vertical",
                               command=self._tree_eg.yview)
        self._tree_eg.configure(yscrollcommand=sb_eg.set)
        sb_eg.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10))
        self._tree_eg.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0,8))

    def _build_panel_analisis(self):
        f = self._panel("Analisis")
        self._ta = tk.Text(f, bg=BG3, fg=TEXT,
                           font=("Consolas", 12), bd=0, padx=20, pady=16,
                           relief="flat", wrap="word", state="disabled")
        self._ta.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        for tag, col in [("title", ACCENT), ("ok", GREEN), ("warn", YELLOW),
                          ("info", PURPLE), ("muted", MUTED), ("val", ORANGE),
                          ("red", RED)]:
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
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)

    # ──────────── PARSEAR INPUTS ────────────
    def _parse_inputs(self):
        env = _math_env()
        def _ev(s):
            return float(eval(s.strip(), {"__builtins__": {}}, env))
        xs = [_ev(v) for v in self.e_xs.get().split(",")]
        ys = [_ev(v) for v in self.e_ys.get().split(",")]
        if len(xs) != len(ys):
            raise ValueError("x_i e y_i deben tener la misma cantidad de valores.")
        if len(set(xs)) != len(xs):
            raise ValueError("Los valores de x_i deben ser distintos.")
        # x_eval es OPCIONAL
        xeval_str = self.e_xeval.get().strip()
        x_eval = _ev(xeval_str) if xeval_str else None
        return np.array(xs), np.array(ys), x_eval

    def _get_fexpr(self):
        return self.e_fx.get().strip()

    def _eval_fx(self, x_val):
        expr = self._get_fexpr()
        if not expr:
            return None
        return evaluar_expr(expr, x_val)

    # ──────────── CALCULAR ────────────
    def _calcular(self):
        try:
            xs, ys, x_eval = self._parse_inputs()
            self._xs     = xs
            self._ys     = ys
            self._x_eval = x_eval

            # construir P(x) simbolico
            terminos, Px_sym = construir_Px_simbolico(xs.tolist(), ys.tolist())
            self._Px_sym  = Px_sym
            self._terminos = terminos

            # evaluar en x* si se dio
            px = None
            if x_eval is not None:
                px = polinomio_lagrange(x_eval, xs, ys)
            self._px = px

            # tabla numerica (solo si hay x_eval)
            if x_eval is not None:
                filas, total = tabla_li_numerica(xs, ys, x_eval)
                self._render_tabla(filas, total)
            else:
                self._render_tabla_vacia()

            self._render_pasos(xs, ys, x_eval, px, terminos, Px_sym)
            self._render_error_local(xs, ys, x_eval, px)
            self._render_error_global(xs, ys, x_eval, px, Px_sym)
            self._render_analisis(xs, ys, x_eval, px, Px_sym)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── GRAFICAR ────────────
    def _graficar(self):
        try:
            xs, ys, x_eval = self._parse_inputs()
            margin  = max(1.0, (max(xs) - min(xs)) * 0.35)
            x_plot  = np.linspace(min(xs) - margin, max(xs) + margin, 600)
            y_poly  = [polinomio_lagrange(xi, xs, ys) for xi in x_plot]

            ax = self._ax
            ax.clear()
            self._style_ax(ax)
            ax.plot(x_plot, y_poly, color=ACCENT, linewidth=2, label="P(x)")

            expr = self._get_fexpr()
            if expr:
                try:
                    y_real = [evaluar_expr(expr, xi) for xi in x_plot]
                    ax.plot(x_plot, y_real, color=GREEN, linewidth=1.5,
                            linestyle="--", label="f(x) real")
                except Exception:
                    pass

            ax.scatter(xs, ys, color=ORANGE, zorder=5, s=70,
                       label="Puntos (x_i, y_i)")

            if x_eval is not None:
                px = polinomio_lagrange(x_eval, xs, ys)
                ax.scatter([x_eval], [px], color=PURPLE, zorder=6, s=90,
                           label=f"P({x_eval:.3f}) = {px:.5f}")

            ax.axhline(0, color=BORDER, linewidth=0.8)
            ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
            self._canvas.draw()
            self._show_tab("Grafico")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ══════════════════════════════════════
    # RENDER: TABLA
    # ══════════════════════════════════════
    def _render_tabla(self, filas, total):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for r in filas:
            self._tree.insert("", "end", values=(
                r["i"],
                f"{r['xi']:.6f}",
                f"{r['yi']:.6f}",
                f"{r['Li']:.8f}",
                f"{r['yiLi']:.8f}",
            ))
        self._tree.insert("", "end", values=(
            "—", "—", "—", "TOTAL  P(x*) =", f"{total:.8f}"
        ))

    def _render_tabla_vacia(self):
        for row in self._tree.get_children():
            self._tree.delete(row)
        self._tree.insert("", "end", values=(
            "—", "—", "—", "Ingresa x* para ver", "la evaluacion"
        ))

    # ══════════════════════════════════════
    # RENDER: PASO A PASO — estilo cuaderno como la foto
    # ══════════════════════════════════════
    def _render_pasos(self, xs, ys, x_eval, px, terminos, Px_sym):
        si = self._si
        for w in si.winfo_children():
            w.destroy()

        n     = len(xs)
        grado = n - 1

        # ── BLOQUE 1: Datos
        _seccion(si, "Datos del problema", ACCENT)
        c = _card(si, ACCENT)
        _c_titulo(c, "Funcion y puntos dados:", ACCENT)
        if self._get_fexpr():
            _c_formula(c, f"f(x) = {self._get_fexpr()}", ACCENT)
        _c_formula(c, f"x pertenece a [{_fmt_num(min(xs))}, {_fmt_num(max(xs))}]")
        if x_eval is not None:
            _c_formula(c, f"Evaluar en:  x* = {_fmt_num(x_eval)}", YELLOW)
        _c_formula(c, f"n = {grado}  (grado del polinomio)")
        _espacio(c)
        # tabla de puntos como en la foto
        header = "   x  |  " + "  |  ".join(f"x_{i}={_fmt_num(xs[i])}" for i in range(n))
        _c_texto(c, header, MUTED)
        vals_row = "y=f(x)| " + "  |  ".join(f"{ys[i]:.5g}" for i in range(n))
        _c_texto(c, vals_row, ACCENT)
        _espacio(c)

        # ── BLOQUE 2: Formula general
        _seccion(si, "Formula de Lagrange", PURPLE)
        c = _card(si, PURPLE)
        _c_titulo(c, "Formula general:", PURPLE)
        _c_formula(c, "P_n(x)  =  sum_{i=0}^{n}  y_i  *  L_i(x)", PURPLE)
        _c_formula(c, "")
        _c_titulo(c, "Donde cada base L_i(x) es:", MUTED)
        _c_formula(c, "L_i(x)  =   prod_{j=0, j!=i}^{n}   (x - x_j)")
        _c_formula(c, "            ─────────────────────────────────")
        _c_formula(c, "                     (x_i - x_j)")
        _espacio(c)

        # ── BLOQUE 3: Calculo de cada L_i(x) como FUNCION DE x
        _seccion(si, f"Iteraciones — Calculo de cada L_i(x)", ORANGE)

        for t in terminos:
            i   = t["i"]
            xi  = t["xi"]
            yi  = t["yi"]
            r   = li_simbolico(i, xs.tolist())

            c = _card(si, ORANGE)

            # header tipo cuaderno: "Iteracion i+1: i=k, j!=k"
            j_vals = [j for j in range(n) if j != i]
            j_str  = ", ".join(str(j) for j in j_vals)
            _c_titulo(c,
                f"Iteracion {i+1}:  i = {i},  j = {j_str}",
                ORANGE)

            # Formula L_i(x) con fraccion visible
            num_str = "  *  ".join(r["num_parts"])
            den_str = "  *  ".join(r["den_parts"])
            _c_formula(c, f"L_{i}(x)  =  {num_str}")
            _c_formula(c, "          " + "─" * max(len(num_str), len(den_str)+2))
            _c_formula(c, f"             {den_str}")
            _espacio(c, 4)

            # sustitucion numerica en denominador
            den_vals_str = "  *  ".join(
                f"({_fmt_num(xi - xs[j])})" for j in range(n) if j != i
            )
            den_prod = t["den_val"]
            _c_formula(c, f"       =  {num_str}")
            _c_formula(c, "          " + "─" * (len(num_str) + 4))
            _c_formula(c, f"             {den_vals_str}  =  {_fmt_num(den_prod)}")
            _espacio(c, 4)

            # numerador expandido
            num_poly_str = _poly_to_str(r["num_expanded"])
            _c_formula(c, f"       =  {num_poly_str}")
            _c_formula(c, "          " + "─" * (len(num_poly_str) + 4))
            _c_formula(c, f"             {_fmt_num(den_prod)}")
            _espacio(c, 4)

            # resultado L_i(x) como polinomio en x
            li_str = _poly_to_str(r["li_expr"])
            _c_resultado_box(c, f"L_{i}(x)  =  {li_str}", ORANGE)
            _espacio(c, 4)

            # y_i * L_i(x) con y evaluado
            yi_str = f"{yi:.5g}" if abs(yi - round(yi)) > 1e-9 else str(int(round(yi)))
            term_str = _poly_to_str(t["termino_expanded"])
            _c_formula(c, f"y_{i}  *  L_{i}(x)  =  {yi_str}  *  ({li_str})")
            _c_resultado_box(c, f"y_{i} * L_{i}(x)  =  {term_str}", PURPLE)
            _espacio(c)

        # ── BLOQUE 4: P(x) como suma de terminos
        _seccion(si, "Construccion de P_n(x)", GREEN)
        c = _card(si, GREEN)
        _c_titulo(c, f"P_{grado}(x) =  suma de todos los terminos:", GREEN)

        # escribir como en la foto: [1/2(x^2-5x+6)] * e + ...
        partes_Px = []
        for t in terminos:
            i      = t["i"]
            yi_str = f"{t['yi']:.5g}" if abs(t['yi'] - round(t['yi'])) > 1e-9 else str(int(round(t['yi'])))
            li_s   = _poly_to_str(t["li_expr"])
            partes_Px.append(f"[{li_s}] * {yi_str}")

        _c_formula(c, f"P_{grado}(x) = " + "\n              + ".join(partes_Px))
        _espacio(c, 6)

        # P(x) expandido final
        Px_str = _poly_to_str(Px_sym)
        _c_resultado_box(c, f"P_{grado}(x)  =  {Px_str}", GREEN)
        _espacio(c, 6)

        # si hay x_eval, mostrar evaluacion numerica
        if x_eval is not None and px is not None:
            _c_sep(c)
            _c_titulo(c, f"Evaluacion en x* = {_fmt_num(x_eval)}:", YELLOW)
            _c_formula(c,
                f"P_{grado}({_fmt_num(x_eval)})  =  {Px_str.replace('x', '('+_fmt_num(x_eval)+')')}")
            _c_resultado_box(c, f"P_{grado}({_fmt_num(x_eval)})  =  {px:.8f}", YELLOW)

    # ══════════════════════════════════════
    # RENDER: ERROR LOCAL — como la foto
    # ══════════════════════════════════════
    def _render_error_local(self, xs, ys, x_eval, px):
        si = self._si_el
        for w in si.winfo_children():
            w.destroy()

        _seccion(si, "ERROR LOCAL", RED)
        c = _card(si, RED)
        _c_titulo(c, "Definicion:", RED)
        _c_formula(c, "|E(x*)| = |f(x*) - P_n(x*)|", RED)
        _c_formula(c, "Referencia: Caceres pag. 22", MUTED)
        _espacio(c)

        # verificar que tenemos lo necesario
        fexpr = self._get_fexpr()
        if not fexpr:
            _c_texto(c, "Necesitas ingresar f(x) en el sidebar.", YELLOW)
            _c_texto(c, "Ejemplo:  exp(x),  sin(x),  x**2", MUTED)
            return
        if x_eval is None:
            _c_texto(c, "Necesitas ingresar x* (punto de evaluacion).", YELLOW)
            return
        _espacio(c)

        # datos
        _seccion(si, "PASO 1 — Valores conocidos", ACCENT)
        c = _card(si, ACCENT)
        _c_titulo(c, "Datos del problema:", ACCENT)
        _c_formula(c, f"f(x) = {fexpr}", ACCENT)
        _c_formula(c, f"x*   = {_fmt_num(x_eval)}", YELLOW)
        _c_formula(c, f"x* pertenece al intervalo [{_fmt_num(min(xs))}, {_fmt_num(max(xs))}]")
        _espacio(c)

        # evaluar f(x*)
        _seccion(si, "PASO 2 — Calcular f(x*)", GREEN)
        c = _card(si, GREEN)
        try:
            fx_val = evaluar_expr(fexpr, x_eval)
        except Exception as e:
            _c_texto(c, f"Error al evaluar f(x*): {e}", RED)
            return
        _c_titulo(c, f"Evaluamos f(x) en x* = {_fmt_num(x_eval)}:", GREEN)
        _c_formula(c, f"f({_fmt_num(x_eval)})  =  {fexpr}  con  x = {_fmt_num(x_eval)}")
        _c_resultado_box(c, f"f({_fmt_num(x_eval)})  =  {fx_val:.8f}", GREEN)
        _espacio(c)

        # evaluar P(x*)
        _seccion(si, "PASO 3 — Calcular P(x*)", PURPLE)
        c = _card(si, PURPLE)
        Px_str = _poly_to_str(self._Px_sym) if self._Px_sym is not None else "P(x)"
        _c_titulo(c, f"Evaluamos el polinomio en x* = {_fmt_num(x_eval)}:", PURPLE)
        _c_formula(c, f"P({_fmt_num(x_eval)})  =  {Px_str}  con  x = {_fmt_num(x_eval)}")
        _c_resultado_box(c, f"P({_fmt_num(x_eval)})  =  {px:.8f}", PURPLE)
        _espacio(c)

        # error local
        _seccion(si, "PASO 4 — Error local", RED)
        c = _card(si, RED)
        err = abs(fx_val - px)
        _c_titulo(c, "Aplicamos la formula:", RED)
        _c_formula(c, f"|E({_fmt_num(x_eval)})| = |f(x*) - P(x*)|")
        _c_formula(c,
            f"         = |{fx_val:.8f} - {px:.8f}|")
        _c_formula(c,
            f"         = |{fx_val - px:.8f}|")
        _c_resultado_box(c, f"|E({_fmt_num(x_eval)})|  =  {err:.6f}  ≈  {err:.2e}", RED)
        _espacio(c)

        # interpretacion
        _seccion(si, "PASO 5 — Interpretacion", YELLOW)
        c = _card(si, YELLOW)
        if err < 1e-6:
            cal, col = "MUY BUENA  (error < 1e-6)", GREEN
        elif err < 1e-3:
            cal, col = "BUENA  (error < 1e-3)", ACCENT
        elif err < 1e-1:
            cal, col = "ACEPTABLE  (error < 0.1)", YELLOW
        else:
            cal, col = "ALTA — agregar mas puntos", RED
        _c_igual(c, "Precision", cal, col)
        if fx_val != 0:
            _c_igual(c, "Error relativo",
                     f"{abs(err/fx_val)*100:.4f} %", MUTED)
        _espacio(c)

    # ══════════════════════════════════════
    # RENDER: ERROR GLOBAL — como la foto
    # ══════════════════════════════════════
    def _render_error_global(self, xs, ys, x_eval, px, Px_sym):
        si = self._si_eg
        for w in si.winfo_children():
            w.destroy()

        n     = len(xs)
        grado = n - 1
        fexpr = self._get_fexpr()
        a     = float(min(xs))
        b     = float(max(xs))

        # definicion
        _seccion(si, "ERROR GLOBAL — Cota teorica", ORANGE)
        c = _card(si, ORANGE)
        _c_titulo(c, "Formula (Caceres pag. 21):", ORANGE)
        _c_formula(c, "|E(x)| <=  max|f^(n+1)(xi)|  *  |prod(x - x_i)|")
        _c_formula(c, "           ─────────────────")
        _c_formula(c, "                 (n+1)!")
        _espacio(c)

        # paso 1: derivada n-esima
        _seccion(si, "PASO 1 — Derivada n-esima  f^(n+1)(x)", TEAL)
        c = _card(si, TEAL)
        orden = n
        _c_titulo(c, f"Con n+1 = {n} puntos, el grado es {grado}, necesitamos:", TEAL)
        _c_formula(c, f"f^({orden})(x)  =  derivada de orden {orden} de f(x)")
        _c_formula(c, f"(n+1)!  =  {orden}!  =  {math.factorial(orden)}", GREEN)
        _espacio(c)

        # si hay f(x), mostrar derivada simbolica
        if fexpr:
            try:
                f_sym   = sp.sympify(fexpr)
                for _ in range(orden):
                    f_sym = sp.diff(f_sym, _x)
                f_sym_s = sp.simplify(f_sym)
                _c_titulo(c, "Calculamos la derivada:", TEAL)
                _c_formula(c, f"f(x)      = {fexpr}", ACCENT)
                _c_formula(c, f"f^({orden})(x)  = {str(f_sym_s)}", GREEN)
            except Exception:
                pass
        _espacio(c)

        # paso 2: intervalo
        _seccion(si, "PASO 2 — Intervalo de analisis", PURPLE)
        c = _card(si, PURPLE)
        _c_titulo(c, "Intervalo basado en los nodos:", PURPLE)
        _c_formula(c, f"[a, b]  =  [{_fmt_num(a)},  {_fmt_num(b)}]")
        _espacio(c)

        # paso 3: omega(x) = prod(x - x_i)
        _seccion(si, "PASO 3 — Funcion  omega(x) = prod(x - x_i)", GREEN)
        c = _card(si, GREEN)
        _c_titulo(c, "Definimos:", GREEN)
        omega_sym_str = "  *  ".join(f"(x - {_fmt_num(xi)})" for xi in xs)
        _c_formula(c, f"omega(x)  =  {omega_sym_str}", GREEN)
        _espacio(c, 4)

        # calcular omega con sympy
        r_omega = hallar_max_omega(xs.tolist(), a, b)
        _c_formula(c, f"omega(x)  =  {r_omega['omega_expr']}", MUTED)
        _espacio(c)

        # paso 4: hallar maximo de |omega(x)| via omega'(x)=0
        _seccion(si, "PASO 4 — Hallar max |omega(x)| via  omega'(x) = 0", YELLOW)
        c = _card(si, YELLOW)
        _c_titulo(c, "Como en la foto del cuaderno:", YELLOW)
        _c_formula(c, f"omega'(x)  =  {r_omega['domega_expr']}", YELLOW)
        _espacio(c, 4)
        _c_formula(c, "Resolvemos omega'(x) = 0  para hallar los puntos criticos:")
        if r_omega["criticos"]:
            for k, cv in enumerate(r_omega["criticos"]):
                ov = abs(float(r_omega["omega_sym"].subs(_x, cv)))
                _c_formula(c, f"x_{k+1}  =  {cv:.6f}   ->   |omega(x_{k+1})|  =  {ov:.6f}")
        else:
            _c_formula(c, "Sin criticos en el intervalo.")
        _espacio(c, 4)
        _c_formula(c, "Evaluamos tambien en los extremos del intervalo:")
        for pt, val in r_omega["evals"]:
            _c_formula(c, f"omega({_fmt_num(pt)})  =  {val:.6f}")
        _espacio(c, 4)
        _c_resultado_box(c,
            f"max |omega(x)| en [{_fmt_num(a)}, {_fmt_num(b)}]  =  {r_omega['max_omega']:.6f}",
            YELLOW)
        _espacio(c)

        # paso 5: M_(n+1)
        _seccion(si, f"PASO 5 — Maximo de  |f^({orden})(x)|  en el intervalo", RED)
        c = _card(si, RED)
        if not fexpr:
            _c_texto(c, "Ingresa f(x) para calcular M numericamente.", YELLOW)
            return
        try:
            # calcular M con todos los puntos del barrido
            puntos_scan = 300
            xs_scan = np.linspace(a, b, puntos_scan)
            vals_scan = []
            for xi in xs_scan:
                try:
                    vals_scan.append(abs(derivada_numerica_orden(fexpr, xi, orden)))
                except Exception:
                    vals_scan.append(float("nan"))
            vals_arr = np.array(vals_scan)
            idx_max  = int(np.nanargmax(vals_arr))
            x_max    = xs_scan[idx_max]
            M        = float(vals_arr[idx_max])

            # Mostrar muestra de 5 puntos representativos
            _c_titulo(c,
                f"Evaluamos |f^({orden})(x)| en {puntos_scan} puntos del intervalo [{_fmt_num(a)}, {_fmt_num(b)}]:",
                RED)
            indices_muestra = [0, puntos_scan//4, puntos_scan//2, 3*puntos_scan//4, puntos_scan-1]
            for idx in indices_muestra:
                xi_v = xs_scan[idx]
                vi   = vals_arr[idx]
                marca = "  <-- MAXIMO" if idx == idx_max else ""
                _c_formula(c,
                    f"   x = {xi_v:.6f}   |f^({orden})(x)| = {vi:.8f}{marca}",
                    RED if marca else MUTED)
            _c_resultado_box(c,
                f"x_max = {x_max:.6f}   M_{orden} = max|f^({orden})(x)| = {M:.8f}",
                RED)

            # renderizar la nueva pestana EG Grafico
            self._render_eg_grafico(xs, fexpr, orden, a, b,
                                    xs_scan, vals_arr, idx_max, x_max, M)
        except Exception as e:
            _c_texto(c, f"Error al calcular: {e}", RED)
            return
        _espacio(c)

        # paso 6: cota final
        _seccion(si, "PASO 6 — Cota del error global", ORANGE)
        c = _card(si, ORANGE)
        facto = math.factorial(orden)
        cota  = M / facto * r_omega["max_omega"]
        _c_titulo(c, "Reemplazamos en la formula:", ORANGE)
        _c_formula(c,
            f"|E(x)| <=  M_{orden} / {orden}!  *  max|omega(x)|")
        _c_formula(c,
            f"        <=  {M:.4f} / {facto}  *  {r_omega['max_omega']:.6f}")
        _c_resultado_box(c,
            f"Cota de error global  <=  {cota:.2e}  =  {cota:.8f}",
            ORANGE)
        _espacio(c)

        # paso 7: comparar con error real si aplica
        if fexpr and x_eval is not None and px is not None:
            _seccion(si, "PASO 7 — Verificacion con error local", ACCENT)
            c = _card(si, ACCENT)
            try:
                fx_val   = evaluar_expr(fexpr, x_eval)
                err_real = abs(fx_val - px)
                _c_titulo(c, f"En x* = {_fmt_num(x_eval)}:", ACCENT)
                _c_igual(c, "Error real  |f(x*) - P(x*)|",
                         f"{err_real:.2e}", ACCENT)
                _c_igual(c, "Cota teorica", f"{cota:.2e}", ORANGE)
                ok  = err_real <= cota * (1 + 1e-6)
                msg = "OK — error real <= cota" if ok else "REVISAR (cota numerica puede ser imprecisa)"
                _c_resultado_box(c, msg, GREEN if ok else YELLOW)
            except Exception:
                pass
            _espacio(c)

    # ══════════════════════════════════════
    # RENDER: EG GRAFICO — pestaña dedicada
    # ══════════════════════════════════════
    def _render_eg_grafico(self, xs, fexpr, orden, a, b,
                           xs_scan, vals_arr, idx_max, x_max, M):
        """
        Grafico de |f^(n+1)(x)| evaluado en:
          1) curva continua en [a,b]
          2) puntos destacados en cada x_i
          3) tabla x_i vs |f^(n+1)(x_i)|
        """
        # ── 1. calcular |f^(orden)(x_i)| en cada nodo ────────────
        vals_nodos = []
        for xi in xs:
            try:
                v = abs(derivada_numerica_orden(fexpr, float(xi), orden))
            except Exception:
                v = float("nan")
            vals_nodos.append(v)

        vals_nodos_arr = np.array(vals_nodos)
        idx_nodo_max   = int(np.nanargmax(vals_nodos_arr)) if not np.all(np.isnan(vals_nodos_arr)) else 0

        # ── 2. grafico ────────────────────────────────────────────
        ax = self._ax_eg
        ax.clear()
        self._style_ax(ax)

        # curva continua |f^(n+1)(x)|
        xs_ok  = xs_scan[~np.isnan(vals_arr)]
        ys_ok  = vals_arr[~np.isnan(vals_arr)]
        ax.plot(xs_ok, ys_ok, color=RED, linewidth=1.8,
                label=f"|f^({orden})(x)|  continua", alpha=0.7, zorder=2)
        ax.fill_between(xs_ok, ys_ok, alpha=0.08, color=RED)

        # puntos en cada x_i — normales en ACCENT, maximo en YELLOW
        for k, (xi, vi) in enumerate(zip(xs, vals_nodos_arr)):
            if np.isnan(vi):
                continue
            es_max = (k == idx_nodo_max)
            color_pt = YELLOW if es_max else ACCENT
            size_pt  = 120    if es_max else 60
            ax.scatter([float(xi)], [vi], color=color_pt,
                       s=size_pt, zorder=5,
                       label=(f"MAX: x_{k}={_fmt_num(float(xi))}  val={vi:.4f}"
                              if es_max else f"x_{k}={_fmt_num(float(xi))}"))
            # etiqueta encima de cada punto
            ax.annotate(
                f"x_{k}",
                xy=(float(xi), vi),
                xytext=(float(xi), vi + (max(ys_ok)-min(ys_ok))*0.08 if len(ys_ok) else vi+0.1),
                color=color_pt, fontsize=8, ha="center",
            )

        # linea horizontal en el maximo de los nodos
        ax.axhline(vals_nodos_arr[idx_nodo_max], color=YELLOW,
                   linewidth=1.0, linestyle="--", alpha=0.8)

        ax.set_xlabel("x_i  (nodos)", color=MUTED, fontsize=9)
        ax.set_ylabel(f"|f^({orden})(x)|", color=MUTED, fontsize=9)
        xi_max_nodo = float(xs[idx_nodo_max])
        M_nodo      = vals_nodos_arr[idx_nodo_max]
        ax.set_title(
            f"|f^({orden})(x_i)|  evaluada en cada nodo   "
            f"→  MAXIMO en  x_{idx_nodo_max} = {_fmt_num(xi_max_nodo)}  "
            f"valor = {M_nodo:.6f}",
            color=TEXT, fontsize=9, pad=6)
        # leyenda solo con los primeros 6 items para no saturar
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:7], labels[:7],
                  facecolor=BG3, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=8)
        self._canvas_eg.draw()

        # ── 3. tabla treeview ────────────────────────────────────
        for row in self._tree_eg.get_children():
            self._tree_eg.delete(row)

        for k, (xi, vi) in enumerate(zip(xs, vals_nodos_arr)):
            xi_f = float(xi)
            vi_s = f"{vi:.8f}" if not np.isnan(vi) else "—"
            es_max_s = "*** MAXIMO ***" if k == idx_nodo_max else ""
            self._tree_eg.insert("", "end", values=(
                k, f"{xi_f:.6f}", vi_s, es_max_s
            ))
            # colorear la fila del maximo
            if k == idx_nodo_max:
                self._tree_eg.item(
                    self._tree_eg.get_children()[-1],
                    tags=("maximo",))

        self._tree_eg.tag_configure("maximo", foreground=YELLOW)

    # ══════════════════════════════════════
    # RENDER: ANALISIS
    # ══════════════════════════════════════
    def _render_analisis(self, xs, ys, x_eval, px, Px_sym):
        ta = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        n = len(xs)
        w("ANALISIS COMPLETO — LAGRANGE\n", "title")
        w("Ref: Caceres, Modelado y Simulacion, 2 ed. 2026, pag. 20-23\n\n", "muted")

        w("DATOS\n", "title")
        w(f"  Puntos:  n+1 = {n}  ->  grado = {n-1}\n")
        w(f"  x_i = {[round(float(v),6) for v in xs]}\n", "info")
        w(f"  y_i = {[round(float(v),6) for v in ys]}\n\n", "info")

        w("POLINOMIO\n", "title")
        Px_str = _poly_to_str(Px_sym)
        w(f"  P(x) = {Px_str}\n\n", "ok")

        if x_eval is not None and px is not None:
            w("EVALUACION\n", "title")
            w(f"  x*   = {x_eval:.8f}\n")
            w(f"  P(x*) = "); w(f"{px:.8f}\n", "ok")
            fexpr = self._get_fexpr()
            if fexpr:
                try:
                    fx = evaluar_expr(fexpr, x_eval)
                    err = abs(fx - px)
                    w(f"  f(x*) = "); w(f"{fx:.8f}\n", "info")
                    w(f"  Error local = "); w(f"{err:.2e}\n\n", "warn")
                except Exception:
                    pass

        w("TEOREMA (Caceres pag. 20)\n", "title")
        w("  Existencia: para n+1 puntos distintos siempre existe\n")
        w("              un unico polinomio de grado <= n.\n", "ok")
        w("  Unicidad:   el polinomio es unico.\n", "ok")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = LagrangeApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()