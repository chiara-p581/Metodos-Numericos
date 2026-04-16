"""
Comparador de Métodos Numéricos para EDOs de Primer Orden
==========================================================
Métodos implementados:
  • Euler (RK1)
  • RK2 — Heun (a=1) o Punto Medio (a=1/2)
  • RK4 Clásico

Fórmulas:
──────────────────────────────────────────────
EULER:
  y_{n+1} = y_n + h * f(x_n, y_n)

RK2 - HEUN (a=1):
  k1 = f(x_n, y_n)
  k2 = f(x_n + h, y_n + h*k1)
  y_{n+1} = y_n + (h/2)*(k1 + k2)

RK2 - PUNTO MEDIO (a=1/2):
  k1 = f(x_n, y_n)
  k2 = f(x_n + h/2, y_n + (h/2)*k1)
  y_{n+1} = y_n + h*k2

RK4 CLÁSICO:
  k1 = f(x_n,       y_n)
  k2 = f(x_n + h/2, y_n + (h/2)*k1)
  k3 = f(x_n + h/2, y_n + (h/2)*k2)
  k4 = f(x_n + h,   y_n + h*k3)
  y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
──────────────────────────────────────────────
"""

import tkinter as tk
from tkinter import messagebox, ttk
import math
import re
import numpy as np
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ══════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════
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
PINK   = "#ff7eb3"
CYAN   = "#56d364"

# Colores por método
COL_EULER  = "#f85149"
COL_RK2    = "#f0883e"
COL_RK4    = "#58a6ff"
COL_EXACTA = "#3fb950"


# ══════════════════════════════════════════════════════
# EVALUACIÓN SEGURA
# ══════════════════════════════════════════════════════
def _make_env(x_val=None, y_val=None):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env.update({"np": np, "exp": math.exp, "sin": math.sin, "cos": math.cos,
                "log": math.log, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
                "tan": math.tan, "sinh": math.sinh, "cosh": math.cosh,
                "abs": abs, "ln": math.log})
    if x_val is not None: env["x"] = x_val
    if y_val is not None: env["y"] = y_val
    return env

def _evaluar_f(fexpr: str, x_val: float, y_val: float) -> float:
    try:
        val = eval(fexpr, {"__builtins__": {}}, _make_env(x_val, y_val))
        return float(val)
    except ZeroDivisionError:
        return 0.0
    except Exception as e:
        raise ValueError(f"Error evaluando f(x,y)='{fexpr}' en x={x_val:.4f}, y={y_val:.4f}: {e}")

def _parse_float(s: str, campo: str) -> float:
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env.update({"pi": math.pi, "e": math.e})
    try:
        return float(eval(s.strip(), {"__builtins__": {}}, env))
    except Exception as exc:
        raise ValueError(f"Valor inválido en '{campo}': {s!r}  ({exc})")


# ══════════════════════════════════════════════════════
# LÓGICA NUMÉRICA
# ══════════════════════════════════════════════════════
def euler_paso(fexpr, x, y, h):
    k1 = _evaluar_f(fexpr, x, y)
    return {"k1": k1, "y_nuevo": y + h * k1, "x_n": x, "y_n": y, "x_n1": x + h}

def rk2_heun_paso(fexpr, x, y, h):
    k1 = _evaluar_f(fexpr, x, y)
    k2 = _evaluar_f(fexpr, x + h, y + h * k1)
    return {"k1": k1, "k2": k2, "prom": (k1+k2)/2,
            "y_nuevo": y + (h/2)*(k1+k2), "x_n": x, "y_n": y, "x_n1": x+h}

def rk2_midpoint_paso(fexpr, x, y, h):
    k1 = _evaluar_f(fexpr, x, y)
    k2 = _evaluar_f(fexpr, x + h/2, y + (h/2)*k1)
    return {"k1": k1, "k2": k2, "prom": k2,
            "y_nuevo": y + h*k2, "x_n": x, "y_n": y, "x_n1": x+h}

def rk4_paso(fexpr, x, y, h):
    k1 = _evaluar_f(fexpr, x,       y)
    k2 = _evaluar_f(fexpr, x + h/2, y + (h/2)*k1)
    k3 = _evaluar_f(fexpr, x + h/2, y + (h/2)*k2)
    k4 = _evaluar_f(fexpr, x + h,   y + h*k3)
    prom = (k1 + 2*k2 + 2*k3 + k4)/6
    return {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "prom": prom,
            "y_nuevo": y + h*prom, "x_n": x, "y_n": y, "x_n1": x+h}

def calcular_metodo(metodo_fn, fexpr, x0, y0, h, pasos):
    resultados, x, y = [], x0, y0
    for i in range(pasos):
        p = metodo_fn(fexpr, x, y, h)
        p["i"] = i
        resultados.append(p)
        x, y = p["x_n1"], p["y_nuevo"]
    return resultados

def _solucion_exacta_sympy(fexpr: str, x0: float, y0: float, xs: np.ndarray):
    try:
        x_sym = sp.Symbol("x")
        y_fn  = sp.Function("y")
        fstr  = fexpr.replace("^", "**")
        fstr_sym = re.sub(r'(?<![a-zA-Z_])y(?![a-zA-Z_(])', 'y(x)', fstr)
        f_sym = sp.sympify(fstr_sym, locals={"x": x_sym, "y": y_fn})
        ode   = sp.Eq(y_fn(x_sym).diff(x_sym), f_sym)
        sol   = sp.dsolve(ode, y_fn(x_sym))
        C1    = sp.Symbol("C1")
        eq_ci = sol.rhs.subs(x_sym, x0) - y0
        c_val = sp.solve(eq_ci, C1)
        if not c_val:
            return None, None
        sol_particular = sol.rhs.subs(C1, c_val[0])
        sol_str = str(sol_particular)
        f_exacta = sp.lambdify(x_sym, sol_particular, "numpy")
        vals = np.array([float(f_exacta(xi)) for xi in xs], dtype=float)
        return vals, sol_str
    except Exception:
        return None, None

def _evaluar_exacta_manual(expr_str: str, xs: np.ndarray):
    try:
        resultados = []
        for xi in xs:
            val = eval(expr_str, {"__builtins__": {}}, _make_env(xi, None))
            resultados.append(float(val))
        return np.array(resultados, dtype=float)
    except Exception as e:
        raise ValueError(f"Error en solución exacta manual: {e}")


# ══════════════════════════════════════════════════════
# CLASIFICACIÓN Y RESOLUCIÓN ANALÍTICA DINÁMICA
# ══════════════════════════════════════════════════════

def _sp_str(expr):
    """Convierte expresión sympy a string legible."""
    return str(expr)

def _sp_pretty(expr):
    """Sympy pretty pero compacto."""
    try:
        s = sp.latex(expr)
        # Fallback to str for display
        return str(expr)
    except:
        return str(expr)

def clasificar_edo(fexpr):
    """
    Clasifica la EDO dy/dx = f(x,y).
    Retorna (tipo, P_std, Q_std) donde:
      - tipo: 'solo_x' | 'linear' | 'separable' | 'general'
      - Para 'linear': P_std, Q_std tal que dy/dx + P_std*y = Q_std
      - Para 'separable': g(x), h(y) tal que dy/dx = g(x)*h(y)
      - Para 'solo_x': None, None
    """
    x_sym = sp.Symbol('x')
    y_sym = sp.Symbol('ytmp')

    fstr = fexpr.replace('^', '**')
    fstr_ysym = re.sub(r'(?<![a-zA-Z_])y(?![a-zA-Z_(])', 'ytmp', fstr)

    try:
        f_alg = sp.sympify(fstr_ysym, locals={'x': x_sym, 'ytmp': y_sym})
    except Exception:
        return 'general', None, None

    has_y = y_sym in f_alg.free_symbols

    if not has_y:
        return 'solo_x', None, None

    # Linear check: f is degree-1 polynomial in y
    try:
        poly = sp.Poly(f_alg, y_sym)
        if poly.degree() == 1:
            A = poly.nth(1)   # coeff of y  (dy/dx = A*y + B)
            B = poly.nth(0)   # constant
            if y_sym not in A.free_symbols and y_sym not in B.free_symbols:
                P_std = -A   # standard: dy/dx + P*y = Q  =>  P = -A
                Q_std = B
                return 'linear', P_std, Q_std
    except Exception:
        pass

    # Separable check: f = g(x) * h(y)
    try:
        fy1 = f_alg.subs(y_sym, 1)
        if fy1 != 0:
            ratio = sp.simplify(f_alg / fy1)
            if y_sym in ratio.free_symbols and x_sym not in ratio.free_symbols:
                return 'separable', fy1, ratio   # g(x)=fy1, h(y)=ratio
        fx0 = f_alg.subs(x_sym, 0)
        if fx0 != 0:
            ratio2 = sp.simplify(f_alg / fx0)
            if y_sym not in ratio2.free_symbols:
                return 'separable', ratio2, fx0
    except Exception:
        pass

    return 'general', None, None


def resolver_edo_analitico(fexpr, x0, y0):
    """
    Resuelve la EDO simbólicamente con SymPy y retorna un dict con toda
    la información para mostrar paso a paso.
    """
    x_sym = sp.Symbol('x')
    y_fn  = sp.Function('y')
    C1    = sp.Symbol('C1')

    fstr     = fexpr.replace('^', '**')
    fstr_sym = re.sub(r'(?<![a-zA-Z_])y(?![a-zA-Z_(])', 'y(x)', fstr)

    tipo, extra1, extra2 = clasificar_edo(fexpr)

    result = {
        "tipo": tipo,
        "fexpr": fexpr,
        "x0": x0, "y0": y0,
        "extra1": extra1,
        "extra2": extra2,
        "sol_general": None,
        "C1_val": None,
        "sol_particular": None,
        "sol_str": None,
        "error": None,
    }

    try:
        f_sym = sp.sympify(fstr_sym, locals={"x": x_sym, "y": y_fn})
        ode   = sp.Eq(y_fn(x_sym).diff(x_sym), f_sym)
        sol   = sp.dsolve(ode, y_fn(x_sym))
        result["sol_general"] = sol.rhs

        eq_ci = sol.rhs.subs(x_sym, x0) - y0
        c_val = sp.solve(eq_ci, C1)
        if c_val:
            result["C1_val"] = c_val[0]
            result["sol_particular"] = sp.simplify(sol.rhs.subs(C1, c_val[0]))
            result["sol_str"] = str(result["sol_particular"])
    except Exception as e:
        result["error"] = str(e)

    # For linear: compute mu and integral step by step
    if tipo == 'linear' and extra1 is not None:
        P_std = extra1
        Q_std = extra2
        try:
            mu_exp_expr  = sp.integrate(P_std, x_sym)
            mu_expr      = sp.exp(mu_exp_expr)
            integral_muQ = sp.integrate(mu_expr * Q_std, x_sym)
            result["mu_exp"]      = mu_exp_expr
            result["mu"]          = mu_expr
            result["integral_muQ"] = integral_muQ
        except Exception:
            pass

    # For separable: compute integrals
    if tipo == 'separable' and extra1 is not None:
        g_x = extra1  # g(x)
        h_y = extra2  # h(y)
        y_sym = sp.Symbol('y')
        try:
            integral_gx = sp.integrate(g_x, x_sym)
            # integral of 1/h(y) dy
            integral_1hy = sp.integrate(1/h_y.subs(sp.Symbol('ytmp'), y_sym), y_sym)
            result["integral_gx"]  = integral_gx
            result["integral_1hy"] = integral_1hy
            result["g_x"] = g_x
            result["h_y"] = h_y
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════
# HELPERS DE WIDGETS
# ══════════════════════════════════════════════════════
def _lbl(parent, text, fg=MUTED, font=("Consolas", 11), bg=None):
    return tk.Label(parent, text=text, bg=bg or BG2, fg=fg, font=font)

def _entry(parent, default: str, width=None):
    kw = dict(bg=BG3, fg=TEXT, insertbackground=TEXT,
              font=("Consolas", 12), bd=0, relief="flat",
              highlightthickness=1, highlightbackground=BORDER,
              highlightcolor=ACCENT)
    if width: kw["width"] = width
    e = tk.Entry(parent, **kw)
    e.insert(0, default)
    return e

def _labeled_entry(parent, label: str, default: str):
    _lbl(parent, label).pack(anchor="w")
    e = _entry(parent, default)
    e.pack(fill=tk.X, ipady=6, pady=(2, 8))
    return e

def _btn(parent, text: str, cmd, color=ACCENT, fg="#000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 11, "bold"),
                 padx=12, pady=8, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h: str) -> str:
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0, int(r*.75)), max(0, int(g*.75)), max(0, int(b*.75)))

def _scrollable_frame(parent):
    wrap = tk.Frame(parent, bg=BG)
    wrap.pack(fill=tk.BOTH, expand=True)
    cvs  = tk.Canvas(wrap, bg=BG, highlightthickness=0)
    vsb  = tk.Scrollbar(wrap, orient="vertical", command=cvs.yview)
    cvs.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = tk.Frame(cvs, bg=BG)
    win   = cvs.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>",
               lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
    cvs.bind("<Configure>",
             lambda e: cvs.itemconfig(win, width=e.width))
    cvs.bind_all("<MouseWheel>",
                 lambda e: cvs.yview_scroll(int(-1*(e.delta/120)), "units"))
    return inner

def _style_ax(ax):
    ax.set_facecolor(BG2)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)

def _seccion(parent, titulo: str, color=ACCENT):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill=tk.X, padx=10, pady=(14, 4))
    tk.Frame(f, bg=color, width=4).pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(f, text=f"  {titulo}", bg=BG, fg=color,
             font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=4)

def _card(parent, color=ACCENT):
    outer = tk.Frame(parent, bg=BG)
    outer.pack(fill=tk.X, padx=10, pady=4)
    tk.Frame(outer, bg=color, width=3).pack(side=tk.LEFT, fill=tk.Y)
    inner = tk.Frame(outer, bg=BG2)
    inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return inner

def _ctitulo(parent, texto, color=TEAL):
    tk.Label(parent, text=texto, bg=BG2, fg=color,
             font=("Consolas", 12, "bold", "underline"),
             anchor="w").pack(fill=tk.X, padx=16, pady=(10, 2))

def _cformula(parent, texto, color=MUTED, indent=1):
    tk.Label(parent, text="   "*indent + texto, bg=BG2, fg=color,
             font=("Consolas", 11), justify="left",
             anchor="w").pack(fill=tk.X, padx=18, pady=1)

def _cigual(parent, izq, der, color_der=GREEN):
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=18, pady=2)
    tk.Label(row, text="   "+izq+"  =  ", bg=BG2, fg=MUTED,
             font=("Consolas", 11)).pack(side=tk.LEFT)
    tk.Label(row, text=der, bg=BG2, fg=color_der,
             font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

def _cbox(parent, texto, color=GREEN):
    outer = tk.Frame(parent, bg=color, padx=2, pady=2)
    outer.pack(fill=tk.X, padx=20, pady=6)
    inner = tk.Frame(outer, bg=BG3)
    inner.pack(fill=tk.BOTH)
    tk.Label(inner, text="  "+texto, bg=BG3, fg=color,
             font=("Consolas", 12, "bold"),
             padx=10, pady=8, anchor="w").pack(fill=tk.X)

def _csep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=6)

def _gap(parent, h=6):
    tk.Frame(parent, bg=BG2, height=h).pack()

def _tabla_header(parent, cols, widths, color=ACCENT):
    row = tk.Frame(parent, bg=BG3)
    row.pack(fill=tk.X, padx=10, pady=(4, 0))
    for col, w in zip(cols, widths):
        tk.Label(row, text=col, bg=BG3, fg=color,
                 font=("Consolas", 10, "bold"),
                 width=w, anchor="center").pack(side=tk.LEFT, padx=1, pady=3)

def _tabla_fila(parent, valores, widths, colores=None, bg=BG2):
    row = tk.Frame(parent, bg=bg)
    row.pack(fill=tk.X, padx=10, pady=0)
    for i, (v, w) in enumerate(zip(valores, widths)):
        fg = colores[i] if colores and i < len(colores) else TEXT
        tk.Label(row, text=str(v), bg=bg, fg=fg,
                 font=("Consolas", 10),
                 width=w, anchor="center").pack(side=tk.LEFT, padx=1, pady=2)


# ══════════════════════════════════════════════════════
# EJEMPLOS PREDEFINIDOS
# ══════════════════════════════════════════════════════
EJEMPLOS_REF = [
    {
        "nombre":    "dy/dx = x+y, y(0)=1, h=0.1",
        "fexpr":     "x + y",
        "x0":        "0", "y0": "1", "h": "0.1", "xf": "1",
        "exacta":    "2*exp(x) - x - 1",
        "desc":      "Exacta: 2eˣ - x - 1",
    },
    {
        "nombre":    "dy/dx = x²+y, y(0)=1, h=0.2",
        "fexpr":     "x**2 + y",
        "x0":        "0", "y0": "1", "h": "0.2", "xf": "1",
        "exacta":    "",
        "desc":      "Sin solución analítica cerrada simple",
    },
    {
        "nombre":    "dy/dx = -2y, y(0)=1, h=0.1",
        "fexpr":     "-2*y",
        "x0":        "0", "y0": "1", "h": "0.1", "xf": "1",
        "exacta":    "exp(-2*x)",
        "desc":      "Exacta: e^(-2x)",
    },
    {
        "nombre":    "dy/dx = y·cos(x), y(0)=1, h=0.2",
        "fexpr":     "y*cos(x)",
        "x0":        "0", "y0": "1", "h": "0.2", "xf": "2",
        "exacta":    "exp(sin(x))",
        "desc":      "Exacta: e^sin(x)",
    },
    {
        "nombre":    "dy/dx = y·sin(x), y(0)=1, h=0.2",
        "fexpr":     "y*sin(x)",
        "x0":        "0", "y0": "1", "h": "0.2", "xf": "2",
        "exacta":    "exp(1 - cos(x))",
        "desc":      "Exacta: e^(1-cos(x))  [Separable]",
    },
    {
        "nombre":    "dy/dx = x*y+y, y(0)=1, h=0.1",
        "fexpr":     "x*y + y",
        "x0":        "0", "y0": "1", "h": "0.1", "xf": "0.8",
        "exacta":    "exp(x + x**2/2)",
        "desc":      "Exacta: e^(x + x²/2)",
    },
]


# ══════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════
class ComparadorEDOApp(tk.Frame):

    TABS_MAIN = [
        ("📊 Comparación",   "comparacion"),
        ("📋 Euler",         "euler"),
        ("🔵 RK2",           "rk2"),
        ("🔷 RK4",           "rk4"),
        ("📈 Gráfico",       "grafico"),
        ("📐 Pendientes",    "pendientes"),
        ("🔍 Paso a Paso",   "pasos"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Comparador de Métodos Numéricos — EDO de 1er Orden")
            master.configure(bg=BG)
            master.geometry("1500x880")
            master.minsize(1200, 700)
        self._data = None
        self._build_ui()

    def _build_ui(self):
        self._topbar()
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._sidebar(body)
        self._main_area(body)

    def _topbar(self):
        bar = tk.Frame(self, bg=BG2, height=52)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="  🧮  Comparador de Métodos Numéricos para EDO — Euler · RK2 · RK4",
                 bg=BG2, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar, text="Euler (RK1)  |  RK2 Heun/Midpoint  |  RK4 Clásico  |  Solución Analítica  ",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=16)

    # ── Sidebar ──────────────────────────────────────
    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=360)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        cvs = tk.Canvas(sb, bg=BG2, highlightthickness=0)
        vsb = tk.Scrollbar(sb, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        wrap = tk.Frame(cvs, bg=BG2)
        win  = cvs.create_window((0, 0), window=wrap, anchor="nw")
        wrap.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>", lambda e: cvs.itemconfig(win, width=e.width))
        cvs.bind("<MouseWheel>", lambda e: cvs.yview_scroll(int(-1*(e.delta/120)), "units"))
        p = tk.Frame(wrap, bg=BG2)
        p.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # Ejemplos
        _lbl(p, "EJEMPLOS DE REFERENCIA", fg=YELLOW,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        for ej in EJEMPLOS_REF:
            self._ej_btn(p, ej)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        # EDO
        _lbl(p, "ECUACIÓN DIFERENCIAL  dy/dx = f(x, y)", fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._ef = _labeled_entry(p, "f(x, y) =", "x + y")

        # Condición inicial
        _lbl(p, "CONDICIÓN INICIAL", fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._ex0 = _labeled_entry(p, "x₀", "0")
        self._ey0 = _labeled_entry(p, "y₀  (condición inicial)", "1")

        # Parámetros
        _lbl(p, "PARÁMETROS", fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._eh  = _labeled_entry(p, "h  — tamaño del paso", "0.1")
        self._exf = _labeled_entry(p, "X Final", "1")

        _lbl(p, "n — número de pasos (calculado)").pack(anchor="w")
        self._en = _entry(p, "10")
        self._en.config(state="readonly")
        self._en.pack(fill=tk.X, ipady=6, pady=(2, 8))

        # Variante RK2
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=4)
        _lbl(p, "VARIANTE RK2", fg=PURPLE,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._rk2_var = tk.StringVar(value="heun")
        row = tk.Frame(p, bg=BG2)
        row.pack(fill=tk.X)
        for val, txt in [("heun", "Heun (a=1)"), ("midpoint", "Punto Medio (a=½)")]:
            tk.Radiobutton(row, text=txt, variable=self._rk2_var, value=val,
                           bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, activeforeground=ACCENT,
                           font=("Consolas", 10)).pack(side=tk.LEFT, padx=6)

        # Solución analítica manual
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _lbl(p, "SOLUCIÓN EXACTA (opcional)", fg=GREEN,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        _lbl(p, "y(x) = ... (si la conoces, se usa para error)", fg=MUTED,
             font=("Consolas", 9)).pack(anchor="w")
        self._e_exacta = _labeled_entry(p, "Ej: 2*exp(x) - x - 1", "")

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)
        _btn(p, "  ▶  RESOLVER EDO  ", self._calcular, TEAL, "#000").pack(fill=tk.X, pady=4)
        _btn(p, "  ↺  LIMPIAR  ", self._limpiar, BG3, MUTED).pack(fill=tk.X, pady=2)

    def _ej_btn(self, parent, ej):
        f = tk.Frame(parent, bg=BG3, cursor="hand2")
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=ej["nombre"], bg=BG3, fg=ACCENT,
                 font=("Consolas", 9, "bold"), anchor="w", padx=8, pady=3).pack(fill=tk.X)
        tk.Label(f, text=ej["desc"], bg=BG3, fg=MUTED,
                 font=("Consolas", 8), anchor="w", padx=10, pady=1).pack(fill=tk.X)
        for w in [f] + list(f.winfo_children()):
            w.bind("<Button-1>", lambda e, d=ej: self._cargar_ej(d))
            w.bind("<Enter>",    lambda e, fr=f: fr.config(bg=BG))
            w.bind("<Leave>",    lambda e, fr=f: fr.config(bg=BG3))

    def _cargar_ej(self, ej):
        for widget, key in [(self._ef, "fexpr"), (self._ex0, "x0"),
                            (self._ey0, "y0"), (self._eh, "h"),
                            (self._e_exacta, "exacta")]:
            widget.delete(0, tk.END)
            widget.insert(0, ej[key])
        # xf field
        self._exf.delete(0, tk.END)
        self._exf.insert(0, ej.get("xf", "1"))
        self._calcular()

    def _limpiar(self):
        for w in [self._ef, self._ex0, self._ey0, self._eh, self._exf, self._e_exacta]:
            w.delete(0, tk.END)
        self._ef.insert(0, "x + y"); self._ex0.insert(0, "0")
        self._ey0.insert(0, "1"); self._eh.insert(0, "0.1"); self._exf.insert(0, "1")
        self._data = None

    # ── Área principal con pestañas ───────────────────
    def _main_area(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tbar = tk.Frame(right, bg=BG2, height=48)
        tbar.pack(fill=tk.X)
        tbar.pack_propagate(False)
        self._tbtns   = {}
        self._tframes = {}
        for label, key in self.TABS_MAIN:
            b = tk.Label(tbar, text=label, bg=BG2, fg=MUTED,
                         font=("Segoe UI", 11), padx=14, pady=14, cursor="hand2")
            b.pack(side=tk.LEFT)
            b.bind("<Button-1>", lambda e, k=key: self._show_tab(k))
            self._tbtns[key] = b
        self._panels = tk.Frame(right, bg=BG)
        self._panels.pack(fill=tk.BOTH, expand=True)
        self._build_all_tabs()
        self._show_tab("comparacion")

    def _show_tab(self, name):
        for n, b in self._tbtns.items():
            b.config(fg=TEXT if n == name else MUTED,
                     bg=BG3 if n == name else BG2)
        for n, f in self._tframes.items():
            if n == name:
                f.pack(fill=tk.BOTH, expand=True)
            else:
                f.pack_forget()

    def _new_panel(self, name):
        f = tk.Frame(self._panels, bg=BG)
        self._tframes[name] = f
        return f

    def _build_all_tabs(self):
        self._si_comparacion = _scrollable_frame(self._new_panel("comparacion"))
        self._si_euler       = _scrollable_frame(self._new_panel("euler"))
        self._si_rk2         = _scrollable_frame(self._new_panel("rk2"))
        self._si_rk4         = _scrollable_frame(self._new_panel("rk4"))
        self._build_tab_grafico()
        self._build_tab_pendientes()
        self._build_tab_pasos()

    def _build_tab_grafico(self):
        f = self._new_panel("grafico")
        ctrl = tk.Frame(f, bg=BG2)
        ctrl.pack(fill=tk.X)
        tk.Label(ctrl, text="  Mostrar curvas:", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=8, pady=8)
        self._chk_vars = {}
        checks = [("Euler", COL_EULER), ("RK2", COL_RK2),
                  ("RK4", COL_RK4), ("Exacta", COL_EXACTA)]
        for lbl, col in checks:
            v = tk.BooleanVar(value=True)
            self._chk_vars[lbl] = v
            cb = tk.Checkbutton(ctrl, text=lbl, variable=v, bg=BG2, fg=col,
                                selectcolor=BG3, activebackground=BG2,
                                activeforeground=col, font=("Segoe UI", 10, "bold"),
                                command=self._render_grafico)
            cb.pack(side=tk.LEFT, padx=8)
        self._fig_graf = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_graf  = self._fig_graf.add_subplot(111)
        _style_ax(self._ax_graf)
        self._cvs_graf = FigureCanvasTkAgg(self._fig_graf, master=f)
        self._cvs_graf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_tab_pendientes(self):
        f = self._new_panel("pendientes")
        self._fig_pend = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_pend  = self._fig_pend.add_subplot(111)
        _style_ax(self._ax_pend)
        self._cvs_pend = FigureCanvasTkAgg(self._fig_pend, master=f)
        self._cvs_pend.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_tab_pasos(self):
        f = self._new_panel("pasos")
        ctrl = tk.Frame(f, bg=BG2)
        ctrl.pack(fill=tk.X)
        tk.Label(ctrl, text="  Ver paso a paso de:", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=8, pady=8)
        self._paso_metodo = tk.StringVar(value="euler")
        opciones = [("analitico", "Analítico", GREEN), ("euler", "Euler", COL_EULER),
                    ("rk2", "RK2", COL_RK2), ("rk4", "RK4", COL_RK4)]
        for val, lbl, col in opciones:
            rb = tk.Radiobutton(ctrl, text=lbl, variable=self._paso_metodo,
                                value=val, bg=BG2, fg=col, selectcolor=BG3,
                                activebackground=BG2, activeforeground=col,
                                font=("Segoe UI", 10, "bold"),
                                command=self._render_pasos)
            rb.pack(side=tk.LEFT, padx=10)
        self._si_pasos = _scrollable_frame(f)

    # ══════════════════════════════════════════════════
    # CÁLCULO PRINCIPAL
    # ══════════════════════════════════════════════════
    def _calcular(self):
        try:
            fexpr  = self._ef.get().strip()
            if not fexpr: raise ValueError("Ingresa la EDO f(x, y).")
            x0     = _parse_float(self._ex0.get(), "x0")
            y0     = _parse_float(self._ey0.get(), "y0")
            h      = _parse_float(self._eh.get(),  "h")
            xf     = _parse_float(self._exf.get(), "X Final")
            if xf <= x0: raise ValueError("X Final debe ser mayor a x0")

            pasos = int(round((xf - x0) / h))

            self._en.config(state="normal")
            self._en.delete(0, tk.END)
            self._en.insert(0, str(pasos))
            self._en.config(state="readonly")

            rk2var = self._rk2_var.get()
            if pasos < 1: raise ValueError("El intervalo es demasiado pequeño para este h.")
            if h <= 0:    raise ValueError("h debe ser > 0.")

            xs = np.array([x0 + i*h for i in range(pasos+1)])

            rk2_fn = rk2_heun_paso if rk2var == "heun" else rk2_midpoint_paso
            euler_data = calcular_metodo(euler_paso,  fexpr, x0, y0, h, pasos)
            rk2_data   = calcular_metodo(rk2_fn,      fexpr, x0, y0, h, pasos)
            rk4_data   = calcular_metodo(rk4_paso,    fexpr, x0, y0, h, pasos)

            y_euler = np.array([y0] + [p["y_nuevo"] for p in euler_data])
            y_rk2   = np.array([y0] + [p["y_nuevo"] for p in rk2_data])
            y_rk4   = np.array([y0] + [p["y_nuevo"] for p in rk4_data])

            exacta_str  = self._e_exacta.get().strip()
            exacta_vals = None
            exacta_formula = None

            if exacta_str:
                try:
                    exacta_vals    = _evaluar_exacta_manual(exacta_str, xs)
                    exacta_formula = exacta_str
                except Exception as ex:
                    messagebox.showwarning("Solución exacta",
                        f"No se pudo evaluar la solución exacta ingresada:\n{ex}")

            if exacta_vals is None:
                exacta_vals, exacta_formula = _solucion_exacta_sympy(fexpr, x0, y0, xs)

            # Resolver analíticamente para el paso a paso
            analitico = resolver_edo_analitico(fexpr, x0, y0)

            self._data = {
                "fexpr": fexpr, "x0": x0, "y0": y0, "h": h, "pasos": pasos,
                "rk2var": rk2var, "xs": xs,
                "euler_data": euler_data, "rk2_data": rk2_data, "rk4_data": rk4_data,
                "y_euler": y_euler, "y_rk2": y_rk2, "y_rk4": y_rk4,
                "exacta": exacta_vals, "exacta_formula": exacta_formula,
                "analitico": analitico,
            }

            self._render_comparacion()
            self._render_tabla_metodo("euler")
            self._render_tabla_metodo("rk2")
            self._render_tabla_metodo("rk4")
            self._render_grafico()
            self._render_pendientes()
            self._render_pasos()
            self._show_tab("comparacion")

        except Exception as exc:
            messagebox.showerror("Error de entrada", str(exc))

    # ══════════════════════════════════════════════════
    # RENDER: TABLA COMPARACIÓN
    # ══════════════════════════════════════════════════
    def _render_comparacion(self):
        d  = self._data
        si = self._si_comparacion
        for w in si.winfo_children():
            w.destroy()

        rk2_nom = "Heun" if d["rk2var"] == "heun" else "Midpoint"
        tiene_exacta = d["exacta"] is not None

        _seccion(si, "TABLA COMPARATIVA — Todos los Métodos", TEAL)
        c = _card(si, TEAL)
        _cformula(c, f"EDO:  dy/dx = {d['fexpr']}", ACCENT)
        _cformula(c, f"CI:   y({d['x0']:.4g}) = {d['y0']:.6g}", MUTED)
        _cformula(c, f"h = {d['h']:.4g}  |  {d['pasos']} pasos  |  x final = {d['x0']+d['pasos']*d['h']:.4g}", MUTED)
        if d["exacta_formula"]:
            _cformula(c, f"y exacta: {d['exacta_formula']}", GREEN)
        _gap(c)

        if tiene_exacta:
            cols   = ["n", "Xn", "Euler", "RK2", "RK4", "Y Real", "Err Euler", "Err RK2", "Err RK4"]
            widths = [4, 8, 11, 11, 11, 11, 10, 10, 10]
        else:
            cols   = ["n", "Xn", "Euler", "RK2", "RK4"]
            widths = [4, 12, 14, 14, 14]

        _tabla_header(si, cols, widths, TEAL)

        for i in range(len(d["xs"])):
            xi = d["xs"][i]
            ye = d["y_euler"][i]
            yr2 = d["y_rk2"][i]
            yr4 = d["y_rk4"][i]
            bg_row = BG3 if i % 2 == 0 else BG2

            if tiene_exacta:
                yex = d["exacta"][i]
                e_e, e_r2, e_r4 = abs(ye-yex), abs(yr2-yex), abs(yr4-yex)
                vals = [str(i), f"{xi:.2f}", f"{ye:.6f}", f"{yr2:.6f}", f"{yr4:.6f}",
                        f"{yex:.6f}", f"{e_e:.2e}", f"{e_r2:.2e}", f"{e_r4:.2e}"]
                colores = [MUTED, TEXT, COL_EULER, COL_RK2, COL_RK4, GREEN, YELLOW, YELLOW, YELLOW]
            else:
                vals = [str(i), f"{xi:.2f}", f"{ye:.6f}", f"{yr2:.6f}", f"{yr4:.6f}"]
                colores = [MUTED, TEXT, COL_EULER, COL_RK2, COL_RK4]

            _tabla_fila(si, vals, widths, colores, bg_row)

        if tiene_exacta:
            _gap(si, 8)
            _seccion(si, "RESUMEN DE ERRORES MÁXIMOS", YELLOW)
            c2 = _card(si, YELLOW)
            _ctitulo(c2, "Error máximo  |y_num - y_exacta|:", YELLOW)
            for lbl, yv, col in [("Euler   (RK1)", d["y_euler"], COL_EULER),
                                  (f"RK2 {rk2_nom:8s}", d["y_rk2"],   COL_RK2),
                                  ("RK4 Clásico  ", d["y_rk4"],   COL_RK4)]:
                err = float(np.max(np.abs(yv - d["exacta"])))
                _cigual(c2, lbl, f"{err:.4e}", col)
            _csep(c2)
            _cformula(c2, "Orden de error esperado:", MUTED)
            _cformula(c2, "Euler → O(h¹)   RK2 → O(h²)   RK4 → O(h⁴)", ACCENT)
            _gap(c2)

        _gap(si, 6)
        _seccion(si, "FÓRMULAS DE CADA MÉTODO", ACCENT)
        infos = [
            ("EULER (RK1)", COL_EULER,
             ["y_{n+1} = y_n + h · f(x_n, y_n)",
              "— Una sola evaluación de f por paso",
              "— Error local O(h²), global O(h)"]),
            (f"RK2 — {('HEUN (a=1)' if d['rk2var']=='heun' else 'PUNTO MEDIO (a=½)')}", COL_RK2,
             ["k₁ = f(x_n, y_n)",
              ("k₂ = f(x_n + h,   y_n + h·k₁)    [Heun]"
               if d['rk2var']=='heun' else
               "k₂ = f(x_n + h/2, y_n + h/2·k₁)  [Midpoint]"),
              ("y_{n+1} = y_n + (h/2)·(k₁ + k₂)  [Heun]"
               if d['rk2var']=='heun' else
               "y_{n+1} = y_n + h·k₂              [Midpoint]"),
              "— Error local O(h³), global O(h²)"]),
            ("RK4 CLÁSICO", COL_RK4,
             ["k₁ = f(x_n,       y_n)",
              "k₂ = f(x_n + h/2, y_n + h/2·k₁)",
              "k₃ = f(x_n + h/2, y_n + h/2·k₂)",
              "k₄ = f(x_n + h,   y_n + h·k₃)",
              "y_{n+1} = y_n + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)",
              "— Error local O(h⁵), global O(h⁴)"]),
        ]
        for titulo, col, lineas in infos:
            c3 = _card(si, col)
            _ctitulo(c3, titulo, col)
            for ln in lineas:
                _cformula(c3, ln, MUTED if ln.startswith("—") else TEXT)
            _gap(c3, 4)

    # ══════════════════════════════════════════════════
    # RENDER: TABLA INDIVIDUAL POR MÉTODO
    # ══════════════════════════════════════════════════
    def _render_tabla_metodo(self, metodo: str):
        d   = self._data
        si  = {"euler": self._si_euler, "rk2": self._si_rk2, "rk4": self._si_rk4}[metodo]
        col = {"euler": COL_EULER, "rk2": COL_RK2, "rk4": COL_RK4}[metodo]
        nom = {
            "euler": "Euler (RK1)",
            "rk2":   f"RK2 — {'Heun' if d['rk2var']=='heun' else 'Punto Medio'}",
            "rk4":   "RK4 Clásico",
        }[metodo]
        data_pasos = {"euler": d["euler_data"], "rk2": d["rk2_data"], "rk4": d["rk4_data"]}[metodo]

        for w in si.winfo_children():
            w.destroy()

        tiene_exacta = d["exacta"] is not None

        _seccion(si, f"TABLA — {nom}", col)
        c = _card(si, col)
        _cformula(c, f"EDO:  dy/dx = {d['fexpr']}", ACCENT)
        _cformula(c, f"CI:   y({d['x0']:.4g}) = {d['y0']:.6g}    h = {d['h']:.4g}    n = {d['pasos']}", MUTED)
        if tiene_exacta:
            _cformula(c, f"y exacta: {d['exacta_formula']}", GREEN)
        _gap(c)

        if metodo == "euler":
            if tiene_exacta:
                cols   = ["n", "xn", "yn", "yn+1 (Euler)", "yreal"]
                widths = [4,   12,   14,   16,              14]
            else:
                cols   = ["n", "xn", "yn", "yn+1 (Euler)"]
                widths = [4,   12,   14,   16]
        elif metodo == "rk2":
            cols   = ["n", "xn", "yn", "k1", "k2", "yn+1 (RK2)"]
            widths = [4,   10,   13,   13,   13,   14]
        else:
            cols   = ["n", "xn", "yn", "k1", "k2", "k3", "k4", "yn+1 (RK4)"]
            widths = [3,   9,    11,   11,   11,   11,   11,   12]

        _tabla_header(si, cols, widths, col)

        for paso in data_pasos:
            i  = paso["i"]
            xn = paso["x_n"]
            yn = paso["y_n"]
            yn1 = paso["y_nuevo"]
            bg_row = BG3 if i % 2 == 0 else BG2

            if metodo == "euler":
                if tiene_exacta:
                    yreal = d["exacta"][i]
                    vals = [str(i), f"{xn:.4f}", f"{yn:.7f}", f"{yn1:.7f}", f"{yreal:.7f}"]
                    cols_color = [MUTED, TEXT, col, COL_EULER, COL_EXACTA]
                else:
                    vals = [str(i), f"{xn:.4f}", f"{yn:.7f}", f"{yn1:.7f}"]
                    cols_color = [MUTED, TEXT, col, COL_EULER]
            elif metodo == "rk2":
                vals = [str(i), f"{xn:.4f}", f"{yn:.7f}",
                        f"{paso['k1']:.6f}", f"{paso['k2']:.6f}", f"{yn1:.7f}"]
                cols_color = [MUTED, TEXT, col, GREEN, TEAL, COL_RK2]
            else:
                vals = [str(i), f"{xn:.4f}", f"{yn:.7f}",
                        f"{paso['k1']:.5f}", f"{paso['k2']:.5f}",
                        f"{paso['k3']:.5f}", f"{paso['k4']:.5f}", f"{yn1:.7f}"]
                cols_color = [MUTED, TEXT, col, GREEN, TEAL, PURPLE, ORANGE, COL_RK4]

            _tabla_fila(si, vals, widths, cols_color, bg_row)

        _gap(si, 10)
        _seccion(si, f"DETALLE DE PENDIENTES — {nom}", col)

        for paso in data_pasos:
            i  = paso["i"]
            xn = paso["x_n"]
            yn = paso["y_n"]
            cv = _card(si, col)
            _ctitulo(cv, f"Paso {i+1}:  x_{i} = {xn:.6f}   y_{i} = {yn:.8f}", col)

            if metodo == "euler":
                _cigual(cv, f"k1 = f({xn:.4f}, {yn:.6f})", f"{paso['k1']:.8f}", GREEN)
                _cigual(cv, f"y_{i+1} = {yn:.6f} + {d['h']:.4f} × {paso['k1']:.6f}",
                         f"{paso['y_nuevo']:.8f}", col)
                if tiene_exacta:
                    yreal = d["exacta"][i]
                    _csep(cv)
                    _cigual(cv, f"yreal(x={xn:.4f})", f"{yreal:.8f}", COL_EXACTA)
                    ea = abs(paso["y_nuevo"] - d["exacta"][i+1])
                    _cigual(cv, "error abs. (yn+1 vs exacta)", f"{ea:.3e}",
                             GREEN if ea<1e-4 else (YELLOW if ea<1e-2 else RED))
            elif metodo == "rk2":
                _cigual(cv, f"k1 = f({xn:.4f}, {yn:.6f})", f"{paso['k1']:.8f}", GREEN)
                x2 = xn+d["h"] if d["rk2var"]=="heun" else xn+d["h"]/2
                _cigual(cv, f"k2 = f({x2:.4f}, ...)", f"{paso['k2']:.8f}", TEAL)
                _cigual(cv, "promedio k", f"{paso['prom']:.8f}", YELLOW)
                _cigual(cv, f"y_{i+1} (corrector)", f"{paso['y_nuevo']:.8f}", col)
                if tiene_exacta:
                    ea = abs(paso["y_nuevo"] - d["exacta"][i+1])
                    _csep(cv)
                    _cigual(cv, f"y exacta(x={paso['x_n1']:.4f})", f"{d['exacta'][i+1]:.8f}", COL_EXACTA)
                    _cigual(cv, "error abs.", f"{ea:.3e}",
                             GREEN if ea<1e-4 else (YELLOW if ea<1e-2 else RED))
            else:
                _cigual(cv, "k1", f"{paso['k1']:.8f}", GREEN)
                _cigual(cv, "k2", f"{paso['k2']:.8f}", TEAL)
                _cigual(cv, "k3", f"{paso['k3']:.8f}", PURPLE)
                _cigual(cv, "k4", f"{paso['k4']:.8f}", ORANGE)
                _cigual(cv, "prom = (k1+2k2+2k3+k4)/6", f"{paso['prom']:.8f}", YELLOW)
                _cigual(cv, f"y_{i+1}", f"{paso['y_nuevo']:.8f}", col)
                if tiene_exacta:
                    ea = abs(paso["y_nuevo"] - d["exacta"][i+1])
                    _csep(cv)
                    _cigual(cv, f"y exacta(x={paso['x_n1']:.4f})", f"{d['exacta'][i+1]:.8f}", COL_EXACTA)
                    _cigual(cv, "error abs.", f"{ea:.3e}",
                             GREEN if ea<1e-4 else (YELLOW if ea<1e-2 else RED))
            _gap(cv, 4)

    # ══════════════════════════════════════════════════
    # RENDER: GRÁFICO COMPARATIVO
    # ══════════════════════════════════════════════════
    def _render_grafico(self):
        if not self._data:
            return
        d  = self._data
        ax = self._ax_graf
        ax.clear()
        _style_ax(ax)
        xs = d["xs"]

        if self._chk_vars["Euler"].get():
            ax.plot(xs, d["y_euler"], color=COL_EULER, linewidth=2,
                    marker="o", markersize=4, label="Euler (RK1)", zorder=3)
        if self._chk_vars["RK2"].get():
            rk2_nom = "Heun" if d["rk2var"]=="heun" else "Midpoint"
            ax.plot(xs, d["y_rk2"], color=COL_RK2, linewidth=2,
                    marker="s", markersize=4, label=f"RK2 {rk2_nom}", zorder=3)
        if self._chk_vars["RK4"].get():
            ax.plot(xs, d["y_rk4"], color=COL_RK4, linewidth=2,
                    marker="^", markersize=4, label="RK4 Clásico", zorder=4)
        if self._chk_vars["Exacta"].get() and d["exacta"] is not None:
            xs_dense = np.linspace(xs[0], xs[-1], 400)
            exacta_dense, _ = _solucion_exacta_sympy(d["fexpr"], d["x0"], d["y0"], xs_dense)
            if exacta_dense is not None:
                ax.plot(xs_dense, exacta_dense, color=COL_EXACTA, linewidth=2,
                        linestyle="--", label="Exacta (analítica)", zorder=5, alpha=0.9)
            ax.plot(xs, d["exacta"], color=COL_EXACTA,
                    marker="D", markersize=3, linewidth=0,
                    alpha=0.6, label="Exacta (puntos)", zorder=5)

        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y(x)", fontsize=9)
        ax.set_title(
            f"Comparación — dy/dx = {d['fexpr']}  |  y({d['x0']}) = {d['y0']}  |  h = {d['h']}",
            color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="best")
        self._fig_graf.tight_layout()
        self._cvs_graf.draw()

    # ══════════════════════════════════════════════════
    # RENDER: CAMPO DE PENDIENTES
    # ══════════════════════════════════════════════════
    def _render_pendientes(self):
        if not self._data:
            return
        d  = self._data
        ax = self._ax_pend
        ax.clear()
        _style_ax(ax)
        xs = d["xs"]
        h  = d["h"]

        y_all = np.concatenate([d["y_euler"], d["y_rk2"], d["y_rk4"]])
        x_min = xs[0] - h*0.5
        x_max = xs[-1] + h*0.5
        y_min_r = float(np.min(y_all))
        y_max_r = float(np.max(y_all))
        dy_r = max(abs(y_max_r - y_min_r)*0.3, 0.5)

        xg = np.linspace(x_min, x_max, 20)
        yg = np.linspace(y_min_r - dy_r, y_max_r + dy_r, 15)
        Xg, Yg = np.meshgrid(xg, yg)
        try:
            DY = np.array([[_evaluar_f(d["fexpr"], xi, yi) for xi in xg] for yi in yg])
            DX = np.ones_like(DY)
            nm = np.sqrt(DX**2 + DY**2); nm[nm==0] = 1
            ax.quiver(Xg, Yg, DX/nm, DY/nm, color=BORDER, alpha=0.45,
                      scale=30, width=0.003, headwidth=2)
        except Exception:
            pass

        ax.plot(xs, d["y_euler"], color=COL_EULER, lw=2, marker="o", ms=4, label="Euler", zorder=3)
        ax.plot(xs, d["y_rk2"],   color=COL_RK2,   lw=2, marker="s", ms=4,
                label=f"RK2 {'Heun' if d['rk2var']=='heun' else 'Midpoint'}", zorder=3)
        ax.plot(xs, d["y_rk4"],   color=COL_RK4,   lw=2, marker="^", ms=4, label="RK4", zorder=4)
        if d["exacta"] is not None:
            xs_d = np.linspace(xs[0], xs[-1], 300)
            ed, _ = _solucion_exacta_sympy(d["fexpr"], d["x0"], d["y0"], xs_d)
            if ed is not None:
                ax.plot(xs_d, ed, color=COL_EXACTA, lw=1.5, ls="--", alpha=0.8, label="Exacta", zorder=5)

        p0  = d["rk4_data"][0]
        xn  = p0["x_n"]; yn = p0["y_n"]
        esc = h * 0.4
        def _flecha(ax, x, y, dy_f, color, lbl):
            ax.annotate("", xy=(x+esc, y+dy_f*esc), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
            ax.plot([], [], color=color, label=lbl, lw=2)

        _flecha(ax, xn,         yn,               p0["k1"], GREEN,  "k₁ inicio")
        _flecha(ax, xn+h/2, yn+(h/2)*p0["k1"],   p0["k2"], TEAL,   "k₂ mitad")
        _flecha(ax, xn+h/2, yn+(h/2)*p0["k2"],   p0["k3"], PURPLE, "k₃ mitad")
        _flecha(ax, xn+h,   yn+h*p0["k3"],       p0["k4"], ORANGE, "k₄ final")

        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Campo de direcciones + trayectorias  |  paso 1: x={xn:.3f}",
                     color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, loc="best")
        self._fig_pend.tight_layout()
        self._cvs_pend.draw()

    # ══════════════════════════════════════════════════
    # RENDER: PASO A PASO
    # ══════════════════════════════════════════════════
    def _render_pasos(self):
        if not self._data:
            return
        si = self._si_pasos
        for w in si.winfo_children():
            w.destroy()

        metodo = self._paso_metodo.get()
        d = self._data

        if metodo == "analitico":
            self._render_desarrollo_analitico()
            return

        col = {"euler": COL_EULER, "rk2": COL_RK2, "rk4": COL_RK4}[metodo]
        nom = {
            "euler": "Euler (RK1)",
            "rk2":   f"RK2 — {'Heun (a=1)' if d['rk2var']=='heun' else 'Punto Medio (a=½)'}",
            "rk4":   "RK4 Clásico",
        }[metodo]
        pasos_d = {"euler": d["euler_data"], "rk2": d["rk2_data"], "rk4": d["rk4_data"]}[metodo]

        _seccion(si, f"FÓRMULA GENERAL — {nom}", col)
        cf = _card(si, col)
        if metodo == "euler":
            _cformula(cf, "y_{n+1} = y_n + h · f(x_n, y_n)", GREEN)
        elif metodo == "rk2":
            _cformula(cf, "k₁ = f(x_n, y_n)", GREEN)
            if d["rk2var"] == "heun":
                _cformula(cf, "k₂ = f(x_n + h, y_n + h·k₁)", TEAL)
                _cformula(cf, "y_{n+1} = y_n + (h/2)·(k₁ + k₂)", ACCENT)
            else:
                _cformula(cf, "k₂ = f(x_n + h/2, y_n + (h/2)·k₁)", TEAL)
                _cformula(cf, "y_{n+1} = y_n + h·k₂", ACCENT)
        else:
            _cformula(cf, "k₁ = f(x_n, y_n)", GREEN)
            _cformula(cf, "k₂ = f(x_n + h/2, y_n + h/2·k₁)", TEAL)
            _cformula(cf, "k₃ = f(x_n + h/2, y_n + h/2·k₂)", PURPLE)
            _cformula(cf, "k₄ = f(x_n + h, y_n + h*k₃)", ORANGE)
            _cformula(cf, "y_{n+1} = y_n + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)", ACCENT)

        _seccion(si, f"ITERACIONES DETALLADAS — {nom}", col)
        for paso in pasos_d:
            i = paso["i"]
            xn, yn = paso["x_n"], paso["y_n"]
            cv = _card(si, TEAL if i%2==0 else col)
            _ctitulo(cv, f"Paso {i+1}: x_{i} = {xn:.4f} | y_{i} = {yn:.6f}", TEAL if i%2==0 else col)

            if metodo == "euler":
                _cigual(cv, "k₁", f"{paso['k1']:.6f}", GREEN)
            elif metodo == "rk2":
                _cigual(cv, "k₁", f"{paso['k1']:.6f}", GREEN)
                _cigual(cv, "k₂", f"{paso['k2']:.6f}", TEAL)
            else:
                _cigual(cv, "k₁", f"{paso['k1']:.6f}", GREEN)
                _cigual(cv, "k₂", f"{paso['k2']:.6f}", TEAL)
                _cigual(cv, "k₃", f"{paso['k3']:.6f}", PURPLE)
                _cigual(cv, "k₄", f"{paso['k4']:.6f}", ORANGE)

            _csep(cv)
            _cigual(cv, f"y_{i+1} calculado", f"{paso['y_nuevo']:.9f}", col)

            if d["exacta"] is not None:
                yex = d["exacta"][i+1]
                err = abs(paso["y_nuevo"] - yex)
                _cigual(cv, "y Real", f"{yex:.9f}", COL_EXACTA)
                _cigual(cv, "Error Abs.", f"{err:.3e}", YELLOW)

    # ══════════════════════════════════════════════════
    # RENDER: DESARROLLO ANALÍTICO DINÁMICO
    # ══════════════════════════════════════════════════
    def _render_desarrollo_analitico(self):
        d   = self._data
        si  = self._si_pasos
        for w in si.winfo_children():
            w.destroy()

        ana  = d["analitico"]
        tipo = ana["tipo"]
        fexpr = d["fexpr"]
        x0, y0 = d["x0"], d["y0"]
        x_sym = sp.Symbol("x")

        _seccion(si, f"RESOLUCIÓN ANALÍTICA — EDO: dy/dx = {fexpr}", GREEN)

        # ── Error de resolución ───────────────────────
        if ana["error"] and ana["sol_particular"] is None:
            c_err = _card(si, RED)
            _ctitulo(c_err, "SymPy no pudo resolver esta EDO analíticamente", RED)
            _cformula(c_err, f"Error: {ana['error']}", MUTED)
            _cformula(c_err, "Puedes intentar resolverla manualmente o usar métodos numéricos.", YELLOW)
            _gap(c_err, 4)
            return

        # ── TIPO: EDO con solo x  (dy/dx = g(x)) ─────
        if tipo == "solo_x":
            self._pasos_solo_x(si, ana, fexpr, x0, y0, x_sym)

        # ── TIPO: Lineal ──────────────────────────────
        elif tipo == "linear":
            self._pasos_lineal(si, ana, fexpr, x0, y0, x_sym)

        # ── TIPO: Separable ───────────────────────────
        elif tipo == "separable":
            self._pasos_separable(si, ana, fexpr, x0, y0, x_sym)

        # ── TIPO: General (SymPy resolvió pero no clasif.)
        else:
            self._pasos_general(si, ana, fexpr, x0, y0, x_sym)

        # ── Verificación numérica ─────────────────────
        if d["exacta"] is not None and ana["sol_particular"] is not None:
            _gap(si, 8)
            _seccion(si, "VERIFICACIÓN NUMÉRICA — y(x) vs métodos", COL_EXACTA)
            cols   = ["x", "y exacta", "Euler", "RK2", "RK4"]
            widths = [10, 14, 14, 14, 14]
            _tabla_header(si, cols, widths, COL_EXACTA)
            for i, xi in enumerate(d["xs"]):
                yex  = d["exacta"][i]
                ye   = d["y_euler"][i]
                yr2  = d["y_rk2"][i]
                yr4  = d["y_rk4"][i]
                bg_r = BG3 if i%2==0 else BG2
                vals = [f"{xi:.4f}", f"{yex:.8f}", f"{ye:.8f}", f"{yr2:.8f}", f"{yr4:.8f}"]
                cc   = [TEXT, COL_EXACTA, COL_EULER, COL_RK2, COL_RK4]
                _tabla_fila(si, vals, widths, cc, bg_r)

    # ── Pasos: dy/dx = g(x) ───────────────────────────
    def _pasos_solo_x(self, si, ana, fexpr, x0, y0, x_sym):
        c1 = _card(si, ACCENT)
        _ctitulo(c1, "Tipo detectado: EDO de variable separable — solo f(x)", ACCENT)
        _cformula(c1, "dy/dx = g(x)  =>  dy = g(x) dx", TEXT)
        _cformula(c1, "Integrar ambos lados:", MUTED)
        _cformula(c1, "y = ∫ g(x) dx + C", GREEN)
        _gap(c1, 4)

        c2 = _card(si, GREEN)
        _ctitulo(c2, "Paso 1 — Integración directa", GREEN)
        _cigual(c2, "g(x)", str(fexpr), TEXT)
        integral = sp.integrate(sp.sympify(fexpr.replace("^","**"), locals={"x": x_sym}), x_sym)
        _cigual(c2, "∫ g(x) dx", str(integral), TEAL)
        _cigual(c2, "y(x)", f"{integral} + C", ACCENT)
        _gap(c2, 4)

        c3 = _card(si, YELLOW)
        _ctitulo(c3, f"Paso 2 — Condición inicial: y({x0}) = {y0}", YELLOW)
        C1 = sp.Symbol("C1")
        eq_val = integral.subs(x_sym, x0)
        C_val  = y0 - float(eq_val)
        _cformula(c3, f"{y0} = {eq_val} + C", TEXT)
        _cigual(c3, "C", f"{C_val:.6g}", YELLOW)
        _gap(c3, 4)

        c4 = _card(si, COL_EXACTA)
        _ctitulo(c4, "Solución Particular", COL_EXACTA)
        sol = ana["sol_particular"]
        _cformula(c4, f"y(x) = {sol}", COL_EXACTA)
        _gap(c4, 4)

    # ── Pasos: Lineal dy/dx + P(x)y = Q(x) ───────────
    def _pasos_lineal(self, si, ana, fexpr, x0, y0, x_sym):
        P_std = ana["extra1"]   # P(x) in dy/dx + P*y = Q
        Q_std = ana["extra2"]
        mu    = ana.get("mu")
        mu_exp= ana.get("mu_exp")
        int_mq= ana.get("integral_muQ")

        c1 = _card(si, ACCENT)
        _ctitulo(c1, "Tipo detectado: EDO Lineal de Primer Orden", ACCENT)
        _cformula(c1, "Forma estándar:   dy/dx + P(x)·y = Q(x)", TEXT)
        _csep(c1)
        _cformula(c1, f"EDO ingresada:    dy/dx = {fexpr}", TEXT)
        _cformula(c1, f"Reescrita:        dy/dx + ({P_std})·y = {Q_std}", TEXT)
        _csep(c1)
        _cigual(c1, "P(x)", str(P_std), COL_RK2)
        _cigual(c1, "Q(x)", str(Q_std), COL_RK2)
        _gap(c1, 4)

        c2 = _card(si, TEAL)
        _ctitulo(c2, "Paso 1 — Factor Integrante  μ(x) = e^{ ∫P(x)dx }", TEAL)
        _cformula(c2, f"μ(x) = e^{{ ∫ ({P_std}) dx }}", TEXT)
        if mu_exp is not None:
            _cformula(c2, f"     = e^{{ {mu_exp} }}", TEXT)
            _cigual(c2, "μ(x)", str(mu), GREEN)
        _csep(c2)
        _cformula(c2, "Al multiplicar ambos lados por μ(x):", MUTED)
        _cformula(c2, "d/dx [ μ(x)·y ]  =  μ(x)·Q(x)", GREEN)
        _gap(c2, 4)

        c3 = _card(si, PURPLE)
        _ctitulo(c3, "Paso 2 — Integración", PURPLE)
        if mu is not None and int_mq is not None:
            _cformula(c3, f"μ(x)·Q(x) = {mu}·({Q_std})", TEXT)
            _cformula(c3, f"          = {sp.expand(mu * Q_std)}", TEXT)
            _csep(c3)
            _cformula(c3, f"∫ μ(x)·Q(x) dx = {int_mq}", GREEN)
        _cformula(c3, "μ(x)·y = ∫ μ(x)·Q(x) dx + C", ACCENT)
        _gap(c3, 4)

        c4 = _card(si, ORANGE)
        _ctitulo(c4, "Paso 3 — Solución general  y = (1/μ)·[∫μQ dx + C]", ORANGE)
        sol_gen = ana["sol_general"]
        _cformula(c4, f"y(x) = {sol_gen}", GREEN)
        _gap(c4, 4)

        c5 = _card(si, YELLOW)
        _ctitulo(c5, f"Paso 4 — Condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        C1 = sp.Symbol("C1")
        eq_val = sol_gen.subs(x_sym, x0)
        _cformula(c5, f"y({x0:.4g}) = {eq_val} = {y0:.6g}", TEXT)
        C_val  = ana["C1_val"]
        _cigual(c5, "C", str(C_val), YELLOW)
        _gap(c5, 4)

        c6 = _card(si, COL_EXACTA)
        _ctitulo(c6, "Solución Particular", COL_EXACTA)
        _cformula(c6, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c6, 4)

    # ── Pasos: Separable ──────────────────────────────
    def _pasos_separable(self, si, ana, fexpr, x0, y0, x_sym):
        g_x = ana.get("g_x")
        h_y = ana.get("h_y")
        int_gx  = ana.get("integral_gx")
        int_1hy = ana.get("integral_1hy")
        y_sym_s = sp.Symbol("y")

        c1 = _card(si, ACCENT)
        _ctitulo(c1, "Tipo detectado: EDO Separable", ACCENT)
        _cformula(c1, "Forma:  dy/dx = g(x)·h(y)", TEXT)
        _cformula(c1, "Separando variables:   dy/h(y) = g(x) dx", MUTED)
        _cformula(c1, "Integrando:  ∫ dy/h(y) = ∫ g(x) dx + C", GREEN)
        _gap(c1, 4)

        c2 = _card(si, TEAL)
        _ctitulo(c2, "Paso 1 — Identificación de g(x) y h(y)", TEAL)
        _cformula(c2, f"dy/dx = {fexpr}", TEXT)
        if g_x is not None and h_y is not None:
            g_display = str(g_x)
            h_display = str(h_y).replace("ytmp", "y")
            _cigual(c2, "g(x)", g_display, COL_RK2)
            _cigual(c2, "h(y)", h_display, COL_RK4)
        _gap(c2, 4)

        c3 = _card(si, PURPLE)
        _ctitulo(c3, "Paso 2 — Integración de ambos lados", PURPLE)
        if int_gx is not None:
            _cigual(c3, "∫ g(x) dx", str(int_gx), GREEN)
        if int_1hy is not None:
            int_1hy_str = str(int_1hy).replace("y", "y")
            _cigual(c3, "∫ dy/h(y)", int_1hy_str, TEAL)
        _cformula(c3, "Igualando:  ∫dy/h(y) = ∫g(x)dx + C", ACCENT)
        _gap(c3, 4)

        c4 = _card(si, ORANGE)
        _ctitulo(c4, "Paso 3 — Solución general (implícita o explícita)", ORANGE)
        sol_gen = ana["sol_general"]
        _cformula(c4, f"y(x) = {sol_gen}", GREEN)
        _gap(c4, 4)

        c5 = _card(si, YELLOW)
        _ctitulo(c5, f"Paso 4 — Condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        C1  = sp.Symbol("C1")
        eq_val = sol_gen.subs(x_sym, x0)
        _cformula(c5, f"y({x0:.4g}) = {eq_val} = {y0:.6g}", TEXT)
        C_val = ana["C1_val"]
        _cigual(c5, "C1", str(C_val), YELLOW)
        _gap(c5, 4)

        c6 = _card(si, COL_EXACTA)
        _ctitulo(c6, "Solución Particular", COL_EXACTA)
        _cformula(c6, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c6, 4)

    # ── Pasos: General (SymPy lo resolvió pero tipo desconocido)
    def _pasos_general(self, si, ana, fexpr, x0, y0, x_sym):
        c1 = _card(si, YELLOW)
        _ctitulo(c1, "Resolución por SymPy (tipo no clasificado manualmente)", YELLOW)
        _cformula(c1, f"EDO:  dy/dx = {fexpr}", TEXT)
        _cformula(c1, f"CI:   y({x0:.4g}) = {y0:.6g}", MUTED)
        _csep(c1)
        _cformula(c1, "SymPy resolvió la EDO. Solución general:", MUTED)
        _cformula(c1, f"y(x) = {ana['sol_general']}", GREEN)
        _gap(c1, 4)

        c2 = _card(si, YELLOW)
        _ctitulo(c2, f"Condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        C1  = sp.Symbol("C1")
        _cigual(c2, "C1", str(ana["C1_val"]), YELLOW)
        _gap(c2, 4)

        c3 = _card(si, COL_EXACTA)
        _ctitulo(c3, "Solución Particular", COL_EXACTA)
        _cformula(c3, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c3, 4)

        cn = _card(si, MUTED)
        _cformula(cn, "NOTA: Este tipo de EDO requiere un método de resolución específico", YELLOW)
        _cformula(cn, "(separable, exacta, Bernoulli, factor integrante especial, etc.)", MUTED)
        _cformula(cn, "El resultado numérico final es correcto; el desarrollo detallado", MUTED)
        _cformula(cn, "paso a paso está disponible para EDOs lineales y separables.", MUTED)
        _gap(cn, 4)


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = ComparadorEDOApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()