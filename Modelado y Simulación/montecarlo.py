"""
Metodo de Montecarlo — Integracion Numerica
============================================
Formulas implementadas (segun profesor):

  1D (metodo promedio):
    x_i ~ U(a, b),  f vectorizada con lambdify
    I_hat   = (b-a) * mean(f(x_i))
    sigma_f = sqrt( 1/(n-1) * sum((f(x_i) - f_bar)^2) )   [ddof=1]
    EE      = sigma_f / sqrt(n)
    sigma_I = (b-a) * sigma_f / sqrt(n)
    IC      = I_hat +- z * sigma_I

  1D (hit-or-miss):
    Se genera rectangulo contenedor [a,b] x [y_min, y_max]
    rect_area = (b-a) * (y_max - y_min)
    I_hat = (exitos / n) * rect_area

  2D (limites fijos):
    x_i ~ U(a,b),  y_i ~ U(c,d)
    area    = (b-a)(d-c)
    I_hat   = area * mean(f(x_i, y_i))
    sigma_f = std(f(x_i,y_i), ddof=1)
    sigma_I = area * sigma_f / sqrt(n)
    IC      = I_hat +- z * sigma_I

  2D (limites variables en y: c(x), d(x)):
    valores_i = (d(x_i) - c(x_i)) * f(x_i, y_i)
    I_hat = (b-a) * mean(valores_i)

  IC usa distribucion t de Student (como en ventana estadistica del profesor)

  Nivel de confianza: usuario elige o ingresa manualmente -> z calculado automaticamente
"""

import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
import sympy as sp
from scipy import stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy.polynomial.legendre import leggauss


# ══════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════
BG      = "#0d1117"
BG2     = "#161b22"
BG3     = "#1c2128"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
PURPLE  = "#bc8cff"
ORANGE  = "#f0883e"
TEAL    = "#39d0d8"

# Niveles predefinidos: nivel% -> z
_NIVELES_Z = {
    "90%":   1.645,
    "95%":   1.960,
    "99%":   2.576,
    "99.7%": 3.000,
}

_x_sym = sp.Symbol("x")
_y_sym = sp.Symbol("y")


# ══════════════════════════════════════════════════════
# EJERCICIOS DE REFERENCIA (Ej 5 y Ej 6 del TP)
# ══════════════════════════════════════════════════════
EJERCICIOS_REF = [
    {
        "nombre": "Ej 5 — integral de sin(x) de 0 a pi  |  n=10000  IC 95%",
        "fexpr":  "sin(x)",
        "a": "0", "b": "pi",
        "c": "0", "d": "1",
        "n": "10000",
        "nivel": "95%",
        "dim": "1D",
        "descripcion": (
            "I = integral de 0 a pi de sin(x) dx\n"
            "n = 10 000   |   IC al 95%   |   Exacto = 2.0"
        ),
    },
    {
        "nombre": "Ej 6 — integral doble e^(x+y)  x:[0,2] y:[1,3]  n=50000  IC 90%",
        "fexpr":  "exp(x+y)",
        "a": "0", "b": "2",
        "c": "1", "d": "3",
        "n": "50000",
        "nivel": "90%",
        "dim": "2D",
        "descripcion": (
            "I = integral de 0 a 2 de integral de 1 a 3 de e^(x+y) dy dx\n"
            "n = 50 000   |   IC al 90%"
        ),
    },
]


# ══════════════════════════════════════════════════════
# FUNCIONES AUXILIARES DE EVALUACION
# ══════════════════════════════════════════════════════

def _make_lambdify_1d(fexpr: str):
    """
    Crea una funcion vectorizada usando sp.lambdify (metodo del profesor).
    Retorna f tal que f(xs_array) -> array de valores.
    """
    x = sp.Symbol('x')
    f_sym = sp.sympify(fexpr)
    return sp.lambdify(x, f_sym, "numpy")


def _make_lambdify_2d(fexpr: str):
    """
    Crea una funcion vectorizada 2D usando sp.lambdify.
    Retorna f tal que f(xs, ys) -> array de valores.
    """
    x, y = sp.symbols('x y')
    f_sym = sp.sympify(fexpr)
    return sp.lambdify((x, y), f_sym, "numpy")


def _evaluar_expr_x(expr: str, x_val: float) -> float:
    """Evalua una expresion de x en un valor escalar (para limites variables)."""
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["x"] = x_val
    try:
        return float(eval(expr, {"__builtins__": {}}, env))
    except Exception as e:
        raise ValueError(f"Error evaluando limite '{expr}' con x={x_val}: {e}")


# ══════════════════════════════════════════════════════
# LOGICA MONTECARLO — METODO PROMEDIO 1D
# ══════════════════════════════════════════════════════
def montecarlo_1d(fexpr: str, a: float, b: float, n: int,
                  nivel: str, z: float, semilla=None) -> dict:
    """
    Montecarlo 1D — Metodo promedio (como en el script del profesor).
    
    Usa sp.lambdify para evaluacion vectorizada con numpy.
    
    Formulas:
      I_hat   = (b-a) * mean(f(x_i))
      sigma_f = std(f(x_i), ddof=1)
      EE      = sigma_f / sqrt(n)
      sigma_I = (b-a) * EE
      IC      = I_hat +/- z * sigma_I
    """
    rng = np.random.default_rng(semilla)
    xs  = rng.uniform(a, b, n)

    # Evaluacion vectorizada con lambdify (metodo del profesor)
    f = _make_lambdify_1d(fexpr)
    fvals = np.nan_to_num(f(xs)).astype(float)

    f_bar   = float(np.mean(fvals))
    vol     = b - a
    I_hat   = vol * f_bar
    sigma_f = float(np.std(fvals, ddof=1))
    EE      = sigma_f / math.sqrt(n)
    sigma_I = vol * EE
    margen  = z * sigma_I

    # Hit-or-miss (adicional, como en el profesor)
    xs_dense = np.linspace(a, b, 1000)
    ys_dense = np.nan_to_num(f(xs_dense))
    y_min = min(0.0, float(np.min(ys_dense)))
    y_max = max(0.0, float(np.max(ys_dense)))
    ys_hom = rng.uniform(y_min, y_max, n)
    success_mask = ((ys_hom >= 0) & (ys_hom <= fvals)) | ((ys_hom <= 0) & (ys_hom >= fvals))
    rect_area = (b - a) * (y_max - y_min)
    I_hitormiss = float(np.sum(success_mask) / n * rect_area)

    # Gauss-Legendre como referencia (metodo del profesor)
    try:
        n_gauss = 5
        nodes, weights = leggauss(n_gauss)
        trans_nodes = 0.5 * (nodes + 1) * (b - a) + a
        gauss_val = float(0.5 * (b - a) * np.sum(weights * f(trans_nodes)))
    except Exception:
        gauss_val = None

    return {
        "dim": "1D",
        "fexpr": fexpr,
        "a": a, "b": b,
        "n": n, "vol": vol,
        "nivel": nivel, "z": z,
        "xs": xs, "fvals": fvals,
        "ys_hom": ys_hom, "success_mask": success_mask,
        "y_min": y_min, "y_max": y_max,
        "f_bar": f_bar,
        "I_hat": I_hat,
        "I_hitormiss": I_hitormiss,
        "gauss_val": gauss_val,
        "sigma_f": sigma_f,
        "EE": EE,
        "sigma_I": sigma_I,
        "margen": margen,
        "ic_lo": I_hat - margen,
        "ic_hi": I_hat + margen,
    }


# ══════════════════════════════════════════════════════
# LOGICA MONTECARLO 2D
# ══════════════════════════════════════════════════════
def montecarlo_2d(fexpr: str, a: float, b: float, c_expr: str, d_expr: str,
                  n: int, nivel: str, z: float, semilla=None) -> dict:
    """
    Montecarlo 2D.
    
    - Si c y d son constantes numericas: metodo directo del profesor
        area = (b-a)*(d-c),  I_hat = area * mean(f(xi, yi))
    - Si c y d son expresiones en x: limites variables
        valores_i = (d(xi)-c(xi)) * f(xi, yi),  I_hat = (b-a) * mean(valores_i)
    
    Usa sp.lambdify para evaluacion vectorizada.
    IC con z fijo (igual que 1D).
    """
    rng = np.random.default_rng(semilla)
    f   = _make_lambdify_2d(fexpr)

    # Determinar si los limites son constantes o funciones de x
    def _es_constante(expr_str: str) -> tuple:
        """Retorna (True, valor_float) si es constante, (False, None) si no."""
        env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        env["pi"] = math.pi; env["e"] = math.e
        try:
            val = float(eval(expr_str.strip(), {"__builtins__": {}}, env))
            return True, val
        except Exception:
            return False, None

    c_cte, c_val = _es_constante(c_expr)
    d_cte, d_val = _es_constante(d_expr)

    if c_cte and d_cte:
        # ── Limites fijos: metodo directo del profesor ──
        c_num = c_val
        d_num = d_val
        xs = rng.uniform(a, b, n)
        ys = rng.uniform(c_num, d_num, n)
        fvals = np.nan_to_num(f(xs, ys)).astype(float)

        area    = (b - a) * (d_num - c_num)
        f_bar   = float(np.mean(fvals))
        I_hat   = area * f_bar
        sigma_f = float(np.std(fvals, ddof=1))
        EE      = sigma_f / math.sqrt(n)
        sigma_I = area * EE
        margen  = z * sigma_I
        vol     = area

        return {
            "dim": "2D",
            "fexpr": fexpr,
            "a": a, "b": b,
            "c": c_expr, "d": d_expr,
            "n": n, "vol": vol,
            "nivel": nivel, "z": z,
            "xs": xs, "ys": ys, "fvals": fvals,
            "f_bar": f_bar,
            "I_hat": I_hat,
            "sigma_f": sigma_f,
            "EE": EE,
            "sigma_I": sigma_I,
            "margen": margen,
            "ic_lo": I_hat - margen,
            "ic_hi": I_hat + margen,
            "limites_variables": False,
        }
    else:
        # ── Limites variables en y: formula con pesos ──
        xs = rng.uniform(a, b, n)
        ys     = np.empty(n)
        pesos  = np.empty(n)
        for i, xi in enumerate(xs):
            c_i = _evaluar_expr_x(c_expr, xi)
            d_i = _evaluar_expr_x(d_expr, xi)
            ys[i]    = rng.uniform(c_i, d_i)
            pesos[i] = d_i - c_i

        fvals   = np.nan_to_num(f(xs, ys)).astype(float)
        valores = pesos * fvals

        f_bar   = float(np.mean(valores))
        vol     = b - a
        I_hat   = vol * f_bar
        sigma_f = float(np.std(valores, ddof=1))
        EE      = sigma_f / math.sqrt(n)
        sigma_I = vol * EE
        margen  = z * sigma_I

        return {
            "dim": "2D",
            "fexpr": fexpr,
            "a": a, "b": b,
            "c": c_expr, "d": d_expr,
            "n": n, "vol": vol,
            "nivel": nivel, "z": z,
            "xs": xs, "ys": ys, "fvals": fvals,
            "f_bar": f_bar,
            "I_hat": I_hat,
            "sigma_f": sigma_f,
            "EE": EE,
            "sigma_I": sigma_I,
            "margen": margen,
            "ic_lo": I_hat - margen,
            "ic_hi": I_hat + margen,
            "limites_variables": True,
        }


def _integral_analitica_1d(fexpr: str, a: float, b: float):
    """Calcula el valor exacto con sympy. Retorna float o None."""
    try:
        fstr  = fexpr.replace("log(", "ln(")
        f_sym = sp.sympify(fstr)
        F     = sp.integrate(f_sym, _x_sym)
        return float((F.subs(_x_sym, b) - F.subs(_x_sym, a)).evalf())
    except Exception:
        return None


def _convergencia(fexpr: str, a: float, b: float, n_max: int,
                  semilla=None, pasos: int = 55):
    """
    Convergencia acumulada — usa lambdify como el profesor.
    Retorna (ns, I_vals, std_vals) para poder graficar banda +-1 std.
    """
    rng   = np.random.default_rng(semilla)
    f     = _make_lambdify_1d(fexpr)
    ns    = np.unique(np.geomspace(1, n_max, pasos).astype(int))
    I_vals   = []
    std_vals = []
    for n in ns:
        xs    = rng.uniform(a, b, n)
        fvals = np.nan_to_num(f(xs)).astype(float)
        mean_ = float(np.mean(fvals)) * (b - a)
        std_  = (float(np.std(fvals, ddof=1)) if n > 1 else 0.0) * (b - a)
        I_vals.append(mean_)
        std_vals.append(std_)
    return ns, np.array(I_vals), np.array(std_vals)


# ══════════════════════════════════════════════════════
# HELPERS DE WIDGETS  (sin cambios de diseno)
# ══════════════════════════════════════════════════════
def _lbl(parent, text, fg=MUTED, font=("Consolas", 11), bg=None):
    return tk.Label(parent, text=text, bg=bg or BG2, fg=fg, font=font)

def _entry(parent, default: str):
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 12), bd=0, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT)
    e.insert(0, default)
    return e

def _labeled_entry(parent, label: str, default: str):
    _lbl(parent, label).pack(anchor="w")
    e = _entry(parent, default)
    e.pack(fill=tk.X, ipady=6, pady=(2, 8))
    return e

def _btn(parent, text: str, cmd, color=ACCENT, fg="#000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 12, "bold"),
                 padx=14, pady=9, cursor="hand2")
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


# ── Bloques visuales ──────────────────────────────────
def _seccion(parent, titulo: str, color=ACCENT):
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

def _ctitulo(parent, texto: str, color=TEAL):
    tk.Label(parent, text=texto, bg=BG2, fg=color,
             font=("Consolas", 12, "bold", "underline"),
             anchor="w").pack(fill=tk.X, padx=16, pady=(10, 2))

def _cformula(parent, texto: str, color=MUTED, indent=1):
    tk.Label(parent, text="   "*indent + texto, bg=BG2, fg=color,
             font=("Consolas", 12), justify="left",
             anchor="w").pack(fill=tk.X, padx=18, pady=1)

def _cigual(parent, izq: str, der: str, color_der=GREEN):
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=18, pady=2)
    tk.Label(row, text="   "+izq+"  =  ", bg=BG2, fg=MUTED,
             font=("Consolas", 12)).pack(side=tk.LEFT)
    tk.Label(row, text=der, bg=BG2, fg=color_der,
             font=("Consolas", 12, "bold")).pack(side=tk.LEFT)

def _cbox(parent, texto: str, color=GREEN):
    outer = tk.Frame(parent, bg=color, padx=2, pady=2)
    outer.pack(fill=tk.X, padx=20, pady=6)
    inner = tk.Frame(outer, bg=BG3)
    inner.pack(fill=tk.BOTH)
    tk.Label(inner, text="  "+texto, bg=BG3, fg=color,
             font=("Consolas", 13, "bold"),
             padx=10, pady=8, anchor="w").pack(fill=tk.X)

def _csep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=6)

def _gap(parent, h=6):
    tk.Frame(parent, bg=BG2, height=h).pack()


# ══════════════════════════════════════════════════════
# APLICACION PRINCIPAL
# ══════════════════════════════════════════════════════
class MontecarloApp(tk.Frame):

    TABS = [
        ("Resultado",   "📋"),
        ("Paso a paso", "🔍"),
        ("Convergencia","📉"),
        ("Distribucion","📊"),
        ("Dispersion",  "🔵"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Montecarlo — Integracion Numerica")
            master.configure(bg=BG)
            master.geometry("1400x820")
            master.minsize(1100, 640)
        self._r = None
        self._build_ui()

    # ─────────────────────────────────────────────────
    def _build_ui(self):
        self._topbar()
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._sidebar(body)
        self._main_area(body)

    def _topbar(self):
        bar = tk.Frame(self, bg=BG2, height=48)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="  Montecarlo — Integracion Numerica",
                 bg=BG2, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="I_hat = vol * f_bar   |   "
                      "sigma_I = vol * sigma_f / sqrt(n)   |   "
                      "IC = I_hat +/- z * sigma_I",
                 bg=BG2, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    # ── Sidebar ───────────────────────────────────────
    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=340)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        cvs = tk.Canvas(sb, bg=BG2, highlightthickness=0)
        vsb = tk.Scrollbar(sb, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        wrap = tk.Frame(cvs, bg=BG2)
        win  = cvs.create_window((0, 0), window=wrap, anchor="nw")
        wrap.bind("<Configure>",
                  lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>",
                 lambda e: cvs.itemconfig(win, width=e.width))
        cvs.bind("<MouseWheel>",
                 lambda e: cvs.yview_scroll(int(-1*(e.delta/120)), "units"))

        p = tk.Frame(wrap, bg=BG2)
        p.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # ── Ejercicios de referencia
        _lbl(p, "EJERCICIOS DE REFERENCIA (TP)", fg=YELLOW,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        for ej in EJERCICIOS_REF:
            self._ej_btn(p, ej)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=10)

        # ── Dimension
        _lbl(p, "DIMENSION", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._dim_var = tk.StringVar(value="1D")
        for v, t in [("1D","1D  —  una variable"),
                     ("2D","2D  —  integral doble")]:
            tk.Radiobutton(p, text=t, variable=self._dim_var, value=v,
                           bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, font=("Segoe UI", 11),
                           command=self._toggle_2d).pack(anchor="w", pady=1)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        # ── Funcion y limites
        _lbl(p, "FUNCION Y LIMITES", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._ef = _labeled_entry(p, "f(x)  — funcion a integrar", "sin(x)")
        self._ea = _labeled_entry(p, "a  — limite inferior  x", "0")
        self._eb = _labeled_entry(p, "b  — limite superior  x", "pi")

        self._frm2d = tk.Frame(p, bg=BG2)
        _lbl(self._frm2d, "c  — limite inferior  y  (numero o expr. en x)").pack(anchor="w")
        self._ec = _entry(self._frm2d, "0")
        self._ec.pack(fill=tk.X, ipady=6, pady=(2, 8))
        _lbl(self._frm2d, "d  — limite superior  y  (numero o expr. en x)").pack(anchor="w")
        self._ed = _entry(self._frm2d, "1")
        self._ed.pack(fill=tk.X, ipady=6, pady=(2, 8))
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        # ── Muestras
        _lbl(p, "MUESTRAS", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._en = _labeled_entry(p, "n  — cantidad de muestras", "10000")
        _lbl(p, "Semilla aleatoria  (dejar vacio = aleatorio)").pack(anchor="w")
        self._esemilla = _entry(p, "42")
        self._esemilla.pack(fill=tk.X, ipady=6, pady=(2, 8))
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        # ── Nivel de confianza
        _lbl(p, "NIVEL DE CONFIANZA", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self._nivel_var = tk.StringVar(value="95%")
        for niv, z in _NIVELES_Z.items():
            tk.Radiobutton(p, text=f"{niv}   (z = {z})",
                           variable=self._nivel_var, value=niv,
                           bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, font=("Segoe UI", 11),
                           command=self._on_radio_nivel).pack(anchor="w", pady=1)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=(8, 4))

        # entrada manual
        _lbl(p, "O ingresa nivel personalizado (%):").pack(anchor="w")
        manual_row = tk.Frame(p, bg=BG2)
        manual_row.pack(fill=tk.X, pady=(2, 6))
        self._e_nivel_manual = _entry(manual_row, "")
        self._e_nivel_manual.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        tk.Label(manual_row, text=" %", bg=BG2, fg=MUTED,
                 font=("Consolas", 12)).pack(side=tk.LEFT)

        # z resultante (solo lectura)
        zrow = tk.Frame(p, bg=BG2)
        zrow.pack(fill=tk.X, pady=(0, 8))
        tk.Label(zrow, text="z calculado = ", bg=BG2, fg=MUTED,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self._lbl_z = tk.Label(zrow, text="1.9600", bg=BG2, fg=GREEN,
                               font=("Consolas", 13, "bold"))
        self._lbl_z.pack(side=tk.LEFT)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(p, "  CALCULAR  ", self._calcular, ACCENT).pack(fill=tk.X, pady=4)

    def _ej_btn(self, parent, ej: dict):
        f = tk.Frame(parent, bg=BG3, cursor="hand2")
        f.pack(fill=tk.X, pady=3)
        tk.Label(f, text=ej["nombre"], bg=BG3, fg=ACCENT,
                 font=("Consolas", 10, "bold"),
                 anchor="w", padx=8, pady=4).pack(fill=tk.X)
        tk.Label(f, text=ej["descripcion"], bg=BG3, fg=MUTED,
                 font=("Consolas", 9), anchor="w", justify="left",
                 padx=10, pady=2).pack(fill=tk.X)
        for w in [f] + list(f.winfo_children()):
            w.bind("<Button-1>", lambda e, d=ej: self._cargar_ej(d))
            w.bind("<Enter>", lambda e, fr=f: fr.config(bg=BG))
            w.bind("<Leave>", lambda e, fr=f: fr.config(bg=BG3))

    def _cargar_ej(self, ej: dict):
        for widget, val in [(self._ef,  ej["fexpr"]),
                            (self._ea,  ej["a"]),
                            (self._eb,  ej["b"]),
                            (self._ec,  ej["c"]),
                            (self._ed,  ej["d"]),
                            (self._en,  ej["n"]),
                            (self._e_nivel_manual, "")]:
            widget.delete(0, tk.END)
            widget.insert(0, val)
        self._nivel_var.set(ej["nivel"])
        self._lbl_z.config(text=f"{_NIVELES_Z[ej['nivel']]:.4f}")
        self._dim_var.set(ej["dim"])
        self._toggle_2d()
        self._calcular()

    def _toggle_2d(self):
        if self._dim_var.get() == "2D":
            self._frm2d.pack(fill=tk.X)
        else:
            self._frm2d.pack_forget()

    def _on_radio_nivel(self):
        niv = self._nivel_var.get()
        self._lbl_z.config(text=f"{_NIVELES_Z[niv]:.4f}")
        self._e_nivel_manual.delete(0, tk.END)

    # ── Main area con tabs ────────────────────────────
    def _main_area(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tbar = tk.Frame(right, bg=BG2, height=44)
        tbar.pack(fill=tk.X)
        tbar.pack_propagate(False)
        self._tbtns   = {}
        self._tframes = {}
        for name, icon in self.TABS:
            b = tk.Label(tbar, text=f"{icon} {name}",
                         bg=BG2, fg=MUTED,
                         font=("Segoe UI", 12), padx=16, pady=12,
                         cursor="hand2")
            b.pack(side=tk.LEFT)
            b.bind("<Button-1>", lambda e, n=name: self._show_tab(n))
            self._tbtns[name] = b

        self._panels = tk.Frame(right, bg=BG)
        self._panels.pack(fill=tk.BOTH, expand=True)

        self._build_tab_resultado()
        self._build_tab_pasos()
        self._build_tab_convergencia()
        self._build_tab_distribucion()
        self._build_tab_dispersion()
        self._show_tab("Resultado")

    def _show_tab(self, name: str):
        for n, b in self._tbtns.items():
            b.config(fg=TEXT if n == name else MUTED)
        for n, f in self._tframes.items():
            if n == name:
                f.pack(fill=tk.BOTH, expand=True)
            else:
                f.pack_forget()

    def _new_panel(self, name: str) -> tk.Frame:
        f = tk.Frame(self._panels, bg=BG)
        self._tframes[name] = f
        return f

    def _build_tab_resultado(self):
        self._si_res = _scrollable_frame(self._new_panel("Resultado"))

    def _build_tab_pasos(self):
        self._si_paso = _scrollable_frame(self._new_panel("Paso a paso"))

    def _build_tab_convergencia(self):
        f = self._new_panel("Convergencia")
        self._fig_conv = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_conv  = self._fig_conv.add_subplot(111)
        _style_ax(self._ax_conv)
        self._cvs_conv = FigureCanvasTkAgg(self._fig_conv, master=f)
        self._cvs_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_tab_distribucion(self):
        f = self._new_panel("Distribucion")
        self._fig_dist = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_dist  = self._fig_dist.add_subplot(111)
        _style_ax(self._ax_dist)
        self._cvs_dist = FigureCanvasTkAgg(self._fig_dist, master=f)
        self._cvs_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_tab_dispersion(self):
        f = self._new_panel("Dispersion")
        self._fig_disp = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_disp  = self._fig_disp.add_subplot(111)
        _style_ax(self._ax_disp)
        self._cvs_disp = FigureCanvasTkAgg(self._fig_disp, master=f)
        self._cvs_disp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────────────────────────────
    # PARSE y CALCULO
    # ─────────────────────────────────────────────────
    def _parse_float(self, s: str, campo: str) -> float:
        env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        env["pi"] = math.pi
        env["e"]  = math.e
        try:
            return float(eval(s.strip(), {"__builtins__": {}}, env))
        except Exception as exc:
            raise ValueError(f"Valor invalido en '{campo}': {s!r}  ({exc})")

    def _resolve_z(self) -> tuple:
        """
        Prioridad: campo manual > radio button.
        Si el usuario ingreso un % manualmente, calcula z con la
        aproximacion de la distribucion normal estandar.
        """
        manual = self._e_nivel_manual.get().strip()
        if manual:
            try:
                nivel_num = float(manual.replace("%", ""))
            except ValueError:
                raise ValueError(
                    f"Nivel manual invalido: {manual!r}\n"
                    "Ingresa un numero entre 50 y 100 (ej: 97.5)")
            if not (50 < nivel_num < 100):
                raise ValueError("El nivel de confianza debe estar entre 50% y 100%.")
            nivel_str = f"{nivel_num:.2f}%"
            alpha = 1.0 - nivel_num / 100.0
            p     = 1.0 - alpha / 2.0
            # Aproximacion racional de Abramowitz & Stegun 26.2.17
            t  = math.sqrt(-2.0 * math.log(1.0 - p))
            c  = (2.515517, 0.802853, 0.010328)
            d  = (1.432788, 0.189269, 0.001308)
            z  = t - (c[0] + c[1]*t + c[2]*t**2) / (
                     1.0 + d[0]*t + d[1]*t**2 + d[2]*t**3)
        else:
            nivel_str = self._nivel_var.get()
            z         = _NIVELES_Z[nivel_str]

        self._lbl_z.config(text=f"{z:.4f}")
        return nivel_str, z

    def _calcular(self):
        try:
            fexpr = self._ef.get().strip()
            if not fexpr:
                raise ValueError("Ingresa la funcion f(x).")
            a   = self._parse_float(self._ea.get(), "a")
            b   = self._parse_float(self._eb.get(), "b")
            n   = int(self._parse_float(self._en.get(), "n"))
            if n < 10:
                raise ValueError("n debe ser al menos 10.")
            s_str   = self._esemilla.get().strip()
            semilla = int(s_str) if s_str else None
            nivel, z = self._resolve_z()
            dim = self._dim_var.get()

            if dim == "1D":
                r = montecarlo_1d(fexpr, a, b, n, nivel, z, semilla)
                r["I_analitica"] = _integral_analitica_1d(fexpr, a, b)
            else:
                c_expr = self._ec.get().strip()
                d_expr = self._ed.get().strip()
                r = montecarlo_2d(fexpr, a, b, c_expr, d_expr, n, nivel, z, semilla)
                r["I_analitica"] = None

            self._r = r
            self._render_resultado(r)
            self._render_pasos(r)
            self._render_convergencia(r)
            self._render_distribucion(r)
            self._render_dispersion(r)
            self._show_tab("Resultado")

        except Exception as exc:
            messagebox.showerror("Error de entrada", str(exc))

    # ─────────────────────────────────────────────────
    # RENDER: RESULTADO
    # ─────────────────────────────────────────────────
    def _render_resultado(self, r: dict):
        si = self._si_res
        for w in si.winfo_children():
            w.destroy()

        # Encabezado integral
        _seccion(si, "RESULTADO  —  Estimacion de la integral", GREEN)
        c = _card(si, GREEN)
        if r["dim"] == "1D":
            _cformula(c,
                f"I  =  integral de {r['a']:.4g} a {r['b']:.4g}  "
                f"de  {r['fexpr']}  dx", ACCENT)
            _cformula(c, f"vol = b - a = {r['vol']:.6g}", MUTED)
            # Mostrar hit-or-miss y Gauss como referencia
            if r.get("I_hitormiss") is not None:
                _cigual(c, "I_hat  (metodo promedio)", f"{r['I_hat']:.8f}", GREEN)
                _cigual(c, "I_hat  (hit-or-miss)",     f"{r['I_hitormiss']:.8f}", TEAL)
                if r.get("gauss_val") is not None:
                    _cigual(c, "Gauss-Legendre (ref.)", f"{r['gauss_val']:.8f}", PURPLE)
                _gap(c, 4)
        else:
            _cformula(c,
                f"I  =  integral doble de  {r['fexpr']}", ACCENT)
            _cformula(c,
                f"x en [{r['a']:.4g}, {r['b']:.4g}]   "
                f"y en [{r['c']}, {r['d']}]", ACCENT)
            if r.get("limites_variables"):
                _cformula(c, "Limites de y variables — formula con pesos", YELLOW)
            _cformula(c, f"vol = {r['vol']:.6g}", MUTED)
        _cbox(c, f"I_hat  =  {r['I_hat']:.8f}", GREEN)
        _gap(c)

        if r.get("I_analitica") is not None:
            ia  = r["I_analitica"]
            err = abs(r["I_hat"] - ia)
            pct = abs(err / ia * 100) if ia != 0 else 0.0
            _cigual(c, "Valor exacto (sympy)", f"{ia:.8f}", TEAL)
            _cigual(c, "Error absoluto",       f"{err:.3e}", YELLOW)
            _cigual(c, "Error relativo",       f"{pct:.4f} %", YELLOW)
            _gap(c)

        # Estadisticas
        _seccion(si, "Estadisticas", PURPLE)
        c = _card(si, PURPLE)
        _cigual(c, "f_bar  (media de f(x_i))",       f"{r['f_bar']:.8f}", ACCENT)
        _csep(c)
        _cigual(c, "sigma_f  (desv. std de f(x_i))", f"{r['sigma_f']:.8f}", PURPLE)
        _cformula(c, "sigma_f = sqrt( sum(f(xi)-f_bar)^2 / (n-1) )  [ddof=1]", MUTED)
        _csep(c)
        _cigual(c, "EE  = sigma_f / sqrt(n)",          f"{r['EE']:.8f}", ORANGE)
        _csep(c)
        _cformula(c, ">>> Desv. std de la INTEGRAL:", GREEN)
        _cigual(c, "sigma_I  = vol * sigma_f / sqrt(n)", f"{r['sigma_I']:.8f}", GREEN)
        _cformula(c,
            f"        = {r['vol']:.6g} x {r['sigma_f']:.6f} / sqrt({r['n']:,})",
            MUTED)
        _gap(c)

        # IC
        _seccion(si,
                 f"Intervalo de confianza al {r['nivel']}  "
                 f"(z = {r['z']:.4f})",
                 TEAL)
        c = _card(si, TEAL)
        _ctitulo(c, "IC  =  I_hat  +/-  z * sigma_I", TEAL)
        _cformula(c,
            f"   =  {r['I_hat']:.6f}  +/-  "
            f"{r['z']:.4f} x {r['sigma_I']:.6f}")
        _cformula(c,
            f"   =  {r['I_hat']:.6f}  +/-  {r['margen']:.6f}")
        _cbox(c,
              f"IC {r['nivel']}:   "
              f"[ {r['ic_lo']:.6f}  ,  {r['ic_hi']:.6f} ]   "
              f"margen = {r['margen']:.6f}",
              TEAL)
        if r.get("I_analitica") is not None:
            dentro = r["ic_lo"] <= r["I_analitica"] <= r["ic_hi"]
            _cbox(c,
                  ("Valor exacto DENTRO del IC  v" if dentro
                   else "Valor exacto FUERA del IC  — aumentar n"),
                  GREEN if dentro else YELLOW)
        _gap(c)

        # IC con t de Student (como en ventana estadistica del profesor)
        if r["n"] > 1:
            _seccion(si,
                     f"IC con t de Student al {r['nivel']}  (n={r['n']:,})",
                     ORANGE)
            c = _card(si, ORANGE)
            conf_num = float(r["nivel"].replace("%","")) / 100.0
            t_val    = float(stats.t.ppf(0.5 + conf_num / 2.0, r["n"] - 1))
            # El IC con t usa el error estandar de la media de f(xi)*vol
            stderr_I = r["sigma_I"]   # = vol * sigma_f / sqrt(n), igual que sigma_I
            ic_t_lo  = r["I_hat"] - t_val * stderr_I
            ic_t_hi  = r["I_hat"] + t_val * stderr_I
            _cformula(c, f"t({r['nivel']}, gl={r['n']-1}) = {t_val:.4f}  (vs z = {r['z']:.4f})", MUTED)
            _cformula(c, "IC_t = I_hat +/- t * sigma_I")
            _cbox(c,
                  f"IC_t {r['nivel']}:   "
                  f"[ {ic_t_lo:.6f}  ,  {ic_t_hi:.6f} ]",
                  ORANGE)
            _cformula(c,
                "Nota: con n grande t -> z y ambos IC coinciden.",
                MUTED)
            _gap(c)

    # ─────────────────────────────────────────────────
    # RENDER: PASO A PASO
    # ─────────────────────────────────────────────────
    def _render_pasos(self, r: dict):
        si = self._si_paso
        for w in si.winfo_children():
            w.destroy()

        dim = r["dim"]

        # PASO 1
        _seccion(si, "PASO 1 — Plantear la integral", ACCENT)
        c = _card(si, ACCENT)
        if dim == "1D":
            _cformula(c,
                f"I = integral de {r['a']:.4g} a {r['b']:.4g}  f(x) dx",
                ACCENT)
            _cformula(c, f"f(x) = {r['fexpr']}", ACCENT)
            _cformula(c,
                f"vol = b - a = {r['b']:.4g} - {r['a']:.4g} = {r['vol']:.6g}",
                GREEN)
        else:
            _cformula(c,
                f"I = integral de {r['a']:.4g} a {r['b']:.4g}  "
                f"integral de {r['c']} a {r['d']}  "
                f"f(x,y) dy dx", ACCENT)
            _cformula(c, f"f(x,y) = {r['fexpr']}", ACCENT)
            if r.get("limites_variables"):
                _cformula(c, "Limites variables: se usa formula con pesos (d(x)-c(x))*f(x,y)", YELLOW)
            else:
                _cformula(c,
                    f"area = (b-a)(d-c) = {r['vol']:.6g}", GREEN)

        # PASO 2
        _seccion(si, "PASO 2 — Generar muestras uniformes con lambdify", PURPLE)
        c = _card(si, PURPLE)
        _cformula(c, "Evaluacion vectorizada: sp.lambdify(x, f_sym, 'numpy')", TEAL)
        if dim == "1D":
            _cformula(c,
                f"x_i  ~  U({r['a']:.4g}, {r['b']:.4g})"
                f"   i = 1, ..., {r['n']:,}", PURPLE)
        else:
            _cformula(c,
                f"x_i ~ U({r['a']:.4g}, {r['b']:.4g})   "
                f"y_i ~ U({r['c']}, {r['d']})",
                PURPLE)
        _cformula(c, f"n = {r['n']:,}  muestras generadas")
        _ctitulo(c, "Primeras 5 muestras:", MUTED)
        for k in range(min(5, r["n"])):
            xi, fi = r["xs"][k], r["fvals"][k]
            if dim == "2D":
                yi = r["ys"][k]
                _cformula(c,
                    f"x_{k+1}={xi:.5f}  y_{k+1}={yi:.5f}"
                    f"  ->  f = {fi:.8f}", MUTED)
            else:
                _cformula(c,
                    f"x_{k+1} = {xi:.8f}   ->   "
                    f"f(x_{k+1}) = {fi:.8f}", MUTED)
        if r["n"] > 5:
            _cformula(c, f"... ({r['n']-5:,} muestras mas)", MUTED)

        # PASO 2b — Hit-or-miss (solo 1D)
        if dim == "1D" and r.get("I_hitormiss") is not None:
            _seccion(si, "PASO 2b — Hit-or-miss (metodo de puntos de exito)", TEAL)
            c = _card(si, TEAL)
            _cformula(c, "Rectangulo contenedor: [a,b] x [y_min, y_max]", TEAL)
            _cformula(c, f"y_min = {r['y_min']:.6f}   y_max = {r['y_max']:.6f}", MUTED)
            rect = (r["b"] - r["a"]) * (r["y_max"] - r["y_min"])
            exitos = int(np.sum(r["success_mask"]))
            _cformula(c, f"rect_area = (b-a) * (y_max - y_min) = {rect:.6f}", MUTED)
            _cformula(c, f"Exitos = {exitos:,}   /   n = {r['n']:,}", MUTED)
            _cformula(c, "I_hm = (exitos / n) * rect_area")
            _cbox(c, f"I_hm = {r['I_hitormiss']:.8f}", TEAL)
            if r.get("gauss_val") is not None:
                _cformula(c, f"Gauss-Legendre (5 pts, referencia) = {r['gauss_val']:.8f}", PURPLE)

        # PASO 3
        _seccion(si, "PASO 3 — Media de f(x_i)", GREEN)
        c = _card(si, GREEN)
        _cformula(c, "f_bar  =  (1/n) * sum( f(x_i) )")
        _cformula(c,
            f"      =  {np.sum(r['fvals']):.6f}  /  {r['n']:,}")
        _cbox(c, f"f_bar  =  {r['f_bar']:.8f}", GREEN)

        # PASO 4
        _seccion(si, "PASO 4 — Integral estimada (metodo promedio)", ACCENT)
        c = _card(si, ACCENT)
        if dim == "2D" and not r.get("limites_variables"):
            _cformula(c, "I_hat  =  area * mean(f(xi, yi))")
            _cformula(c,
                f"       =  {r['vol']:.6g}  x  {r['f_bar']:.8f}")
        else:
            _cformula(c, "I_hat  =  vol * f_bar")
            _cformula(c,
                f"       =  {r['vol']:.6g}  x  {r['f_bar']:.8f}")
        _cbox(c, f"I_hat  =  {r['I_hat']:.8f}", ACCENT)

        # PASO 5
        _seccion(si, "PASO 5 — Desvio estandar de f(x_i)  -->  sigma_f  [ddof=1]", ORANGE)
        c = _card(si, ORANGE)
        _cformula(c,
            "sigma_f  =  sqrt( (1/(n-1)) * sum( (f(xi) - f_bar)^2 ) )")
        _cformula(c, "                                           ^^ ddof=1", TEAL)
        _cbox(c, f"sigma_f  =  {r['sigma_f']:.8f}", ORANGE)

        # PASO 6
        _seccion(si,
                 "PASO 6 — Error estandar  EE  y  "
                 "Desv. std de la integral  sigma_I",
                 YELLOW)
        c = _card(si, YELLOW)
        _cformula(c, "EE  =  sigma_f / sqrt(n)")
        _cformula(c,
            f"    =  {r['sigma_f']:.6f} / sqrt({r['n']:,})")
        _cformula(c,
            f"    =  {r['sigma_f']:.6f} / {math.sqrt(r['n']):.4f}")
        _cigual(c, "EE", f"{r['EE']:.8f}", YELLOW)
        _csep(c)
        _cformula(c, ">>> sigma_I  =  vol * sigma_f / sqrt(n)", GREEN)
        _cformula(c, "             =  vol * EE", GREEN)
        _cformula(c,
            f"             =  {r['vol']:.6g}  x  {r['EE']:.8f}",
            MUTED)
        _cbox(c,
              f"sigma_I  =  {r['sigma_I']:.8f}"
              f"   <-- desvio std de la INTEGRAL",
              GREEN)

        # PASO 7
        _seccion(si,
                 f"PASO 7 — Intervalo de confianza al {r['nivel']}",
                 TEAL)
        c = _card(si, TEAL)
        _cformula(c, f"Nivel elegido: {r['nivel']}   -->   z = {r['z']:.4f}")
        _cformula(c, "IC  =  I_hat  +/-  z * sigma_I")
        _cformula(c,
            f"    =  {r['I_hat']:.6f}  +/-  "
            f"{r['z']:.4f} x {r['sigma_I']:.6f}")
        _cformula(c,
            f"    =  {r['I_hat']:.6f}  +/-  {r['margen']:.6f}")
        _cbox(c,
              f"IC {r['nivel']}:  "
              f"[ {r['ic_lo']:.6f} ,  {r['ic_hi']:.6f} ]",
              TEAL)

        # PASO 7b — IC con t de Student
        if r["n"] > 1:
            _seccion(si,
                     f"PASO 7b — IC con t de Student al {r['nivel']} (metodo del profesor)",
                     ORANGE)
            c = _card(si, ORANGE)
            conf_num = float(r["nivel"].replace("%","")) / 100.0
            t_val    = float(stats.t.ppf(0.5 + conf_num / 2.0, r["n"] - 1))
            ic_t_lo  = r["I_hat"] - t_val * r["sigma_I"]
            ic_t_hi  = r["I_hat"] + t_val * r["sigma_I"]
            _cformula(c, "El profesor usa dist. t de Student para IC exacto:")
            _cformula(c,
                f"t({r['nivel']}, gl = n-1 = {r['n']-1}) = {t_val:.6f}")
            _cformula(c, "IC_t = I_hat +/- t * sigma_I")
            _cformula(c,
                f"     = {r['I_hat']:.6f} +/- {t_val:.4f} x {r['sigma_I']:.6f}")
            _cbox(c,
                  f"IC_t {r['nivel']}:  "
                  f"[ {ic_t_lo:.6f} ,  {ic_t_hi:.6f} ]",
                  ORANGE)
            _cformula(c,
                f"Con n={r['n']:,}: z={r['z']:.4f}  vs  t={t_val:.4f}  "
                f"(diferencia = {abs(t_val - r['z']):.6f})",
                MUTED)

        # PASO 8 (solo si hay exacto)
        if r.get("I_analitica") is not None:
            _seccion(si, "PASO 8 — Verificacion con valor exacto", GREEN)
            c = _card(si, GREEN)
            ia  = r["I_analitica"]
            err = abs(r["I_hat"] - ia)
            _cigual(c, "I exacta (sympy)", f"{ia:.8f}", TEAL)
            _cigual(c, "I estimada",       f"{r['I_hat']:.8f}", GREEN)
            _cigual(c, "Error absoluto",   f"{err:.3e}", YELLOW)
            if ia != 0:
                _cigual(c, "Error relativo",
                        f"{abs(err/ia)*100:.4f} %", YELLOW)
            dentro = r["ic_lo"] <= ia <= r["ic_hi"]
            _cbox(c,
                  ("Valor exacto DENTRO del IC  ok" if dentro
                   else "Valor exacto FUERA del IC  -- aumentar n"),
                  GREEN if dentro else YELLOW)

    # ─────────────────────────────────────────────────
    # RENDER: CONVERGENCIA
    # ─────────────────────────────────────────────────
    def _render_convergencia(self, r: dict):
        """
        Convergencia con banda +-1 std acumulada (metodo del profesor).
        """
        ax = self._ax_conv
        ax.clear()
        _style_ax(ax)
        try:
            s_str   = self._esemilla.get().strip()
            semilla = int(s_str) if s_str else None
            ns, I_vals, std_vals = _convergencia(
                r["fexpr"], r["a"], r["b"], r["n"], semilla)

            ax.plot(ns, I_vals, color=ACCENT, linewidth=1.5,
                    label="MC promedio acumulado")

            # Banda +-1 std acumulada (igual que el profesor)
            ax.fill_between(ns,
                            I_vals - std_vals,
                            I_vals + std_vals,
                            color=MUTED, alpha=0.25,
                            label="±1 std acumulado")

            if r.get("I_analitica") is not None:
                ax.axhline(r["I_analitica"], color=GREEN,
                           linewidth=1.5, linestyle="--",
                           label=f"Exacto = {r['I_analitica']:.6f}")

            if r.get("gauss_val") is not None:
                ax.axhline(r["gauss_val"], color=PURPLE,
                           linewidth=1.2, linestyle=":",
                           label=f"Gauss-Legendre = {r['gauss_val']:.6f}")

            ax.set_xscale("log")
            ax.set_xlabel("n  (escala log)", color=MUTED, fontsize=9)
            ax.set_ylabel("Estimacion acumulada", color=MUTED, fontsize=9)
            ax.set_title(
                f"Convergencia  |  f(x) = {r['fexpr']}  "
                f"[{r['a']:.3g}, {r['b']:.3g}]",
                color=TEXT, fontsize=10, pad=8)
            ax.legend(facecolor=BG3, edgecolor=BORDER,
                      labelcolor=TEXT, fontsize=9)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error: {exc}",
                    transform=ax.transAxes, color=RED,
                    ha="center", va="center")
        self._cvs_conv.draw()

    # ─────────────────────────────────────────────────
    # RENDER: DISTRIBUCION
    # ─────────────────────────────────────────────────
    def _render_distribucion(self, r: dict):
        """
        Histograma con curva normal superpuesta (metodo del profesor).
        Ajustado por volumen: muestra f(xi)*(b-a).
        """
        ax = self._ax_dist
        ax.clear()
        _style_ax(ax)

        vol    = r["vol"]
        data_v = r["fvals"] * vol   # ajustado por volumen, como en el profesor

        media  = float(np.mean(data_v))
        std_v  = float(np.std(data_v, ddof=1))

        n_bins = min(40, max(5, r["n"] // 50))
        ax.hist(data_v, bins=n_bins, color=PURPLE, alpha=0.75,
                edgecolor=BG, density=True,
                label="f(x_i) * vol")

        # Curva normal superpuesta (metodo del profesor)
        if std_v > 0:
            x_norm = np.linspace(data_v.min(), data_v.max(), 300)
            y_norm = stats.norm.pdf(x_norm, media, std_v)
            ax.plot(x_norm, y_norm, color=ORANGE, linewidth=2,
                    label="Distribucion Normal ajustada")

        ax.axvline(media, color=YELLOW, linewidth=2,
                   label=f"Media = {media:.4f}")
        ax.axvline(media + std_v, color=GREEN,
                   linewidth=1.2, linestyle="--",
                   label="Media ±1 std")
        ax.axvline(media - std_v, color=GREEN,
                   linewidth=1.2, linestyle="--")

        ax.set_xlabel("f(x_i) * (b-a)", color=MUTED, fontsize=9)
        ax.set_ylabel("Densidad", color=MUTED, fontsize=9)
        ax.set_title(
            f"Distribucion muestral  |  "
            f"sigma_f = {r['sigma_f']:.4f}   n = {r['n']:,}",
            color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)
        self._cvs_dist.draw()

    # ─────────────────────────────────────────────────
    # RENDER: DISPERSION
    # ─────────────────────────────────────────────────
    def _render_dispersion(self, r: dict):
        ax = self._ax_disp
        ax.clear()
        _style_ax(ax)
        N_plot = min(2000, len(r["xs"]))
        idx    = np.random.choice(len(r["xs"]), N_plot, replace=False)

        if r["dim"] == "1D":
            # 1D: scatter x vs f(x), con curva real encima
            ax.scatter(r["xs"][idx], r["fvals"][idx],
                       color=ACCENT, s=4, alpha=0.5,
                       label=f"{N_plot:,} / {len(r['xs']):,} puntos")
            # Graficar la curva real con lambdify
            try:
                f     = _make_lambdify_1d(r["fexpr"])
                x_plt = np.linspace(r["a"], r["b"], 500)
                y_plt = np.nan_to_num(f(x_plt))
                ax.plot(x_plt, y_plt, color=GREEN, linewidth=1.5,
                        label=f"f(x)={r['fexpr']}", zorder=3)
                # Relleno bajo la curva
                ax.fill_between(x_plt, 0, y_plt,
                                color=GREEN, alpha=0.07)
                # Puntos de hit-or-miss (solo algunos)
                if "success_mask" in r:
                    sm  = r["success_mask"][idx]
                    ys_h = r["ys_hom"][idx]
                    ax.scatter(r["xs"][idx][~sm], ys_h[~sm],
                               color=RED, s=3, alpha=0.3, label="H-o-M fallidos")
                    ax.scatter(r["xs"][idx][sm],  ys_h[sm],
                               color=TEAL, s=3, alpha=0.3, label="H-o-M exitos")
            except Exception:
                pass
        else:
            # 2D: scatter coloreado por f(x,y)
            sc = ax.scatter(r["xs"][idx], r["ys"][idx],
                            c=r["fvals"][idx], cmap="viridis",
                            s=5, alpha=0.6,
                            label=f"{N_plot:,} puntos")
            self._fig_disp.colorbar(sc, ax=ax, label="f(x,y)")

        ax.axhline(r["f_bar"], color=YELLOW, linewidth=1.8,
                   linestyle="--",
                   label=f"f_bar = {r['f_bar']:.4f}")
        ax.set_xlabel("x_i", color=MUTED, fontsize=9)
        ax.set_ylabel("f(x_i)" if r["dim"] == "1D" else "y_i",
                      color=MUTED, fontsize=9)
        ax.set_title(
            f"Dispersion  |  f = {r['fexpr']}",
            color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)
        self._cvs_disp.draw()


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = MontecarloApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()