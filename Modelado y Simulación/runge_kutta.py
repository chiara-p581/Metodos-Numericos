"""
Metodo de Runge-Kutta de Cuarto Orden (RK4)
============================================
Formulas implementadas:

  EDO:  dy/dx = f(x, y)   con y(x0) = y0

  RK4 estandar (un paso de tamano h):
    k1 = f(x_n,       y_n)
    k2 = f(x_n + h/2, y_n + (h/2)*k1)
    k3 = f(x_n + h/2, y_n + (h/2)*k2)
    k4 = f(x_n + h,   y_n + h*k3)

    y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    x_{n+1} = x_n + h

  Interpretacion geometrica:
    k1 = pendiente al inicio del intervalo
    k2 = primera corrección a mitad de intervalo
    k3 = segunda correccion a mitad de intervalo (mas precisa)
    k4 = pendiente al final del intervalo
    => promedio ponderado: (k1 + 2k2 + 2k3 + k4) / 6
"""

import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ══════════════════════════════════════════════════════
# PALETA  (identica a los demas modulos)
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


# ══════════════════════════════════════════════════════
# EVALUACION SEGURA DE EXPRESIONES
# ══════════════════════════════════════════════════════
def _make_env(x_val=None, y_val=None):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["np"] = np
    env["exp"] = math.exp
    env["sin"] = math.sin
    env["cos"] = math.cos
    env["log"] = math.log
    env["sqrt"] = math.sqrt
    env["pi"]  = math.pi
    env["e"]   = math.e
    if x_val is not None:
        env["x"] = x_val
    if y_val is not None:
        env["y"] = y_val
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
    env["pi"] = math.pi; env["e"] = math.e
    try:
        return float(eval(s.strip(), {"__builtins__": {}}, env))
    except Exception as exc:
        raise ValueError(f"Valor invalido en '{campo}': {s!r}  ({exc})")


# ══════════════════════════════════════════════════════
# LOGICA RK4
# ══════════════════════════════════════════════════════
def rk4_paso(fexpr: str, x: float, y: float, h: float) -> dict:
    """
    Ejecuta UN paso de RK4 y devuelve todas las pendientes intermedias.
    """
    k1 = _evaluar_f(fexpr, x,       y)
    k2 = _evaluar_f(fexpr, x + h/2, y + (h/2)*k1)
    k3 = _evaluar_f(fexpr, x + h/2, y + (h/2)*k2)
    k4 = _evaluar_f(fexpr, x + h,   y + h*k3)
    y_nuevo = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return {
        "k1": k1, "k2": k2, "k3": k3, "k4": k4,
        "prom": (k1 + 2*k2 + 2*k3 + k4) / 6,
        "y_nuevo": y_nuevo,
    }

def rk4_completo(fexpr: str, x0: float, y0: float,
                 h: float, pasos: int) -> list:
    """
    Ejecuta `pasos` iteraciones de RK4.
    Retorna lista de dicts con todos los datos de cada paso.
    """
    resultados = []
    x, y = x0, y0
    for i in range(pasos):
        paso = rk4_paso(fexpr, x, y, h)
        paso["i"]  = i
        paso["x_n"] = x
        paso["y_n"] = y
        paso["x_n1"] = x + h
        resultados.append(paso)
        x = paso["x_n1"]
        y = paso["y_nuevo"]
    return resultados

def _solucion_exacta_sympy(fexpr: str, x0: float, y0: float,
                           xs: np.ndarray):
    """
    Intenta resolver la EDO exactamente con sympy.
    Retorna array de valores o None si no puede.
    """
    try:
        x_sym = sp.Symbol("x")
        y_sym = sp.Function("y")
        fstr  = fexpr.replace("^", "**")
        # Sustituir x e y en la expresion sympy
        f_sym = sp.sympify(fstr.replace("y", "y(x)"),
                           locals={"x": x_sym, "y": y_sym(x_sym)})
        ode   = sp.Eq(y_sym(x_sym).diff(x_sym), f_sym)
        sol   = sp.dsolve(ode, y_sym(x_sym))
        # Aplicar condicion inicial
        C1    = sp.Symbol("C1")
        eq_ci = sol.rhs.subs(x_sym, x0) - y0
        c_val = sp.solve(eq_ci, C1)
        if not c_val:
            return None
        sol_particular = sol.rhs.subs(C1, c_val[0])
        f_exacta = sp.lambdify(x_sym, sol_particular, "numpy")
        return np.array([float(f_exacta(xi)) for xi in xs])
    except Exception:
        return None


# ══════════════════════════════════════════════════════
# HELPERS DE WIDGETS  (mismo patron que demas modulos)
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

def _tabla_header(parent, cols: list, widths: list, color=ACCENT):
    row = tk.Frame(parent, bg=BG3)
    row.pack(fill=tk.X, padx=18, pady=(4, 0))
    for col, w in zip(cols, widths):
        tk.Label(row, text=col, bg=BG3, fg=color,
                 font=("Consolas", 11, "bold"),
                 width=w, anchor="center").pack(side=tk.LEFT, padx=2, pady=3)

def _tabla_fila(parent, valores: list, widths: list,
                colores: list = None, bg=BG2):
    row = tk.Frame(parent, bg=bg)
    row.pack(fill=tk.X, padx=18, pady=0)
    for i, (v, w) in enumerate(zip(valores, widths)):
        fg = colores[i] if colores and i < len(colores) else TEXT
        tk.Label(row, text=str(v), bg=bg, fg=fg,
                 font=("Consolas", 11),
                 width=w, anchor="center").pack(side=tk.LEFT, padx=2, pady=2)


# ══════════════════════════════════════════════════════
# EJEMPLOS DE REFERENCIA
# ══════════════════════════════════════════════════════
EJEMPLOS_REF = [
    {
        "nombre":  "Ej. Profe — dy/dx = x+y   y(0)=1   h=0.1   10 pasos",
        "fexpr":   "x + y",
        "x0":      "0",
        "y0":      "1",
        "h":       "0.1",
        "pasos":   "10",
        "descripcion": "Exacta: y = 2*exp(x) - x - 1\nh=0.1  |  10 pasos",
    },
    {
        "nombre":  "dy/dx = -2*y   y(0)=1   h=0.1   10 pasos",
        "fexpr":   "-2*y",
        "x0":      "0",
        "y0":      "1",
        "h":       "0.1",
        "pasos":   "10",
        "descripcion": "Exacta: y = exp(-2x)\nh=0.1  |  10 pasos",
    },
    {
        "nombre":  "dy/dx = y*cos(x)   y(0)=1   h=0.2   15 pasos",
        "fexpr":   "y*cos(x)",
        "x0":      "0",
        "y0":      "1",
        "h":       "0.2",
        "pasos":   "15",
        "descripcion": "Exacta: y = exp(sin(x))\nh=0.2  |  15 pasos",
    },
]


# ══════════════════════════════════════════════════════
# APLICACION PRINCIPAL
# ══════════════════════════════════════════════════════
class RungeKuttaApp(tk.Frame):

    TABS = [
        ("Resultado",   "📋"),
        ("Paso a paso", "🔍"),
        ("Grafico",     "📈"),
        ("Pendientes",  "📐"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Runge-Kutta RK4 — Ecuaciones Diferenciales")
            master.configure(bg=BG)
            master.geometry("1400x820")
            master.minsize(1100, 640)
        self._resultados = None
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
        tk.Label(bar, text="  Runge-Kutta RK4 — EDO de primer orden",
                 bg=BG2, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="y_{n+1} = y_n + (h/6)*(k1 + 2k2 + 2k3 + k4)   "
                      "|   k2,k3 en x+h/2   |   k4 en x+h",
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

        # ── Ejemplos de referencia
        _lbl(p, "EJEMPLOS DE REFERENCIA", fg=YELLOW,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        for ej in EJEMPLOS_REF:
            self._ej_btn(p, ej)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=10)

        # ── EDO
        _lbl(p, "ECUACION DIFERENCIAL", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._ef = _labeled_entry(p, "dy/dx = f(x, y)", "x + y")
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        # ── Condicion inicial
        _lbl(p, "CONDICION INICIAL", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._ex0 = _labeled_entry(p, "x0  — valor inicial de x", "0")
        self._ey0 = _labeled_entry(p, "y0  — valor inicial de y  (condicion)", "1")
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        # ── Parametros
        _lbl(p, "PARAMETROS RK4", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._eh  = _labeled_entry(p, "h  — tamano del paso", "0.1")
        self._en  = _labeled_entry(p, "n  — numero de pasos", "10")
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        _btn(p, "  CALCULAR RK4  ", self._calcular, ACCENT).pack(fill=tk.X, pady=4)

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
                            (self._ex0, ej["x0"]),
                            (self._ey0, ej["y0"]),
                            (self._eh,  ej["h"]),
                            (self._en,  ej["pasos"])]:
            widget.delete(0, tk.END)
            widget.insert(0, val)
        self._calcular()

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
        self._build_tab_grafico()
        self._build_tab_pendientes()
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

    def _build_tab_grafico(self):
        f = self._new_panel("Grafico")
        self._fig_graf = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_graf  = self._fig_graf.add_subplot(111)
        _style_ax(self._ax_graf)
        self._cvs_graf = FigureCanvasTkAgg(self._fig_graf, master=f)
        self._cvs_graf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_tab_pendientes(self):
        f = self._new_panel("Pendientes")
        self._fig_pend = Figure(figsize=(9, 5), facecolor=BG)
        self._ax_pend  = self._fig_pend.add_subplot(111)
        _style_ax(self._ax_pend)
        self._cvs_pend = FigureCanvasTkAgg(self._fig_pend, master=f)
        self._cvs_pend.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ─────────────────────────────────────────────────
    # CALCULO
    # ─────────────────────────────────────────────────
    def _calcular(self):
        try:
            fexpr = self._ef.get().strip()
            if not fexpr:
                raise ValueError("Ingresa la EDO f(x, y).")
            x0    = _parse_float(self._ex0.get(), "x0")
            y0    = _parse_float(self._ey0.get(), "y0")
            h     = _parse_float(self._eh.get(),  "h")
            pasos = int(_parse_float(self._en.get(), "n"))
            if pasos < 1:
                raise ValueError("n debe ser al menos 1.")
            if h <= 0:
                raise ValueError("h debe ser mayor que 0.")

            # Ejecutar RK4
            resultados = rk4_completo(fexpr, x0, y0, h, pasos)

            # Intentar solucion exacta
            xs_num = np.array([x0 + i*h for i in range(pasos+1)])
            exacta = _solucion_exacta_sympy(fexpr, x0, y0, xs_num)

            self._resultados = {
                "fexpr":     fexpr,
                "x0": x0, "y0": y0, "h": h, "pasos": pasos,
                "pasos_data": resultados,
                "xs":        xs_num,
                "exacta":    exacta,
            }

            self._render_resultado()
            self._render_pasos()
            self._render_grafico()
            self._render_pendientes()
            self._show_tab("Resultado")

        except Exception as exc:
            messagebox.showerror("Error de entrada", str(exc))

    # ─────────────────────────────────────────────────
    # RENDER: RESULTADO — tabla comparativa
    # ─────────────────────────────────────────────────
    def _render_resultado(self):
        r  = self._resultados
        si = self._si_res
        for w in si.winfo_children():
            w.destroy()

        # Encabezado
        _seccion(si, "RESULTADO  —  Tabla comparativa RK4 vs Exacta", GREEN)
        c = _card(si, GREEN)
        _cformula(c, f"EDO:   dy/dx = {r['fexpr']}", ACCENT)
        _cformula(c, f"CI:    y({r['x0']:.4g}) = {r['y0']:.6g}", MUTED)
        _cformula(c, f"h = {r['h']:.4g}   |   {r['pasos']} pasos   "
                     f"|   x final = {r['x0'] + r['pasos']*r['h']:.4g}", MUTED)
        _gap(c)

        tiene_exacta = r["exacta"] is not None

        # Cabecera tabla
        if tiene_exacta:
            cols   = ["  i  ", "  x_n  ", "  y_RK4  ", "  y_exacta  ", "  error abs  ", "  error rel%  "]
            widths = [5, 10, 14, 14, 14, 14]
        else:
            cols   = ["  i  ", "  x_n  ", "  y_RK4  "]
            widths = [5, 12, 16]

        _tabla_header(si, cols, widths, ACCENT)

        # Reconstruir y values
        y_vals = [r["y0"]]
        for p in r["pasos_data"]:
            y_vals.append(p["y_nuevo"])

        for i, (xi, yi_rk4) in enumerate(zip(r["xs"], y_vals)):
            if tiene_exacta:
                yi_ex  = r["exacta"][i]
                err_a  = abs(yi_rk4 - yi_ex)
                err_r  = abs(err_a / yi_ex * 100) if yi_ex != 0 else 0.0
                valores = [
                    str(i),
                    f"{xi:.4f}",
                    f"{yi_rk4:.8f}",
                    f"{yi_ex:.8f}",
                    f"{err_a:.2e}",
                    f"{err_r:.4f}",
                ]
                # Color: verde si error pequeño, amarillo si medio, rojo si grande
                if err_a < 1e-4:
                    c_err = GREEN
                elif err_a < 1e-2:
                    c_err = YELLOW
                else:
                    c_err = RED
                colores = [MUTED, TEXT, ACCENT, TEAL, c_err, c_err]
            else:
                valores = [str(i), f"{xi:.4f}", f"{yi_rk4:.8f}"]
                colores = [MUTED, TEXT, ACCENT]

            bg_row = BG3 if i % 2 == 0 else BG2
            _tabla_fila(si, valores, widths, colores, bg_row)

        _gap(si, 12)

        # Resumen error total
        if tiene_exacta:
            y_vals_arr = np.array(y_vals)
            err_max    = float(np.max(np.abs(y_vals_arr - r["exacta"])))
            err_medio  = float(np.mean(np.abs(y_vals_arr - r["exacta"])))
            _seccion(si, "Resumen de error", YELLOW)
            c2 = _card(si, YELLOW)
            _cigual(c2, "Error maximo  (max |y_RK4 - y_exacta|)",
                    f"{err_max:.4e}", RED if err_max > 1e-3 else GREEN)
            _cigual(c2, "Error medio   (mean |y_RK4 - y_exacta|)",
                    f"{err_medio:.4e}", YELLOW)
            _cformula(c2,
                "RK4 es de orden 4: el error global es O(h^4)",
                MUTED)
            _gap(c2)

    # ─────────────────────────────────────────────────
    # RENDER: PASO A PASO
    # ─────────────────────────────────────────────────
    def _render_pasos(self):
        r  = self._resultados
        si = self._si_paso
        for w in si.winfo_children():
            w.destroy()

        # Formula general
        _seccion(si, "FORMULA GENERAL RK4", ACCENT)
        c = _card(si, ACCENT)
        _ctitulo(c, "Runge-Kutta de orden 4 — un paso:", ACCENT)
        _cformula(c, "k1 = f(x_n,        y_n)",                         GREEN)
        _cformula(c, "k2 = f(x_n + h/2,  y_n + (h/2)*k1)",             TEAL)
        _cformula(c, "k3 = f(x_n + h/2,  y_n + (h/2)*k2)",             PURPLE)
        _cformula(c, "k4 = f(x_n + h,    y_n + h*k3)",                  ORANGE)
        _csep(c)
        _cformula(c, "y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)", ACCENT)
        _cformula(c, "prom_k  = (k1 + 2*k2 + 2*k3 + k4) / 6  [prom. ponderado]", MUTED)
        _gap(c)

        # Interpretacion geometrica
        _seccion(si, "INTERPRETACION GEOMETRICA", YELLOW)
        c = _card(si, YELLOW)
        _ctitulo(c, "Cada ki representa una pendiente estimada:", YELLOW)
        _cformula(c, "k1 — pendiente al INICIO del intervalo  (punto actual)", GREEN)
        _cformula(c, "k2 — 1ra correccion a MITAD del intervalo  (usa k1)", TEAL)
        _cformula(c, "k3 — 2da correccion a MITAD del intervalo  (usa k2, mas precisa)", PURPLE)
        _cformula(c, "k4 — pendiente al FINAL del intervalo  (usa k3)", ORANGE)
        _csep(c)
        _cformula(c, "Promedio ponderado: k1 y k4 pesan 1, k2 y k3 pesan 2", MUTED)
        _cformula(c, "=> Mayor peso a las pendientes intermedias", MUTED)
        _cformula(c, "=> Error de truncacion local O(h^5), global O(h^4)", MUTED)
        _gap(c)

        # Detalle de cada paso
        _seccion(si, f"DETALLE PASO A PASO  ({r['pasos']} iteraciones)", PURPLE)

        y_actual = r["y0"]
        for paso in r["pasos_data"]:
            i   = paso["i"]
            xn  = paso["x_n"]
            yn  = paso["y_n"]
            h   = r["h"]

            c = _card(si, TEAL if i % 2 == 0 else PURPLE)
            _ctitulo(c,
                f"Paso {i+1}:   x_{i} = {xn:.6f}   y_{i} = {yn:.8f}",
                TEAL if i % 2 == 0 else PURPLE)

            _cigual(c, f"k1 = f({xn:.4f}, {yn:.6f})",
                    f"{paso['k1']:.8f}", GREEN)
            _cigual(c,
                f"k2 = f({xn+h/2:.4f}, {yn+(h/2)*paso['k1']:.6f})",
                f"{paso['k2']:.8f}", TEAL)
            _cigual(c,
                f"k3 = f({xn+h/2:.4f}, {yn+(h/2)*paso['k2']:.6f})",
                f"{paso['k3']:.8f}", PURPLE)
            _cigual(c,
                f"k4 = f({xn+h:.4f}, {yn+h*paso['k3']:.6f})",
                f"{paso['k4']:.8f}", ORANGE)
            _csep(c)
            _cigual(c, "prom_k = (k1 + 2k2 + 2k3 + k4) / 6",
                    f"{paso['prom']:.8f}", YELLOW)
            _cigual(c, f"y_{i+1}  = {yn:.6f} + {h:.4f} * {paso['prom']:.6f}",
                    f"{paso['y_nuevo']:.8f}", ACCENT)

            if r["exacta"] is not None:
                y_ex  = r["exacta"][i+1]
                err_a = abs(paso["y_nuevo"] - y_ex)
                _cigual(c, f"y exacta en x={paso['x_n1']:.4f}",
                        f"{y_ex:.8f}", TEAL)
                _cigual(c, "error absoluto",
                        f"{err_a:.3e}",
                        GREEN if err_a < 1e-4 else (YELLOW if err_a < 1e-2 else RED))
            _gap(c, 4)

    # ─────────────────────────────────────────────────
    # RENDER: GRAFICO
    # ─────────────────────────────────────────────────
    def _render_grafico(self):
        r  = self._resultados
        ax = self._ax_graf
        ax.clear()
        _style_ax(ax)

        # Reconstruir y_vals RK4
        y_vals = [r["y0"]] + [p["y_nuevo"] for p in r["pasos_data"]]
        xs     = r["xs"]

        ax.plot(xs, y_vals, color=ACCENT, linewidth=2,
                marker="o", markersize=5, markerfacecolor=ACCENT,
                label="RK4 (numerico)", zorder=3)

        if r["exacta"] is not None:
            # Curva exacta mas densa
            xs_dense = np.linspace(xs[0], xs[-1], 400)
            try:
                fexpr = r["fexpr"]
                x0, y0_val = r["x0"], r["y0"]
                exacta_dense = _solucion_exacta_sympy(fexpr, x0, y0_val, xs_dense)
                if exacta_dense is not None:
                    ax.plot(xs_dense, exacta_dense,
                            color=GREEN, linewidth=1.5, linestyle="--",
                            label="Solucion exacta", zorder=2)
            except Exception:
                pass
            ax.plot(xs, r["exacta"], color=GREEN,
                    marker="s", markersize=4, markerfacecolor=GREEN,
                    linewidth=0, alpha=0.7, label="Exacta (puntos)", zorder=4)

            # Error en cada punto
            errs = np.abs(np.array(y_vals) - r["exacta"])
            ax2  = ax.twinx()
            ax2.bar(xs, errs, width=r["h"]*0.3, color=RED, alpha=0.35,
                    label="Error abs.")
            ax2.set_ylabel("Error absoluto", color=RED, fontsize=8)
            ax2.tick_params(colors=RED, labelsize=7)
            ax2.spines["right"].set_color(RED)
            ax2.legend(facecolor=BG3, edgecolor=BORDER,
                       labelcolor=TEXT, fontsize=8, loc="upper right")

        ax.set_xlabel("x", color=MUTED, fontsize=9)
        ax.set_ylabel("y", color=MUTED, fontsize=9)
        ax.set_title(
            f"RK4  |  dy/dx = {r['fexpr']}  |  y({r['x0']}) = {r['y0']}",
            color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9, loc="upper left")
        self._fig_graf.tight_layout()
        self._cvs_graf.draw()

    # ─────────────────────────────────────────────────
    # RENDER: PENDIENTES (campo de direcciones + k1..k4)
    # ─────────────────────────────────────────────────
    def _render_pendientes(self):
        r  = self._resultados
        ax = self._ax_pend
        ax.clear()
        _style_ax(ax)

        xs    = r["xs"]
        h     = r["h"]
        fexpr = r["fexpr"]

        # Reconstruir y_vals
        y_vals = [r["y0"]] + [p["y_nuevo"] for p in r["pasos_data"]]

        # Campo de direcciones
        x_min = xs[0] - h * 0.5
        x_max = xs[-1] + h * 0.5
        y_min_r = min(y_vals)
        y_max_r = max(y_vals)
        dy_r  = max(abs(y_max_r - y_min_r) * 0.3, 0.5)

        nx, ny = 18, 14
        xg = np.linspace(x_min, x_max, nx)
        yg = np.linspace(y_min_r - dy_r, y_max_r + dy_r, ny)
        Xg, Yg = np.meshgrid(xg, yg)

        try:
            DY = np.array([[_evaluar_f(fexpr, xi, yi)
                            for xi in xg] for yi in yg])
            DX = np.ones_like(DY)
            norm = np.sqrt(DX**2 + DY**2)
            norm[norm == 0] = 1
            ax.quiver(Xg, Yg, DX/norm, DY/norm,
                      color=BORDER, alpha=0.5, scale=30,
                      width=0.003, headwidth=2)
        except Exception:
            pass

        # Trayectoria RK4
        ax.plot(xs, y_vals, color=ACCENT, linewidth=2,
                marker="o", markersize=6, zorder=4,
                label="Trayectoria RK4")

        # Dibujar k1..k4 en el primer paso (representativo)
        p0 = r["pasos_data"][0]
        xn, yn = p0["x_n"], p0["y_n"]
        escala = h * 0.4

        def _flecha(ax, x, y, dy, color, label, ls="-"):
            dx_arr = escala
            dy_arr = dy * escala
            ax.annotate("", xy=(x + dx_arr, y + dy_arr),
                        xytext=(x, y),
                        arrowprops=dict(arrowstyle="->",
                                        color=color, lw=1.8,
                                        linestyle=ls))
            ax.plot([], [], color=color, label=label, lw=2)

        _flecha(ax, xn, yn,           p0["k1"], GREEN,  "k1 — inicio")
        _flecha(ax, xn+h/2, yn+(h/2)*p0["k1"], p0["k2"], TEAL, "k2 — mitad/k1")
        _flecha(ax, xn+h/2, yn+(h/2)*p0["k2"], p0["k3"], PURPLE, "k3 — mitad/k2")
        _flecha(ax, xn+h,   yn+h*p0["k3"],     p0["k4"], ORANGE, "k4 — final")

        if r["exacta"] is not None:
            xs_dense = np.linspace(xs[0], xs[-1], 300)
            try:
                ed = _solucion_exacta_sympy(fexpr, r["x0"], r["y0"], xs_dense)
                if ed is not None:
                    ax.plot(xs_dense, ed, color=GREEN,
                            linewidth=1.2, linestyle="--", alpha=0.6,
                            label="Exacta", zorder=2)
            except Exception:
                pass

        ax.set_xlabel("x", color=MUTED, fontsize=9)
        ax.set_ylabel("y", color=MUTED, fontsize=9)
        ax.set_title(
            f"Campo de direcciones + pendientes RK4  |  Paso 1: x={xn:.3f}",
            color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)
        self._fig_pend.tight_layout()
        self._cvs_pend.draw()


# ══════════════════════════════════════════════════════
# ENTRY POINT standalone
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = RungeKuttaApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()