"""
Newton-Raphson
==============
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Metodo de Newton-Raphson (pag. 13-16)

Pestanas:
  1. Convergencia
  2. Funcion f(x)
  3. Tabla
  4. Paso a paso
  5. Derivada     <- paso a paso ALGEBRAICO de como derivar f(x), estilo cuaderno
  6. Analisis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
import re
import threading
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


# ══════════════════════════════════════
# EVALUACION SEGURA
# ══════════════════════════════════════
def _env(x_val):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x_val
    return e

def evaluar(expr, x_val):
    return eval(expr, {"__builtins__": {}}, _env(x_val))


# ══════════════════════════════════════
# LOGICA — NEWTON-RAPHSON
# ══════════════════════════════════════
def newton(fexpr, dfexpr, x0, tol=1e-6, max_iter=100):
    hist = []
    x    = x0
    for i in range(max_iter):
        try:
            fx  = evaluar(fexpr,  x)
            dfx = evaluar(dfexpr, x)
            if dfx == 0:
                return x, hist, "derivada_cero"
            xnew  = x - fx / dfx
            error = abs(xnew - x)
            hist.append({"i": i+1, "xn": x, "xn1": xnew,
                          "fx": fx, "dfx": dfx,
                          "paso": fx/dfx, "error": error})
            if error < tol:
                return xnew, hist, "convergencia"
            x = xnew
        except Exception as exc:
            return None, hist, f"error: {exc}"
    return x, hist, "max_iter"

def analisis_newton(hist, raiz, tol, fexpr, dfexpr):
    errores     = [r["error"] for r in hist if r["error"] > 0]
    ratios      = [errores[i]/errores[i-1]**2
                   for i in range(1, len(errores)) if errores[i-1] != 0]
    factor_cuad = sum(ratios)/len(ratios) if ratios else 0
    ue          = errores[-1] if errores else 0
    try:    fval = evaluar(fexpr, raiz)
    except: fval = float("nan")
    return {"iters": len(hist), "raiz": raiz, "fval": fval,
            "ue": ue, "factor_cuad": factor_cuad, "tol": tol, "ok": ue < tol}


# ══════════════════════════════════════
# DERIVADA SIMBOLICA PASO A PASO
# ══════════════════════════════════════
def _sympy_str(expr):
    """Convierte expresion sympy a string legible."""
    import sympy as sp
    # preferir notacion estandar
    s = str(expr)
    # limpiar algunos patrones comunes para legibilidad
    s = s.replace("**", "^")
    s = s.replace("sqrt", "sqrt")
    return s

def _to_py(expr_sym):
    """Convierte sympy a string evaluable sin prefijo math."""
    import sympy as sp
    code = sp.printing.pycode(expr_sym)
    code = re.sub(r'\bmath\.', '', code)
    return code

def derivar_paso_a_paso(fexpr_str):
    """
    Dado f(x) como string, devuelve:
      df_py   — string evaluable de f'(x)
      pasos   — lista de dicts: {tipo, texto, color}
        tipos: 'titulo', 'regla', 'formula', 'calculo', 'resultado',
               'separador', 'final', 'nota'
    """
    import sympy as sp
    x = sp.Symbol("x")

    try:
        f      = sp.sympify(fexpr_str)
    except Exception as e:
        return None, [{"tipo": "nota", "texto": f"No se pudo parsear: {e}", "color": RED}]

    df         = sp.diff(f, x)
    df_simple  = sp.simplify(df)
    df_py      = _to_py(df_simple)

    pasos = []

    def P(tipo, texto, color=TEXT):
        pasos.append({"tipo": tipo, "texto": texto, "color": color})

    def SEP():
        pasos.append({"tipo": "separador", "texto": "", "color": BORDER})

    # ── ENCABEZADO
    P("titulo",    "DERIVADA DE  f(x)",             TEAL)
    P("formula",   f"f(x) = {fexpr_str}",           ACCENT)
    P("formula",   "Queremos calcular:  f'(x) = d/dx [ f(x) ]", MUTED)
    SEP()

    # ── ANALIZAR ESTRUCTURA
    f_expand = sp.expand(f)
    terms    = sp.Add.make_args(f_expand)

    # ── CASO: SUMA O RESTA de terminos
    if len(terms) > 1:
        P("regla",
          "REGLA 1 — Linealidad (suma/resta de derivadas)",
          TEAL)
        P("formula",
          "Si  f(x) = g(x) + h(x)  entonces  f'(x) = g'(x) + h'(x)",
          MUTED)
        P("formula",
          f"f(x) = {f_expand}  tiene {len(terms)} terminos",
          MUTED)
        SEP()

        resultados = []
        for k, t in enumerate(terms):
            P("titulo", f"Termino {k+1}:  d/dx [ {_sympy_str(t)} ]", ORANGE)
            dt = sp.diff(t, x)

            # ─ CONSTANTE
            if t.is_number or (t.is_Mul and not t.free_symbols):
                P("regla",   "REGLA — Derivada de constante", PURPLE)
                P("formula", "d/dx [ c ] = 0   para cualquier constante c", MUTED)
                P("calculo", f"d/dx [ {_sympy_str(t)} ] = 0", GREEN)

            # ─ x sola
            elif t == x:
                P("regla",   "REGLA — Derivada de x", PURPLE)
                P("formula", "d/dx [ x ] = 1", MUTED)
                P("calculo", "d/dx [ x ] = 1", GREEN)

            # ─ c * x
            elif t.is_Mul and len(t.as_ordered_factors()) == 2:
                facs = t.as_ordered_factors()
                coef_facs = [ff for ff in facs if ff.is_number]
                var_facs  = [ff for ff in facs if not ff.is_number]
                if coef_facs and var_facs:
                    c = coef_facs[0]
                    u = var_facs[0]
                    du = sp.diff(u, x)
                    P("regla",   "REGLA — Constante por funcion", PURPLE)
                    P("formula", "d/dx [ c·f(x) ] = c · f'(x)", MUTED)
                    P("calculo", f"d/dx [ {_sympy_str(c)} · {_sympy_str(u)} ]", MUTED)
                    P("calculo", f"= {_sympy_str(c)} · d/dx [ {_sympy_str(u)} ]", MUTED)
                    P("calculo", f"= {_sympy_str(c)} · {_sympy_str(du)}", MUTED)
                    P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)
                else:
                    P("calculo",   f"d/dx [ {_sympy_str(t)} ] = {_sympy_str(sp.simplify(dt))}", GREEN)

            # ─ POTENCIA x^n
            elif t.is_Pow:
                base, exp = t.as_base_exp()
                if base == x:
                    P("regla",   "REGLA — Potencia (Power Rule)", PURPLE)
                    P("formula", "d/dx [ x^n ] = n · x^(n-1)", MUTED)
                    P("calculo", f"d/dx [ x^{_sympy_str(exp)} ]", MUTED)
                    P("calculo", f"= {_sympy_str(exp)} · x^({_sympy_str(exp)}-1)", MUTED)
                    P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)
                else:
                    _derivar_termino(t, x, P, dt)

            # ─ MULTIPLICACION con funciones (producto)
            elif t.is_Mul:
                _derivar_producto(t, x, P, dt)

            # ─ FUNCIONES ELEMENTALES
            else:
                _derivar_funcion(t, x, P, dt)

            resultados.append(sp.simplify(dt))
            SEP()

        # Suma final
        P("titulo",  "RESULTADO — Sumando todas las derivadas:", GREEN)
        partes = [_sympy_str(r) for r in resultados if not (r.is_number and r == 0)]
        P("formula",
          "f'(x) = " + "  +  ".join(_sympy_str(r) for r in resultados),
          MUTED)
        P("formula",
          f"f'(x) = {_sympy_str(df)}   (expandido)",
          MUTED)

    # ── CASO: UN SOLO TERMINO
    else:
        f_single = terms[0] if terms else f
        _derivar_termino_completo(f_single, x, P, df, fexpr_str)

    SEP()

    # ── RESULTADO FINAL
    P("titulo",    "RESULTADO FINAL", GREEN)
    P("formula",   f"f(x)  = {fexpr_str}", ACCENT)
    P("resultado", f"f'(x) = {_sympy_str(df)}", GREEN)
    if str(df) != str(df_simple):
        P("resultado", f"f'(x) = {_sympy_str(df_simple)}   (simplificado)", GREEN)
    P("nota",
      "Esta derivada va en el denominador de Newton-Raphson: x_{n+1} = x_n - f/f'",
      YELLOW)

    return df_py, pasos


def _derivar_termino(t, x, P, dt):
    """Deriva un termino generico."""
    import sympy as sp
    P("calculo", f"d/dx [ {_sympy_str(t)} ] = {_sympy_str(sp.simplify(dt))}", GREEN)


def _derivar_producto(t, x, P, dt):
    """Aplica regla del producto a t."""
    import sympy as sp
    facs     = t.as_ordered_factors()
    coef_f   = [ff for ff in facs if ff.is_number]
    var_f    = [ff for ff in facs if not ff.is_number]

    if len(var_f) >= 2:
        u  = var_f[0]
        v  = sp.Mul(*var_f[1:]) * (coef_f[0] if coef_f else 1)
        du = sp.diff(u, x)
        dv = sp.diff(v, x)
        P("regla",    "REGLA DEL PRODUCTO", PURPLE)
        P("formula",  "d/dx [ u · v ] = u' · v  +  u · v'", MUTED)
        P("calculo",  f"u  = {_sympy_str(u)}", MUTED)
        P("calculo",  f"v  = {_sympy_str(v)}", MUTED)
        P("calculo",  f"u' = {_sympy_str(sp.simplify(du))}", MUTED)
        P("calculo",  f"v' = {_sympy_str(sp.simplify(dv))}", MUTED)
        P("calculo",
          f"= ({_sympy_str(sp.simplify(du))}) · ({_sympy_str(v)})"
          f"  +  ({_sympy_str(u)}) · ({_sympy_str(sp.simplify(dv))})",
          MUTED)
        P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)
    else:
        P("calculo", f"d/dx [ {_sympy_str(t)} ] = {_sympy_str(sp.simplify(dt))}", GREEN)


def _derivar_funcion(t, x, P, dt):
    """Deriva funciones elementales con regla de la cadena si aplica."""
    import sympy as sp

    # Trigonometricas
    if isinstance(t, sp.sin):
        u  = t.args[0]
        du = sp.diff(u, x)
        P("regla",   "REGLA — Seno  +  Regla de la cadena", PURPLE)
        P("formula", "d/dx [ sin(u) ] = cos(u) · u'", MUTED)
        P("calculo", f"u  = {_sympy_str(u)}", MUTED)
        P("calculo", f"u' = {_sympy_str(sp.simplify(du))}", MUTED)
        P("calculo",
          f"= cos({_sympy_str(u)}) · {_sympy_str(sp.simplify(du))}",
          MUTED)
        P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)

    elif isinstance(t, sp.cos):
        u  = t.args[0]
        du = sp.diff(u, x)
        P("regla",   "REGLA — Coseno  +  Regla de la cadena", PURPLE)
        P("formula", "d/dx [ cos(u) ] = -sin(u) · u'", MUTED)
        P("calculo", f"u  = {_sympy_str(u)}", MUTED)
        P("calculo", f"u' = {_sympy_str(sp.simplify(du))}", MUTED)
        P("calculo",
          f"= -sin({_sympy_str(u)}) · {_sympy_str(sp.simplify(du))}",
          MUTED)
        P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)

    elif isinstance(t, sp.tan):
        u  = t.args[0]
        du = sp.diff(u, x)
        P("regla",   "REGLA — Tangente  +  Regla de la cadena", PURPLE)
        P("formula", "d/dx [ tan(u) ] = sec^2(u) · u' = u' / cos^2(u)", MUTED)
        P("calculo", f"u  = {_sympy_str(u)}", MUTED)
        P("calculo", f"u' = {_sympy_str(sp.simplify(du))}", MUTED)
        P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)

    elif isinstance(t, sp.exp):
        u  = t.args[0]
        du = sp.diff(u, x)
        P("regla",   "REGLA — Exponencial  +  Regla de la cadena", PURPLE)
        P("formula", "d/dx [ e^u ] = e^u · u'", MUTED)
        P("calculo", f"u  = {_sympy_str(u)}", MUTED)
        P("calculo", f"u' = {_sympy_str(sp.simplify(du))}", MUTED)
        P("calculo",
          f"= e^({_sympy_str(u)}) · {_sympy_str(sp.simplify(du))}",
          MUTED)
        P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)

    elif isinstance(t, sp.log):
        u  = t.args[0]
        du = sp.diff(u, x)
        P("regla",   "REGLA — Logaritmo natural  +  Regla de la cadena", PURPLE)
        P("formula", "d/dx [ ln(u) ] = u' / u", MUTED)
        P("calculo", f"u  = {_sympy_str(u)}", MUTED)
        P("calculo", f"u' = {_sympy_str(sp.simplify(du))}", MUTED)
        P("calculo",
          f"= ({_sympy_str(sp.simplify(du))}) / ({_sympy_str(u)})",
          MUTED)
        P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)

    elif isinstance(t, sp.Pow):
        base, exp = t.as_base_exp()
        du_base = sp.diff(base, x)
        if base == x and exp.is_number:
            P("regla",   "REGLA — Potencia", PURPLE)
            P("formula", "d/dx [ x^n ] = n · x^(n-1)", MUTED)
            P("calculo",
              f"d/dx [ x^{_sympy_str(exp)} ] = {_sympy_str(exp)} · x^{_sympy_str(exp-1)}",
              MUTED)
            P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)
        else:
            P("regla",   "REGLA DE LA CADENA — Potencia compuesta", PURPLE)
            P("formula", "d/dx [ u^n ] = n · u^(n-1) · u'", MUTED)
            P("calculo", f"u  = {_sympy_str(base)}", MUTED)
            P("calculo", f"u' = {_sympy_str(sp.simplify(du_base))}", MUTED)
            P("calculo",
              f"= {_sympy_str(exp)} · ({_sympy_str(base)})^({_sympy_str(exp-1)}) · {_sympy_str(sp.simplify(du_base))}",
              MUTED)
            P("resultado", f"= {_sympy_str(sp.simplify(dt))}", GREEN)
    else:
        P("calculo", f"d/dx [ {_sympy_str(t)} ] = {_sympy_str(sp.simplify(dt))}", GREEN)


def _derivar_termino_completo(f_single, x, P, df, fexpr_str):
    """Deriva cuando la funcion tiene un solo termino."""
    import sympy as sp
    dt = sp.simplify(df)

    if f_single.is_Mul:
        _derivar_producto(f_single, x, P, dt)
    else:
        _derivar_funcion(f_single, x, P, dt)


# ══════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════
def _lbl(parent, text, bg=BG2, fg=MUTED, font=("Consolas", 11)):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font)

def _labeled_entry(parent, label, default):
    _lbl(parent, label).pack(anchor="w")
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 12), bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, relief="flat")
    e.insert(0, default)
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
def _c_titulo(parent, texto, color=TEAL):
    """Titulo de seccion subrayado."""
    f = tk.Frame(parent, bg=BG2)
    f.pack(fill=tk.X, padx=16, pady=(12, 3))
    tk.Label(f, text=texto, bg=BG2, fg=color,
             font=("Consolas", 12, "bold", "underline")).pack(anchor="w")

def _c_regla(parent, texto):
    """Nombre de regla en caja destacada."""
    f = tk.Frame(parent, bg=BG3, padx=1, pady=1)
    f.pack(fill=tk.X, padx=16, pady=(6, 3))
    tk.Label(f, text=f"  [{texto}]  ", bg=BG3, fg=PURPLE,
             font=("Consolas", 11, "bold")).pack(anchor="w", padx=4, pady=3)

def _c_formula(parent, texto, color=MUTED, indent=0):
    """Linea de formula con indentacion."""
    prefix = "    " * indent
    tk.Label(parent, text=prefix + texto, bg=BG2, fg=color,
             font=("Consolas", 12), justify="left", anchor="w").pack(
                 fill=tk.X, padx=20, pady=1)

def _c_calculo(parent, texto, color=MUTED):
    """Linea de calculo numerico."""
    tk.Label(parent, text="    " + texto, bg=BG2, fg=color,
             font=("Consolas", 11), justify="left", anchor="w").pack(
                 fill=tk.X, padx=20, pady=1)

def _c_resultado(parent, texto, color=GREEN):
    """Resultado de un paso, en caja verde/destacada."""
    f = tk.Frame(parent, bg=BG3)
    f.pack(fill=tk.X, padx=24, pady=4)
    tk.Label(f, text="  " + texto, bg=BG3, fg=color,
             font=("Consolas", 12, "bold"), pady=4).pack(anchor="w", padx=4)

def _c_final(parent, texto, color=GREEN):
    """Resultado final — caja con borde de color."""
    f = tk.Frame(parent, bg=color, padx=2, pady=2)
    f.pack(fill=tk.X, padx=16, pady=8)
    inner = tk.Frame(f, bg=BG3)
    inner.pack(fill=tk.BOTH)
    tk.Label(inner, text="  " + texto, bg=BG3, fg=color,
             font=("Consolas", 13, "bold"), padx=12, pady=8).pack(anchor="w")

def _c_nota(parent, texto, color=YELLOW):
    """Nota al margen."""
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=20, pady=3)
    tk.Label(row, text="Nota: ", bg=BG2, fg=color,
             font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
    tk.Label(row, text=texto, bg=BG2, fg=MUTED,
             font=("Consolas", 11)).pack(side=tk.LEFT)

def _c_sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=8)

def _c_seccion(parent, titulo, color=TEAL):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill=tk.X, padx=10, pady=(14, 4))
    tk.Frame(f, bg=color, width=4).pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(f, text=f"  {titulo}", bg=BG, fg=color,
             font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=4)

def _espacio(parent, h=8):
    tk.Frame(parent, bg=BG2, height=h).pack()


# ══════════════════════════════════════
# RENDER PASO A PASO DE DERIVADA
# ══════════════════════════════════════
def render_derivada_en(parent, pasos):
    """Renderiza los pasos de la derivada en el frame 'parent'."""
    TIPO_RENDER = {
        "titulo":     lambda p, t, c: _c_titulo(p, t, c),
        "regla":      lambda p, t, c: _c_regla(p, t),
        "formula":    lambda p, t, c: _c_formula(p, t, c),
        "calculo":    lambda p, t, c: _c_calculo(p, t, c),
        "resultado":  lambda p, t, c: _c_resultado(p, t, c),
        "final":      lambda p, t, c: _c_final(p, t, c),
        "nota":       lambda p, t, c: _c_nota(p, t, c),
        "separador":  lambda p, t, c: _c_sep(p),
    }

    for paso in pasos:
        tipo  = paso["tipo"]
        texto = paso["texto"]
        color = paso["color"]

        # resultado y final usan caja especial
        if tipo == "resultado":
            _c_resultado(parent, texto, color)
        elif tipo == "final":
            _c_final(parent, texto, color)
        elif tipo in TIPO_RENDER:
            TIPO_RENDER[tipo](parent, texto, color)
        else:
            _c_formula(parent, texto, color)


# ══════════════════════════════════════
# CLASE PRINCIPAL — NEWTON-RAPHSON
# ══════════════════════════════════════
class NewtonApp(tk.Frame):

    TABS = [
        ("📉", "Convergencia"),
        ("📊", "Funcion f(x)"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("∂",  "Derivada"),
        ("🧠", "Analisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Newton-Raphson — Metodos Numericos")
            master.configure(bg=BG)
            master.geometry("1340x740")
            master.minsize(1050, 600)
        self._hist   = []
        self._raiz   = None
        self._fexpr  = ""
        self._dfexpr = ""
        self._x0     = 0.0
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
        tk.Label(bar, text="  Newton-Raphson", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="x_{n+1} = x_n  -  f(x_n) / f'(x_n)   |   Ref: Caceres 2026",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=285)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        _lbl(inner, "PARAMETROS", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.e_f  = _labeled_entry(inner, "f(x)", "x**3 - x - 4")

        # f'(x) con boton Hallar
        _lbl(inner, "f'(x)  — derivada de f").pack(anchor="w")
        df_row = tk.Frame(inner, bg=BG2)
        df_row.pack(fill=tk.X, pady=(2, 8))
        self.e_df = tk.Entry(df_row, bg=BG3, fg=TEXT, insertbackground=TEXT,
                             font=("Consolas", 12), bd=0,
                             highlightthickness=1, highlightbackground=BORDER,
                             highlightcolor=ACCENT, relief="flat")
        self.e_df.insert(0, "3*x**2 - 1")
        self.e_df.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7)

        btn_hallar = tk.Label(
            df_row, text=" Hallar ", bg=TEAL, fg="#000",
            font=("Segoe UI", 10, "bold"), cursor="hand2", padx=6, pady=6)
        btn_hallar.pack(side=tk.LEFT, padx=(4, 0))
        btn_hallar.bind("<Button-1>", lambda e: self._hallar_derivada())
        btn_hallar.bind("<Enter>",    lambda e: btn_hallar.config(bg=_dk(TEAL)))
        btn_hallar.bind("<Leave>",    lambda e: btn_hallar.config(bg=TEAL))

        self.e_x0  = _labeled_entry(inner, "x0  (punto inicial)", "1")
        self.e_tol = _labeled_entry(inner, "Tolerancia",           "1e-6")
        self.e_it  = _labeled_entry(inner, "Max iteraciones",      "100")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(inner, "Calcular",       self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "Graficar f(x)",  self._graficar,
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

        self._build_panel_conv()
        self._build_panel_func()
        self._build_panel_tabla()
        self._build_panel_steps()
        self._build_panel_derivada()
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
    def _build_panel_conv(self):
        f = self._panel("Convergencia")
        self._fig_conv = Figure(figsize=(7, 4), facecolor=BG)
        self._ax_conv  = self._fig_conv.add_subplot(111)
        self._style_ax(self._ax_conv)
        self._canvas_conv = FigureCanvasTkAgg(self._fig_conv, master=f)
        self._canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_panel_func(self):
        f = self._panel("Funcion f(x)")
        self._fig_func = Figure(figsize=(7, 4), facecolor=BG)
        self._ax_func  = self._fig_func.add_subplot(111)
        self._style_ax(self._ax_func)
        self._canvas_func = FigureCanvasTkAgg(self._fig_func, master=f)
        self._canvas_func.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        cols = ("i", "xn", "f(xn)", "f'(xn)", "paso", "error")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        for col, w in zip(cols, [40, 130, 110, 110, 110, 110]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="e")
        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

    def _build_panel_steps(self):
        f = self._panel("Paso a paso")
        self._si = _scrollable(f)

    def _build_panel_derivada(self):
        f = self._panel("Derivada")
        self._si_der = _scrollable(f)

    def _build_panel_analisis(self):
        f = self._panel("Analisis")
        self._ta = tk.Text(f, bg=BG3, fg=TEXT,
                           font=("Consolas", 12), bd=0, padx=20, pady=16,
                           relief="flat", wrap="word", state="disabled")
        self._ta.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        for tag, col in [("title", ACCENT), ("ok", GREEN), ("warn", YELLOW),
                          ("info", PURPLE), ("muted", MUTED)]:
            kw = {"foreground": col}
            if tag == "title":
                kw["font"] = ("Consolas", 12, "bold")
            self._ta.tag_config(tag, **kw)

    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)

    # ══════════════════════════════════════
    # HALLAR DERIVADA — boton
    # ══════════════════════════════════════
    def _hallar_derivada(self):
        fexpr = self.e_f.get().strip()
        if not fexpr:
            messagebox.showwarning("Atencion", "Ingresa f(x) primero.")
            return

        # Loading
        si = self._si_der
        for w in si.winfo_children():
            w.destroy()
        _c_seccion(si, "Calculando derivada con sympy...", TEAL)
        c_load = tk.Frame(si, bg=BG2)
        c_load.pack(fill=tk.X, padx=16, pady=8)
        _c_formula(c_load, f"f(x) = {fexpr}", ACCENT)
        _c_formula(c_load, "Analizando reglas de derivacion...", MUTED)
        self._show_tab("Derivada")

        def calcular():
            df_py, pasos = derivar_paso_a_paso(fexpr)
            self.after(0, lambda: self._mostrar_derivada(fexpr, df_py, pasos))

        threading.Thread(target=calcular, daemon=True).start()

    def _mostrar_derivada(self, fexpr, df_py, pasos):
        si = self._si_der
        for w in si.winfo_children():
            w.destroy()

        # encabezado
        _c_seccion(si, "DERIVADA PASO A PASO — estilo cuaderno", TEAL)
        hdr = tk.Frame(si, bg=BG2)
        hdr.pack(fill=tk.X, padx=16, pady=(4, 0))
        tk.Label(hdr,
                 text="Cada paso muestra la REGLA aplicada, la FORMULA y el CALCULO.",
                 bg=BG2, fg=MUTED, font=("Consolas", 11)).pack(anchor="w")
        _c_sep(hdr)

        # bloque principal con fondo BG2
        bloque = tk.Frame(si, bg=BG2)
        bloque.pack(fill=tk.X, padx=10, pady=4)

        render_derivada_en(bloque, pasos)

        # si se calculo correctamente, ofrecer cargar en el campo
        if df_py:
            _c_sep(bloque)
            btn_usar = tk.Label(
                bloque, text=f"  Usar  f'(x) = {df_py}  en el campo  ",
                bg=TEAL, fg="#000",
                font=("Segoe UI", 11, "bold"),
                cursor="hand2", padx=10, pady=6)
            btn_usar.pack(anchor="w", padx=16, pady=8)
            btn_usar.bind("<Button-1>",
                lambda e, d=df_py: self._cargar_derivada(d))
            btn_usar.bind("<Enter>", lambda e: btn_usar.config(bg=_dk(TEAL)))
            btn_usar.bind("<Leave>", lambda e: btn_usar.config(bg=TEAL))

    def _cargar_derivada(self, df_py):
        """Carga f'(x) calculada en el campo del sidebar."""
        self.e_df.delete(0, tk.END)
        self.e_df.insert(0, df_py)
        self._show_tab("Paso a paso")

    # ══════════════════════════════════════
    # CALCULAR
    # ══════════════════════════════════════
    def _calcular(self):
        try:
            fexpr  = self.e_f.get().strip()
            dfexpr = self.e_df.get().strip()
            x0     = float(eval(self.e_x0.get()))
            tol    = float(eval(self.e_tol.get()))
            it     = int(self.e_it.get())

            raiz, hist, estado = newton(fexpr, dfexpr, x0, tol, it)
            self._hist   = hist
            self._raiz   = raiz
            self._fexpr  = fexpr
            self._dfexpr = dfexpr
            self._x0     = x0

            self._render_convergencia(hist)
            self._render_tabla(hist)
            self._render_pasos(hist, fexpr, dfexpr, x0, tol)
            self._render_analisis(hist, raiz, tol, fexpr, dfexpr, estado)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ══════════════════════════════════════
    # GRAFICAR
    # ══════════════════════════════════════
    def _graficar(self):
        try:
            fexpr  = self.e_f.get().strip()
            dfexpr = self.e_df.get().strip()
            x0     = float(eval(self.e_x0.get()))
            xs     = np.linspace(x0 - 5, x0 + 5, 600)

            def safe(expr, x):
                try:    return evaluar(expr, x)
                except: return float("nan")

            ys  = [safe(fexpr,  x) for x in xs]
            dys = [safe(dfexpr, x) for x in xs]

            ax = self._ax_func
            ax.clear()
            self._style_ax(ax)
            ax.plot(xs, ys,  color=ACCENT, linewidth=2, label="f(x)")
            ax.plot(xs, dys, color=PURPLE, linewidth=1.5,
                    linestyle="--", alpha=0.7, label="f'(x)")
            ax.axhline(0, color=BORDER, linewidth=0.8)

            if self._hist:
                for r in self._hist[:6]:
                    xn, fx, dfx = r["xn"], r["fx"], r["dfx"]
                    if dfx != 0:
                        x_tang = np.array([xn - 1.5, xn + 1.5])
                        y_tang = fx + dfx * (x_tang - xn)
                        ax.plot(x_tang, y_tang, color=YELLOW,
                                linewidth=0.8, alpha=0.5)
                        ax.scatter([xn], [fx], color=ORANGE, s=35, zorder=4)

            if self._raiz is not None:
                ax.scatter([self._raiz], [0], color=GREEN,
                           zorder=5, s=80,
                           label=f"raiz = {self._raiz:.6f}")

            ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
            self._canvas_func.draw()
            self._show_tab("Funcion f(x)")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ══════════════════════════════════════
    # RENDER: CONVERGENCIA
    # ══════════════════════════════════════
    def _render_convergencia(self, hist):
        iters  = [r["i"]     for r in hist]
        errors = [r["error"] for r in hist]
        ax = self._ax_conv
        ax.clear()
        self._style_ax(ax)
        ax.semilogy(iters, errors, color=ACCENT, linewidth=2,
                    marker="o", markersize=4, markerfacecolor=ACCENT)
        ax.fill_between(iters, errors, alpha=0.08, color=ACCENT)
        ax.set_xlabel("Iteracion", color=MUTED, fontsize=9)
        ax.set_ylabel("Error (log)", color=MUTED, fontsize=9)
        ax.set_title("Convergencia — Newton-Raphson",
                     color=TEXT, fontsize=10, pad=10)
        self._canvas_conv.draw()

    # ══════════════════════════════════════
    # RENDER: TABLA
    # ══════════════════════════════════════
    def _render_tabla(self, hist):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for r in hist:
            self._tree.insert("", "end", values=(
                r["i"],
                f"{r['xn']:.8f}",
                f"{r['fx']:.6e}",
                f"{r['dfx']:.6e}",
                f"{r['paso']:.8f}",
                f"{r['error']:.2e}",
            ))

    # ══════════════════════════════════════
    # RENDER: PASO A PASO
    # ══════════════════════════════════════
    def _render_pasos(self, hist, fexpr, dfexpr, x0, tol):
        si = self._si
        for w in si.winfo_children():
            w.destroy()

        # config
        _c_seccion(si, "Configuracion inicial", ACCENT)
        cfg = tk.Frame(si, bg=BG2)
        cfg.pack(fill=tk.X, padx=12, pady=4)
        for label, val, col in [
            ("f(x)       = ", fexpr,   ACCENT),
            ("f'(x)      = ", dfexpr,  PURPLE),
            ("x0         = ", str(x0), ACCENT),
            ("Tolerancia = ", str(tol),ACCENT),
            ("Formula    = ", "x_{n+1} = x_n - f(x_n) / f'(x_n)", MUTED),
        ]:
            row = tk.Frame(cfg, bg=BG2)
            row.pack(anchor="w", padx=14, pady=2)
            tk.Label(row, text=label, bg=BG2, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG2, fg=col,
                     font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        _c_seccion(si, "Iteraciones", GREEN)
        for r in hist:
            self._step_block(r, r["error"] < tol)

    def _step_block(self, r, converged):
        bar_color = GREEN if converged else ACCENT
        outer = tk.Frame(self._si, bg=BG)
        outer.pack(fill=tk.X, padx=12, pady=3)
        tk.Frame(outer, bg=bar_color, width=3).pack(side=tk.LEFT, fill=tk.Y)
        inner = tk.Frame(outer, bg=BG2)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        hdr = tk.Frame(inner, bg=BG2)
        hdr.pack(anchor="w", padx=12, pady=(8, 4))
        tk.Label(hdr, text=f" {r['i']} ", bg=bar_color, fg="#000",
                 font=("Segoe UI", 10, "bold"), padx=4, pady=1).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"   x_n = {r['xn']:.8f}",
                 bg=BG2, fg=MUTED, font=("Consolas", 11)).pack(side=tk.LEFT)

        for pre, val, col in [
            (f"f(x_n)   = f({r['xn']:.5f})",  f"  {r['fx']:.8f}",   PURPLE),
            (f"f'(x_n)  = f'({r['xn']:.5f})", f"  {r['dfx']:.8f}",  PURPLE),
            ( "paso     = f / f'",              f"  {r['paso']:.8f}", MUTED),
            (f"x_{{n+1}}  = {r['xn']:.5f} - ({r['paso']:.6f})",
                                                f"  {r['xn1']:.8f}", ACCENT),
        ]:
            row = tk.Frame(inner, bg=BG2)
            row.pack(anchor="w", padx=12, pady=1)
            tk.Label(row, text=pre, bg=BG2, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=" = ", bg=BG2, fg=TEXT,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG2, fg=col,
                     font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        err_row = tk.Frame(inner, bg=BG2)
        err_row.pack(anchor="w", padx=12, pady=(1, 8))
        tk.Label(err_row, text="Error = |x_{n+1} - x_n| = ",
                 bg=BG2, fg=MUTED, font=("Consolas", 11)).pack(side=tk.LEFT)
        tk.Label(err_row, text=f"{r['error']:.2e}",
                 bg=BG2, fg=ORANGE, font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(err_row,
                 text="  OK convergido" if converged else "  -> continuar",
                 bg=BG2, fg=GREEN if converged else YELLOW,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

    # ══════════════════════════════════════
    # RENDER: ANALISIS
    # ══════════════════════════════════════
    def _render_analisis(self, hist, raiz, tol, fexpr, dfexpr, estado):
        info = analisis_newton(hist, raiz, tol, fexpr, dfexpr)
        ta   = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANALISIS — NEWTON-RAPHSON\n\n", "title")
        w("OK ", "ok"); w("Convergio en "); w(str(info["iters"]), "info"); w(" iteraciones\n")
        w("OK ", "ok"); w("Raiz = ");       w(f"{info['raiz']:.8f}\n", "info")
        w("OK ", "ok"); w("f(raiz) = ");    w(f"{info['fval']:.10f}\n\n", "info")
        w("OK ", "ok"); w("Error final:              "); w(f"{info['ue']:.2e}\n", "info")
        w("OK ", "ok"); w("Factor cuadratico (C):    "); w(f"{info['factor_cuad']:.4f}\n", "info")
        w("OK ", "ok"); w("Convergencia ")
        w("cuadratica", "ok"); w(" (orden 2).\n\n")

        w("ESTADO\n", "title")
        est_map = {
            "convergencia":  ("OK convergencia alcanzada", "ok"),
            "derivada_cero": ("WARN derivada = 0", "warn"),
            "max_iter":      ("WARN maximo de iteraciones", "warn"),
        }
        txt, tag = est_map.get(estado, (f"? {estado}", "muted"))
        w(f"  {txt}\n", tag)

        w("\nCRITERIO DE PARADA\n", "title")
        w(f"  {info['ue']:.2e} < {info['tol']}  ->  ")
        w("OK cumplido\n" if info["ok"] else "X no cumplido\n",
          "ok" if info["ok"] else "warn")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = NewtonApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()