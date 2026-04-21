"""
Integracion Numerica — Reglas de Newton-Cotes
===============================================
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Reglas de Newton-Cotes (pag. 27-35)

CORRECCIONES v2:
  - Indeterminaciones explicadas DENTRO de cada paso a paso por metodo
  - Error de truncamiento: manejo robusto de NaN/zoo/inf en derivadas
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
    """
    Evalua f(x) en x_val. Si la evaluacion directa produce NaN o infinito
    (indeterminacion), usa sympy para calcular el limite en ese punto.
    """
    try:
        val = eval(expr, {"__builtins__": {}}, _env(x_val))
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            raise ValueError("indeterminacion")
        return float(val)
    except Exception:
        try:
            f_sym = sp.sympify(expr)
            lim = sp.limit(f_sym, _x, sp.Rational(x_val).limit_denominator(10**9))
            val_lim = float(lim.evalf())
            if math.isnan(val_lim) or math.isinf(val_lim):
                return 0.0
            return val_lim
        except Exception:
            return 0.0

def _tiene_indeterminacion(expr, x_val):
    """
    Devuelve True si f(x_val) presenta una indeterminacion numerica.
    """
    try:
        val = eval(expr, {"__builtins__": {}}, _env(x_val))
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return True
        return False
    except Exception:
        return True

def _calcular_limite(expr, x_val):
    """
    Calcula lim_{x -> x_val} f(x) usando sympy.
    Devuelve (valor_float, expresion_sympy_str, pasos_str_list).
    pasos_str_list contiene las lineas de explicacion del proceso.
    """
    try:
        f_sym     = sp.sympify(expr)
        x_sym_val = sp.Rational(x_val).limit_denominator(10**9)

        # Intentar sustitucion directa primero para mostrar la forma indeterminada
        val_directo = f_sym.subs(_x, x_sym_val)
        forma_indet = str(val_directo)

        # Calcular el limite
        lim_expr = sp.limit(f_sym, _x, x_sym_val)
        val_lim  = float(lim_expr.evalf())
        if math.isnan(val_lim) or math.isinf(val_lim):
            val_lim = 0.0

        # Construir explicacion paso a paso
        pasos = []
        pasos.append(f"Sustitucion directa:  f({x_val}) = {expr}  con  x={x_val}")
        pasos.append(f"  => resultado directo = {forma_indet}  (INDETERMINADO)")
        pasos.append(f"Se aplica el calculo de limite:")
        pasos.append(f"  lim_{{x -> {x_val}}}  {expr}")

        # Intentar mostrar factorizacion / simplificacion si hay cociente
        try:
            num, den = sp.fraction(sp.cancel(sp.sympify(expr)))
            if den != 1:
                num_en_x0 = num.subs(_x, x_sym_val)
                den_en_x0 = den.subs(_x, x_sym_val)
                pasos.append(f"  Numerador evaluado en x={x_val}:    {num_en_x0}  = {float(num_en_x0.evalf()) if num_en_x0.is_number else num_en_x0}")
                pasos.append(f"  Denominador evaluado en x={x_val}:  {den_en_x0}  = {float(den_en_x0.evalf()) if den_en_x0.is_number else den_en_x0}")
                pasos.append(f"  => Forma indeterminada  {num_en_x0}/{den_en_x0}")

                # Intentar L'Hopital: derivar num y den
                dnum = sp.diff(num, _x)
                dden = sp.diff(den, _x)
                pasos.append(f"Regla de L'Hopital  =>  d/dx[{num}] / d/dx[{den}]")
                pasos.append(f"  = ({dnum}) / ({dden})")
                val_lhopital = sp.limit(dnum / dden, _x, x_sym_val)
                pasos.append(f"  lim_{{x->{x_val}}}  ({dnum}) / ({dden})  =  {val_lhopital}")
        except Exception:
            # No es un cociente simple; mostrar solo el resultado de sympy
            pasos.append(f"  sympy.limit resuelve la forma indeterminada directamente")
            pasos.append(f"  (usando series de Taylor / reglas internas de SymPy)")

        pasos.append(f"RESULTADO:  lim_{{x -> {x_val}}}  f(x)  =  {lim_expr}  =  {val_lim:.8f}")
        return val_lim, str(lim_expr), pasos

    except Exception as exc:
        return 0.0, "0", [f"No se pudo calcular el limite: {exc}"]

def f_eval_con_registro(expr, x_val, registro_indet):
    """
    Igual que f_eval pero si hay indeterminacion la registra en la lista
    registro_indet como dict con x, lim, lim_sym, pasos.
    """
    if _tiene_indeterminacion(expr, x_val):
        val_lim, lim_sym, pasos = _calcular_limite(expr, x_val)
        registro_indet.append({
            "x":       x_val,
            "lim":     val_lim,
            "lim_sym": lim_sym,
            "pasos":   pasos,
        })
        return val_lim
    try:
        val = eval(expr, {"__builtins__": {}}, _env(x_val))
        return float(val)
    except Exception:
        return 0.0

def f_vec(expr, xs):
    return np.array([f_eval(expr, xi) for xi in xs])


# ══════════════════════════════════════
# LOGICA — METODOS DE INTEGRACION
# ══════════════════════════════════════

def rectangulo_medio(expr, a, b, n):
    h       = (b - a) / n
    medios  = [a + (i - 0.5) * h for i in range(1, n + 1)]
    f_meds  = [f_eval(expr, xi) for xi in medios]
    I       = h * sum(f_meds)
    return I, h, medios, f_meds

def trapecio_simple(expr, a, b):
    fa = f_eval(expr, a)
    fb = f_eval(expr, b)
    I  = (b - a) / 2 * (fa + fb)
    return I, fa, fb

def trapecio_compuesto(expr, a, b, n):
    h    = (b - a) / n
    xs   = [a + i * h for i in range(n + 1)]
    ys   = [f_eval(expr, xi) for xi in xs]
    S    = ys[0] + 2 * sum(ys[1:n]) + ys[-1]
    I    = (h / 2) * S
    return I, h, xs, ys, S

def simpson13_simple(expr, a, b):
    h  = (b - a) / 2
    m  = (a + b) / 2
    fa = f_eval(expr, a)
    fm = f_eval(expr, m)
    fb = f_eval(expr, b)
    I  = (h / 3) * (fa + 4 * fm + fb)
    return I, h, m, fa, fm, fb

def simpson13_compuesto(expr, a, b, n):
    if n % 2 != 0:
        n += 1
    h   = (b - a) / n
    xs  = [a + i * h for i in range(n + 1)]
    ys  = [f_eval(expr, xi) for xi in xs]
    S_imp = sum(ys[i] for i in range(1, n, 2))
    S_par = sum(ys[i] for i in range(2, n - 1, 2))
    S   = ys[0] + 4 * S_imp + 2 * S_par + ys[-1]
    I   = (h / 3) * S
    return I, h, xs, ys, S_imp, S_par, n

def simpson38_simple(expr, a, b):
    h  = (b - a) / 3
    x1 = a + h
    x2 = a + 2 * h
    fa = f_eval(expr, a)
    f1 = f_eval(expr, x1)
    f2 = f_eval(expr, x2)
    fb = f_eval(expr, b)
    I  = (3 * h / 8) * (fa + 3 * f1 + 3 * f2 + fb)
    return I, h, x1, x2, fa, f1, f2, fb

def simpson38_compuesto(expr, a, b, n):
    while n % 3 != 0:
        n += 1
    h  = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    ys = [f_eval(expr, xi) for xi in xs]
    S_mult3 = sum(ys[i] for i in range(3, n - 2, 3))
    S_rest  = sum(ys[i] for i in range(1, n) if i % 3 != 0)
    S = ys[0] + 3 * S_rest + 2 * S_mult3 + ys[-1]
    I = (3 * h / 8) * S
    return I, h, xs, ys, n

def integral_analitica(expr, a, b):
    try:
        f_sym = sp.sympify(expr)
        F     = sp.integrate(f_sym, _x)

        try:
            F_b = float(F.subs(_x, b).evalf())
            if math.isnan(F_b) or math.isinf(F_b):
                F_b = float(sp.limit(F, _x, b).evalf())
        except Exception:
            F_b = float(sp.limit(F, _x, b).evalf())

        try:
            F_a = float(F.subs(_x, a).evalf())
            if math.isnan(F_a) or math.isinf(F_a):
                F_a = float(sp.limit(F, _x, a).evalf())
        except Exception:
            F_a = float(sp.limit(F, _x, a).evalf())

        val = F_b - F_a
        F_pretty = str(sp.simplify(F))
        return val, F_pretty, F_b, F_a
    except Exception:
        return None, None, None, None


# ══════════════════════════════════════
# ERROR DE TRUNCAMIENTO — CORREGIDO v2
# ══════════════════════════════════════
def _evaluar_derivada_segura(f_deriv, xs_scan):
    """
    Evalua la derivada simbolica en un array de puntos, ignorando
    puntos que producen NaN, zoo, inf o cualquier excepcion.
    Retorna (M, xi_max):
      M      — maximo de |f_deriv(x)| sobre xs_scan  (None si no hay puntos validos)
      xi_max — valor de x donde se alcanzo ese maximo
    """
    best_val = None
    best_xi  = None
    for xi in xs_scan:
        try:
            v   = f_deriv.subs(_x, xi).evalf()
            v_f = complex(v)
            if math.isnan(v_f.real) or math.isinf(v_f.real):
                continue
            av = abs(v_f.real)
            if best_val is None or av > best_val:
                best_val = av
                best_xi  = float(xi)
        except Exception:
            continue
    return best_val, best_xi


def error_truncamiento(expr, a, b, n, metodo, xi_usuario=None):
    """
    Calcula el error de truncamiento con las formulas exactas del enunciado.

    Si xi_usuario es None  => usa max|f^(k)| en [a,b] como cota superior.
    Si xi_usuario es float => evalua la formula exacta con signo en ese punto.

    Formulas exactas (con signo):
      Trapecio compuesto:     ET = -((b-a)^3 / (12*n^2))  * f''(xi)
      Simpson 1/3 simple:     ET = -(h^5 / 90)            * f''''(xi)   h=(b-a)/2
      Simpson 1/3 compuesto:  ET = -((b-a)^5 / (180*n^4)) * f''''(xi)
      Simpson 3/8 simple:     ET = -((3*h^5) / 80)        * f''''(xi)   h=(b-a)/3
      Simpson 3/8 compuesto:  ET = -((b-a)^5 / 6480)      * f''''(xi)

    Para Trapecio simple y Rectangulo medio se mantiene la formula original
    (cota con signo negativo).

    Retorna (ET, mensaje_error, info).
    info contiene: deriv_str, M, xi_max, orden_k, formula, modo,
                   val_deriv_con_signo (None si es cota).
    Ref: Caceres pag. 28-31
    """
    info = {}
    try:
        f_sym   = sp.sympify(expr)
        xs_scan = np.linspace(a, b, 500)

        # ── Orden de derivada segun metodo ──────────
        if "Trapecio" in metodo or "Rectangulo" in metodo:
            f_k     = sp.diff(f_sym, _x, 2)
            orden_k = 2
        else:
            f_k     = sp.diff(f_sym, _x, 4)
            orden_k = 4

        info["orden_k"]   = orden_k
        info["deriv_str"] = str(sp.simplify(f_k))

        # ── Valor de la derivada en xi ───────────────
        if xi_usuario is not None:
            # Evaluacion exacta en el punto pedido
            try:
                v   = f_k.subs(_x, xi_usuario).evalf()
                v_f = complex(v)
                if math.isnan(v_f.real) or math.isinf(v_f.real):
                    return None, f"Indeterminacion en xi={xi_usuario}", info
                val_deriv = float(v_f.real)
            except Exception as e:
                return None, f"No se pudo evaluar f^({orden_k})({xi_usuario}): {e}", info
            xi_eval  = xi_usuario
            M        = abs(val_deriv)
            info["modo"]               = "exacto"
            info["M"]                  = M
            info["xi_max"]             = xi_eval
            info["val_deriv_con_signo"] = val_deriv
        else:
            # Cota: maximo en [a,b]
            M, xi_max = _evaluar_derivada_segura(f_k, xs_scan)
            if M is None:
                return None, f"No se pudo evaluar f^({orden_k})(x) en [a,b]", info
            xi_eval   = xi_max
            val_deriv = None   # solo tenemos |M|, no el signo
            info["modo"]               = "cota"
            info["M"]                  = M
            info["xi_max"]             = xi_eval
            info["val_deriv_con_signo"] = None

        # ── Calculo del ET segun la formula del metodo ──
        h_s13  = (b - a) / 2    # h para Simpson 1/3 simple
        h_s38  = (b - a) / 3    # h para Simpson 3/8 simple

        if "Trapecio compuesto" in metodo:
            coef = -((b - a) ** 3) / (12 * n ** 2)
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -((b-a)^3 / (12*n^2)) * f''(xi)\n"
                f"   = -(({b}-{a})^3 / (12*{n}^2)) * f''({xi_eval:.6f})\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        elif "Simpson 1/3 simple" in metodo:
            coef = -(h_s13 ** 5) / 90
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -(h^5 / 90) * f''''(xi)          [h = (b-a)/2]\n"
                f"   h = ({b}-{a})/2 = {h_s13:.6f}\n"
                f"   = -({h_s13:.6f}^5 / 90) * f''''({xi_eval:.6f})\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        elif "Simpson 1/3 compuesto" in metodo:
            coef = -((b - a) ** 5) / (180 * n ** 4)
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -((b-a)^5 / (180*n^4)) * f''''(xi)\n"
                f"   = -(({b}-{a})^5 / (180*{n}^4)) * f''''({xi_eval:.6f})\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        elif "Simpson 3/8 simple" in metodo:
            coef = -(3 * h_s38 ** 5) / 80
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -((3*h^5) / 80) * f''''(xi)      [h = (b-a)/3]\n"
                f"   h = ({b}-{a})/3 = {h_s38:.6f}\n"
                f"   = -((3*{h_s38:.6f}^5) / 80) * f''''({xi_eval:.6f})\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        elif "Simpson 3/8 compuesto" in metodo:
            coef = -((b - a) ** 5) / 6480
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -((b-a)^5 / 6480) * f''''(xi)\n"
                f"   = -(({b}-{a})^5 / 6480) * f''''({xi_eval:.6f})\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        elif "Trapecio simple" in metodo:
            coef = -((b - a) ** 3) / 12
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -((b-a)^3 / 12) * f''(xi)\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        else:  # Rectangulo medio
            coef = -((b - a) ** 3) / (24 * n ** 2)
            ET   = coef * val_deriv if val_deriv is not None else abs(coef) * M
            deriv_val_str = f"{val_deriv:.8f}" if val_deriv is not None else f"|max| = {M:.8f}"
            info["formula"] = (
                f"ET = -((b-a)^3 / (24*n^2)) * f''(xi)\n"
                f"   coeficiente = {coef:.8f}\n"
                f"   f''({xi_eval:.6f}) = {deriv_val_str}\n"
                f"   ET = {ET:.8f}"
            )

        info["ET"] = ET
        return ET, None, info

    except Exception as exc:
        return None, f"Error al calcular: {exc}", info


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
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=18, pady=1)
    tk.Label(row, text=f"   i={i}  x_{i}={xi:.6f}  f(x_{i})={fxi:.8f}  peso={peso}",
             bg=BG2, fg=color, font=("Consolas", 11)).pack(side=tk.LEFT)


# ══════════════════════════════════════
# HELPER: BLOQUE DE INDETERMINACION INLINE
# Muestra el limite detallado para UN punto indeterminado.
# Se llama desde dentro de cada _pasos_XXXX al evaluar un nodo.
# ══════════════════════════════════════
def _c_bloque_limite(parent, expr, x_val, val_lim, pasos, color=ORANGE):
    """
    Muestra un sub-bloque dentro de la card de pasos que explica
    como se resuelve la indeterminacion en x=x_val mediante limite.
    """
    # Separador visual suave
    tk.Frame(parent, bg=ORANGE, height=1).pack(fill=tk.X, padx=20, pady=(6, 2))

    tk.Label(parent,
             text=f"   ⚠  Indeterminacion en  x = {x_val}  — resolucion por limite",
             bg=BG2, fg=ORANGE,
             font=("Consolas", 11, "bold")).pack(anchor="w", padx=18, pady=(2, 0))

    for linea in pasos:
        # Color segun el contenido de la linea
        if "RESULTADO" in linea or "lim_sym" in linea:
            c = GREEN
        elif "INDETERMINADO" in linea or "indeterminado" in linea:
            c = RED
        elif "L'Hopital" in linea or "L'Hopital" in linea:
            c = YELLOW
        elif "sympy" in linea or "Taylor" in linea:
            c = PURPLE
        else:
            c = MUTED
        tk.Label(parent, text=f"      {linea}",
                 bg=BG2, fg=c,
                 font=("Consolas", 11),
                 justify="left", anchor="w").pack(fill=tk.X, padx=18, pady=0)

    _c_resultado_box(parent,
        f"lim_{{x -> {x_val}}}  f(x)  =  {val_lim:.8f}  (valor usado en el calculo)",
        ORANGE)
    tk.Frame(parent, bg=ORANGE, height=1).pack(fill=tk.X, padx=20, pady=(2, 6))


def _c_eval_nodo(parent, expr, x_val, color_normal=GREEN):
    val_lim, lim_sym, pasos = _calcular_limite(expr, x_val)

    if _tiene_indeterminacion(expr, x_val):
        # Mostrar evaluacion con indeterminacion + limite
        _c_formula(parent,
            f"f({x_val})  =  {expr}  con  x = {x_val}",
            MUTED)
        _c_bloque_limite(parent, expr, x_val, val_lim, pasos, ORANGE)
        return val_lim
    else:
        # Evaluacion normal
        val = f_eval(expr, x_val)
        _c_igual(parent, f"f({x_val})", f"{val:.8f}", color_normal)
        return val


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
        sb = tk.Frame(parent, bg=BG2, width=300)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        container = tk.Frame(sb, bg=BG2)
        container.pack(fill=tk.BOTH, expand=True)

        canvas    = tk.Canvas(container, bg=BG2, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        inner  = tk.Frame(canvas, bg=BG2)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(window, width=e.width))

        def _scroll_sidebar(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_scroll(e):
            canvas.bind_all("<MouseWheel>", _scroll_sidebar)

        def _unbind_scroll(e):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_scroll)
        canvas.bind("<Leave>", _unbind_scroll)
        inner.bind("<Enter>",  _bind_scroll)
        inner.bind("<Leave>",  _unbind_scroll)

        self.after(150, lambda: canvas.configure(scrollregion=canvas.bbox("all")))

        inner.configure(padx=14, pady=14)

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

        # ── NUEVO: xi para el error de truncamiento ──────────────
        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=6)
        _lbl(inner, "ERROR DE TRUNCAMIENTO", fg=YELLOW,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 2))
        _lbl(inner, "xi — punto de evaluacion (opcional)", fg=MUTED,
             font=("Consolas", 10)).pack(anchor="w")
        _lbl(inner, "Con xi: ET exacto con signo en ese punto",
             fg=MUTED, font=("Consolas", 9)).pack(anchor="w")
        _lbl(inner, "Sin xi: cota  |ET| con max|f''| o max|f''''|",
             fg=MUTED, font=("Consolas", 9)).pack(anchor="w")
        self.e_xi_error = _entry(inner, "")
        self.e_xi_error.pack(fill=tk.X, ipady=5, pady=(4, 8))
        # ─────────────────────────────────────────────────────────

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

    # ── NUEVO: parsear xi para error de truncamiento ─────────────
    def _parse_xi_error(self):
        """Parsea el campo xi. Retorna float o None si esta vacio."""
        s = self.e_xi_error.get().strip()
        if not s:
            return None
        env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        env["pi"] = math.pi; env["e"] = math.e
        try:
            return float(eval(s, {"__builtins__": {}}, env))
        except Exception as exc:
            raise ValueError(f"xi invalido: {s!r} — {exc}")

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
        r = {"metodo": metodo, "fexpr": fexpr, "a": a, "b": b, "n": n}

        indet = []

        if metodo == "Rectangulo medio":
            I, h, medios, f_meds = rectangulo_medio(fexpr, a, b, n)
            for xi in medios:
                f_eval_con_registro(fexpr, xi, indet)
            r.update({"I": I, "h": h, "medios": medios, "f_meds": f_meds})
            r["nodos"] = [(i, medios[i], f_meds[i], "1") for i in range(n)]

        elif metodo == "Trapecio simple":
            I, fa, fb = trapecio_simple(fexpr, a, b)
            for xi in (a, b):
                f_eval_con_registro(fexpr, xi, indet)
            r.update({"I": I, "fa": fa, "fb": fb})
            r["nodos"] = [(0, a, fa, "1"), (1, b, fb, "1")]

        elif metodo == "Trapecio compuesto":
            I, h, xs, ys, S = trapecio_compuesto(fexpr, a, b, n)
            for xi in xs:
                f_eval_con_registro(fexpr, xi, indet)
            r.update({"I": I, "h": h, "xs": xs, "ys": ys, "S": S})
            pesos = ["1"] + ["2"] * (n - 1) + ["1"]
            r["nodos"] = [(i, xs[i], ys[i], pesos[i]) for i in range(n+1)]

        elif metodo == "Simpson 1/3 simple":
            I, h, m, fa, fm, fb = simpson13_simple(fexpr, a, b)
            for xi in (a, m, b):
                f_eval_con_registro(fexpr, xi, indet)
            r.update({"I": I, "h": h, "m": m, "fa": fa, "fm": fm, "fb": fb})
            r["nodos"] = [(0, a, fa, "1"), (1, m, fm, "4"), (2, b, fb, "1")]

        elif metodo == "Simpson 1/3 compuesto":
            I, h, xs, ys, S_imp, S_par, n2 = simpson13_compuesto(fexpr, a, b, n)
            for xi in xs:
                f_eval_con_registro(fexpr, xi, indet)
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
            for xi in (a, x1, x2, b):
                f_eval_con_registro(fexpr, xi, indet)
            r.update({"I": I, "h": h, "x1": x1, "x2": x2,
                       "fa": fa, "f1": f1, "f2": f2, "fb": fb})
            r["nodos"] = [(0, a, fa, "1"), (1, x1, f1, "3"),
                          (2, x2, f2, "3"), (3, b, fb, "1")]

        elif metodo == "Simpson 3/8 compuesto":
            I, h, xs, ys, n2 = simpson38_compuesto(fexpr, a, b, n)
            for xi in xs:
                f_eval_con_registro(fexpr, xi, indet)
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

        # Deduplicar indeterminaciones
        vistas = set()
        indet_uniq = []
        for d in indet:
            key = round(d["x"], 12)
            if key not in vistas:
                vistas.add(key)
                indet_uniq.append(d)
        r["indeterminaciones"] = indet_uniq

        # Solucion analitica
        analitica_str = self.e_analitica.get().strip()
        if analitica_str:
            try:
                env2 = {k:v for k,v in math.__dict__.items() if not k.startswith("__")}
                r["I_analitica"]     = float(eval(analitica_str, {"__builtins__":{}}, env2))
                r["I_analitica_str"] = analitica_str
                r["F_antiderivada"]  = None
                r["F_en_b"]          = None
                r["F_en_a"]          = None
            except Exception:
                r["I_analitica"] = None
        else:
            val, F_str, F_b, F_a = integral_analitica(fexpr, a, b)
            r["I_analitica"]     = val
            r["I_analitica_str"] = F_str
            r["F_antiderivada"]  = F_str
            r["F_en_b"]          = F_b
            r["F_en_a"]          = F_a

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
            ax.plot(x_plot, y_plot, color=ACCENT, linewidth=2.5,
                    label=f"f(x) = {fexpr}", zorder=3)

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
                if "compuesto" in metodo.lower():
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
                if "compuesto" in metodo.lower():
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

            val_anal, F_str, F_b, F_a = integral_analitica(fexpr, a, b)
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
            self._tree.insert("", "end", values=(
                i, f"{xi:.6f}", f"{fxi:.8f}", peso, f"{fxi:.8f}"
            ))
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

        # ── RESOLUCION ANALITICA
        self._pasos_analitica(si, fexpr, a, b, r)

        # ── RESULTADO FINAL
        _seccion(si, "RESULTADO FINAL", GREEN)
        c = _card(si, GREEN)
        _c_resultado_box(c,
            f"I  =  integral f(x) dx  aprox  {r['I']:.8f}", GREEN)
        if r.get("I_analitica") is not None:
            err = abs(r["I"] - r["I_analitica"])
            _c_igual(c, "Valor analitico", f"{r['I_analitica']:.8f}", TEAL)
            _c_igual(c, "|E_absoluto| = |analitico - numerico|",
                     f"{err:.2e}", YELLOW)
            if r["I_analitica"] != 0:
                _c_igual(c, "Error relativo",
                         f"{abs(err/r['I_analitica'])*100:.4f} %", YELLOW)
        _espacio(c)

    # ══════════════════════════════════════
    # HELPER INTERNO: evalua un punto y
    # muestra limite si hay indeterminacion.
    # Usado dentro de cada _pasos_XXXX.
    # ══════════════════════════════════════
    def _mostrar_eval(self, card, fexpr, x_val, label=None, color_normal=GREEN):
        """
        Muestra la evaluacion de f(x_val) dentro de una card.
        Si hay indeterminacion despliega el bloque de limite completo.
        Retorna el valor numerico.
        """
        lbl = label if label is not None else f"f({x_val})"

        if _tiene_indeterminacion(fexpr, x_val):
            val_lim, lim_sym, pasos = _calcular_limite(fexpr, x_val)
            _c_formula(card,
                f"{lbl}  =  {fexpr}  |  x = {x_val}  =>  INDETERMINACION",
                ORANGE)
            _c_bloque_limite(card, fexpr, x_val, val_lim, pasos, ORANGE)
            return val_lim
        else:
            val = f_eval(fexpr, x_val)
            _c_igual(card, lbl, f"{val:.8f}", color_normal)
            return val

    # ── PASO ANALITICA ─────────────────────────────────────────────
    def _pasos_analitica(self, si, fexpr, a, b, r):
        _seccion(si, "Resolucion analitica exacta", TEAL)
        c = _card(si, TEAL)
        _c_titulo(c, "Calculo de la integral por antiderivacion:", TEAL)

        if r.get("F_antiderivada") is not None:
            F_str = r["F_antiderivada"]
            F_b   = r["F_en_b"]
            F_a   = r["F_en_a"]
            val   = r["I_analitica"]

            _c_formula(c, f"Sea  F(x)  la antiderivada de  f(x) = {fexpr}:", TEAL)
            _c_igual(c, "F(x)", F_str, TEAL)
            _espacio(c, 4)
            _c_formula(c, "Aplicando el Teorema Fundamental del Calculo:", MUTED)
            _c_formula(c, f"I  =  F(b) - F(a)  =  F({b}) - F({a})", MUTED)

            try:
                f_sym = sp.sympify(fexpr)
                F_sym = sp.integrate(f_sym, _x)
                test_b = F_sym.subs(_x, b)
                test_a = F_sym.subs(_x, a)
                if str(test_b) in ("nan", "zoo", "oo", "-oo") or \
                   str(test_a) in ("nan", "zoo", "oo", "-oo"):
                    _c_formula(c,
                        "Indeterminacion en los limites — se aplica limite de sympy:",
                        YELLOW)
                    _c_formula(c,
                        f"lim_{{x->{b}}} F(x)  =  {F_b:.8f}", YELLOW)
                    _c_formula(c,
                        f"lim_{{x->{a}}} F(x)  =  {F_a:.8f}", YELLOW)
                else:
                    _c_igual(c, f"F({b})", f"{F_b:.8f}", GREEN)
                    _c_igual(c, f"F({a})", f"{F_a:.8f}", GREEN)
            except Exception:
                _c_igual(c, f"F({b})", f"{F_b:.8f}" if F_b is not None else "—", GREEN)
                _c_igual(c, f"F({a})", f"{F_a:.8f}" if F_a is not None else "—", GREEN)

            _espacio(c, 4)
            if val is not None:
                _c_resultado_box(c,
                    f"I  =  {F_b:.8f} - {F_a:.8f}  =  {val:.8f}", TEAL)
            else:
                _c_formula(c, "No se pudo calcular la integral analitica.", RED)

        elif r.get("I_analitica") is not None:
            _c_formula(c, "Valor ingresado manualmente por el usuario:", MUTED)
            _c_resultado_box(c,
                f"I_analitica  =  {r['I_analitica']:.8f}", TEAL)
        else:
            try:
                val, F_str, F_b, F_a = integral_analitica(fexpr, a, b)
                if val is not None:
                    _c_formula(c, f"F(x)  =  {F_str}", TEAL)
                    _c_igual(c, f"F({b})", f"{F_b:.8f}", GREEN)
                    _c_igual(c, f"F({a})", f"{F_a:.8f}", GREEN)
                    _c_resultado_box(c,
                        f"I  =  {F_b:.8f} - {F_a:.8f}  =  {val:.8f}", TEAL)
                else:
                    _c_formula(c,
                        "No se pudo calcular la integral analitica con sympy.", RED)
            except Exception:
                _c_formula(c, "No se pudo calcular la integral analitica.", RED)

        _espacio(c)

    # ══════════════════════════════════════
    # PASOS POR METODO — con limite inline
    # ══════════════════════════════════════

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
        for i in range(len(r["medios"])):
            xi  = r["medios"][i]
            fxi = r["f_meds"][i]
            xl  = a + i * h
            xr  = a + (i + 1) * h
            _c_formula(c,
                f"i={i+1}:  x_mid = ({xl:.4f} + {xr:.4f})/2 = {xi:.6f}",
                color if i < 3 else MUTED)
            self._mostrar_eval(c, fexpr, xi,
                               label=f"   f({xi:.6f})",
                               color_normal=GREEN)
            suma += fxi
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
        self._mostrar_eval(c, fexpr, a, label=f"f(a) = f({a})", color_normal=GREEN)
        self._mostrar_eval(c, fexpr, b, label=f"f(b) = f({b})", color_normal=GREEN)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        fa = r["fa"]
        fb = r["fb"]
        _c_formula(c, f"I  =  ({b} - {a}) / 2  *  [{fa:.6f} + {fb:.6f}]")
        _c_formula(c, f"   =  {(b-a)/2:.6f}  *  {fa+fb:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_trapecio_compuesto(self, si, fexpr, a, b, n, r, color):
        h   = r["h"]
        xs  = r["xs"]
        ys  = r["ys"]
        _seccion(si, "PASO 1 — Formula (Caceres pag. 28)", color)
        c = _card(si, color)
        _c_titulo(c, "Regla del Trapecio Compuesta:", color)
        _c_formula(c, "I  =  h/2 * [f(a) + 2*f(x_1) + ... + 2*f(x_{n-1}) + f(b)]")
        _c_formula(c, "h  =  (b - a) / n")
        _espacio(c)
        _c_igual(c, "h", f"({b} - {a}) / {n}  =  {h:.6f}", color)
        _espacio(c)

        _seccion(si, "PASO 2 — Evaluar f(x_i) en todos los nodos", color)
        c = _card(si, color)
        _c_titulo(c, "Tabla de nodos — evaluacion con deteccion de indeterminaciones:", color)
        for i in range(len(xs)):
            if i == 0 or i == len(xs) - 1:
                peso_str = "1 (extremo)"
                col_i    = GREEN
            else:
                peso_str = "2 (interior)"
                col_i    = color
            _c_formula(c,
                f"─── Nodo i={i}  |  x_{i} = {xs[i]:.6f}  |  peso = {peso_str}",
                col_i)
            self._mostrar_eval(c, fexpr, xs[i],
                               label=f"   f(x_{i}) = f({xs[i]:.6f})",
                               color_normal=col_i)
        _espacio(c)

        _seccion(si, "PASO 3 — Suma ponderada", color)
        c = _card(si, color)
        S = r["S"]
        _c_formula(c, "S  =  f(a)  +  2*(suma interiores)  +  f(b)")
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
        self._mostrar_eval(c, fexpr, a, label=f"f(a) = f({a})",   color_normal=GREEN)
        self._mostrar_eval(c, fexpr, m, label=f"f(m) = f({m:.6f})", color_normal=color)
        self._mostrar_eval(c, fexpr, b, label=f"f(b) = f({b})",   color_normal=GREEN)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        fa = r["fa"]; fm = r["fm"]; fb = r["fb"]
        S = fa + 4*fm + fb
        _c_formula(c, "S  =  f(a) + 4*f(m) + f(b)")
        _c_formula(c, f"   =  {fa:.6f} + 4*{fm:.6f} + {fb:.6f}")
        _c_igual(c, "S", f"{S:.6f}", color)
        _c_formula(c, f"I  =  h/3 * S  =  {h:.6f}/3 * {S:.6f}")
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    def _pasos_simpson13_compuesto(self, si, fexpr, a, b, n, r, color):
        h  = r["h"]
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

        _seccion(si, "PASO 2 — Clasificar y evaluar nodos", color)
        c = _card(si, color)
        sum_imp = 0.0
        sum_par = 0.0
        for i in range(len(xs)):
            if i == 0 or i == len(xs) - 1:
                etiqueta, col_i = "extremo  peso=1", GREEN
            elif i % 2 == 1:
                etiqueta, col_i = "impar    peso=4", ORANGE
                sum_imp += ys[i]
            else:
                etiqueta, col_i = "par      peso=2", PURPLE
                sum_par += ys[i]
            _c_formula(c,
                f"─── x_{i} = {xs[i]:.4f}  [{etiqueta}]", col_i)
            self._mostrar_eval(c, fexpr, xs[i],
                               label=f"   f(x_{i}) = f({xs[i]:.6f})",
                               color_normal=col_i)
        _espacio(c)

        _seccion(si, "PASO 3 — Sumas parciales y resultado", color)
        c = _card(si, color)
        _c_igual(c, "sum_impares (peso 4)", f"{sum_imp:.6f}", ORANGE)
        _c_igual(c, "sum_pares   (peso 2)", f"{sum_par:.6f}", PURPLE)
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
        self._mostrar_eval(c, fexpr, a,       label=f"f(a)  = f({a})",         color_normal=GREEN)
        self._mostrar_eval(c, fexpr, r["x1"], label=f"f(x1) = f({r['x1']:.6f})", color_normal=color)
        self._mostrar_eval(c, fexpr, r["x2"], label=f"f(x2) = f({r['x2']:.6f})", color_normal=color)
        self._mostrar_eval(c, fexpr, b,       label=f"f(b)  = f({b})",         color_normal=GREEN)
        _espacio(c)

        _seccion(si, "PASO 3 — Aplicar formula", color)
        c = _card(si, color)
        fa = r["fa"]; f1 = r["f1"]; f2 = r["f2"]; fb = r["fb"]
        S = fa + 3*f1 + 3*f2 + fb
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

        _seccion(si, "PASO 2 — Clasificar y evaluar nodos", color)
        c = _card(si, color)
        for i in range(len(xs)):
            if i == 0 or i == len(xs) - 1:
                etq, col_i = "peso=1 (extremo)", GREEN
            elif i % 3 == 0:
                etq, col_i = "peso=2 (mult.3)", PURPLE
            else:
                etq, col_i = "peso=3", color
            _c_formula(c,
                f"─── x_{i} = {xs[i]:.4f}  [{etq}]", col_i)
            self._mostrar_eval(c, fexpr, xs[i],
                               label=f"   f(x_{i}) = f({xs[i]:.6f})",
                               color_normal=col_i)
        _espacio(c)

        _seccion(si, "PASO 3 — Resultado", color)
        c = _card(si, color)
        _c_resultado_box(c, f"I  =  {r['I']:.8f}", color)

    # ══════════════════════════════════════
    # RENDER: ERROR — con xi opcional
    # ══════════════════════════════════════
    def _render_error(self, fexpr, a, b, n, metodo, r):
        si = self._si_err
        for w in si.winfo_children():
            w.destroy()

        color = COLORES_METODO.get(metodo, ACCENT)

        # Leer xi del campo
        try:
            xi_usuario = self._parse_xi_error()
        except Exception as exc:
            xi_usuario = None
            messagebox.showwarning("xi invalido", str(exc))

        _seccion(si, "ANALISIS DE ERROR", RED)
        c = _card(si, RED)
        _c_titulo(c, "Tipos de error en integracion numerica:", RED)
        _c_formula(c, "1. Error de truncamiento (ET) — por la formula")
        _c_formula(c, "2. |E_absoluto| = |I_analitico - I_numerico|")
        _c_formula(c, "3. Error relativo  =  |E_abs / I_analitico| * 100%")
        _espacio(c)

        # ── Error de truncamiento teorico
        _seccion(si, "Error de truncamiento teorico", YELLOW)
        c = _card(si, YELLOW)
        ET, ET_msg, info = error_truncamiento(fexpr, a, b, n, metodo, xi_usuario)
        _c_titulo(c, f"Para {metodo}:", YELLOW)

        k = info.get("orden_k", 2)
        k_label = "f''(x)"   if k == 2 else "f''''(x)"

        # Formula generica del metodo
        if "Trapecio compuesto" in metodo:
            _c_formula(c, "ET  =  -((b-a)^3 / (12*n^2))  *  f''(xi)", YELLOW)
        elif "Simpson 1/3 simple" in metodo:
            _c_formula(c, "ET  =  -(h^5 / 90)  *  f''''(xi)           h = (b-a)/2", YELLOW)
        elif "Simpson 1/3 compuesto" in metodo:
            _c_formula(c, "ET  =  -((b-a)^5 / (180*n^4))  *  f''''(xi)", YELLOW)
        elif "Simpson 3/8 simple" in metodo:
            _c_formula(c, "ET  =  -((3*h^5) / 80)  *  f''''(xi)       h = (b-a)/3", YELLOW)
        elif "Simpson 3/8 compuesto" in metodo:
            _c_formula(c, "ET  =  -((b-a)^5 / 6480)  *  f''''(xi)", YELLOW)
        elif "Trapecio simple" in metodo:
            _c_formula(c, "ET  =  -((b-a)^3 / 12)  *  f''(xi)", YELLOW)
        elif "Rectangulo" in metodo:
            _c_formula(c, "ET  =  -((b-a)^3 / (24*n^2))  *  f''(xi)", YELLOW)

        # Modo de calculo
        if xi_usuario is not None:
            _c_formula(c,
                f"Modo: EXACTO — xi = {xi_usuario}  (ingresado por el usuario)",
                TEAL)
        else:
            _c_formula(c,
                "Modo: COTA — se usa max|" + k_label + "| en [a,b]",
                ORANGE)
            _c_formula(c,
                "Para ET exacto con signo, ingresa xi en el campo del sidebar.",
                MUTED)
        _espacio(c, 4)

        if ET is not None:
            deriv_str = info.get("deriv_str", "—")
            M         = info["M"]
            xi_eval   = info["xi_max"]

            # Paso 1: derivada simbolica
            _c_formula(c, f"Paso 1 — {k_label} (derivada simbolica):", TEAL)
            _c_igual(c, k_label, deriv_str, TEAL)
            _espacio(c, 4)

            # Paso 2: valor en xi
            _c_formula(c, f"Paso 2 — Valor en xi = {xi_eval:.6f}:", TEAL)
            val_d = info.get("val_deriv_con_signo")
            if val_d is not None:
                _c_igual(c, f"{k_label} en xi={xi_eval:.6f}", f"{val_d:.8f}", GREEN)
            else:
                _c_igual(c, f"max|{k_label}| en [{a},{b}]  (xi*={xi_eval:.6f})",
                         f"{M:.8f}", ORANGE)
            _espacio(c, 4)

            # Paso 3: sustitucion completa
            _c_formula(c, "Paso 3 — Sustitucion en la formula:", TEAL)
            for linea in info.get("formula", "").split("\n"):
                if linea.strip():
                    _c_formula(c, linea, MUTED)
            _espacio(c, 4)

            if xi_usuario is not None:
                _c_resultado_box(c,
                    f"ET  =  {ET:.8f}   (exacto en xi = {xi_eval:.6f})",
                    YELLOW)
            else:
                _c_resultado_box(c,
                    f"|ET|  <=  {abs(ET):.8f}   (cota, max en xi* = {xi_eval:.4f})",
                    YELLOW)
        else:
            _c_formula(c, ET_msg or "No se pudo calcular.", RED)

        _espacio(c)

        # ── Error absoluto con solucion analitica
        if r.get("I_analitica") is not None:
            _seccion(si, "Error absoluto vs solucion analitica", GREEN)
            c = _card(si, GREEN)
            I_anal  = r["I_analitica"]
            I_num   = r["I"]
            err_abs = abs(I_anal - I_num)
            err_rel = abs(err_abs / I_anal * 100) if I_anal != 0 else 0

            _c_igual(c, "I analitico",  f"{I_anal:.8f}", TEAL)
            _c_igual(c, "I numerico",   f"{I_num:.8f}", color)
            _espacio(c, 4)
            _c_formula(c, "|E_absoluto|  =  |I_analitico - I_numerico|", MUTED)
            _c_formula(c,
                f"|E_absoluto|  =  |{I_anal:.6f} - {I_num:.6f}|", MUTED)
            _c_resultado_box(c,
                f"|E_absoluto|  =  {err_abs:.2e}  =  {err_abs:.8f}", GREEN)
            _espacio(c, 4)
            _c_igual(c, "Error relativo", f"{err_rel:.6f} %", YELLOW)

            if ET is not None:
                ok  = err_abs <= abs(ET) * 1.1
                msg = "OK — |E_abs| <= |ET| teorico" if ok \
                      else "El error real supera la cota (normal en derivadas de alto orden)"
                _c_resultado_box(c, msg, GREEN if ok else YELLOW)
            _espacio(c)

        else:
            _seccion(si, "Solucion analitica", MUTED)
            c = _card(si, MUTED)
            _c_formula(c, "No hay solucion analitica disponible.", MUTED)
            _c_formula(c,
                "Puedes ingresar el valor exacto en el campo del sidebar.", MUTED)
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

        w("RESULTADO NUMERICO\n", "title")
        w(f"  I numerica  = "); w(f"{r['I']:.8f}\n\n", "ok")

        w("SOLUCION ANALITICA\n", "title")
        if r.get("F_antiderivada") is not None:
            w(f"  F(x) = {r['F_antiderivada']}\n", "info")
            if r.get("F_en_b") is not None and r.get("F_en_a") is not None:
                w(f"  F({b}) = {r['F_en_b']:.8f}\n", "muted")
                w(f"  F({a}) = {r['F_en_a']:.8f}\n", "muted")
            w(f"  I analitica = F({b}) - F({a}) = ")
            w(f"{r['I_analitica']:.8f}\n\n", "ok")
        elif r.get("I_analitica") is not None:
            w(f"  I analitica = {r['I_analitica']:.8f}  (ingresado manualmente)\n\n", "ok")
        else:
            w("  No disponible — ingresa el valor en el sidebar\n\n", "muted")

        if r.get("I_analitica") is not None:
            w("COMPARACION Y ERROR\n", "title")
            I_anal  = r["I_analitica"]
            I_num   = r["I"]
            err_abs = abs(I_anal - I_num)
            w(f"  I analitica = "); w(f"{I_anal:.8f}\n", "info")
            w(f"  |E_absoluto| = |{I_anal:.6f} - {I_num:.6f}|\n", "muted")
            w(f"  |E_absoluto| = "); w(f"{err_abs:.2e}\n", "warn")
            if I_anal != 0:
                w(f"  Error rel    = "); w(f"{abs(err_abs/I_anal)*100:.4f} %\n\n", "warn")

        # ── Error de truncamiento en analisis ──
        w("ERROR DE TRUNCAMIENTO\n", "title")
        try:
            xi_usuario = self._parse_xi_error()
        except Exception:
            xi_usuario = None

        ET, ET_msg, info = error_truncamiento(fexpr, a, b, n, metodo, xi_usuario)

        formulas_et = {
            "Trapecio compuesto":    "ET = -((b-a)^3/(12*n^2)) * f''(xi)",
            "Simpson 1/3 simple":    "ET = -(h^5/90) * f''''(xi)   h=(b-a)/2",
            "Simpson 1/3 compuesto": "ET = -((b-a)^5/(180*n^4)) * f''''(xi)",
            "Simpson 3/8 simple":    "ET = -((3*h^5)/80) * f''''(xi)   h=(b-a)/3",
            "Simpson 3/8 compuesto": "ET = -((b-a)^5/6480) * f''''(xi)",
            "Trapecio simple":       "ET = -((b-a)^3/12) * f''(xi)",
            "Rectangulo medio":      "ET = -((b-a)^3/(24*n^2)) * f''(xi)",
        }
        w(f"  Formula: {formulas_et.get(metodo, 'ver tab Error')}\n", "info")

        if ET is not None:
            k       = info.get("orden_k", 2)
            k_label = "f''" if k == 2 else "f''''"
            w(f"  {k_label}(x) = {info.get('deriv_str', '—')}\n", "muted")
            val_d = info.get("val_deriv_con_signo")
            xi_ev = info["xi_max"]
            if val_d is not None:
                w(f"  {k_label}({xi_ev:.6f}) = {val_d:.8f}  [xi exacto]\n", "muted")
                w(f"  ET (exacto en xi={xi_ev:.4f}) = "); w(f"{ET:.8f}\n", "warn")
            else:
                w(f"  max|{k_label}| = {info['M']:.8f}  en xi* = {xi_ev:.6f}\n", "muted")
                w(f"  |ET| <= "); w(f"{abs(ET):.8f}  (cota superior)\n", "warn")
        else:
            w(f"  {ET_msg or 'No disponible'}\n", "red")

        w("\n")
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

        # Indeterminaciones encontradas
        if r.get("indeterminaciones"):
            w("\nINDETERMINACIONES RESUELTAS POR LIMITE\n", "title")
            for d in r["indeterminaciones"]:
                w(f"  x = {d['x']}  =>  lim = {d['lim']:.8f}\n", "warn")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = IntegracionApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()