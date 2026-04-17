"""
Metodo de Montecarlo — Integracion Numerica
============================================

FORMULAS CLAVE (para copiar en el parcial):
─────────────────────────────────────────────────────────────────────────
  METODO PROMEDIO 1D
    Muestras:   x_i  ~  U(a, b)      i = 1, 2, ..., n
    Estimacion: I_hat = (b-a) * (1/n) * Σ f(x_i)
    Media:      f_bar = (1/n) * Σ f(x_i)
    sigma_f   = desv. std de los VALORES f(x_i)   [ddof=1]
              = sqrt[ 1/(n-1) * Σ (f(x_i) - f_bar)² ]
    EE        = Error Estandar de la media
              = sigma_f / sqrt(n)
    sigma_I   = Desv. std de la INTEGRAL estimada
              = (b-a) * sigma_f / sqrt(n)
              = (b-a) * EE
    IC        = I_hat  +-  z * sigma_I
─────────────────────────────────────────────────────────────────────────
  HIT-OR-MISS 1D
    Rectangulo contenedor: [a,b] x [y_min, y_max]
    rect_area = (b-a) * (y_max - y_min)
    I_hat = (cantidad_exitos / n) * rect_area
─────────────────────────────────────────────────────────────────────────
  METODO PROMEDIO 2D (limites fijos)
    x_i ~ U(a,b),   y_i ~ U(c,d)
    area  = (b-a) * (d-c)
    I_hat = area * (1/n) * Σ f(x_i, y_i)
    sigma_f = desv. std de f(x_i, y_i)   [ddof=1]
    sigma_I = area * sigma_f / sqrt(n)
    IC      = I_hat  +-  z * sigma_I
─────────────────────────────────────────────────────────────────────────
  METODO PROMEDIO 2D (limites variables  y in [c(x), d(x)])
    pesos_i   = d(x_i) - c(x_i)
    valores_i = pesos_i * f(x_i, y_i)
    I_hat     = (b-a) * (1/n) * Σ valores_i
─────────────────────────────────────────────────────────────────────────
  INTERVALO DE CONFIANZA
    Con z (normal):     IC = I_hat +- z * sigma_I
    Con t de Student:   IC = I_hat +- t(alpha/2, n-1) * sigma_I
    Nota: para n >= 30, t -> z  y ambos IC son casi iguales
─────────────────────────────────────────────────────────────────────────
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
# PALETA  (identica al resto del proyecto)
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

_NIVELES_Z = {
    "90%":   1.645,
    "95%":   1.960,
    "99%":   2.576,
    "99.7%": 3.000,
}

_x_sym = sp.Symbol("x")
_y_sym = sp.Symbol("y")


# ══════════════════════════════════════════════════════
# EJERCICIOS DE REFERENCIA
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
# MOTOR DE INTEGRACION ANALITICA PASO A PASO
# (adaptado del comparador de EDOs — doc 5)
# ══════════════════════════════════════════════════════

def _is_linear_in(expr, var):
    try:
        return bool(expr.is_polynomial(var) and sp.Poly(expr, var).degree() == 1)
    except Exception:
        return False

def _liate_priority(f, var):
    if f.has(sp.log):                                                      return 0
    if any(f.has(fn) for fn in [sp.asin, sp.acos, sp.atan, sp.acot]):     return 1
    if f.is_polynomial(var):                                               return 2
    if any(f.has(fn) for fn in [sp.sin, sp.cos, sp.tan, sp.cot]):         return 3
    if f.has(sp.exp):                                                      return 4
    return 5

def _detect_integration_method(expr, var):
    x = var
    expr = sp.expand(expr)
    if not expr.free_symbols or x not in expr.free_symbols:
        return ('direct_const', {'value': expr})
    if expr.is_polynomial(x):
        return ('direct_poly', {'terms': sp.Add.make_args(expr)})
    if not expr.is_Mul:
        if isinstance(expr, sp.exp):
            arg = expr.args[0]
            if _is_linear_in(arg, x):
                a = sp.Poly(arg, x).nth(1)
                return ('direct_exp_linear', {'a': a, 'arg': arg, 'expr': expr})
            return ('direct_exp', {'expr': expr})
        for fn in [sp.sin, sp.cos, sp.tan]:
            if isinstance(expr, fn):
                arg = expr.args[0]
                if _is_linear_in(arg, x):
                    a = sp.Poly(arg, x).nth(1)
                    return ('direct_trig_linear', {'fn': fn, 'arg': arg, 'a': a})
                return ('direct_trig', {'fn': fn, 'arg': arg})
        if isinstance(expr, sp.log):
            return ('direct_log', {'arg': expr.args[0]})
        if expr.is_Pow:
            base, exp_p = expr.args
            if _is_linear_in(base, x):
                a = sp.Poly(base, x).nth(1)
                if exp_p == -1:
                    return ('direct_inv_linear', {'base': base, 'a': a})
                return ('direct_pow_linear', {'base': base, 'exp': exp_p, 'a': a})
    try:
        numer, denom = sp.fraction(expr)
        if denom != 1 and denom.is_polynomial(x) and numer.is_polynomial(x):
            pf = sp.apart(expr, x)
            if pf != expr and pf.is_Add:
                return ('partial_fractions', {'numer': numer, 'denom': denom, 'pf_decomp': pf})
    except Exception:
        pass
    sub = _detect_substitution(expr, x)
    if sub:
        return ('substitution', sub)
    if expr.is_Mul:
        factors = sorted(expr.as_ordered_factors(), key=lambda f: _liate_priority(f, x))
        u_p  = factors[0]
        dv_p = sp.Mul(*factors[1:])
        du_p = sp.diff(u_p, x)
        v_p  = sp.integrate(dv_p, x)
        rem  = sp.expand(v_p * du_p)
        method = 'byparts_repeated' if (
            rem.is_Mul and rem != expr and rem != 0
        ) else 'byparts'
        return (method, {'u': u_p, 'dv': dv_p, 'du': du_p, 'v': v_p, 'remaining': rem})
    if expr.is_Add:
        return ('sum_of_terms', {'terms': sp.Add.make_args(expr)})
    return ('general_sympy', {})

def _detect_substitution(expr, var):
    x = var
    for atom in sp.preorder_traversal(expr):
        if atom in (var, expr): continue
        if not isinstance(atom, sp.Expr): continue
        if x not in atom.free_symbols: continue
        if atom.is_Symbol: continue
        if (isinstance(atom, (sp.exp, sp.log, sp.sin, sp.cos, sp.tan))
                or (atom.is_Pow and not atom.is_polynomial(x))
                or (atom.is_Add and not _is_linear_in(atom, x))):
            du = sp.diff(atom, x)
            if du == 0: continue
            try:
                ratio = sp.simplify(expr / du)
                if x not in ratio.free_symbols:
                    return {'u': atom, 'du_dx': du, 'coeff': ratio}
            except Exception:
                pass
    return None

def _generate_integral_steps(integrand_str, var_sym):
    """Genera lista de pasos didacticos para la integral. Retorna list de dicts."""
    x = var_sym
    try:
        expr = sp.sympify(integrand_str.replace('^', '**'), locals={'x': x})
    except Exception as e:
        return [{'type': 'error', 'text': f'No se pudo parsear: {e}'}]
    method, meta = _detect_integration_method(expr, x)
    steps = _dispatch_integral_steps(method, expr, meta, x)
    try:
        result = sp.integrate(expr, x)
        steps.append({'type': 'result', 'text': f'∫ ({expr}) dx  =  {result} + C'})
    except Exception:
        pass
    return steps

def _dispatch_integral_steps(method, expr, meta, x):
    d = {
        'direct_const':       _si_const,
        'direct_poly':        _si_poly,
        'direct_exp_linear':  _si_exp_linear,
        'direct_exp':         _si_exp,
        'direct_trig_linear': _si_trig_linear,
        'direct_trig':        _si_trig,
        'direct_log':         _si_log,
        'direct_pow_linear':  _si_pow_linear,
        'direct_inv_linear':  _si_inv_linear,
        'substitution':       _si_substitution,
        'byparts':            _si_byparts,
        'byparts_repeated':   _si_byparts,
        'partial_fractions':  _si_partial_fractions,
        'sum_of_terms':       _si_sum,
        'general_sympy':      _si_general,
    }
    return d.get(method, _si_general)(expr, meta, x)

def _si_const(expr, meta, x):
    return [
        {'type': 'method',  'text': 'Integral directa — constante'},
        {'type': 'formula', 'text': '∫ k dx  =  k·x + C'},
        {'type': 'calc',    'text': f'∫ ({expr}) dx  =  ({expr})·x + C'},
    ]

def _si_poly(expr, meta, x):
    terms = meta['terms']
    steps = [
        {'type': 'method',  'text': 'Integral directa — polinomio'},
        {'type': 'formula', 'text': '∫ xⁿ dx  =  xⁿ⁺¹/(n+1) + C    [regla de la potencia]'},
    ]
    if len(terms) > 1:
        steps.append({'type': 'info', 'text': f'Integramos {len(terms)} términos por separado (linealidad):'})
        for t in terms:
            r = sp.integrate(t, x)
            steps.append({'type': 'calc', 'text': f'   ∫ ({t}) dx  =  {r}'})
    result = sp.integrate(expr, x)
    steps.append({'type': 'calc', 'text': f'Resultado:  ∫ ({expr}) dx  =  {result} + C'})
    return steps

def _si_exp_linear(expr, meta, x):
    a, arg = meta['a'], meta['arg']
    result = sp.integrate(expr, x)
    return [
        {'type': 'method',  'text': 'Integral directa — exponencial con argumento lineal'},
        {'type': 'formula', 'text': '∫ e^(ax+b) dx  =  e^(ax+b) / a + C'},
        {'type': 'info',    'text': 'La derivada del argumento es la constante a → dividimos por a.'},
        {'type': 'assign',  'label': 'Argumento', 'value': str(arg)},
        {'type': 'assign',  'label': 'a (coef. de x)', 'value': str(a)},
        {'type': 'calc',    'text': f'∫ {expr} dx  =  {expr} / ({a})  =  {result} + C'},
    ]

def _si_exp(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Integral directa — función exponencial'},
        {'type': 'formula', 'text': '∫ eˣ dx  =  eˣ + C'},
        {'type': 'calc',    'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_trig_linear(expr, meta, x):
    fn, arg, a = meta['fn'], meta['arg'], meta['a']
    fn_name = fn.__name__
    formulas = {
        'sin': '∫ sin(ax+b) dx  =  −cos(ax+b)/a + C',
        'cos': '∫ cos(ax+b) dx  =  sin(ax+b)/a + C',
        'tan': '∫ tan(ax+b) dx  =  −ln|cos(ax+b)|/a + C',
    }
    result = sp.integrate(expr, x)
    return [
        {'type': 'method',  'text': f'Integral directa — {fn_name}(ax+b)'},
        {'type': 'formula', 'text': formulas.get(fn_name, f'∫ {fn_name}(ax+b) dx')},
        {'type': 'info',    'text': 'Regla de la cadena inversa: dividimos por el coef. a.'},
        {'type': 'assign',  'label': 'Argumento', 'value': str(arg)},
        {'type': 'assign',  'label': 'a', 'value': str(a)},
        {'type': 'calc',    'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_trig(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': f'Integral básica — {meta["fn"].__name__}(x)'},
        {'type': 'calc',   'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_log(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method',  'text': 'Integral directa — logaritmo natural'},
        {'type': 'formula', 'text': '∫ ln(x) dx  =  x·ln(x) − x + C'},
        {'type': 'calc',    'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_pow_linear(expr, meta, x):
    base, exp_p, a = meta['base'], meta['exp'], meta['a']
    result = sp.integrate(expr, x)
    return [
        {'type': 'method',  'text': 'Sustitución implícita — (ax+b)ⁿ'},
        {'type': 'formula', 'text': '∫ (ax+b)ⁿ dx  =  (ax+b)^(n+1) / [a·(n+1)] + C'},
        {'type': 'assign',  'label': 'u = base', 'value': str(base)},
        {'type': 'assign',  'label': 'n',        'value': str(exp_p)},
        {'type': 'assign',  'label': 'a',        'value': str(a)},
        {'type': 'calc',    'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_inv_linear(expr, meta, x):
    base, a = meta['base'], meta['a']
    result = sp.integrate(expr, x)
    return [
        {'type': 'method',  'text': 'Integral logarítmica — 1/(ax+b)'},
        {'type': 'formula', 'text': '∫ 1/(ax+b) dx  =  (1/a)·ln|ax+b| + C'},
        {'type': 'assign',  'label': 'u = ax+b', 'value': str(base)},
        {'type': 'assign',  'label': 'a',        'value': str(a)},
        {'type': 'calc',    'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_substitution(expr, meta, x):
    u_cand = meta['u']
    du_dx  = meta['du_dx']
    coeff  = meta.get('coeff', sp.Integer(1))
    u_sym  = sp.Symbol('u')
    result = sp.integrate(expr, x)
    steps = [
        {'type': 'method',  'text': 'Método de Sustitución  u = g(x)'},
        {'type': 'formula', 'text': 'Si  u = g(x)  →  du = g\'(x)·dx  →  dx = du/g\'(x)'},
        {'type': 'info',    'text': 'Elegimos u = g(x) tal que g\'(x) aparezca en el integrando.'},
        {'type': 'assign',  'label': 'u',     'value': str(u_cand)},
        {'type': 'assign',  'label': 'du/dx', 'value': str(du_dx)},
        {'type': 'assign',  'label': 'du',    'value': f'({du_dx}) dx'},
        {'type': 'divider', 'text': '→ Despejamos dx y sustituimos:'},
        {'type': 'calc',    'text': f'dx  =  du / ({du_dx})'},
    ]
    if coeff != 1:
        int_in_u = sp.integrate(coeff, u_sym)
        steps.append({'type': 'calc', 'text': f'Integrando en u:  ∫ {coeff} du  =  {int_in_u} + C'})
    else:
        steps.append({'type': 'calc', 'text': '∫ du  =  u + C'})
    steps.append({'type': 'divider', 'text': '→ Volvemos a variable x  (u = g(x)):'})
    steps.append({'type': 'calc',    'text': f'∫ ({expr}) dx  =  {result} + C'})
    return steps

def _si_byparts(expr, meta, x, _depth=0):
    u_p  = meta['u'];  dv_p = meta['dv']
    du_p = meta['du']; v_p  = meta['v']
    rem  = meta['remaining']
    uv   = sp.expand(u_p * v_p)
    ind  = '   ' * _depth
    steps = []
    if _depth == 0:
        steps += [
            {'type': 'method',  'text': 'Integración por Partes'},
            {'type': 'formula', 'text': '∫ u dv  =  u·v  −  ∫ v du'},
            {'type': 'info',    'text': 'Regla LIATE: Log > InvTrig > Algebraica > Trig > Exp'},
        ]
    else:
        steps.append({'type': 'info', 'text': f'{ind}▸ Ronda {_depth+1} — nueva integración por partes:'})
    steps.append({'type': 'assign', 'label': f'{ind}u',  'value': str(u_p)})
    steps.append({'type': 'assign', 'label': f'{ind}dv', 'value': f'{dv_p} dx'})
    steps.append({'type': 'divider', 'text': f'{ind}→ du = derivada de u,  v = integral de dv:'})
    steps.append({'type': 'assign', 'label': f'{ind}du', 'value': f'{du_p} dx'})
    steps.append({'type': 'assign', 'label': f'{ind}v',  'value': str(v_p)})
    steps.append({'type': 'divider', 'text': f'{ind}→ Aplicamos ∫ u dv = u·v − ∫ v·du:'})
    steps.append({'type': 'calc', 'text': f'{ind}∫({expr})dx  =  ({u_p})·({v_p}) − ∫({v_p})·({du_p})dx'})
    steps.append({'type': 'calc', 'text': f'{ind}           =  {uv}  −  ∫ ({rem}) dx'})
    sub_method, sub_meta = _detect_integration_method(rem, x)
    if sub_method in ('byparts', 'byparts_repeated') and _depth < 3:
        steps += _si_byparts(rem, sub_meta, x, _depth + 1)
    else:
        rem_int = sp.integrate(rem, x)
        steps.append({'type': 'calc', 'text': f'{ind}∫ ({rem}) dx  =  {rem_int} + C'})
    if _depth == 0:
        result = sp.integrate(expr, x)
        steps.append({'type': 'divider', 'text': '→ Resultado final:'})
        steps.append({'type': 'calc', 'text': f'∫ ({expr}) dx  =  {sp.simplify(result)} + C'})
    return steps

def _si_partial_fractions(expr, meta, x):
    numer = meta['numer']; denom = meta['denom']; pf = meta['pf_decomp']
    pf_terms = sp.Add.make_args(pf)
    steps = [
        {'type': 'method',  'text': 'Fracciones Parciales'},
        {'type': 'formula', 'text': 'P(x)/Q(x) se descompone en fracciones simples'},
        {'type': 'assign',  'label': 'P(x)', 'value': str(numer)},
        {'type': 'assign',  'label': 'Q(x)', 'value': str(denom)},
        {'type': 'divider', 'text': '→ Descomposición:'},
        {'type': 'calc',    'text': f'{expr}  =  {" + ".join(str(t) for t in pf_terms)}'},
        {'type': 'divider', 'text': '→ Integramos cada fracción:'},
    ]
    for t in pf_terms:
        r = sp.integrate(t, x)
        steps.append({'type': 'calc', 'text': f'   ∫ ({t}) dx  =  {r}'})
    result = sp.integrate(expr, x)
    steps.append({'type': 'calc', 'text': f'∫ ({expr}) dx  =  {sp.simplify(result)} + C'})
    return steps

def _si_sum(expr, meta, x):
    terms = meta['terms']
    steps = [
        {'type': 'method',  'text': 'Linealidad — suma de términos'},
        {'type': 'formula', 'text': '∫ [f+g] dx  =  ∫ f dx  +  ∫ g dx'},
        {'type': 'info',    'text': f'Integramos {len(terms)} términos:'},
    ]
    for t in terms:
        r = sp.integrate(t, x)
        steps.append({'type': 'calc', 'text': f'   ∫ ({t}) dx  =  {r}'})
    result = sp.integrate(expr, x)
    steps.append({'type': 'calc', 'text': f'Total:  ∫ ({expr}) dx  =  {result} + C'})
    return steps

def _si_general(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Resolución simbólica (SymPy — técnica avanzada)'},
        {'type': 'calc',   'text': f'∫ ({expr}) dx  =  {result} + C'},
    ]


# ══════════════════════════════════════════════════════
# RENDER DE CARD DE INTEGRAL (visual)
# ══════════════════════════════════════════════════════
def _render_integral_card(parent, integrand_str, var_sym, titulo=None, borde_color=TEAL):
    """Renderiza un card con el paso a paso de la integral dada."""
    if not integrand_str:
        return
    try:
        steps = _generate_integral_steps(str(integrand_str), var_sym)
    except Exception as e:
        steps = [{'type': 'error', 'text': str(e)}]

    outer = tk.Frame(parent, bg=BG)
    outer.pack(fill=tk.X, padx=10, pady=(2, 6))
    tk.Frame(outer, bg=borde_color, width=3).pack(side=tk.LEFT, fill=tk.Y)
    inner = tk.Frame(outer, bg=BG3)
    inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    if titulo:
        tk.Label(inner, text=titulo, bg=BG3, fg=borde_color,
                 font=("Consolas", 11, "bold", "underline"),
                 anchor="w").pack(fill=tk.X, padx=14, pady=(8, 4))

    for step in steps:
        stype = step.get('type', 'calc')

        if stype == 'method':
            fm = tk.Frame(inner, bg=BG2)
            fm.pack(fill=tk.X, padx=8, pady=(6, 2))
            tk.Label(fm, text=f"  ⟐ {step['text']}", bg=BG2, fg=TEAL,
                     font=("Consolas", 11, "bold"), anchor="w").pack(fill=tk.X, padx=6, pady=4)
        elif stype == 'formula':
            tk.Label(inner, text=f"    {step['text']}", bg=BG3, fg=GREEN,
                     font=("Consolas", 11), anchor="w").pack(fill=tk.X, padx=14, pady=2)
        elif stype == 'info':
            tk.Label(inner, text=f"    {step['text']}", bg=BG3, fg=MUTED,
                     font=("Consolas", 10), anchor="w").pack(fill=tk.X, padx=14, pady=1)
        elif stype == 'assign':
            row = tk.Frame(inner, bg=BG3)
            row.pack(anchor="w", padx=18, pady=2)
            tk.Label(row, text=f"  {step.get('label','?')}  =  ",
                     bg=BG3, fg=MUTED, font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=str(step.get('value', '')),
                     bg=BG3, fg=YELLOW, font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        elif stype == 'calc':
            tk.Label(inner, text=f"    {step['text']}", bg=BG3, fg=TEXT,
                     font=("Consolas", 11), anchor="w").pack(fill=tk.X, padx=14, pady=2)
        elif stype == 'divider':
            tk.Label(inner, text=f"  {step['text']}", bg=BG3, fg=ACCENT,
                     font=("Consolas", 11, "bold"), anchor="w").pack(fill=tk.X, padx=14, pady=(6, 2))
        elif stype == 'result':
            rf = tk.Frame(inner, bg="#1a2a1a", pady=4, padx=6)
            rf.pack(fill=tk.X, padx=14, pady=6)
            tk.Label(rf, text=f"  ✓  {step['text']}", bg="#1a2a1a", fg=GREEN,
                     font=("Consolas", 12, "bold"), anchor="w").pack(fill=tk.X, padx=8, pady=4)
        elif stype == 'error':
            tk.Label(inner, text=f"  ✗ {step['text']}", bg=BG3, fg=RED,
                     font=("Consolas", 10), anchor="w").pack(fill=tk.X, padx=14, pady=2)

    tk.Frame(inner, bg=BG3, height=8).pack()


# ══════════════════════════════════════════════════════
# FUNCIONES DE EVALUACION
# ══════════════════════════════════════════════════════
def _make_lambdify_1d(fexpr: str):
    x = sp.Symbol('x')
    return sp.lambdify(x, sp.sympify(fexpr), "numpy")

def _make_lambdify_2d(fexpr: str):
    x, y = sp.symbols('x y')
    return sp.lambdify((x, y), sp.sympify(fexpr), "numpy")

def _evaluar_expr_x(expr: str, x_val: float) -> float:
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["x"] = x_val
    try:
        return float(eval(expr, {"__builtins__": {}}, env))
    except Exception as e:
        raise ValueError(f"Error evaluando limite '{expr}' con x={x_val}: {e}")


# ══════════════════════════════════════════════════════
# LOGICA MONTECARLO 1D
# ══════════════════════════════════════════════════════
def montecarlo_1d(fexpr, a, b, n, nivel, z, semilla=None):
    rng = np.random.default_rng(semilla)
    xs  = rng.uniform(a, b, n)
    f   = _make_lambdify_1d(fexpr)
    fvals = np.nan_to_num(f(xs)).astype(float)

    f_bar   = float(np.mean(fvals))
    vol     = b - a
    I_hat   = vol * f_bar
    sigma_f = float(np.std(fvals, ddof=1))
    EE      = sigma_f / math.sqrt(n)
    sigma_I = vol * EE
    margen  = z * sigma_I

    # Hit-or-miss
    xs_dense = np.linspace(a, b, 1000)
    ys_dense = np.nan_to_num(f(xs_dense))
    y_min    = min(0.0, float(np.min(ys_dense)))
    y_max    = max(0.0, float(np.max(ys_dense)))
    ys_hom   = rng.uniform(y_min, y_max, n)
    success_mask = ((ys_hom >= 0) & (ys_hom <= fvals)) | ((ys_hom <= 0) & (ys_hom >= fvals))
    rect_area    = (b - a) * (y_max - y_min)
    I_hitormiss  = float(np.sum(success_mask) / n * rect_area)

    # Gauss-Legendre (referencia)
    try:
        nodes, weights = leggauss(5)
        trans_nodes = 0.5 * (nodes + 1) * (b - a) + a
        gauss_val   = float(0.5 * (b - a) * np.sum(weights * f(trans_nodes)))
    except Exception:
        gauss_val = None

    return {
        "dim": "1D", "fexpr": fexpr,
        "a": a, "b": b, "n": n, "vol": vol,
        "nivel": nivel, "z": z,
        "xs": xs, "fvals": fvals,
        "ys_hom": ys_hom, "success_mask": success_mask,
        "y_min": y_min, "y_max": y_max,
        "f_bar": f_bar, "I_hat": I_hat,
        "I_hitormiss": I_hitormiss, "gauss_val": gauss_val,
        "sigma_f": sigma_f, "EE": EE,
        "sigma_I": sigma_I, "margen": margen,
        "ic_lo": I_hat - margen, "ic_hi": I_hat + margen,
    }


# ══════════════════════════════════════════════════════
# LOGICA MONTECARLO 2D
# ══════════════════════════════════════════════════════
def montecarlo_2d(fexpr, a, b, c_expr, d_expr, n, nivel, z, semilla=None):
    rng = np.random.default_rng(semilla)
    f   = _make_lambdify_2d(fexpr)

    def _es_constante(expr_str):
        env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        env["pi"] = math.pi; env["e"] = math.e
        try:
            return True, float(eval(expr_str.strip(), {"__builtins__": {}}, env))
        except Exception:
            return False, None

    c_cte, c_val = _es_constante(c_expr)
    d_cte, d_val = _es_constante(d_expr)

    if c_cte and d_cte:
        xs    = rng.uniform(a, b, n)
        ys    = rng.uniform(c_val, d_val, n)
        fvals = np.nan_to_num(f(xs, ys)).astype(float)
        area    = (b - a) * (d_val - c_val)
        f_bar   = float(np.mean(fvals))
        I_hat   = area * f_bar
        sigma_f = float(np.std(fvals, ddof=1))
        EE      = sigma_f / math.sqrt(n)
        sigma_I = area * EE
        margen  = z * sigma_I
        return {
            "dim": "2D", "fexpr": fexpr,
            "a": a, "b": b, "c": c_expr, "d": d_expr,
            "n": n, "vol": area, "nivel": nivel, "z": z,
            "xs": xs, "ys": ys, "fvals": fvals,
            "f_bar": f_bar, "I_hat": I_hat,
            "sigma_f": sigma_f, "EE": EE,
            "sigma_I": sigma_I, "margen": margen,
            "ic_lo": I_hat - margen, "ic_hi": I_hat + margen,
            "limites_variables": False,
        }
    else:
        xs = rng.uniform(a, b, n)
        ys    = np.empty(n)
        pesos = np.empty(n)
        for i, xi in enumerate(xs):
            c_i      = _evaluar_expr_x(c_expr, xi)
            d_i      = _evaluar_expr_x(d_expr, xi)
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
            "dim": "2D", "fexpr": fexpr,
            "a": a, "b": b, "c": c_expr, "d": d_expr,
            "n": n, "vol": vol, "nivel": nivel, "z": z,
            "xs": xs, "ys": ys, "fvals": fvals,
            "f_bar": f_bar, "I_hat": I_hat,
            "sigma_f": sigma_f, "EE": EE,
            "sigma_I": sigma_I, "margen": margen,
            "ic_lo": I_hat - margen, "ic_hi": I_hat + margen,
            "limites_variables": True,
        }


def _integral_analitica_1d(fexpr, a, b):
    try:
        fstr  = fexpr.replace("log(", "ln(")
        f_sym = sp.sympify(fstr)
        F     = sp.integrate(f_sym, _x_sym)
        return float((F.subs(_x_sym, b) - F.subs(_x_sym, a)).evalf())
    except Exception:
        return None

def _convergencia(fexpr, a, b, n_max, semilla=None, pasos=55):
    rng  = np.random.default_rng(semilla)
    f    = _make_lambdify_1d(fexpr)
    ns   = np.unique(np.geomspace(1, n_max, pasos).astype(int))
    I_vals = []; std_vals = []
    for n in ns:
        xs    = rng.uniform(a, b, n)
        fvals = np.nan_to_num(f(xs)).astype(float)
        I_vals.append(float(np.mean(fvals)) * (b - a))
        std_vals.append((float(np.std(fvals, ddof=1)) if n > 1 else 0.0) * (b - a))
    return ns, np.array(I_vals), np.array(std_vals)


# ══════════════════════════════════════════════════════
# HELPERS DE WIDGETS  (identicos al resto del proyecto)
# ══════════════════════════════════════════════════════
def _lbl(parent, text, fg=MUTED, font=("Consolas", 11), bg=None):
    return tk.Label(parent, text=text, bg=bg or BG2, fg=fg, font=font)

def _entry(parent, default):
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 12), bd=0, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT)
    e.insert(0, default)
    return e

def _labeled_entry(parent, label, default):
    _lbl(parent, label).pack(anchor="w")
    e = _entry(parent, default)
    e.pack(fill=tk.X, ipady=6, pady=(2, 8))
    return e

def _btn(parent, text, cmd, color=ACCENT, fg="#000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 12, "bold"), padx=14, pady=9, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(max(0,int(r*.75)), max(0,int(g*.75)), max(0,int(b*.75)))

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
    inner.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
    cvs.bind("<Configure>",   lambda e: cvs.itemconfig(win, width=e.width))
    cvs.bind_all("<MouseWheel>", lambda e: cvs.yview_scroll(int(-1*(e.delta/120)), "units"))
    return inner

def _style_ax(ax):
    ax.set_facecolor(BG2)
    for s in ax.spines.values(): s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)

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

def _ctitulo(parent, texto, color=TEAL):
    tk.Label(parent, text=texto, bg=BG2, fg=color,
             font=("Consolas", 12, "bold", "underline"),
             anchor="w").pack(fill=tk.X, padx=16, pady=(10, 2))

def _cformula(parent, texto, color=MUTED, indent=1):
    tk.Label(parent, text="   "*indent + texto, bg=BG2, fg=color,
             font=("Consolas", 12), justify="left",
             anchor="w").pack(fill=tk.X, padx=18, pady=1)

def _cigual(parent, izq, der, color_der=GREEN):
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=18, pady=2)
    tk.Label(row, text="   "+izq+"  =  ", bg=BG2, fg=MUTED,
             font=("Consolas", 12)).pack(side=tk.LEFT)
    tk.Label(row, text=der, bg=BG2, fg=color_der,
             font=("Consolas", 12, "bold")).pack(side=tk.LEFT)

def _cbox(parent, texto, color=GREEN):
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

def _parse_float(s, campo):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["pi"] = math.pi; env["e"] = math.e
    try:
        return float(eval(s.strip(), {"__builtins__": {}}, env))
    except Exception as exc:
        raise ValueError(f"Valor invalido en '{campo}': {s!r}  ({exc})")


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
                 text="I_hat = vol × f_bar   |   "
                      "sigma_I = vol × sigma_f / √n   |   "
                      "IC = I_hat ± z × sigma_I",
                 bg=BG2, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

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
        wrap.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>",  lambda e: cvs.itemconfig(win, width=e.width))
        cvs.bind("<MouseWheel>", lambda e: cvs.yview_scroll(int(-1*(e.delta/120)), "units"))

        p = tk.Frame(wrap, bg=BG2)
        p.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        _lbl(p, "EJERCICIOS DE REFERENCIA (TP)", fg=YELLOW,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        for ej in EJERCICIOS_REF:
            self._ej_btn(p, ej)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=10)

        _lbl(p, "DIMENSION", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._dim_var = tk.StringVar(value="1D")
        for v, t in [("1D","1D  —  una variable"),("2D","2D  —  integral doble")]:
            tk.Radiobutton(p, text=t, variable=self._dim_var, value=v,
                           bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, font=("Segoe UI", 11),
                           command=self._toggle_2d).pack(anchor="w", pady=1)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

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

        _lbl(p, "MUESTRAS", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self._en = _labeled_entry(p, "n  — cantidad de muestras", "10000")
        _lbl(p, "Semilla aleatoria  (vacio = aleatorio)").pack(anchor="w")
        self._esemilla = _entry(p, "42")
        self._esemilla.pack(fill=tk.X, ipady=6, pady=(2, 8))
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

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
        _lbl(p, "O ingresa nivel personalizado (%):").pack(anchor="w")
        manual_row = tk.Frame(p, bg=BG2)
        manual_row.pack(fill=tk.X, pady=(2, 6))
        self._e_nivel_manual = _entry(manual_row, "")
        self._e_nivel_manual.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        tk.Label(manual_row, text=" %", bg=BG2, fg=MUTED, font=("Consolas", 12)).pack(side=tk.LEFT)

        zrow = tk.Frame(p, bg=BG2)
        zrow.pack(fill=tk.X, pady=(0, 8))
        tk.Label(zrow, text="z calculado = ", bg=BG2, fg=MUTED,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        self._lbl_z = tk.Label(zrow, text="1.9600", bg=BG2, fg=GREEN,
                               font=("Consolas", 13, "bold"))
        self._lbl_z.pack(side=tk.LEFT)

        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(p, "  CALCULAR  ", self._calcular, ACCENT).pack(fill=tk.X, pady=4)

    def _ej_btn(self, parent, ej):
        f = tk.Frame(parent, bg=BG3, cursor="hand2")
        f.pack(fill=tk.X, pady=3)
        tk.Label(f, text=ej["nombre"], bg=BG3, fg=ACCENT,
                 font=("Consolas", 10, "bold"), anchor="w", padx=8, pady=4).pack(fill=tk.X)
        tk.Label(f, text=ej["descripcion"], bg=BG3, fg=MUTED,
                 font=("Consolas", 9), anchor="w", justify="left",
                 padx=10, pady=2).pack(fill=tk.X)
        for w in [f] + list(f.winfo_children()):
            w.bind("<Button-1>", lambda e, d=ej: self._cargar_ej(d))
            w.bind("<Enter>", lambda e, fr=f: fr.config(bg=BG))
            w.bind("<Leave>", lambda e, fr=f: fr.config(bg=BG3))

    def _cargar_ej(self, ej):
        for widget, val in [(self._ef, ej["fexpr"]), (self._ea, ej["a"]),
                            (self._eb, ej["b"]),  (self._ec, ej["c"]),
                            (self._ed, ej["d"]),  (self._en, ej["n"]),
                            (self._e_nivel_manual, "")]:
            widget.delete(0, tk.END); widget.insert(0, val)
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

    def _main_area(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tbar = tk.Frame(right, bg=BG2, height=44)
        tbar.pack(fill=tk.X)
        tbar.pack_propagate(False)
        self._tbtns = {}; self._tframes = {}
        for name, icon in self.TABS:
            b = tk.Label(tbar, text=f"{icon} {name}", bg=BG2, fg=MUTED,
                         font=("Segoe UI", 12), padx=16, pady=12, cursor="hand2")
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

    def _show_tab(self, name):
        for n, b in self._tbtns.items():
            b.config(fg=TEXT if n == name else MUTED)
        for n, f in self._tframes.items():
            if n == name: f.pack(fill=tk.BOTH, expand=True)
            else:         f.pack_forget()

    def _new_panel(self, name):
        f = tk.Frame(self._panels, bg=BG)
        self._tframes[name] = f
        return f

    def _build_tab_resultado(self):
        self._si_res  = _scrollable_frame(self._new_panel("Resultado"))
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
    def _resolve_z(self):
        manual = self._e_nivel_manual.get().strip()
        if manual:
            try:
                nivel_num = float(manual.replace("%", ""))
            except ValueError:
                raise ValueError(f"Nivel manual invalido: {manual!r}")
            if not (50 < nivel_num < 100):
                raise ValueError("El nivel debe estar entre 50% y 100%.")
            nivel_str = f"{nivel_num:.2f}%"
            alpha = 1.0 - nivel_num / 100.0
            p     = 1.0 - alpha / 2.0
            t  = math.sqrt(-2.0 * math.log(1.0 - p))
            c  = (2.515517, 0.802853, 0.010328)
            d  = (1.432788, 0.189269, 0.001308)
            z  = t - (c[0]+c[1]*t+c[2]*t**2)/(1.0+d[0]*t+d[1]*t**2+d[2]*t**3)
        else:
            nivel_str = self._nivel_var.get()
            z         = _NIVELES_Z[nivel_str]
        self._lbl_z.config(text=f"{z:.4f}")
        return nivel_str, z

    def _calcular(self):
        try:
            fexpr = self._ef.get().strip()
            if not fexpr: raise ValueError("Ingresa la funcion f(x).")
            a   = _parse_float(self._ea.get(), "a")
            b   = _parse_float(self._eb.get(), "b")
            n   = int(_parse_float(self._en.get(), "n"))
            if n < 10: raise ValueError("n debe ser al menos 10.")
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

    # ══════════════════════════════════════════════════
    # RENDER: RESULTADO
    # ══════════════════════════════════════════════════
    def _render_resultado(self, r):
        si = self._si_res
        for w in si.winfo_children(): w.destroy()

        # ── Encabezado ────────────────────────────────
        _seccion(si, "RESULTADO  —  Estimacion de la integral", GREEN)
        c = _card(si, GREEN)
        if r["dim"] == "1D":
            _cformula(c, f"I  =  ∫ de {r['a']:.4g} a {r['b']:.4g}  de  f(x) = {r['fexpr']}  dx", ACCENT)
            _cformula(c, f"vol  =  b - a  =  {r['b']:.4g} - {r['a']:.4g}  =  {r['vol']:.6g}", MUTED)
            if r.get("I_hitormiss") is not None:
                _cigual(c, "I_hat  [metodo promedio]", f"{r['I_hat']:.8f}", GREEN)
                _cigual(c, "I_hat  [hit-or-miss]",     f"{r['I_hitormiss']:.8f}", TEAL)
                if r.get("gauss_val") is not None:
                    _cigual(c, "Gauss-Legendre (ref.)", f"{r['gauss_val']:.8f}", PURPLE)
                _gap(c, 4)
        else:
            _cformula(c, f"I  =  ∫∫  f(x,y) = {r['fexpr']}  dA", ACCENT)
            _cformula(c, f"x ∈ [{r['a']:.4g}, {r['b']:.4g}]   y ∈ [{r['c']}, {r['d']}]", ACCENT)
            if r.get("limites_variables"):
                _cformula(c, "Limites de y variables → formula con pesos", YELLOW)
            _cformula(c, f"vol  =  {r['vol']:.6g}", MUTED)
        _cbox(c, f"I_hat  =  {r['I_hat']:.8f}", GREEN)
        _gap(c)

        if r.get("I_analitica") is not None:
            ia  = r["I_analitica"]
            err = abs(r["I_hat"] - ia)
            pct = abs(err / ia * 100) if ia != 0 else 0.0
            _cigual(c, "Valor exacto  (SymPy)", f"{ia:.8f}", TEAL)
            _cigual(c, "Error absoluto",        f"{err:.3e}", YELLOW)
            _cigual(c, "Error relativo",        f"{pct:.4f} %", YELLOW)
            _gap(c)

        # ── Estadisticas — con explicaciones claras ───
        _seccion(si, "ESTADISTICAS  —  ¿Que mide cada valor?", PURPLE)
        c = _card(si, PURPLE)

        _ctitulo(c, "f_bar  =  media de los valores f(x_i)", ACCENT)
        _cformula(c, "f_bar  =  (1/n) × Σ f(x_i)", MUTED)
        _cformula(c, "Es el promedio de evaluar f en los n puntos aleatorios.", MUTED)
        _cigual(c, "f_bar", f"{r['f_bar']:.8f}", ACCENT)
        _gap(c, 4)
        _csep(c)

        _ctitulo(c, "sigma_f  =  desviacion estandar de los VALORES f(x_i)", PURPLE)
        _cformula(c, "sigma_f  =  sqrt[ 1/(n-1) × Σ (f(xi) - f_bar)² ]   [ddof=1]", MUTED)
        _cformula(c, "Mide CUANTO VARIA f entre los puntos muestreados.", MUTED)
        _cformula(c, "NO es el error de la integral — es la variabilidad de f.", MUTED)
        _cigual(c, "sigma_f", f"{r['sigma_f']:.8f}", PURPLE)
        _gap(c, 4)
        _csep(c)

        _ctitulo(c, "EE  =  Error Estandar de la media", ORANGE)
        _cformula(c, "EE  =  sigma_f / sqrt(n)", MUTED)
        _cformula(c, "Mide el error en la ESTIMACION DE f_bar (la media).", MUTED)
        _cformula(c, f"EE  =  {r['sigma_f']:.6f} / sqrt({r['n']:,})", MUTED)
        _cformula(c, f"EE  =  {r['sigma_f']:.6f} / {math.sqrt(r['n']):.4f}", MUTED)
        _cigual(c, "EE", f"{r['EE']:.8f}", ORANGE)
        _gap(c, 4)
        _csep(c)

        _ctitulo(c, "sigma_I  =  desviacion estandar de la INTEGRAL estimada", GREEN)
        _cformula(c, "sigma_I  =  vol × sigma_f / sqrt(n)  =  vol × EE", MUTED)
        _cformula(c, "ESTE sí es el error de la integral. Es la incerteza en I_hat.", MUTED)
        _cformula(c, "Se obtiene al propagar EE por el factor (b-a).", MUTED)
        _cformula(c, f"sigma_I  =  {r['vol']:.6g} × {r['sigma_f']:.6f} / sqrt({r['n']:,})", MUTED)
        _cformula(c, f"sigma_I  =  {r['vol']:.6g} × {r['EE']:.8f}", MUTED)
        _cigual(c, "sigma_I", f"{r['sigma_I']:.8f}", GREEN)
        _gap(c)

        # ── IC con z ──────────────────────────────────
        _seccion(si,
                 f"INTERVALO DE CONFIANZA al {r['nivel']}  "
                 f"(z = {r['z']:.4f})",
                 TEAL)
        c = _card(si, TEAL)
        _ctitulo(c,
                 "IC  =  I_hat  ±  z × sigma_I   [usando distribucion Normal]",
                 TEAL)
        _cformula(c, "El IC indica el rango donde cae el valor real de la integral", MUTED)
        _cformula(c, f"con una confianza del {r['nivel']}.", MUTED)
        _csep(c)
        _cformula(c, "IC  =  I_hat  ±  z × sigma_I")
        _cformula(c,
            f"   =  {r['I_hat']:.6f}  ±  {r['z']:.4f} × {r['sigma_I']:.6f}")
        _cformula(c,
            f"   =  {r['I_hat']:.6f}  ±  {r['margen']:.6f}")
        _cbox(c,
              f"IC {r['nivel']}:   "
              f"[ {r['ic_lo']:.6f}  ,  {r['ic_hi']:.6f} ]   "
              f"  margen = ± {r['margen']:.6f}",
              TEAL)
        if r.get("I_analitica") is not None:
            dentro = r["ic_lo"] <= r["I_analitica"] <= r["ic_hi"]
            _cbox(c,
                  ("✔  Valor exacto DENTRO del IC" if dentro
                   else "✘  Valor exacto FUERA del IC — aumentar n"),
                  GREEN if dentro else YELLOW)
        _gap(c)

        # ── IC con t de Student ────────────────────────
        if r["n"] > 1:
            _seccion(si,
                     f"INTERVALO DE CONFIANZA con t de Student al {r['nivel']}",
                     ORANGE)
            c = _card(si, ORANGE)
            conf_num = float(r["nivel"].replace("%","")) / 100.0
            t_val    = float(stats.t.ppf(0.5 + conf_num / 2.0, r["n"] - 1))
            ic_t_lo  = r["I_hat"] - t_val * r["sigma_I"]
            ic_t_hi  = r["I_hat"] + t_val * r["sigma_I"]
            _ctitulo(c, "IC_t  =  I_hat  ±  t(α/2, n-1) × sigma_I", ORANGE)
            _cformula(c, "La distribucion t tiene colas mas pesadas que la Normal.", MUTED)
            _cformula(c, "Es mas conservadora; converge a z cuando n → ∞.", MUTED)
            _csep(c)
            _cigual(c,
                    f"t({r['nivel']}, gl = n-1 = {r['n']-1})",
                    f"{t_val:.4f}  (vs z = {r['z']:.4f})", ORANGE)
            _cformula(c, f"IC_t  =  {r['I_hat']:.6f}  ±  {t_val:.4f} × {r['sigma_I']:.6f}")
            _cbox(c,
                  f"IC_t {r['nivel']}:   "
                  f"[ {ic_t_lo:.6f}  ,  {ic_t_hi:.6f} ]",
                  ORANGE)
            _cformula(c,
                f"Con n = {r['n']:,}:  z = {r['z']:.4f}  vs  t = {t_val:.4f}  "
                f"→  diferencia = {abs(t_val - r['z']):.6f}",
                MUTED)
            _cformula(c, "Para n grande (≥ 30) t ≈ z y ambos IC son practicamente iguales.", MUTED)
            _gap(c)

    # ══════════════════════════════════════════════════
    # RENDER: PASO A PASO  (con integrales)
    # ══════════════════════════════════════════════════
    def _render_pasos(self, r):
        si = self._si_paso
        for w in si.winfo_children(): w.destroy()
        dim = r["dim"]

        # ── BLOQUE 0: Notacion y glosario ─────────────
        _seccion(si, "GLOSARIO — ¿Que significa cada simbolo?", YELLOW)
        c = _card(si, YELLOW)
        _ctitulo(c, "Variables y simbolos del metodo", YELLOW)
        defs = [
            ("n",        "cantidad de muestras aleatorias generadas"),
            ("x_i",      "muestra aleatoria i-esima,  x_i ~ U(a, b)"),
            ("f(x_i)",   "valor de la funcion evaluada en x_i"),
            ("f_bar",    "media de todos los f(x_i)  =  (1/n) Σ f(x_i)"),
            ("vol",      "longitud del intervalo  =  b - a  (en 1D)"),
            ("I_hat",    "estimacion de la integral  =  vol × f_bar"),
            ("sigma_f",  "desv. std de los VALORES f(x_i)  — variabilidad de f"),
            ("EE",       "error estandar de la media  =  sigma_f / sqrt(n)"),
            ("sigma_I",  "desv. std de la INTEGRAL  =  vol × EE"),
            ("z",        "cuantil normal segun el nivel de confianza elegido"),
            ("IC",       "intervalo de confianza  =  I_hat ± z × sigma_I"),
        ]
        for sym, desc in defs:
            row = tk.Frame(c, bg=BG2)
            row.pack(anchor="w", padx=18, pady=1)
            tk.Label(row, text=f"  {sym:<10}", bg=BG2, fg=YELLOW,
                     font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
            tk.Label(row, text=f"→  {desc}", bg=BG2, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
        _gap(c)

        # ── PASO 1: Plantear la integral ──────────────
        _seccion(si, "PASO 1 — Plantear la integral a resolver", ACCENT)
        c = _card(si, ACCENT)
        if dim == "1D":
            _cformula(c, f"I  =  ∫ de {r['a']:.4g} a {r['b']:.4g}  f(x) dx", ACCENT)
            _cformula(c, f"f(x)  =  {r['fexpr']}", ACCENT)
            _cformula(c,
                f"vol  =  b - a  =  {r['b']:.4g} - {r['a']:.4g}  =  {r['vol']:.6g}",
                GREEN)
        else:
            _cformula(c,
                f"I  =  ∫∫  f(x,y) dA   con  f(x,y) = {r['fexpr']}", ACCENT)
            _cformula(c,
                f"x ∈ [{r['a']:.4g}, {r['b']:.4g}]   "
                f"y ∈ [{r['c']}, {r['d']}]", ACCENT)
            if r.get("limites_variables"):
                _cformula(c,
                    "Limites de y dependen de x → se usa formula con pesos", YELLOW)
            else:
                _cformula(c,
                    f"area  =  (b-a) × (d-c)  =  {r['vol']:.6g}", GREEN)

        # ── PASO 1b: Calculo analitico (solo 1D) ──────
        if dim == "1D" and r.get("I_analitica") is not None:
            _seccion(si, "PASO 1b — Calculo analitico de la integral (valor exacto)", GREEN)
            # Card con desarrollo de la integral usando el motor paso a paso
            _render_integral_card(
                si,
                integrand_str=r["fexpr"],
                var_sym=_x_sym,
                titulo=f"Calcular  ∫ f(x) dx  =  ∫ {r['fexpr']} dx  (primitiva)",
                borde_color=GREEN,
            )
            # Evaluacion en los limites
            c_lim = _card(si, TEAL)
            _ctitulo(c_lim, "Evaluar la primitiva en los limites [a, b]", TEAL)
            try:
                fstr  = r["fexpr"].replace("log(", "ln(")
                f_sym = sp.sympify(fstr)
                F_sym = sp.integrate(f_sym, _x_sym)
                F_b   = float(F_sym.subs(_x_sym, r["b"]).evalf())
                F_a   = float(F_sym.subs(_x_sym, r["a"]).evalf())
                _cformula(c_lim, f"F(x)  =  {F_sym}", TEAL)
                _cformula(c_lim,
                    f"I  =  F(b) - F(a)  =  F({r['b']:.4g}) - F({r['a']:.4g})",
                    MUTED)
                _cformula(c_lim,
                    f"I  =  {F_b:.8f} - {F_a:.8f}", MUTED)
                _cbox(c_lim,
                      f"I exacta  =  {r['I_analitica']:.8f}", TEAL)
            except Exception:
                _cbox(c_lim, f"I exacta  =  {r['I_analitica']:.8f}", TEAL)
            _gap(c_lim)

        # ── PASO 2: Generar muestras ───────────────────
        _seccion(si, "PASO 2 — Generar n muestras uniformes aleatorias", PURPLE)
        c = _card(si, PURPLE)
        _cformula(c,
            "Se usa sp.lambdify() para evaluar f de forma vectorizada con NumPy.",
            TEAL)
        if dim == "1D":
            _cformula(c,
                f"x_i  ~  U({r['a']:.4g}, {r['b']:.4g})"
                f"   para  i = 1, 2, ..., {r['n']:,}", PURPLE)
        else:
            _cformula(c,
                f"x_i ~ U({r['a']:.4g}, {r['b']:.4g})   "
                f"y_i ~ U({r['c']}, {r['d']})",
                PURPLE)
        _cformula(c, f"Total generadas: n = {r['n']:,}", MUTED)
        _gap(c, 4)
        _ctitulo(c, "Primeras 5 muestras (ejemplo):", MUTED)
        for k in range(min(5, r["n"])):
            xi, fi = r["xs"][k], r["fvals"][k]
            if dim == "2D":
                yi = r["ys"][k]
                _cformula(c,
                    f"i={k+1}:  x={xi:.5f}  y={yi:.5f}  "
                    f"→  f(x,y) = {fi:.8f}", MUTED)
            else:
                _cformula(c,
                    f"i={k+1}:  x={xi:.8f}   →   f(x) = {fi:.8f}", MUTED)
        if r["n"] > 5:
            _cformula(c, f"       ... ({r['n']-5:,} muestras mas)", MUTED)

        # ── PASO 2b: Hit-or-miss (solo 1D) ────────────
        if dim == "1D" and r.get("I_hitormiss") is not None:
            _seccion(si, "PASO 2b — Metodo Hit-or-Miss (puntos de exito)", TEAL)
            c = _card(si, TEAL)
            _ctitulo(c, "Idea: lanzar puntos aleatorios en un rectangulo contenedor", TEAL)
            _cformula(c,
                "Se genera un rectangulo que contiene toda la curva de f(x).", MUTED)
            _cformula(c,
                "Se cuenta cuantos puntos caen DEBAJO de la curva (exitos).", MUTED)
            _cformula(c,
                "La proporcion de exitos × area del rectangulo = integral.", MUTED)
            _csep(c)
            _cformula(c,
                f"Rectangulo: [{r['a']:.4g}, {r['b']:.4g}] × [{r['y_min']:.6f}, {r['y_max']:.6f}]",
                TEAL)
            rect = (r["b"] - r["a"]) * (r["y_max"] - r["y_min"])
            exitos = int(np.sum(r["success_mask"]))
            _cformula(c,
                f"rect_area  =  (b-a) × (y_max - y_min)  =  {rect:.6f}", MUTED)
            _cformula(c, f"Exitos     =  {exitos:,}   de   n = {r['n']:,}  puntos", MUTED)
            _cformula(c, "I_hm  =  (exitos / n)  ×  rect_area")
            _cformula(c,
                f"      =  ({exitos:,} / {r['n']:,})  ×  {rect:.6f}")
            _cbox(c, f"I_hm  =  {r['I_hitormiss']:.8f}", TEAL)
            if r.get("gauss_val") is not None:
                _csep(c)
                _cformula(c,
                    f"Gauss-Legendre (5 puntos, referencia exacta) = {r['gauss_val']:.8f}",
                    PURPLE)

        # ── PASO 3: Calcular f_bar ─────────────────────
        _seccion(si, "PASO 3 — Calcular f_bar  (media de los f(x_i))", GREEN)
        c = _card(si, GREEN)
        _cformula(c,
            "f_bar  =  (1/n) × Σ f(x_i)   [promedio de los valores de f]")
        _cformula(c,
            f"      =  {np.sum(r['fvals']):.6f}  /  {r['n']:,}")
        _cbox(c, f"f_bar  =  {r['f_bar']:.8f}", GREEN)

        # ── PASO 4: I_hat ──────────────────────────────
        _seccion(si, "PASO 4 — Calcular I_hat  (estimacion de la integral)", ACCENT)
        c = _card(si, ACCENT)
        if dim == "2D" and not r.get("limites_variables"):
            _cformula(c, "I_hat  =  area × mean(f(xi, yi))")
            _cformula(c, "       =  area × f_bar")
        else:
            _cformula(c, "I_hat  =  vol × f_bar")
        _cformula(c,
            f"       =  {r['vol']:.6g}  ×  {r['f_bar']:.8f}")
        _cbox(c, f"I_hat  =  {r['I_hat']:.8f}", ACCENT)
        _gap(c, 4)

        # ── PASO 5: sigma_f ────────────────────────────
        _seccion(si,
                 "PASO 5 — Calcular sigma_f  (desviacion estandar de los valores f(x_i))",
                 ORANGE)
        c = _card(si, ORANGE)
        _ctitulo(c, "¿Para que sirve sigma_f?", ORANGE)
        _cformula(c,
            "sigma_f mide CUANTO VARIA f entre los puntos muestreados.", MUTED)
        _cformula(c,
            "Si f es constante → sigma_f = 0 → integral exacta con n=1.", MUTED)
        _cformula(c,
            "Si f cambia mucho → sigma_f grande → necesitamos mas muestras.", MUTED)
        _csep(c)
        _cformula(c,
            "sigma_f  =  sqrt[ (1/(n-1)) × Σ (f(xi) - f_bar)² ]")
        _cformula(c,
            "                           ^^^ ddof=1 (corrección de Bessel)", TEAL)
        _cbox(c, f"sigma_f  =  {r['sigma_f']:.8f}", ORANGE)

        # ── PASO 6: EE y sigma_I ──────────────────────
        _seccion(si,
                 "PASO 6 — Calcular EE y sigma_I  (error de la integral estimada)",
                 YELLOW)
        c = _card(si, YELLOW)
        _ctitulo(c, "Paso 6a — Error Estandar de la media (EE)", YELLOW)
        _cformula(c, "EE  =  sigma_f / sqrt(n)", MUTED)
        _cformula(c, "Mide el error en f_bar. Al aumentar n, EE disminuye.", MUTED)
        _cformula(c,
            f"EE  =  {r['sigma_f']:.6f}  /  sqrt({r['n']:,})")
        _cformula(c,
            f"EE  =  {r['sigma_f']:.6f}  /  {math.sqrt(r['n']):.4f}")
        _cigual(c, "EE", f"{r['EE']:.8f}", YELLOW)
        _csep(c)
        _ctitulo(c, "Paso 6b — sigma_I: desviacion estandar de la INTEGRAL", GREEN)
        _cformula(c, "sigma_I  =  vol × EE  =  vol × sigma_f / sqrt(n)", MUTED)
        _cformula(c,
            "Es la incerteza real sobre I_hat. El IC se construye con este valor.", MUTED)
        _cformula(c,
            f"sigma_I  =  {r['vol']:.6g}  ×  {r['EE']:.8f}")
        _cbox(c,
              f"sigma_I  =  {r['sigma_I']:.8f}  ← error de la integral",
              GREEN)

        # ── PASO 7: IC con z ───────────────────────────
        _seccion(si,
                 f"PASO 7 — Construir el IC al {r['nivel']}  (con z normal)",
                 TEAL)
        c = _card(si, TEAL)
        _ctitulo(c,
            f"IC  =  I_hat  ±  z × sigma_I   [nivel {r['nivel']}]",
            TEAL)
        _cformula(c,
            "El IC dice: 'con probabilidad del nivel elegido, la integral real", MUTED)
        _cformula(c,
            "esta dentro de este intervalo.'", MUTED)
        _csep(c)
        _cformula(c, f"Nivel elegido: {r['nivel']}   →   z = {r['z']:.4f}")
        _cformula(c, "IC  =  I_hat  ±  z × sigma_I")
        _cformula(c,
            f"   =  {r['I_hat']:.6f}  ±  {r['z']:.4f} × {r['sigma_I']:.6f}")
        _cformula(c,
            f"   =  {r['I_hat']:.6f}  ±  {r['margen']:.6f}")
        _cbox(c,
              f"IC {r['nivel']}:  "
              f"[ {r['ic_lo']:.6f} ,  {r['ic_hi']:.6f} ]",
              TEAL)

        # ── PASO 7b: IC con t de Student ───────────────
        if r["n"] > 1:
            _seccion(si,
                     f"PASO 7b — IC con t de Student al {r['nivel']}",
                     ORANGE)
            c = _card(si, ORANGE)
            _ctitulo(c, "IC_t  =  I_hat  ±  t(α/2, n-1) × sigma_I", ORANGE)
            _cformula(c,
                "La distribucion t de Student tiene en cuenta que estimamos sigma", MUTED)
            _cformula(c,
                "desde los propios datos (no conocemos sigma_f teorico).", MUTED)
            _cformula(c,
                "Con n grande (≥ 30) la diferencia con el IC normal es minima.", MUTED)
            _csep(c)
            conf_num = float(r["nivel"].replace("%","")) / 100.0
            t_val    = float(stats.t.ppf(0.5 + conf_num / 2.0, r["n"] - 1))
            ic_t_lo  = r["I_hat"] - t_val * r["sigma_I"]
            ic_t_hi  = r["I_hat"] + t_val * r["sigma_I"]
            _cigual(c,
                    f"t({r['nivel']}, gl = n-1 = {r['n']-1})",
                    f"{t_val:.6f}", ORANGE)
            _cigual(c, f"z normal ({r['nivel']})", f"{r['z']:.6f}", TEAL)
            _cformula(c,
                f"Diferencia t - z  =  {abs(t_val - r['z']):.6f}  "
                f"({'pequeña → n suficiente' if abs(t_val - r['z']) < 0.01 else 'notable → aumentar n'})",
                MUTED)
            _cformula(c, "IC_t = I_hat ± t × sigma_I")
            _cformula(c,
                f"     = {r['I_hat']:.6f} ± {t_val:.4f} × {r['sigma_I']:.6f}")
            _cbox(c,
                  f"IC_t {r['nivel']}:  "
                  f"[ {ic_t_lo:.6f} ,  {ic_t_hi:.6f} ]",
                  ORANGE)

        # ── PASO 8: Verificacion (solo si hay exacto) ──
        if r.get("I_analitica") is not None:
            _seccion(si, "PASO 8 — Verificacion con valor exacto", GREEN)
            c = _card(si, GREEN)
            ia  = r["I_analitica"]
            err = abs(r["I_hat"] - ia)
            _cigual(c, "I exacta (SymPy)", f"{ia:.8f}", TEAL)
            _cigual(c, "I estimada (MC)", f"{r['I_hat']:.8f}", GREEN)
            _cigual(c, "Error absoluto",  f"{err:.3e}", YELLOW)
            if ia != 0:
                _cigual(c, "Error relativo",
                        f"{abs(err/ia)*100:.4f} %", YELLOW)
            dentro = r["ic_lo"] <= ia <= r["ic_hi"]
            _cbox(c,
                  ("✔  Valor exacto DENTRO del IC  →  simulacion correcta" if dentro
                   else "✘  Valor exacto FUERA del IC  →  aumentar n"),
                  GREEN if dentro else YELLOW)
            _gap(c)

    # ══════════════════════════════════════════════════
    # RENDER: CONVERGENCIA
    # ══════════════════════════════════════════════════
    def _render_convergencia(self, r):
        ax = self._ax_conv
        ax.clear(); _style_ax(ax)
        try:
            s_str   = self._esemilla.get().strip()
            semilla = int(s_str) if s_str else None
            ns, I_vals, std_vals = _convergencia(
                r["fexpr"], r["a"], r["b"], r["n"], semilla)
            ax.plot(ns, I_vals, color=ACCENT, linewidth=1.5,
                    label="MC promedio acumulado")
            ax.fill_between(ns, I_vals - std_vals, I_vals + std_vals,
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
            ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error: {exc}",
                    transform=ax.transAxes, color=RED, ha="center", va="center")
        self._cvs_conv.draw()

    # ══════════════════════════════════════════════════
    # RENDER: DISTRIBUCION
    # ══════════════════════════════════════════════════
    def _render_distribucion(self, r):
        ax = self._ax_dist
        ax.clear(); _style_ax(ax)
        vol    = r["vol"]
        data_v = r["fvals"] * vol
        media  = float(np.mean(data_v))
        std_v  = float(np.std(data_v, ddof=1))
        n_bins = min(40, max(5, r["n"] // 50))
        ax.hist(data_v, bins=n_bins, color=PURPLE, alpha=0.75,
                edgecolor=BG, density=True, label="f(x_i) × vol")
        if std_v > 0:
            x_norm = np.linspace(data_v.min(), data_v.max(), 300)
            y_norm = stats.norm.pdf(x_norm, media, std_v)
            ax.plot(x_norm, y_norm, color=ORANGE, linewidth=2,
                    label="Distribucion Normal ajustada")
        ax.axvline(media, color=YELLOW, linewidth=2, label=f"Media = {media:.4f}")
        ax.axvline(media + std_v, color=GREEN, linewidth=1.2, linestyle="--",
                   label="Media ±1 std")
        ax.axvline(media - std_v, color=GREEN, linewidth=1.2, linestyle="--")
        ax.set_xlabel("f(x_i) × (b-a)", color=MUTED, fontsize=9)
        ax.set_ylabel("Densidad", color=MUTED, fontsize=9)
        ax.set_title(
            f"Distribucion muestral  |  sigma_f = {r['sigma_f']:.4f}   n = {r['n']:,}",
            color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        self._cvs_dist.draw()

    # ══════════════════════════════════════════════════
    # RENDER: DISPERSION
    # ══════════════════════════════════════════════════
    def _render_dispersion(self, r):
        ax = self._ax_disp
        ax.clear(); _style_ax(ax)
        N_plot = min(2000, len(r["xs"]))
        idx    = np.random.choice(len(r["xs"]), N_plot, replace=False)

        if r["dim"] == "1D":
            ax.scatter(r["xs"][idx], r["fvals"][idx],
                       color=ACCENT, s=4, alpha=0.5,
                       label=f"{N_plot:,} / {len(r['xs']):,} puntos")
            try:
                f     = _make_lambdify_1d(r["fexpr"])
                x_plt = np.linspace(r["a"], r["b"], 500)
                y_plt = np.nan_to_num(f(x_plt))
                ax.plot(x_plt, y_plt, color=GREEN, linewidth=1.5,
                        label=f"f(x) = {r['fexpr']}", zorder=3)
                ax.fill_between(x_plt, 0, y_plt, color=GREEN, alpha=0.07)
                if "success_mask" in r:
                    sm   = r["success_mask"][idx]
                    ys_h = r["ys_hom"][idx]
                    ax.scatter(r["xs"][idx][~sm], ys_h[~sm],
                               color=RED,  s=3, alpha=0.3, label="H-o-M fallidos")
                    ax.scatter(r["xs"][idx][sm],  ys_h[sm],
                               color=TEAL, s=3, alpha=0.3, label="H-o-M exitos")
            except Exception:
                pass
        else:
            sc = ax.scatter(r["xs"][idx], r["ys"][idx],
                            c=r["fvals"][idx], cmap="viridis",
                            s=5, alpha=0.6, label=f"{N_plot:,} puntos")
            self._fig_disp.colorbar(sc, ax=ax, label="f(x,y)")

        ax.axhline(r["f_bar"], color=YELLOW, linewidth=1.8,
                   linestyle="--", label=f"f_bar = {r['f_bar']:.4f}")
        ax.set_xlabel("x_i", color=MUTED, fontsize=9)
        ax.set_ylabel("f(x_i)" if r["dim"] == "1D" else "y_i",
                      color=MUTED, fontsize=9)
        ax.set_title(f"Dispersion  |  f = {r['fexpr']}", color=TEXT, fontsize=10, pad=8)
        ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
        self._cvs_disp.draw()


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = MontecarloApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()