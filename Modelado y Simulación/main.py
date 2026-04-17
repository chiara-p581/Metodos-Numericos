"""
Comparador de Métodos Numéricos para EDOs de Primer Orden
==========================================================
Métodos implementados:
  • Euler (RK1)
  • RK2 — Heun (a=1) o Punto Medio (a=1/2)
  • RK4 Clásico

Resolución analítica con paso a paso detallado de integrales:
  • Integración directa (polinomios, exp, trig, log)
  • Sustitución u = g(x)
  • Integración por partes (regla LIATE, múltiples rondas)
  • Fracciones parciales
  • Linealidad (suma de términos)
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

COL_EULER  = "#f85149"
COL_RK2    = "#f0883e"
COL_RK4    = "#58a6ff"
COL_EXACTA = "#3fb950"


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE INTEGRACIÓN PASO A PASO
# ══════════════════════════════════════════════════════════════════════════════

def _is_linear_in(expr, var):
    """True si expr es polinomio de grado 1 en var."""
    try:
        return bool(expr.is_polynomial(var) and sp.Poly(expr, var).degree() == 1)
    except Exception:
        return False


def _liate_priority(f, var):
    """Prioridad LIATE (menor = más prioritario para u)."""
    if f.has(sp.log):                                                       return 0
    if any(f.has(fn) for fn in [sp.asin, sp.acos, sp.atan, sp.acot]):      return 1
    if f.is_polynomial(var):                                                return 2
    if any(f.has(fn) for fn in [sp.sin, sp.cos, sp.tan, sp.cot]):          return 3
    if f.has(sp.exp):                                                       return 4
    return 5


def detect_integration_method(expr, var):
    """
    Detecta el método de integración más apropiado.
    Retorna (method_name: str, meta: dict).
    """
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
                b = arg.subs(x, 0)
                return ('direct_exp_linear', {'a': a, 'b': b, 'expr': expr, 'arg': arg})
            return ('direct_exp', {'expr': expr})

        for fn in [sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc]:
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

    # Fracción racional
    try:
        numer, denom = sp.fraction(expr)
        if denom != 1 and denom.is_polynomial(x) and numer.is_polynomial(x):
            pf = sp.apart(expr, x)
            if pf != expr and pf.is_Add:
                return ('partial_fractions', {'numer': numer, 'denom': denom, 'pf_decomp': pf})
    except Exception:
        pass

    # Sustitución
    sub = _detect_substitution_method(expr, x)
    if sub:
        return ('substitution', sub)

    # Integración por partes
    if expr.is_Mul:
        factors = list(expr.as_ordered_factors())
        factors_sorted = sorted(factors, key=lambda f: _liate_priority(f, x))
        u_p  = factors_sorted[0]
        dv_p = sp.Mul(*factors_sorted[1:])
        du_p = sp.diff(u_p, x)
        v_p  = sp.integrate(dv_p, x)
        rem  = sp.expand(v_p * du_p)

        needs_second = (rem.is_Mul and rem != expr and rem != 0
                        and _liate_priority(
                            sorted(rem.as_ordered_factors(),
                                   key=lambda f: _liate_priority(f, x))[0], x) < 5)
        method = 'byparts_repeated' if needs_second else 'byparts'
        return (method, {'u': u_p, 'dv': dv_p, 'du': du_p, 'v': v_p, 'remaining': rem})

    if expr.is_Add:
        return ('sum_of_terms', {'terms': sp.Add.make_args(expr)})

    return ('general_sympy', {})


def _detect_substitution_method(expr, var):
    x = var
    candidates = []
    for atom in sp.preorder_traversal(expr):
        if atom in (var, expr, sp.Integer(1), sp.Integer(-1), sp.Integer(0)):
            continue
        if not isinstance(atom, sp.Expr):
            continue
        if x not in atom.free_symbols:
            continue
        if atom.is_Symbol:
            continue
        if (isinstance(atom, (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh))
                or (atom.is_Pow and not atom.is_polynomial(x))
                or (atom.is_Add and not _is_linear_in(atom, x))):
            candidates.append(atom)

    for u_cand in candidates:
        du = sp.diff(u_cand, x)
        if du == 0:
            continue
        try:
            ratio = sp.simplify(expr / du)
            if x not in ratio.free_symbols:
                return {'u': u_cand, 'du_dx': du, 'coeff': ratio}
        except Exception:
            pass

    try:
        numer, denom = sp.fraction(expr)
        if denom != 1 and x in denom.free_symbols:
            du = sp.diff(denom, x)
            try:
                ratio = sp.simplify(numer / du)
                if x not in ratio.free_symbols:
                    return {'u': denom, 'du_dx': du, 'coeff': ratio, 'type': 'fraction'}
            except Exception:
                pass
    except Exception:
        pass

    return None


def generate_integral_steps(integrand_str, var_sym):
    """
    Genera lista de pasos para mostrar en la UI.
    Retorna list de dicts: {'type': ..., ...}
    Types: 'method', 'formula', 'info', 'assign', 'calc', 'divider', 'result', 'error'
    """
    x = var_sym
    try:
        expr = sp.sympify(integrand_str.replace('^', '**'), locals={'x': x})
    except Exception as e:
        return [{'type': 'error', 'text': f'No se pudo parsear: {e}'}]

    method, meta = detect_integration_method(expr, x)
    steps = _dispatch_integral(method, expr, meta, x)

    try:
        result = sp.integrate(expr, x)
        steps.append({'type': 'result', 'text': f'∫ ({expr}) dx = {result} + C'})
    except Exception:
        pass

    return steps


def _dispatch_integral(method, expr, meta, x):
    dispatch = {
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
    return dispatch.get(method, _si_general)(expr, meta, x)


def _si_const(expr, meta, x):
    return [
        {'type': 'method', 'text': 'Integral directa — constante'},
        {'type': 'formula', 'text': '∫ k dx  =  k·x + C'},
        {'type': 'calc', 'text': f'∫ ({expr}) dx  =  ({expr})·x + C'},
    ]

def _si_poly(expr, meta, x):
    terms = meta['terms']
    steps = [
        {'type': 'method', 'text': 'Integral directa — polinomio'},
        {'type': 'formula', 'text': '∫ xⁿ dx  =  xⁿ⁺¹/(n+1) + C    (regla de la potencia)'},
    ]
    if len(terms) > 1:
        steps.append({'type': 'info', 'text': f'Integramos los {len(terms)} términos por separado (linealidad):'})
        for t in terms:
            r = sp.integrate(t, x)
            steps.append({'type': 'calc', 'text': f'  ∫ ({t}) dx  =  {r}'})
    result = sp.integrate(expr, x)
    steps.append({'type': 'calc', 'text': f'Resultado:  ∫ ({expr}) dx  =  {result} + C'})
    return steps

def _si_exp_linear(expr, meta, x):
    a, b, arg = meta['a'], meta['b'], meta['arg']
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Integral directa — exponencial con argumento lineal'},
        {'type': 'formula', 'text': '∫ e^(ax+b) dx  =  e^(ax+b) / a + C'},
        {'type': 'info', 'text': 'La derivada del argumento es la constante a, dividimos por ella.'},
        {'type': 'assign', 'label': 'Argumento', 'value': str(arg)},
        {'type': 'assign', 'label': 'a  (coeficiente de x)', 'value': str(a)},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {expr} / ({a})  =  {result} + C'},
    ]

def _si_exp(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Integral directa — función exponencial'},
        {'type': 'formula', 'text': '∫ eˣ dx  =  eˣ + C'},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_trig_linear(expr, meta, x):
    fn, arg, a = meta['fn'], meta['arg'], meta['a']
    fn_name = fn.__name__
    antideriv = {
        'sin': '∫ sin(ax+b) dx  =  −cos(ax+b)/a + C',
        'cos': '∫ cos(ax+b) dx  =  sin(ax+b)/a + C',
        'tan': '∫ tan(ax+b) dx  =  −ln|cos(ax+b)|/a + C',
    }
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': f'Integral directa — {fn_name}(ax+b)'},
        {'type': 'formula', 'text': antideriv.get(fn_name, f'∫ {fn_name}(ax+b) dx')},
        {'type': 'info', 'text': 'Por regla de la cadena inversa: dividimos por el coeficiente a.'},
        {'type': 'assign', 'label': 'Argumento', 'value': str(arg)},
        {'type': 'assign', 'label': 'a', 'value': str(a)},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_trig(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': f'Integral básica — {meta["fn"].__name__}(x)'},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_log(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Integral directa — logaritmo natural'},
        {'type': 'formula', 'text': '∫ ln(x) dx  =  x·ln(x) − x + C'},
        {'type': 'info', 'text': 'Verificación: d/dx[x·ln(x) − x] = ln(x) + 1 − 1 = ln(x) ✓'},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_pow_linear(expr, meta, x):
    base, exp_p, a = meta['base'], meta['exp'], meta['a']
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Sustitución implícita — potencia de función lineal'},
        {'type': 'formula', 'text': '∫ (ax+b)ⁿ dx  =  (ax+b)^(n+1) / [a·(n+1)] + C'},
        {'type': 'info', 'text': 'Derivada de (ax+b)^(n+1) = a·(n+1)·(ax+b)^n → dividimos por a·(n+1).'},
        {'type': 'assign', 'label': 'u = base', 'value': str(base)},
        {'type': 'assign', 'label': 'n', 'value': str(exp_p)},
        {'type': 'assign', 'label': 'a', 'value': str(a)},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_inv_linear(expr, meta, x):
    base, a = meta['base'], meta['a']
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Integral logarítmica — 1/(ax+b)'},
        {'type': 'formula', 'text': '∫ 1/(ax+b) dx  =  (1/a)·ln|ax+b| + C'},
        {'type': 'assign', 'label': 'u = ax+b', 'value': str(base)},
        {'type': 'assign', 'label': 'a', 'value': str(a)},
        {'type': 'calc', 'text': f'∫ {expr} dx  =  {result} + C'},
    ]

def _si_substitution(expr, meta, x):
    u_cand = meta['u']
    du_dx  = meta['du_dx']
    coeff  = meta.get('coeff', sp.Integer(1))
    u_sym  = sp.Symbol('u')
    result = sp.integrate(expr, x)
    steps = [
        {'type': 'method', 'text': 'Método de Sustitución  u = g(x)'},
        {'type': 'formula', 'text': 'Si  u = g(x)  →  du = g\'(x)·dx  →  dx = du/g\'(x)'},
        {'type': 'info', 'text': 'Elegimos u = g(x) de modo que g\'(x) aparezca en el integrando.'},
    ]
    steps.append({'type': 'assign', 'label': 'u', 'value': str(u_cand)})
    steps.append({'type': 'assign', 'label': 'du/dx', 'value': str(du_dx)})
    steps.append({'type': 'assign', 'label': 'du', 'value': f'({du_dx}) dx'})
    steps.append({'type': 'divider', 'text': '→ Despejamos dx y sustituimos en la integral:'})
    steps.append({'type': 'calc', 'text': f'dx  =  du / ({du_dx})'})
    if coeff != 1:
        steps.append({'type': 'calc',
                      'text': f'Al sustituir, el integrando queda:  {coeff} du'})
        int_in_u = sp.integrate(coeff, u_sym)
        steps.append({'type': 'calc', 'text': f'∫ {coeff} du  =  {int_in_u} + C'})
    else:
        steps.append({'type': 'calc', 'text': '∫ du  =  u + C'})
    steps.append({'type': 'divider', 'text': '→ Volvemos a la variable original x  (u = g(x)):'})
    steps.append({'type': 'calc', 'text': f'u = {u_cand}'})
    steps.append({'type': 'calc', 'text': f'∫ ({expr}) dx  =  {result} + C'})
    return steps

def _si_byparts(expr, meta, x, _depth=0):
    u_p  = meta['u']
    dv_p = meta['dv']
    du_p = meta['du']
    v_p  = meta['v']
    rem  = meta['remaining']
    uv   = sp.expand(u_p * v_p)
    ind  = '   ' * _depth

    steps = []
    if _depth == 0:
        steps += [
            {'type': 'method', 'text': 'Integración por Partes'},
            {'type': 'formula', 'text': '∫ u dv  =  u·v  −  ∫ v du'},
            {'type': 'info', 'text': 'Regla LIATE para elegir u:  Logarítmica > Inv.Trig > Algebraica > Trigonométrica > Exponencial'},
        ]
    else:
        steps.append({'type': 'info',
                      'text': f'{ind}▸ Ronda {_depth+1} — nueva integración por partes:'})

    steps.append({'type': 'assign', 'label': f'{ind}u',  'value': str(u_p)})
    steps.append({'type': 'assign', 'label': f'{ind}dv', 'value': f'{dv_p} dx'})
    steps.append({'type': 'divider', 'text': f'{ind}→ Derivamos u  e  integramos dv:'})
    steps.append({'type': 'assign', 'label': f'{ind}du', 'value': f'{du_p} dx'})
    steps.append({'type': 'assign', 'label': f'{ind}v',  'value': str(v_p)})
    steps.append({'type': 'divider', 'text': f'{ind}→ Aplicamos la fórmula ∫ u dv = u·v − ∫ v du:'})
    steps.append({'type': 'calc',
                  'text': f'{ind}∫ ({expr}) dx  =  ({u_p})·({v_p})  −  ∫ ({v_p})·({du_p}) dx'})
    steps.append({'type': 'calc',
                  'text': f'{ind}             =  {uv}  −  ∫ ({rem}) dx'})

    sub_method, sub_meta = detect_integration_method(rem, x)
    if sub_method in ('byparts', 'byparts_repeated') and _depth < 3:
        steps.append({'type': 'info',
                      'text': f'{ind}La integral ∫({rem})dx requiere otra ronda de integración por partes:'})
        steps += _si_byparts(rem, sub_meta, x, _depth + 1)
    else:
        rem_int = sp.integrate(rem, x)
        steps.append({'type': 'calc',
                      'text': f'{ind}∫ ({rem}) dx  =  {rem_int} + C'})

    if _depth == 0:
        result = sp.integrate(expr, x)
        steps.append({'type': 'divider', 'text': '→ Resultado final:'})
        steps.append({'type': 'calc',
                      'text': f'∫ ({expr}) dx  =  {sp.simplify(result)} + C'})
    return steps

def _si_partial_fractions(expr, meta, x):
    numer = meta['numer']
    denom = meta['denom']
    pf    = meta['pf_decomp']
    pf_terms = sp.Add.make_args(pf)
    steps = [
        {'type': 'method', 'text': 'Fracciones Parciales'},
        {'type': 'formula', 'text': 'P(x)/Q(x) se descompone en suma de fracciones simples'},
        {'type': 'info', 'text': 'Factorizamos Q(x) y planteamos la descomposición general,'},
        {'type': 'info', 'text': 'luego determinamos las constantes por igualación de coeficientes.'},
    ]
    steps.append({'type': 'assign', 'label': 'P(x)', 'value': str(numer)})
    steps.append({'type': 'assign', 'label': 'Q(x)', 'value': str(denom)})
    steps.append({'type': 'divider', 'text': '→ Descomposición en fracciones parciales:'})
    pf_str = '  +  '.join(str(t) for t in pf_terms)
    steps.append({'type': 'calc', 'text': f'{expr}  =  {pf_str}'})
    steps.append({'type': 'divider', 'text': '→ Integramos cada fracción por separado:'})
    for t in pf_terms:
        r = sp.integrate(t, x)
        steps.append({'type': 'calc', 'text': f'  ∫ ({t}) dx  =  {r}'})
    result = sp.integrate(expr, x)
    steps.append({'type': 'divider', 'text': '→ Resultado:'})
    steps.append({'type': 'calc', 'text': f'∫ ({expr}) dx  =  {sp.simplify(result)} + C'})
    return steps

def _si_sum(expr, meta, x):
    terms = meta['terms']
    method_labels = {
        'direct_poly': 'potencia', 'direct_exp_linear': 'exp lineal',
        'direct_trig_linear': 'trig lineal', 'direct_trig': 'trig básica',
        'substitution': 'sustitución', 'byparts': 'por partes',
        'direct_log': 'log', 'direct_const': 'constante',
    }
    steps = [
        {'type': 'method', 'text': 'Linealidad de la integral — suma de términos'},
        {'type': 'formula', 'text': '∫ [f(x)+g(x)] dx  =  ∫ f(x) dx  +  ∫ g(x) dx'},
        {'type': 'info', 'text': f'Integramos cada uno de los {len(terms)} términos:'},
    ]
    for t in terms:
        sub_m, _ = detect_integration_method(t, x)
        r = sp.integrate(t, x)
        lbl = method_labels.get(sub_m, sub_m)
        steps.append({'type': 'calc', 'text': f'  ∫ ({t}) dx  =  {r}   [{lbl}]'})
    result = sp.integrate(expr, x)
    steps.append({'type': 'calc', 'text': f'Total:  ∫ ({expr}) dx  =  {result} + C'})
    return steps

def _si_general(expr, meta, x):
    result = sp.integrate(expr, x)
    return [
        {'type': 'method', 'text': 'Resolución simbólica (SymPy — técnica avanzada)'},
        {'type': 'info', 'text': 'Esta integral requiere una combinación de métodos o técnicas avanzadas.'},
        {'type': 'calc', 'text': f'∫ ({expr}) dx  =  {result} + C'},
    ]


# ══════════════════════════════════════════════════════
# EVALUACIÓN SEGURA DE EDO
# ══════════════════════════════════════════════════════
def _make_env(x_val=None, y_val=None):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env.update({"np": np, "exp": math.exp, "sin": math.sin, "cos": math.cos,
                "log": math.log, "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
                "tan": math.tan, "sinh": math.sinh, "cosh": math.cosh,
                "abs": abs, "ln": math.log})
    if x_val is not None: 
        env["x"] = x_val
    if y_val is not None: env["y"] = y_val
    return env

def _evaluar_f(fexpr, x_val, y_val):
    try:
        val = eval(fexpr, {"__builtins__": {}}, _make_env(x_val, y_val))
        return float(val)
    except ZeroDivisionError:
        return 0.0
    except Exception as e:
        raise ValueError(f"Error evaluando f(x,y)='{fexpr}' en x={x_val:.4f}, y={y_val:.4f}: {e}")

def _parse_float(s, campo):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env.update({"pi": math.pi, "e": math.e})
    try:
        return float(eval(s.strip(), {"__builtins__": {}}, env))
    except Exception as exc:
        raise ValueError(f"Valor inválido en '{campo}': {s!r}  ({exc})")


# ══════════════════════════════════════════════════════
# LÓGICA NUMÉRICA EDO
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

def _solucion_exacta_sympy(fexpr, x0, y0, xs):
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

def _evaluar_exacta_manual(expr_str, xs):
    try:
        return np.array([
            float(eval(expr_str, {"__builtins__": {}}, _make_env(xi, None)))
            for xi in xs
        ], dtype=float)
    except Exception as e:
        raise ValueError(f"Error en solución exacta manual: {e}")


# ══════════════════════════════════════════════════════
# CLASIFICACIÓN ANALÍTICA
# ══════════════════════════════════════════════════════
def clasificar_edo(fexpr):
    x_sym = sp.Symbol('x')
    y_sym = sp.Symbol('ytmp')
    fstr  = fexpr.replace('^', '**')
    fstr_ysym = re.sub(r'(?<![a-zA-Z_])y(?![a-zA-Z_(])', 'ytmp', fstr)
    try:
        f_alg = sp.sympify(fstr_ysym, locals={'x': x_sym, 'ytmp': y_sym})
    except Exception:
        return 'general', None, None

    has_y = y_sym in f_alg.free_symbols
    if not has_y:
        return 'solo_x', None, None

    try:
        poly = sp.Poly(f_alg, y_sym)
        if poly.degree() == 1:
            A = poly.nth(1)
            B = poly.nth(0)
            if y_sym not in A.free_symbols and y_sym not in B.free_symbols:
                return 'linear', -A, B
    except Exception:
        pass

    try:
        fy1 = f_alg.subs(y_sym, 1)
        if fy1 != 0:
            ratio = sp.simplify(f_alg / fy1)
            if y_sym in ratio.free_symbols and x_sym not in ratio.free_symbols:
                return 'separable', fy1, ratio
        fx0 = f_alg.subs(x_sym, 0)
        if fx0 != 0:
            ratio2 = sp.simplify(f_alg / fx0)
            if y_sym not in ratio2.free_symbols:
                return 'separable', ratio2, fx0
    except Exception:
        pass

    return 'general', None, None


def resolver_edo_analitico(fexpr, x0, y0):
    x_sym = sp.Symbol('x')
    y_fn  = sp.Function('y')
    C1    = sp.Symbol('C1')

    fstr     = fexpr.replace('^', '**')
    fstr_sym = re.sub(r'(?<![a-zA-Z_])y(?![a-zA-Z_(])', 'y(x)', fstr)
    tipo, extra1, extra2 = clasificar_edo(fexpr)

    result = {
        "tipo": tipo, "fexpr": fexpr, "x0": x0, "y0": y0,
        "extra1": extra1, "extra2": extra2,
        "sol_general": None, "C1_val": None,
        "sol_particular": None, "sol_str": None, "error": None,
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

    if tipo == 'linear' and extra1 is not None:
        P_std, Q_std = extra1, extra2
        try:
            mu_exp_expr  = sp.integrate(P_std, x_sym)
            mu_expr      = sp.exp(mu_exp_expr)
            integral_muQ = sp.integrate(mu_expr * Q_std, x_sym)
            result["mu_exp"]      = mu_exp_expr
            result["mu"]          = mu_expr
            result["integral_muQ"] = integral_muQ
        except Exception:
            pass

    if tipo == 'separable' and extra1 is not None:
        g_x, h_y = extra1, extra2
        y_sym = sp.Symbol('y')
        try:
            integral_gx  = sp.integrate(g_x, x_sym)
            integral_1hy = sp.integrate(1/h_y.subs(sp.Symbol('ytmp'), y_sym), y_sym)
            result["integral_gx"]  = integral_gx
            result["integral_1hy"] = integral_1hy
            result["g_x"] = g_x
            result["h_y"] = h_y
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════
# HELPERS DE WIDGETS UI
# ══════════════════════════════════════════════════════
def _lbl(parent, text, fg=MUTED, font=("Consolas", 11), bg=None):
    return tk.Label(parent, text=text, bg=bg or BG2, fg=fg, font=font)

def _entry(parent, default, width=None):
    kw = dict(bg=BG3, fg=TEXT, insertbackground=TEXT,
              font=("Consolas", 12), bd=0, relief="flat",
              highlightthickness=1, highlightbackground=BORDER,
              highlightcolor=ACCENT)
    if width: kw["width"] = width
    e = tk.Entry(parent, **kw)
    e.insert(0, default)
    return e

def _labeled_entry(parent, label, default):
    _lbl(parent, label).pack(anchor="w")
    e = _entry(parent, default)
    e.pack(fill=tk.X, ipady=6, pady=(2, 8))
    return e

def _btn(parent, text, cmd, color=ACCENT, fg="#000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 11, "bold"), padx=12, pady=8, cursor="hand2")
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

def _seccion(parent, titulo, color=ACCENT):
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
             font=("Consolas", 12, "bold"), padx=10, pady=8, anchor="w").pack(fill=tk.X)

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
# WIDGET DE PASOS DE INTEGRAL (nuevo)
# ══════════════════════════════════════════════════════

# Colores para cada tipo de paso en la UI
_STEP_COLORS = {
    'method':  (TEAL,   TEAL,   True,  True),    # titulo, borde, negrita, subrayado
    'formula': (GREEN,  GREEN,  False, False),
    'info':    (MUTED,  MUTED,  False, False),
    'assign':  (TEXT,   YELLOW, False, False),   # label TEXT, value YELLOW
    'calc':    (TEXT,   TEXT,   False, False),
    'divider': (ACCENT, ACCENT, False, False),
    'result':  (GREEN,  GREEN,  True,  False),
    'error':   (RED,    RED,    False, False),
}

def _render_integral_card(parent, integrand_str, var_sym, titulo=None, borde_color=TEAL):
    """
    Renderiza un card completo con el paso a paso de la integral.
    Si la expresión es None o vacía, no renderiza nada.
    """
    if not integrand_str:
        return

    try:
        steps = generate_integral_steps(str(integrand_str), var_sym)
    except Exception as e:
        steps = [{'type': 'error', 'text': str(e)}]

    # Card contenedor
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
            # Cabecera del método con fondo ligeramente destacado
            f_method = tk.Frame(inner, bg=BG2)
            f_method.pack(fill=tk.X, padx=8, pady=(6, 2))
            tk.Label(f_method, text=f"  ⟐ {step['text']}", bg=BG2, fg=TEAL,
                     font=("Consolas", 11, "bold"),
                     anchor="w").pack(fill=tk.X, padx=6, pady=4)

        elif stype == 'formula':
            tk.Label(inner, text=f"    {step['text']}", bg=BG3, fg=GREEN,
                     font=("Consolas", 11), anchor="w",
                     justify="left").pack(fill=tk.X, padx=14, pady=2)

        elif stype == 'info':
            tk.Label(inner, text=f"    {step['text']}", bg=BG3, fg=MUTED,
                     font=("Consolas", 10), anchor="w",
                     justify="left").pack(fill=tk.X, padx=14, pady=1)

        elif stype == 'assign':
            row = tk.Frame(inner, bg=BG3)
            row.pack(anchor="w", padx=18, pady=2)
            lbl_text = f"  {step.get('label','?')}  =  "
            tk.Label(row, text=lbl_text, bg=BG3, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=str(step.get('value', '')),
                     bg=BG3, fg=YELLOW,
                     font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        elif stype == 'calc':
            tk.Label(inner, text=f"    {step['text']}", bg=BG3, fg=TEXT,
                     font=("Consolas", 11), anchor="w",
                     justify="left").pack(fill=tk.X, padx=14, pady=2)

        elif stype == 'divider':
            sep_f = tk.Frame(inner, bg=BG3)
            sep_f.pack(fill=tk.X, padx=14, pady=(6, 2))
            tk.Label(sep_f, text=f"  {step['text']}", bg=BG3, fg=ACCENT,
                     font=("Consolas", 11, "bold"), anchor="w").pack(side=tk.LEFT)

        elif stype == 'result':
            res_f = tk.Frame(inner, bg="#1a2a1a", pady=4, padx=6)
            res_f.pack(fill=tk.X, padx=14, pady=6)
            tk.Label(res_f, text=f"  ✓  {step['text']}",
                     bg="#1a2a1a", fg=GREEN,
                     font=("Consolas", 12, "bold"),
                     anchor="w").pack(fill=tk.X, padx=8, pady=4)

        elif stype == 'error':
            tk.Label(inner, text=f"  ✗ {step['text']}", bg=BG3, fg=RED,
                     font=("Consolas", 10), anchor="w").pack(fill=tk.X, padx=14, pady=2)

    tk.Frame(inner, bg=BG3, height=8).pack()


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
        tk.Label(bar,
                 text="  🧮  Comparador de Métodos Numéricos para EDO — Euler · RK2 · RK4",
                 bg=BG2, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="Euler (RK1)  |  RK2 Heun/Midpoint  |  RK4 Clásico  |  Solución Analítica  ",
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

        _lbl(p, "EJEMPLOS DE REFERENCIA", fg=YELLOW,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        for ej in EJEMPLOS_REF:
            self._ej_btn(p, ej)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        _lbl(p, "ECUACIÓN DIFERENCIAL  dy/dx = f(x, y)", fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._ef = _labeled_entry(p, "f(x, y) =", "x + y")

        _lbl(p, "CONDICIÓN INICIAL", fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._ex0 = _labeled_entry(p, "x₀", "0")
        self._ey0 = _labeled_entry(p, "y₀  (condición inicial)", "1")

        _lbl(p, "PARÁMETROS", fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,4))
        self._eh  = _labeled_entry(p, "h  — tamaño del paso", "0.1")
        self._exf = _labeled_entry(p, "X Final", "1")

        _lbl(p, "n — número de pasos (calculado)").pack(anchor="w")
        self._en = _entry(p, "10")
        self._en.config(state="readonly")
        self._en.pack(fill=tk.X, ipady=6, pady=(2, 8))

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
        for lbl, col in [("Euler", COL_EULER), ("RK2", COL_RK2),
                          ("RK4", COL_RK4), ("Exacta", COL_EXACTA)]:
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
        for val, lbl, col in [("analitico", "Analítico", GREEN),
                               ("euler", "Euler", COL_EULER),
                               ("rk2", "RK2", COL_RK2),
                               ("rk4", "RK4", COL_RK4)]:
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
            fexpr = self._ef.get().strip()
            if not fexpr: raise ValueError("Ingresa la EDO f(x, y).")
            x0  = _parse_float(self._ex0.get(), "x0")
            y0  = _parse_float(self._ey0.get(), "y0")
            h   = _parse_float(self._eh.get(),  "h")
            xf  = _parse_float(self._exf.get(), "X Final")
            if xf <= x0: raise ValueError("X Final debe ser mayor a x0")
            pasos = int(round((xf - x0) / h))
            if pasos < 1: raise ValueError("El intervalo es demasiado pequeño.")
            if h <= 0:    raise ValueError("h debe ser > 0.")

            self._en.config(state="normal")
            self._en.delete(0, tk.END)
            self._en.insert(0, str(pasos))
            self._en.config(state="readonly")

            rk2var = self._rk2_var.get()
            xs     = np.array([x0 + i*h for i in range(pasos+1)])
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
            ye, yr2, yr4 = d["y_euler"][i], d["y_rk2"][i], d["y_rk4"][i]
            bg_row = BG3 if i % 2 == 0 else BG2
            if tiene_exacta:
                yex = d["exacta"][i]
                e_e, e_r2, e_r4 = abs(ye-yex), abs(yr2-yex), abs(yr4-yex)
                vals = [str(i), f"{xi:.2f}", f"{ye:.6f}", f"{yr2:.6f}", f"{yr4:.6f}",
                        f"{yex:.6f}", f"{e_e:.2e}", f"{e_r2:.2e}", f"{e_r4:.2e}"]
                colores = [MUTED, TEXT, COL_EULER, COL_RK2, COL_RK4, GREEN, YELLOW, YELLOW, YELLOW]
            else:
                vals    = [str(i), f"{xi:.2f}", f"{ye:.6f}", f"{yr2:.6f}", f"{yr4:.6f}"]
                colores = [MUTED, TEXT, COL_EULER, COL_RK2, COL_RK4]
            _tabla_fila(si, vals, widths, colores, bg_row)

        if tiene_exacta:
            _gap(si, 8)
            _seccion(si, "RESUMEN DE ERRORES MÁXIMOS", YELLOW)
            c2 = _card(si, YELLOW)
            _ctitulo(c2, "Error máximo  |y_num - y_exacta|:", YELLOW)
            for lbl, yv, col in [("Euler   (RK1)", d["y_euler"], COL_EULER),
                                  (f"RK2 {rk2_nom:8s}", d["y_rk2"], COL_RK2),
                                  ("RK4 Clásico  ", d["y_rk4"], COL_RK4)]:
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
    def _render_tabla_metodo(self, metodo):
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
                widths = [4, 12, 14, 16, 14]
            else:
                cols   = ["n", "xn", "yn", "yn+1 (Euler)"]
                widths = [4, 12, 14, 16]
        elif metodo == "rk2":
            cols   = ["n", "xn", "yn", "k1", "k2", "yn+1 (RK2)"]
            widths = [4, 10, 13, 13, 13, 14]
        else:
            cols   = ["n", "xn", "yn", "k1", "k2", "k3", "k4", "yn+1 (RK4)"]
            widths = [3, 9, 11, 11, 11, 11, 11, 12]

        _tabla_header(si, cols, widths, col)

        for paso in data_pasos:
            i   = paso["i"]
            xn  = paso["x_n"]
            yn  = paso["y_n"]
            yn1 = paso["y_nuevo"]
            bg_row = BG3 if i % 2 == 0 else BG2

            if metodo == "euler":
                if tiene_exacta:
                    yreal = d["exacta"][i]
                    vals  = [str(i), f"{xn:.4f}", f"{yn:.7f}", f"{yn1:.7f}", f"{yreal:.7f}"]
                    cc    = [MUTED, TEXT, col, COL_EULER, COL_EXACTA]
                else:
                    vals  = [str(i), f"{xn:.4f}", f"{yn:.7f}", f"{yn1:.7f}"]
                    cc    = [MUTED, TEXT, col, COL_EULER]
            elif metodo == "rk2":
                vals = [str(i), f"{xn:.4f}", f"{yn:.7f}",
                        f"{paso['k1']:.6f}", f"{paso['k2']:.6f}", f"{yn1:.7f}"]
                cc   = [MUTED, TEXT, col, GREEN, TEAL, COL_RK2]
            else:
                vals = [str(i), f"{xn:.4f}", f"{yn:.7f}",
                        f"{paso['k1']:.5f}", f"{paso['k2']:.5f}",
                        f"{paso['k3']:.5f}", f"{paso['k4']:.5f}", f"{yn1:.7f}"]
                cc   = [MUTED, TEXT, col, GREEN, TEAL, PURPLE, ORANGE, COL_RK4]

            _tabla_fila(si, vals, widths, cc, bg_row)

        _gap(si, 10)
        _seccion(si, f"DETALLE DE PENDIENTES — {nom}", col)

        for paso in data_pasos:
            i   = paso["i"]
            xn  = paso["x_n"]
            yn  = paso["y_n"]
            cv  = _card(si, col)
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
    # RENDER: GRÁFICO
    # ══════════════════════════════════════════════════
    def _render_grafico(self):
        if not self._data: return
        d  = self._data
        ax = self._ax_graf
        ax.clear(); _style_ax(ax)
        xs = d["xs"]
        if self._chk_vars["Euler"].get():
            ax.plot(xs, d["y_euler"], color=COL_EULER, lw=2, marker="o", ms=4,
                    label="Euler (RK1)", zorder=3)
        if self._chk_vars["RK2"].get():
            rk2_nom = "Heun" if d["rk2var"]=="heun" else "Midpoint"
            ax.plot(xs, d["y_rk2"], color=COL_RK2, lw=2, marker="s", ms=4,
                    label=f"RK2 {rk2_nom}", zorder=3)
        if self._chk_vars["RK4"].get():
            ax.plot(xs, d["y_rk4"], color=COL_RK4, lw=2, marker="^", ms=4,
                    label="RK4 Clásico", zorder=4)
        if self._chk_vars["Exacta"].get() and d["exacta"] is not None:
            xs_d = np.linspace(xs[0], xs[-1], 400)
            ed, _ = _solucion_exacta_sympy(d["fexpr"], d["x0"], d["y0"], xs_d)
            if ed is not None:
                ax.plot(xs_d, ed, color=COL_EXACTA, lw=2, ls="--",
                        label="Exacta (analítica)", zorder=5, alpha=0.9)
            ax.plot(xs, d["exacta"], color=COL_EXACTA, marker="D", ms=3,
                    lw=0, alpha=0.6, label="Exacta (puntos)", zorder=5)
        ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y(x)", fontsize=9)
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
        if not self._data: return
        d  = self._data
        ax = self._ax_pend
        ax.clear(); _style_ax(ax)
        xs = d["xs"]; h = d["h"]
        y_all = np.concatenate([d["y_euler"], d["y_rk2"], d["y_rk4"]])
        x_min, x_max = xs[0]-h*0.5, xs[-1]+h*0.5
        y_min_r, y_max_r = float(np.min(y_all)), float(np.max(y_all))
        dy_r = max(abs(y_max_r-y_min_r)*0.3, 0.5)
        xg = np.linspace(x_min, x_max, 20)
        yg = np.linspace(y_min_r-dy_r, y_max_r+dy_r, 15)
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
        ax.plot(xs, d["y_rk2"], color=COL_RK2, lw=2, marker="s", ms=4,
                label=f"RK2 {'Heun' if d['rk2var']=='heun' else 'Midpoint'}", zorder=3)
        ax.plot(xs, d["y_rk4"], color=COL_RK4, lw=2, marker="^", ms=4, label="RK4", zorder=4)
        if d["exacta"] is not None:
            xs_d = np.linspace(xs[0], xs[-1], 300)
            ed, _ = _solucion_exacta_sympy(d["fexpr"], d["x0"], d["y0"], xs_d)
            if ed is not None:
                ax.plot(xs_d, ed, color=COL_EXACTA, lw=1.5, ls="--", alpha=0.8,
                        label="Exacta", zorder=5)
        p0 = d["rk4_data"][0]
        xn, yn = p0["x_n"], p0["y_n"]
        esc = h * 0.4
        def _flecha(ax, x, y, dy_f, color, lbl):
            ax.annotate("", xy=(x+esc, y+dy_f*esc), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
            ax.plot([], [], color=color, label=lbl, lw=2)
        _flecha(ax, xn,       yn,               p0["k1"], GREEN,  "k₁ inicio")
        _flecha(ax, xn+h/2, yn+(h/2)*p0["k1"], p0["k2"], TEAL,   "k₂ mitad")
        _flecha(ax, xn+h/2, yn+(h/2)*p0["k2"], p0["k3"], PURPLE, "k₃ mitad")
        _flecha(ax, xn+h,   yn+h*p0["k3"],     p0["k4"], ORANGE, "k₄ final")
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
        if not self._data: return
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
    # RENDER: DESARROLLO ANALÍTICO (con integrales paso a paso)
    # ══════════════════════════════════════════════════
    def _render_desarrollo_analitico(self):
        d   = self._data
        si  = self._si_pasos
        for w in si.winfo_children():
            w.destroy()

        ana   = d["analitico"]
        tipo  = ana["tipo"]
        fexpr = d["fexpr"]
        x0, y0 = d["x0"], d["y0"]
        x_sym = sp.Symbol("x")

        _seccion(si, f"RESOLUCIÓN ANALÍTICA — EDO: dy/dx = {fexpr}", GREEN)

        if ana["error"] and ana["sol_particular"] is None:
            c_err = _card(si, RED)
            _ctitulo(c_err, "SymPy no pudo resolver esta EDO analíticamente", RED)
            _cformula(c_err, f"Error: {ana['error']}", MUTED)
            _cformula(c_err, "Puedes intentar resolverla manualmente o usar métodos numéricos.", YELLOW)
            _gap(c_err, 4)
            return

        if tipo == "solo_x":
            self._pasos_solo_x(si, ana, fexpr, x0, y0, x_sym)
        elif tipo == "linear":
            self._pasos_lineal(si, ana, fexpr, x0, y0, x_sym)
        elif tipo == "separable":
            self._pasos_separable(si, ana, fexpr, x0, y0, x_sym)
        else:
            self._pasos_general(si, ana, fexpr, x0, y0, x_sym)

        # Verificación numérica
        if d["exacta"] is not None and ana["sol_particular"] is not None:
            _gap(si, 8)
            _seccion(si, "VERIFICACIÓN NUMÉRICA — y(x) vs métodos", COL_EXACTA)
            cols   = ["x", "y exacta", "Euler", "RK2", "RK4"]
            widths = [10, 14, 14, 14, 14]
            _tabla_header(si, cols, widths, COL_EXACTA)
            for i, xi in enumerate(d["xs"]):
                yex  = d["exacta"][i]
                vals = [f"{xi:.4f}", f"{yex:.8f}", f"{d['y_euler'][i]:.8f}",
                        f"{d['y_rk2'][i]:.8f}", f"{d['y_rk4'][i]:.8f}"]
                cc   = [TEXT, COL_EXACTA, COL_EULER, COL_RK2, COL_RK4]
                _tabla_fila(si, vals, widths, cc, BG3 if i%2==0 else BG2)

    # ── Pasos: dy/dx = g(x) ───────────────────────────
    def _pasos_solo_x(self, si, ana, fexpr, x0, y0, x_sym):
        c1 = _card(si, ACCENT)
        _ctitulo(c1, "Tipo detectado: EDO integrable directamente — solo f(x)", ACCENT)
        _cformula(c1, "dy/dx = g(x)  →  dy = g(x) dx  →  y = ∫ g(x) dx + C", TEXT)
        _gap(c1, 4)

        # ── Integral de g(x) con paso a paso ──────────
        _seccion(si, "CÁLCULO DE ∫ g(x) dx — Paso a paso", GREEN)
        _render_integral_card(
            si,
            integrand_str=fexpr.replace('^', '**'),
            var_sym=x_sym,
            titulo=f"Paso 1 — Integrar g(x) = {fexpr}",
            borde_color=GREEN,
        )

        # Condición inicial
        c3 = _card(si, YELLOW)
        _ctitulo(c3, f"Paso 2 — Aplicar condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        try:
            fexpr_sym = sp.sympify(fexpr.replace('^','**'), locals={'x': x_sym})
            integral  = sp.integrate(fexpr_sym, x_sym)
            eq_val    = integral.subs(x_sym, x0)
            C_val     = float(y0 - eq_val)
            _cformula(c3, f"y({x0:.4g}) = {integral} + C  =  {y0:.6g}", TEXT)
            _cformula(c3, f"  →  {eq_val} + C = {y0:.6g}", TEXT)
            _cigual(c3, "C", f"{C_val:.6g}", YELLOW)
        except Exception:
            pass
        _gap(c3, 4)

        c4 = _card(si, COL_EXACTA)
        _ctitulo(c4, "Solución Particular", COL_EXACTA)
        _cformula(c4, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c4, 4)

    # ── Pasos: EDO Lineal ─────────────────────────────
    def _pasos_lineal(self, si, ana, fexpr, x0, y0, x_sym):
        P_std = ana["extra1"]
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

        # ── Integral de P(x) para factor integrante ───
        _seccion(si, "PASO 1 — Factor Integrante  μ(x) = e^{ ∫P(x)dx } — Integral", TEAL)
        _render_integral_card(
            si,
            integrand_str=str(P_std),
            var_sym=x_sym,
            titulo=f"Calcular ∫ P(x) dx  =  ∫ ({P_std}) dx",
            borde_color=TEAL,
        )

        c2 = _card(si, TEAL)
        _ctitulo(c2, "Factor integrante resultante", TEAL)
        if mu_exp is not None:
            _cformula(c2, f"μ(x) = e^{{ ∫({P_std})dx }}  =  e^{{ {mu_exp} }}", TEXT)
            _cigual(c2, "μ(x)", str(mu), GREEN)
        _csep(c2)
        _cformula(c2, "Al multiplicar por μ(x):", MUTED)
        _cformula(c2, "d/dx [ μ(x)·y ]  =  μ(x)·Q(x)", GREEN)
        _gap(c2, 4)

        # ── Integral de μ(x)·Q(x) ─────────────────────
        if mu is not None:
            muQ_str = str(sp.expand(mu * Q_std))
            _seccion(si, "PASO 2 — Integración de μ(x)·Q(x) — Integral", PURPLE)
            _render_integral_card(
                si,
                integrand_str=muQ_str,
                var_sym=x_sym,
                titulo=f"Calcular ∫ μ(x)·Q(x) dx  =  ∫ ({muQ_str}) dx",
                borde_color=PURPLE,
            )

        c3 = _card(si, PURPLE)
        _ctitulo(c3, "Paso 2 — Solución general  y = (1/μ)·[ ∫μQ dx + C ]", PURPLE)
        sol_gen = ana["sol_general"]
        _cformula(c3, f"y(x) = {sol_gen}", GREEN)
        _gap(c3, 4)

        c4 = _card(si, YELLOW)
        _ctitulo(c4, f"Paso 3 — Condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        C1 = sp.Symbol("C1")
        try:
            eq_val = sol_gen.subs(x_sym, x0)
            _cformula(c4, f"y({x0:.4g}) = {eq_val}  =  {y0:.6g}", TEXT)
        except Exception:
            pass
        C_val = ana["C1_val"]
        _cigual(c4, "C₁", str(C_val), YELLOW)
        _gap(c4, 4)

        c5 = _card(si, COL_EXACTA)
        _ctitulo(c5, "Solución Particular", COL_EXACTA)
        _cformula(c5, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c5, 4)

    # ── Pasos: EDO Separable ──────────────────────────
    def _pasos_separable(self, si, ana, fexpr, x0, y0, x_sym):
        g_x  = ana.get("g_x")
        h_y  = ana.get("h_y")
        y_sym = sp.Symbol('y')

        c1 = _card(si, ACCENT)
        _ctitulo(c1, "Tipo detectado: EDO Separable", ACCENT)
        _cformula(c1, "Forma:  dy/dx = g(x)·h(y)", TEXT)
        _cformula(c1, "Separando:   dy / h(y) = g(x) dx", MUTED)
        _cformula(c1, "Integrando ambos lados:  ∫ dy/h(y) = ∫ g(x) dx + C", GREEN)
        _gap(c1, 4)

        c_id = _card(si, TEAL)
        _ctitulo(c_id, "Identificación de g(x) y h(y)", TEAL)
        _cformula(c_id, f"dy/dx = {fexpr}", TEXT)
        if g_x is not None and h_y is not None:
            _cigual(c_id, "g(x)", str(g_x), COL_RK2)
            _cigual(c_id, "h(y)", str(h_y).replace("ytmp", "y"), COL_RK4)
        _gap(c_id, 4)

        # ── Integral de g(x) ──────────────────────────
        if g_x is not None:
            _seccion(si, "PASO 2a — Integral del lado derecho: ∫ g(x) dx", GREEN)
            _render_integral_card(
                si,
                integrand_str=str(g_x),
                var_sym=x_sym,
                titulo=f"Calcular ∫ g(x) dx  =  ∫ ({g_x}) dx",
                borde_color=GREEN,
            )

        # ── Integral de 1/h(y) dy ─────────────────────
        if h_y is not None:
            h_y_in_y = h_y.subs(sp.Symbol('ytmp'), y_sym)
            inv_hy   = 1 / h_y_in_y
            _seccion(si, "PASO 2b — Integral del lado izquierdo: ∫ dy/h(y)", PURPLE)
            _render_integral_card(
                si,
                integrand_str=str(inv_hy),
                var_sym=y_sym,
                titulo=f"Calcular ∫ dy/h(y)  =  ∫ ({inv_hy}) dy",
                borde_color=PURPLE,
            )

        c3 = _card(si, ORANGE)
        _ctitulo(c3, "Paso 3 — Solución general (igualando integrales)", ORANGE)
        _cformula(c3, f"y(x) = {ana['sol_general']}", GREEN)
        _gap(c3, 4)

        c4 = _card(si, YELLOW)
        _ctitulo(c4, f"Paso 4 — Condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        C1  = sp.Symbol("C1")
        sol_gen = ana["sol_general"]
        try:
            eq_val = sol_gen.subs(x_sym, x0)
            _cformula(c4, f"y({x0:.4g}) = {eq_val}  =  {y0:.6g}", TEXT)
        except Exception:
            pass
        _cigual(c4, "C₁", str(ana["C1_val"]), YELLOW)
        _gap(c4, 4)

        c5 = _card(si, COL_EXACTA)
        _ctitulo(c5, "Solución Particular", COL_EXACTA)
        _cformula(c5, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c5, 4)

    # ── Pasos: General ────────────────────────────────
    def _pasos_general(self, si, ana, fexpr, x0, y0, x_sym):
        c1 = _card(si, YELLOW)
        _ctitulo(c1, "Resolución por SymPy (tipo no clasificado manualmente)", YELLOW)
        _cformula(c1, f"EDO:  dy/dx = {fexpr}", TEXT)
        _cformula(c1, f"CI:   y({x0:.4g}) = {y0:.6g}", MUTED)
        _csep(c1)
        _cformula(c1, "Solución general:", MUTED)
        _cformula(c1, f"y(x) = {ana['sol_general']}", GREEN)
        _gap(c1, 4)

        c2 = _card(si, YELLOW)
        _ctitulo(c2, f"Condición inicial: y({x0:.4g}) = {y0:.6g}", YELLOW)
        _cigual(c2, "C₁", str(ana["C1_val"]), YELLOW)
        _gap(c2, 4)

        c3 = _card(si, COL_EXACTA)
        _ctitulo(c3, "Solución Particular", COL_EXACTA)
        _cformula(c3, f"y(x) = {ana['sol_particular']}", COL_EXACTA)
        _gap(c3, 4)

        cn = _card(si, MUTED)
        _cformula(cn, "NOTA: Para ver el detalle completo de integrales paso a paso,", YELLOW)
        _cformula(cn, "esta EDO requiere clasificación manual (separable, lineal, Bernoulli, etc.)", MUTED)
        _gap(cn, 4)


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = ComparadorEDOApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()