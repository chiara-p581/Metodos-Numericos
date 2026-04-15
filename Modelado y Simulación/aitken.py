"""
Aitken Delta^2
==============
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Metodo de Aceleracion Aitken (pag. 13-14)

Formula de Aitken (Caceres, pag. 13):
    x*_n = x_n  -  (x_{n+1} - x_n)^2 / (x_{n+2} - 2*x_{n+1} + x_n)

Dos modos de uso:
  A) Solo g(x)     — ingresa directamente la funcion de iteracion
  B) Desde f(x)    — el programa halla g(x) y muestra el desarrollo

Como se halla g(x) desde f(x):
  Partiendo de f(x) = 0, se despeja x de distintas formas:
    1. Suma:     g(x) = f(x) + x
    2. Algebraica: sympy.solve(f, x)  (si es posible)
    3. Newton:   g(x) = x - f(x)/f'(x)
    4. Relaj:    g(x) = x - alpha * f(x)
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


# ══════════════════════════════════════
# EVALUACION SEGURA
# ══════════════════════════════════════
def _env(x_val=None):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    if x_val is not None:
        e["x"] = x_val
    return e

def evaluar(expr, x_val):
    return eval(expr, {"__builtins__": {}}, _env(x_val))

def derivada_num(expr, x_val, h=1e-7):
    return (evaluar(expr, x_val + h) - evaluar(expr, x_val - h)) / (2 * h)


# ══════════════════════════════════════
# HALLAR g(x) DESDE f(x)
# ══════════════════════════════════════
def _sympy_to_py(expr_sym):
    """Convierte sympy a string evaluable sin prefijo math."""
    try:
        import sympy as sp
        code = sp.printing.pycode(expr_sym)
        code = re.sub(r'\bmath\.', '', code)
        return code
    except Exception:
        return None

def _es_evaluable(code, x_val=1.0):
    if not code:
        return False
    try:
        v = evaluar(code, x_val)
        return isinstance(v, (int, float)) and v == v and abs(v) < 1e15
    except Exception:
        return False

def hallar_g_desde_f(fexpr, x0=1.0):
    """
    Dado f(x), genera candidatos a g(x) con:
      - desarrollo algebraico paso a paso
      - derivada simbolica g'(x)
      - verificacion |g'(x0)| < 1  (convergencia)
    """
    import sympy as sp

    x_sym = sp.Symbol("x")
    candidatos = []
    vistos     = set()

    def _agregar(g_expr_str, metodo, pasos):
        if not g_expr_str or g_expr_str in vistos:
            return
        if not _es_evaluable(g_expr_str, x0):
            return
        vistos.add(g_expr_str)

        # derivada simbolica de g(x)
        try:
            g_sym      = sp.sympify(g_expr_str)
            dg_sym     = sp.diff(g_sym, x_sym)
            dg_simple  = sp.simplify(dg_sym)
            dg_str     = str(dg_simple)
            dg_val     = abs(float(dg_simple.subs(x_sym, x0)))
            converge   = dg_val < 1
        except Exception:
            dg_str  = "No se pudo calcular"
            dg_val  = float("nan")
            converge = False

        candidatos.append({
            "expr":     g_expr_str,
            "metodo":   metodo,
            "pasos":    pasos,        # lista de (tipo, texto)
            "gp_sym":   dg_str,
            "gp_val":   dg_val,
            "converge": converge,
        })

    try:
        f_sym    = sp.sympify(fexpr)
        df_sym   = sp.diff(f_sym, x_sym)
        df_str   = str(df_sym)
        df_simp  = str(sp.simplify(df_sym))

        # ── METODO 1: g(x) = f(x) + x  (Caceres pag.12)
        g1  = sp.simplify(f_sym + x_sym)
        e1  = _sympy_to_py(g1)
        _agregar(e1, "Suma directa  [Caceres pag.12]", [
            ("sub",    "Partimos de:"),
            ("val",    f"   f(x) = {fexpr} = 0"),
            ("",       ""),
            ("sub",    "Sumamos  x  a ambos lados de  f(x) = 0:"),
            ("form",   "   f(x) + x  =  0 + x"),
            ("form",   f"   ({fexpr}) + x  =  x"),
            ("",       ""),
            ("sub",    "Entonces definimos  g(x):"),
            ("res",    "   g(x)  =  f(x) + x"),
            ("res",    f"   g(x)  =  ({fexpr}) + x"),
            ("res",    f"   g(x)  =  {e1}"),
            ("",       ""),
            ("sub",    "Verificacion:  g(p) = p  si y solo si  f(p) = 0"),
        ])

        # ── METODO 2: despejes algebraicos
        try:
            for k, sol in enumerate(sp.solve(f_sym, x_sym)):
                if sol.has(sp.I):
                    continue
                sol_str = str(sp.simplify(sol))
                e2      = _sympy_to_py(sp.simplify(sol))
                _agregar(e2, f"Despeje algebraico #{k+1}", [
                    ("sub",  "Partimos de:"),
                    ("val",  f"   f(x) = {fexpr} = 0"),
                    ("",     ""),
                    ("sub",  "Resolvemos algebraicamente (sympy.solve):"),
                    ("form", f"   {fexpr}  =  0"),
                    ("",     ""),
                    ("sub",  "Soluciones:"),
                    ("res",  f"   x  =  {sol_str}"),
                    ("",     ""),
                    ("sub",  "Esta g(x) es una constante (la raiz exacta)."),
                    ("sub",  "La iteracion converge en 1 solo paso."),
                    ("res",  f"   g(x)  =  {e2}"),
                ])
        except Exception:
            pass

        # ── METODO 3: g(x) = x - f(x)/f'(x)  (Newton)
        try:
            gn     = sp.simplify(x_sym - f_sym / df_sym)
            e3     = _sympy_to_py(gn)
            gn_str = str(sp.simplify(gn))
            _agregar(e3, "Tipo Newton  g(x) = x - f/f'", [
                ("sub",  "Partimos de:"),
                ("val",  f"   f(x) = {fexpr}"),
                ("",     ""),
                ("sub",  "Paso 1 — Calculamos  f'(x):"),
                ("form", f"   f'(x)  =  d/dx [ {fexpr} ]"),
                ("res",  f"   f'(x)  =  {df_str}"),
                ("res",  f"   f'(x)  =  {df_simp}   (simplificado)"),
                ("",     ""),
                ("sub",  "Paso 2 — Formula Newton como punto fijo:"),
                ("form", "   g(x)  =  x  -  f(x) / f'(x)"),
                ("",     ""),
                ("sub",  "Paso 3 — Sustituimos:"),
                ("form", f"   g(x)  =  x  -  ({fexpr})"),
                ("form", f"                  ──────────────────"),
                ("form", f"                  ({df_simp})"),
                ("",     ""),
                ("sub",  "Resultado simplificado:"),
                ("res",  f"   g(x)  =  {e3}"),
                ("",     ""),
                ("sub",  "Ventaja: converge cuadraticamente (orden 2)."),
            ])
        except Exception:
            pass

        # ── METODO 4: relajacion
        for alpha, desc in [("0.5","media"), ("0.1","suave"), ("0.01","muy suave")]:
            e4 = f"x - {alpha}*({fexpr})"
            _agregar(e4, f"Relajacion  alpha = {alpha}  ({desc})", [
                ("sub",  "Partimos de:"),
                ("val",  f"   f(x) = {fexpr} = 0"),
                ("",     ""),
                ("sub",  f"Paso 1 — Multiplicamos ambos lados por alpha = {alpha}:"),
                ("form", f"   {alpha} * ({fexpr})  =  0"),
                ("",     ""),
                ("sub",  "Paso 2 — Sumamos  x  a ambos lados:"),
                ("form", f"   x  -  {alpha} * ({fexpr})  =  x"),
                ("",     ""),
                ("sub",  "Resultado:"),
                ("res",  f"   g(x)  =  x  -  {alpha} * f(x)"),
                ("res",  f"   g(x)  =  {e4}"),
                ("",     ""),
                ("sub",  "Condicion de convergencia:"),
                ("form", f"   |g'(x)|  =  |1 - {alpha} * f'(x)|  <  1"),
                ("form", f"   Necesitamos:  0  <  {alpha} * f'(x)  <  2"),
            ])

    except Exception:
        # fallback si sympy no esta disponible
        for alpha, desc in [("0.5","media"), ("0.1","suave")]:
            e = f"x - {alpha}*({fexpr})"
            _agregar(e, f"Relajacion alpha={alpha}", [
                ("res", f"g(x) = {e}")
            ])

    return candidatos



# ══════════════════════════════════════
# LOGICA AITKEN
# ══════════════════════════════════════
def aitken(gexpr, x0, tol=1e-6, max_iter=100):
    hist = []
    x    = x0
    for i in range(max_iter):
        try:
            x1  = evaluar(gexpr, x)
            x2  = evaluar(gexpr, x1)
            num = (x1 - x) ** 2
            den = x2 - 2*x1 + x
            if den == 0:
                return x, hist, "division_cero"
            xhat  = x - num / den
            error = abs(xhat - x)
            hist.append({"i": i+1, "xn": x, "x1": x1, "x2": x2,
                          "xhat": xhat, "error": error})
            if error < tol:
                return xhat, hist, "convergencia"
            x = xhat
        except Exception as exc:
            return None, hist, f"error: {exc}"
    return x, hist, "max_iter"

def analisis_aitken(hist, raiz, tol, gexpr):
    errores = [r["error"] for r in hist if r["error"] > 0]
    ratios  = [errores[i]/errores[i-1] for i in range(1, len(errores))
               if errores[i-1] != 0]
    factor  = sum(ratios)/len(ratios) if ratios else 0
    tipo    = "rapida" if factor < 0.1 else ("moderada" if factor < 0.5 else "lenta")
    ue      = errores[-1] if errores else 0
    return {"iters": len(hist), "raiz": raiz, "ue": ue,
            "factor": factor, "tipo": tipo, "tol": tol, "ok": ue < tol}


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
    wrap = tk.Frame(parent, bg=BG)
    wrap.pack(fill=tk.BOTH, expand=True)
    sc  = tk.Canvas(wrap, bg=BG, highlightthickness=0)
    vsb = tk.Scrollbar(wrap, orient="vertical", command=sc.yview)
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

def _seccion(parent, titulo, color=ACCENT):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill=tk.X, padx=12, pady=(14, 4))
    tk.Frame(f, bg=color, width=4).pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(f, text=f"  {titulo}", bg=BG, fg=color,
             font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=4)

def _card(parent, color=ACCENT):
    outer = tk.Frame(parent, bg=BG)
    outer.pack(fill=tk.X, padx=12, pady=4)
    tk.Frame(outer, bg=color, width=3).pack(side=tk.LEFT, fill=tk.Y)
    inner = tk.Frame(outer, bg=BG2)
    inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return inner

def _linea(parent, label, valor, color_val=ACCENT):
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=14, pady=2)
    tk.Label(row, text=label, bg=BG2, fg=MUTED,
             font=("Consolas", 11)).pack(side=tk.LEFT)
    tk.Label(row, text=valor, bg=BG2, fg=color_val,
             font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

def _espacio(parent, h=8):
    tk.Frame(parent, bg=BG2, height=h).pack()


# ══════════════════════════════════════
# CLASE PRINCIPAL — AITKEN
# ══════════════════════════════════════
class AitkenApp(tk.Frame):

    TABS = [
        ("📉", "Convergencia"),
        ("📊", "Grafico g(x)"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🧮", "Hallar g(x)"),
        ("🧠", "Analisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Aitken - Metodos Numericos")
            master.configure(bg=BG)
            master.geometry("1340x740")
            master.minsize(1050, 600)
        self._hist  = []
        self._raiz  = None
        self._gexpr = ""
        self._x0    = 0.0
        self._candidatos = []
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
        tk.Label(bar, text="  Aitken Delta^2", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="x* = xn - (x1-xn)^2 / (x2 - 2*x1 + xn)   |   Caceres 2026 pag.13",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=300)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # ── MODO selector
        _lbl(inner, "MODO DE INGRESO", fg=MUTED,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        self._modo = tk.StringVar(value="g")
        frame_modo = tk.Frame(inner, bg=BG2)
        frame_modo.pack(fill=tk.X, pady=(0, 8))

        for val, label in [("g", "Solo g(x)"), ("f", "Desde f(x)  [halla g(x)]")]:
            tk.Radiobutton(
                frame_modo, text=label, variable=self._modo, value=val,
                bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2,
                font=("Segoe UI", 11), command=self._on_modo_change,
            ).pack(anchor="w", pady=2)

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

        # ── campo f(x)  (visible solo en modo f)
        self._frame_f = tk.Frame(inner, bg=BG2)
        self._frame_f.pack(fill=tk.X)
        self.e_f = _labeled_entry(self._frame_f, "f(x)  — la funcion original", "cos(x) - x")

        # ── campo g(x)
        _lbl(inner, "g(x)  — funcion de iteracion").pack(anchor="w")
        g_row = tk.Frame(inner, bg=BG2)
        g_row.pack(fill=tk.X, pady=(2, 8))
        self.e_g = tk.Entry(g_row, bg=BG3, fg=TEXT, insertbackground=TEXT,
                            font=("Consolas", 12), bd=0,
                            highlightthickness=1, highlightbackground=BORDER,
                            highlightcolor=ACCENT, relief="flat")
        self.e_g.insert(0, "cos(x) + x")
        self.e_g.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7)

        # boton hallar g(x)
        self._btn_hallar = tk.Label(
            g_row, text=" Hallar ", bg=PURPLE, fg="#000",
            font=("Segoe UI", 10, "bold"), cursor="hand2", padx=6, pady=6)
        self._btn_hallar.pack(side=tk.LEFT, padx=(4, 0))
        self._btn_hallar.bind("<Button-1>", lambda e: self._hallar_g())
        self._btn_hallar.bind("<Enter>",    lambda e: self._btn_hallar.config(bg=_dk(PURPLE)))
        self._btn_hallar.bind("<Leave>",    lambda e: self._btn_hallar.config(bg=PURPLE))

        self.e_x0  = _labeled_entry(inner, "x0  (punto inicial)", "0.5")
        self.e_tol = _labeled_entry(inner, "Tolerancia",           "1e-6")
        self.e_it  = _labeled_entry(inner, "Max iteraciones",      "100")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(inner, "Calcular",        self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "Graficar g(x)",   self._graficar, color=BG3, fg=ACCENT).pack(fill=tk.X, pady=3)

        self._on_modo_change()

    def _on_modo_change(self):
        modo = self._modo.get()
        if modo == "f":
            self._frame_f.pack(fill=tk.X, before=self.e_g.master)
        else:
            self._frame_f.pack_forget()

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
        self._build_panel_hallar_g()
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

    # ──────────── PANELS ────────────
    def _build_panel_conv(self):
        f = self._panel("Convergencia")
        self._fig_conv = Figure(figsize=(7, 4), facecolor=BG)
        self._ax_conv  = self._fig_conv.add_subplot(111)
        self._style_ax(self._ax_conv)
        self._canvas_conv = FigureCanvasTkAgg(self._fig_conv, master=f)
        self._canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_panel_func(self):
        f = self._panel("Grafico g(x)")
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
        cols = ("i", "xn", "x1=g(xn)", "x2=g(x1)", "x* (Aitken)", "error")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        for col, w in zip(cols, [40, 130, 130, 130, 130, 110]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="e")
        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

    def _build_panel_steps(self):
        f = self._panel("Paso a paso")
        self._si = _scrollable(f)

    def _build_panel_hallar_g(self):
        f = self._panel("Hallar g(x)")
        self._si_g = _scrollable(f)

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
    # HALLAR g(x)
    # ══════════════════════════════════════
    def _hallar_g(self):
        fexpr = self.e_f.get().strip()
        if not fexpr:
            messagebox.showwarning("Atencion", "Ingresa f(x) para hallar g(x).")
            return
        try:
            x0 = float(eval(self.e_x0.get()))
        except Exception:
            x0 = 1.0

        # loading en la tab
        for w in self._si_g.winfo_children():
            w.destroy()
        _seccion(self._si_g, "Calculando g(x) desde f(x)...", YELLOW)
        c = _card(self._si_g, YELLOW)
        _linea(c, "f(x) = ", fexpr, YELLOW)
        _linea(c, "Buscando despejes con sympy...", "", MUTED)
        self._show_tab("Hallar g(x)")

        def calcular():
            candidatos = hallar_g_desde_f(fexpr, x0)
            self._candidatos = candidatos
            self.after(0, lambda: self._render_hallar_g(fexpr, candidatos))

        threading.Thread(target=calcular, daemon=True).start()

    def _render_hallar_g(self, fexpr, candidatos):
        si = self._si_g
        for w in si.winfo_children():
            w.destroy()

        # ── COLORES por tipo de linea
        ESTILOS = {
            "sub":  (MUTED,   ("Consolas", 11)),
            "val":  (YELLOW,  ("Consolas", 11, "bold")),
            "form": (MUTED,   ("Consolas", 11)),
            "res":  (GREEN,   ("Consolas", 12, "bold")),
            "":     (MUTED,   ("Consolas", 4)),
        }
        COLORES_METODO = [GREEN, ORANGE, PURPLE, ACCENT, YELLOW, RED]

        # ── ENCABEZADO
        _seccion(si, "COMO SE HALLA g(x) DESDE f(x)", ACCENT)
        c = _card(si, ACCENT)
        _linea(c, "f(x) = ", fexpr, ACCENT)
        _linea(c, "Objetivo: ", "reescribir  f(x) = 0  como  x = g(x)", MUTED)
        _linea(c, "Criterio: ", "|g'(x0)| < 1  =>  converge", GREEN)
        _linea(c, "          ", "|g'(x0)| >= 1 =>  puede diverger", RED)
        _espacio(c)

        if not candidatos:
            _seccion(si, "Sin resultados", RED)
            c = _card(si, RED)
            _linea(c, "No se encontraron g(x) validas.", "", RED)
            return

        # ── RESUMEN primero
        _seccion(si, "RESUMEN — Todas las g(x) encontradas", PURPLE)
        c = _card(si, PURPLE)
        for k, cand in enumerate(candidatos):
            color_est = GREEN if cand["converge"] else RED
            estado    = "CONVERGE" if cand["converge"] else "DIVERGE "
            tk.Label(c, text=f"  [{estado}]  |g'(x0)| = {cand['gp_val']:.4f}",
                     bg=BG2, fg=color_est,
                     font=("Consolas", 11, "bold")).pack(anchor="w", padx=14, pady=2)
            tk.Label(c, text=f"           {cand['metodo']}",
                     bg=BG2, fg=MUTED,
                     font=("Consolas", 11)).pack(anchor="w", padx=14)
            tk.Label(c, text=f"           g(x) = {cand['expr']}",
                     bg=BG2, fg=color_est,
                     font=("Consolas", 11)).pack(anchor="w", padx=14, pady=(0, 6))
        _espacio(c)

        # ── DETALLE de cada candidato
        for k, cand in enumerate(candidatos):
            color_m   = COLORES_METODO[k % len(COLORES_METODO)]
            color_est = GREEN if cand["converge"] else RED
            estado    = "CONVERGE" if cand["converge"] else "DIVERGE"
            icono     = "OK" if cand["converge"] else "X"

            _seccion(si, f"METODO {k+1} — {cand['metodo']}", color_m)
            c = _card(si, color_m)

            # badge estado
            hdr = tk.Frame(c, bg=BG2)
            hdr.pack(anchor="w", padx=14, pady=(8, 4))
            tk.Label(hdr, text=f" {icono} {estado} ",
                     bg=color_est, fg="#000",
                     font=("Segoe UI", 10, "bold"),
                     padx=8, pady=3).pack(side=tk.LEFT)
            tk.Label(hdr, text=f"   |g'(x0)| = {cand['gp_val']:.6f}",
                     bg=BG2, fg=color_est,
                     font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=8)

            # pasos algebraicos
            tk.Frame(c, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=4)
            for tipo, texto in cand["pasos"]:
                if tipo == "" or texto == "":
                    tk.Frame(c, bg=BG2, height=5).pack()
                    continue
                fg_c, font_c = ESTILOS.get(tipo, (MUTED, ("Consolas", 11)))
                tk.Label(c, text=texto, bg=BG2, fg=fg_c, font=font_c,
                         anchor="w").pack(anchor="w", padx=18, pady=1)

            # verificacion g'(x)
            tk.Frame(c, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=6)
            tk.Label(c, text="Verificacion de convergencia:",
                     bg=BG2, fg=MUTED,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14)
            tk.Label(c, text=f"   g'(x)  =  {cand['gp_sym']}",
                     bg=BG2, fg=PURPLE,
                     font=("Consolas", 11)).pack(anchor="w", padx=14, pady=2)
            tk.Label(c, text=f"   g'(x0) =  g'({self._x0:.4f})  =  {cand['gp_val']:.6f}",
                     bg=BG2, fg=PURPLE,
                     font=("Consolas", 11)).pack(anchor="w", padx=14, pady=2)
            tk.Label(c,
                     text=f"   |g'(x0)| = {cand['gp_val']:.6f}  {'< 1  =>  CONVERGE' if cand['converge'] else '>= 1  =>  DIVERGE'}",
                     bg=BG2, fg=color_est,
                     font=("Consolas", 12, "bold")).pack(anchor="w", padx=14, pady=4)

            # resultado final + boton usar
            res_frame = tk.Frame(c, bg=BG3)
            res_frame.pack(fill=tk.X, padx=14, pady=(0, 4))
            tk.Label(res_frame, text="  g(x) = ", bg=BG3, fg=MUTED,
                     font=("Consolas", 12)).pack(side=tk.LEFT, pady=6)
            tk.Label(res_frame, text=cand["expr"], bg=BG3, fg=color_m,
                     font=("Consolas", 13, "bold")).pack(side=tk.LEFT)

            if cand["converge"]:
                btn = tk.Label(c, text="  Usar esta g(x)  ",
                               bg=color_m, fg="#000",
                               font=("Segoe UI", 10, "bold"),
                               cursor="hand2", padx=8, pady=4)
                btn.pack(anchor="w", padx=14, pady=(0, 10))
                expr_cap = cand["expr"]
                btn.bind("<Button-1>", lambda e, ex=expr_cap: self._usar_g(ex))
                btn.bind("<Enter>",    lambda e, b=btn, col=color_m: b.config(bg=_dk(col)))
                btn.bind("<Leave>",    lambda e, b=btn, col=color_m: b.config(bg=col))
            else:
                tk.Label(c, text="  No recomendada — diverge en x0  ",
                         bg=BG3, fg=RED,
                         font=("Segoe UI", 10), padx=8, pady=4).pack(
                             anchor="w", padx=14, pady=(0, 10))

        # ── NOTA FINAL
        _seccion(si, "CONCLUSION — Como elegir g(x)", YELLOW)
        c = _card(si, YELLOW)
        for linea in [
            "1. Elige una g(x) marcada como  CONVERGE.",
            "   (|g'(x0)| < 1 cerca de la raiz).",
            "",
            "2. Si varias convergen, preferi la de menor |g'(x0)|.",
            "   Menor |g'| = convergencia mas rapida.",
            "",
            "3. Tipo Newton tiene orden 2 (el mas rapido).",
            "4. Aitken acelera cualquier g(x) convergente.",
            "",
            "5. Si ninguna converge, proba otro x0 mas cercano",
            "   a la raiz o usa Newton directamente.",
        ]:
            if linea == "":
                _espacio(c, 4)
            else:
                tk.Label(c, text=linea, bg=BG2, fg=MUTED,
                         font=("Consolas", 11)).pack(anchor="w", padx=14, pady=1)
        _espacio(c)


    def _usar_g(self, expr):
        """Carga la expresion en el campo g(x) y cambia el modo."""
        self.e_g.delete(0, tk.END)
        self.e_g.insert(0, expr)
        self._show_tab("Paso a paso")

    # ══════════════════════════════════════
    # CALCULAR
    # ══════════════════════════════════════
    def _calcular(self):
        try:
            # si modo f(x), primero hallar g si el campo esta vacio o es el default
            if self._modo.get() == "f":
                fexpr = self.e_f.get().strip()
                if fexpr and not self.e_g.get().strip():
                    messagebox.showinfo("Atencion",
                        "Primero usa el boton 'Hallar' para obtener g(x),\n"
                        "luego presiona Calcular.")
                    return

            gexpr = self.e_g.get().strip()
            if not gexpr:
                messagebox.showwarning("Atencion", "Ingresa g(x).")
                return

            x0  = float(eval(self.e_x0.get()))
            tol = float(eval(self.e_tol.get()))
            it  = int(self.e_it.get())

            raiz, hist, estado = aitken(gexpr, x0, tol, it)
            self._hist  = hist
            self._raiz  = raiz
            self._gexpr = gexpr
            self._x0    = x0

            self._render_convergencia(hist)
            self._render_tabla(hist)
            self._render_pasos(hist, gexpr, x0, tol)
            self._render_analisis(hist, raiz, tol, gexpr, estado)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ══════════════════════════════════════
    # GRAFICAR
    # ══════════════════════════════════════
    def _graficar(self):
        try:
            gexpr = self.e_g.get().strip()
            x0    = float(eval(self.e_x0.get()))
            xs    = np.linspace(x0 - 2, x0 + 2, 500)

            def safe(x):
                try:    return evaluar(gexpr, x)
                except: return float("nan")

            gys = [safe(x) for x in xs]
            ax  = self._ax_func
            ax.clear()
            self._style_ax(ax)
            ax.plot(xs, gys, color=PURPLE, linewidth=2, label="g(x)")
            ax.plot(xs, xs,  color=GREEN,  linewidth=1, linestyle=":", label="y = x")
            ax.axhline(0, color=BORDER, linewidth=0.8)

            if self._raiz is not None:
                ax.scatter([self._raiz], [self._raiz], color=ORANGE,
                           zorder=5, s=70, label=f"punto fijo = {self._raiz:.6f}")

            ax.legend(facecolor=BG3, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
            self._canvas_func.draw()
            self._show_tab("Grafico g(x)")
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
        ax.set_title("Convergencia — Aitken Delta^2", color=TEXT, fontsize=10, pad=10)
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
                f"{r['x1']:.8f}",
                f"{r['x2']:.8f}",
                f"{r['xhat']:.8f}",
                f"{r['error']:.2e}",
            ))

    # ══════════════════════════════════════
    # RENDER: PASO A PASO
    # ══════════════════════════════════════
    def _render_pasos(self, hist, gexpr, x0, tol):
        si = self._si
        for w in si.winfo_children():
            w.destroy()

        # ── config inicial
        _seccion(si, "Configuracion inicial", ACCENT)
        c = _card(si, ACCENT)
        _linea(c, "g(x)       = ", gexpr, PURPLE)
        _linea(c, "x0         = ", str(x0), ACCENT)
        _linea(c, "Tolerancia = ", str(tol), ACCENT)
        _linea(c, "Formula    = ", "x* = xn - (x1-xn)^2 / (x2 - 2*x1 + xn)", MUTED)
        _espacio(c)

        # ── iteraciones
        _seccion(si, "Iteraciones", GREEN)
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
        tk.Label(hdr, text=f"   xn = {r['xn']:.8f}",
                 bg=BG2, fg=MUTED, font=("Consolas", 11)).pack(side=tk.LEFT)

        num = (r['x1'] - r['xn'])**2
        den = r['x2'] - 2*r['x1'] + r['xn']

        for pre, val, col in [
            (f"x1 = g(xn) = g({r['xn']:.5f})",  f"  {r['x1']:.8f}",  PURPLE),
            (f"x2 = g(x1) = g({r['x1']:.5f})",  f"  {r['x2']:.8f}",  PURPLE),
            ( "num = (x1 - xn)^2",               f"  {num:.8f}",      MUTED),
            ( "den = x2 - 2*x1 + xn",            f"  {den:.8f}",      MUTED),
            ( "x*  = xn - num / den",             f"  {r['xhat']:.8f}", ACCENT),
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
        tk.Label(err_row, text="Error = |x* - xn| = ",
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
    def _render_analisis(self, hist, raiz, tol, gexpr, estado):
        info = analisis_aitken(hist, raiz, tol, gexpr)
        ta   = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANALISIS — AITKEN Delta^2\n\n", "title")
        w("OK ", "ok"); w(f"Convergio en "); w(str(info["iters"]), "info"); w(" iteraciones\n")
        w("OK ", "ok"); w(f"Punto fijo = "); w(f"{info['raiz']:.8f}\n\n", "info")
        w("OK ", "ok"); w(f"Error final:         "); w(f"{info['ue']:.2e}\n", "info")
        w("OK ", "ok"); w(f"Factor de reduccion: "); w(f"{info['factor']:.4f}", "info")
        w(f"  ->  convergencia "); w(info["tipo"] + "\n", "ok")
        w("OK ", "ok"); w("Aitken Delta^2 acelera punto fijo.\n\n")

        w("ESTADO\n", "title")
        est_map = {
            "convergencia":  ("OK convergencia alcanzada", "ok"),
            "division_cero": ("WARN division por cero — den = 0", "warn"),
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
    app  = AitkenApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()