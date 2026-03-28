"""
Polinomio Interpolante de Lagrange
===================================
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Polinomio Interpolante de Lagrange (pag. 20-23)

Pestanas:
  1. Grafico
  2. Tabla L_i
  3. Paso a paso    — construccion completa del polinomio
  4. Error Local    — |f(x) - P(x)| en un punto dado
  5. Error Global   — cota teorica segun Caceres pag. 21-22
  6. Analisis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
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
# LOGICA — LAGRANGE  (Caceres pag. 20-23)
# ══════════════════════════════════════
def _math_env(x_val=None):
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["np"] = np
    if x_val is not None:
        env["x"] = x_val
    return env

def evaluar_expr(expr, x_val):
    return eval(expr, {"__builtins__": {}}, _math_env(x_val))

def base_lagrange(i, x, xs):
    """L_i(x) = prod_{j!=i} (x-x_j)/(x_i-x_j)  —  Caceres pag.20"""
    li = 1.0
    for j in range(len(xs)):
        if j != i:
            li *= (x - xs[j]) / (xs[i] - xs[j])
    return li

def polinomio_lagrange(x, xs, ys):
    """P(x) = sum y_i * L_i(x)  —  Caceres pag.20"""
    return sum(float(ys[i]) * base_lagrange(i, x, xs) for i in range(len(xs)))

def reconstruir_poly1d(xs, ys):
    """Devuelve np.poly1d con coeficientes del polinomio."""
    n    = len(xs)
    poly = np.poly1d([0.0])
    for i in range(n):
        li = np.poly1d([1.0])
        for j in range(n):
            if j != i:
                li = li * np.poly1d([1.0, -float(xs[j])]) / (float(xs[i]) - float(xs[j]))
        poly = poly + float(ys[i]) * li
    return poly

def derivada_numerica(expr, x, orden=1, h=1e-5):
    """Derivada numerica de orden 1..5 en el punto x."""
    f = lambda v: evaluar_expr(expr, v)
    for _ in range(orden):
        fx = lambda v, fn=f: (fn(v+h) - fn(v-h)) / (2*h)
        f  = fx
    return f(x)

def max_derivada_intervalo(expr, orden, a, b, puntos=500):
    """Busca max |f^(n+1)(x)| en [a,b] numericamente."""
    xs_scan = np.linspace(a, b, puntos)
    vals = []
    for xi in xs_scan:
        try:
            vals.append(abs(derivada_numerica(expr, xi, orden)))
        except Exception:
            pass
    return max(vals) if vals else float("nan")

def tabla_li(xs, ys, x_eval):
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
                 font=("Segoe UI", 13, "bold"),
                 padx=12, pady=8, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h):
    r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0,int(r*.8)), max(0,int(g*.8)), max(0,int(b*.8)))

def _scrollable_frame(parent):
    """Devuelve (canvas, inner_frame) con scroll vertical."""
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
    return sc, inner


# ══════════════════════════════════════
# BLOQUES VISUALES REUTILIZABLES
# ══════════════════════════════════════
def _seccion(parent, titulo, color=ACCENT):
    """Titulo de seccion con linea de color."""
    f = tk.Frame(parent, bg=BG)
    f.pack(fill=tk.X, padx=12, pady=(14, 4))
    tk.Frame(f, bg=color, width=4).pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(f, text=f"  {titulo}", bg=BG, fg=color,
             font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=4)
    return f

def _card(parent, color=ACCENT):
    """Card con barra lateral de color."""
    outer = tk.Frame(parent, bg=BG)
    outer.pack(fill=tk.X, padx=12, pady=4)
    tk.Frame(outer, bg=color, width=3).pack(side=tk.LEFT, fill=tk.Y)
    inner = tk.Frame(outer, bg=BG2)
    inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return inner

def _linea(parent, label, valor, color_val=ACCENT, mono=True):
    """Fila: label = valor."""
    row = tk.Frame(parent, bg=BG2)
    row.pack(anchor="w", padx=14, pady=2)
    font = ("Consolas", 11) if mono else ("Segoe UI", 11)
    tk.Label(row, text=label, bg=BG2, fg=MUTED, font=font).pack(side=tk.LEFT)
    tk.Label(row, text=valor, bg=BG2, fg=color_val,
             font=(font[0], font[1], "bold")).pack(side=tk.LEFT)

def _badge(parent, texto, color):
    tk.Label(parent, text=f" {texto} ", bg=color, fg="#000",
             font=("Segoe UI", 10, "bold"), padx=6, pady=2).pack(
                 side=tk.LEFT, padx=(0, 8))

def _espacio(parent, h=8):
    tk.Frame(parent, bg=BG2, height=h).pack()


# ══════════════════════════════════════
# CLASE PRINCIPAL
# ══════════════════════════════════════
class LagrangeApp(tk.Frame):

    TABS = [
        ("📊", "Grafico"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🔴", "Error Local"),
        ("🌐", "Error Global"),
        ("🧠", "Analisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Lagrange — Metodos Numericos")
            master.configure(bg=BG)
            master.geometry("1340x740")
            master.minsize(1050, 600)
        self._poly   = None
        self._xs     = np.array([])
        self._ys     = np.array([])
        self._x_eval = 0.0
        self._px     = 0.0
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
        sb = tk.Frame(parent, bg=BG2, width=350)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)
        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        _lbl(inner, "PUNTOS  (x_i, y_i)", fg=MUTED,
             font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 4))
        _lbl(inner, "Acepta: numeros, pi, e, sqrt(2), sin(pi/2)...").pack(anchor="w")

        self.e_xs   = _labeled_entry(inner, "x_i  (separados por coma)", "0, 1, 2, 3, 4")
        self.e_ys   = _labeled_entry(inner, "y_i  (separados por coma)", "1, 2, 0, 2, 3")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=4)
        _lbl(inner, "EVALUAR EN", fg=MUTED,
             font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4, 4))
        self.e_xeval = _labeled_entry(inner, "x a evaluar  P(x)", "1.5")

        _lbl(inner, "f(x) real  —  necesaria para calcular errores").pack(anchor="w")
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

    # ──────────── PANEL: GRAFICO ────────────
    def _build_panel_grafico(self):
        f = self._panel("Grafico")
        self._fig = Figure(figsize=(8, 5), facecolor=BG)
        self._ax  = self._fig.add_subplot(111)
        self._style_ax(self._ax)
        self._canvas = FigureCanvasTkAgg(self._fig, master=f)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────── PANEL: TABLA ────────────
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
        cols = ("i", "x_i", "y_i", "L_i(x)", "y_i * L_i(x)")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        for col, w in zip(cols, [50, 130, 130, 160, 160]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="e")
        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

    # ──────────── PANEL: PASO A PASO ────────────
    def _build_panel_steps(self):
        f = self._panel("Paso a paso")
        _, self._si = _scrollable_frame(f)

    # ──────────── PANEL: ERROR LOCAL ────────────
    def _build_panel_error_local(self):
        f = self._panel("Error Local")
        _, self._si_el = _scrollable_frame(f)

    # ──────────── PANEL: ERROR GLOBAL ────────────
    def _build_panel_error_global(self):
        f = self._panel("Error Global")
        _, self._si_eg = _scrollable_frame(f)

    # ──────────── PANEL: ANALISIS ────────────
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
        x_eval = _ev(self.e_xeval.get())
        return np.array(xs), np.array(ys), x_eval

    def _get_fexpr(self):
        return self.e_fx.get().strip()

    def _eval_fx(self, x):
        expr = self._get_fexpr()
        if not expr:
            return None
        return evaluar_expr(expr, x)

    # ──────────── CALCULAR ────────────
    def _calcular(self):
        try:
            xs, ys, x_eval = self._parse_inputs()
            self._xs     = xs
            self._ys     = ys
            self._x_eval = x_eval

            poly = reconstruir_poly1d(xs, ys)
            self._poly = poly

            px = polinomio_lagrange(x_eval, xs, ys)
            self._px = px

            filas, total = tabla_li(xs, ys, x_eval)

            self._render_tabla(filas, total)
            self._render_pasos(xs, ys, x_eval, px, filas, poly)
            self._render_error_local(xs, ys, x_eval, px)
            self._render_error_global(xs, ys, x_eval, px)
            self._render_analisis(xs, ys, x_eval, px, poly)
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
            ax.plot(x_plot, y_poly, color=ACCENT, linewidth=2, label="P(x) — Lagrange")

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
            "—", "—", "—", "TOTAL  P(x) =", f"{total:.8f}"
        ))

    # ══════════════════════════════════════
    # RENDER: PASO A PASO — construccion del polinomio
    # ══════════════════════════════════════
    def _render_pasos(self, xs, ys, x_eval, px, filas, poly):
        si = self._si
        for w in si.winfo_children():
            w.destroy()

        n     = len(xs)
        grado = n - 1

        # ── BLOQUE 1: Datos del problema
        _seccion(si, "PASO 1 — Datos del problema", ACCENT)
        c = _card(si, ACCENT)
        _linea(c, "Cantidad de puntos:  ", f"n+1 = {n}  →  polinomio de grado {grado}")
        _linea(c, "Puntos dados:        ", "")
        for i in range(n):
            _linea(c, f"   x_{i} = {xs[i]:.6f}", f"   y_{i} = {ys[i]:.6f}", PURPLE)
        _linea(c, "Evaluar en:          ", f"x = {x_eval:.8f}", YELLOW)
        _espacio(c)

        # ── BLOQUE 2: Formula general
        _seccion(si, "PASO 2 — Formula general de Lagrange", PURPLE)
        c = _card(si, PURPLE)
        _linea(c, "Formula:  ", "P(x) = y_0*L_0(x) + y_1*L_1(x) + ... + y_n*L_n(x)")
        _linea(c, "Donde:    ", "L_i(x) = producto de (x - x_j)/(x_i - x_j)  para j != i", MUTED)
        _espacio(c)

        # ── BLOQUE 3: Calculo de cada L_i(x)
        _seccion(si, "PASO 3 — Calculo de cada base L_i(x)", ORANGE)
        for r in filas:
            i  = r["i"]
            xi = r["xi"]
            yi = r["yi"]

            c = _card(si, ORANGE)

            # header con badge
            hdr = tk.Frame(c, bg=BG2)
            hdr.pack(anchor="w", padx=14, pady=(8, 4))
            _badge(hdr, f"  i = {i}  ", ORANGE)
            tk.Label(hdr, text=f"x_{i} = {xi:.6f}   |   y_{i} = {yi:.6f}",
                     bg=BG2, fg=MUTED, font=("Consolas", 11)).pack(side=tk.LEFT)

            # formula simbolica
            num_sym = " · ".join(
                f"(x - x_{j})" for j in range(n) if j != i
            )
            den_sym = " · ".join(
                f"(x_{i} - x_{j})" for j in range(n) if j != i
            )
            _linea(c, "Formula simbolica:", f"L_{i}(x) = [ {num_sym} ] / [ {den_sym} ]", MUTED)

            # sustitucion del numerador
            num_sust = " · ".join(
                f"({x_eval:.4f} - {xs[j]:.4f})" for j in range(n) if j != i
            )
            num_vals = [x_eval - xs[j] for j in range(n) if j != i]
            num_prod = 1.0
            for v in num_vals:
                num_prod *= v
            num_str2 = " · ".join(f"{v:.6f}" for v in num_vals)
            _linea(c, "Numerador:        ", f"{num_sust}")
            _linea(c, "             =    ", f"{num_str2}  =  {num_prod:.8f}", GREEN)

            # sustitucion del denominador
            den_sust = " · ".join(
                f"({xi:.4f} - {xs[j]:.4f})" for j in range(n) if j != i
            )
            den_vals = [xi - float(xs[j]) for j in range(n) if j != i]
            den_prod = 1.0
            for v in den_vals:
                den_prod *= v
            den_str2 = " · ".join(f"{v:.6f}" for v in den_vals)
            _linea(c, "Denominador:      ", f"{den_sust}")
            _linea(c, "             =    ", f"{den_str2}  =  {den_prod:.8f}", RED)

            # resultado L_i
            _linea(c, f"L_{i}(x) =         ",
                   f"{num_prod:.8f} / {den_prod:.8f}  =  {r['Li']:.8f}", ACCENT)

            # contribucion y_i * L_i
            _linea(c, f"y_{i} * L_{i}(x) =  ",
                   f"{yi:.6f} * {r['Li']:.8f}  =  {r['yiLi']:.8f}", PURPLE)
            _espacio(c)

        # ── BLOQUE 4: Suma final P(x)
        _seccion(si, "PASO 4 — Construccion final  P(x)", GREEN)
        c = _card(si, GREEN)

        suma_partes = "\n             +  ".join(
            f"y_{r['i']}*L_{r['i']}(x) = {r['yiLi']:.8f}" for r in filas
        )
        _linea(c, "P(x) =  ", f"y_0*L_0 + y_1*L_1 + ... + y_{n-1}*L_{n-1}")
        _linea(c, "     =  ", suma_partes, MUTED)
        _linea(c, f"P({x_eval:.4f}) = ", f"{px:.8f}", GREEN)

        # Polinomio expandido
        try:
            coefs = np.round(poly.coefficients, 6)
            terminos = []
            grd = len(coefs) - 1
            for k, c_k in enumerate(coefs):
                exp = grd - k
                if abs(c_k) < 1e-10:
                    continue
                if exp == 0:
                    terminos.append(f"{c_k:+.4f}")
                elif exp == 1:
                    terminos.append(f"{c_k:+.4f}x")
                else:
                    terminos.append(f"{c_k:+.4f}x^{exp}")
            poly_str = "  ".join(terminos) if terminos else "0"
            c2 = _card(si, PURPLE)
            _linea(c2, "Polinomio expandido:", "")
            _linea(c2, "P(x) = ", poly_str, PURPLE)
            _espacio(c2)
        except Exception:
            pass

    # ══════════════════════════════════════
    # RENDER: ERROR LOCAL
    # ══════════════════════════════════════
    def _render_error_local(self, xs, ys, x_eval, px):
        si = self._si_el
        for w in si.winfo_children():
            w.destroy()

        fexpr = self._get_fexpr()

        # ── DEFINICION
        _seccion(si, "ERROR LOCAL — Definicion", RED)
        c = _card(si, RED)
        _linea(c, "Formula:     ", "|E(x)| = |f(x) - P(x)|", RED)
        _linea(c, "Referencia:  ", "Caceres, pag. 22 — 'Error local'", MUTED)
        _linea(c, "Significado: ",
               "diferencia entre el valor real y el polinomio en un punto dado", MUTED)
        _espacio(c)

        if not fexpr:
            _seccion(si, "Atencion", YELLOW)
            c = _card(si, YELLOW)
            _linea(c, "Para calcular el error local necesitas ingresar", "", YELLOW)
            _linea(c, "la funcion real f(x) en el campo del sidebar.", "", YELLOW)
            _linea(c, "Ejemplo:  sin(x),  exp(x),  x**2", "", MUTED)
            return

        # ── PASO 1: valores conocidos
        _seccion(si, "PASO 1 — Valores conocidos", ACCENT)
        c = _card(si, ACCENT)
        _linea(c, "Punto de evaluacion:  ", f"x = {x_eval:.8f}", YELLOW)
        _linea(c, "Funcion real:         ", f"f(x) = {fexpr}", ACCENT)
        _linea(c, "Polinomio:            ", f"P(x) de grado {len(xs)-1}", ACCENT)
        _espacio(c)

        # ── PASO 2: evaluar f(x)
        _seccion(si, "PASO 2 — Evaluar f(x) en el punto", GREEN)
        c = _card(si, GREEN)
        try:
            fx_val = evaluar_expr(fexpr, x_eval)
            _linea(c, f"f({x_eval:.6f}) = ", f"{fexpr}  evaluada en x = {x_eval:.6f}", MUTED)
            _linea(c, "Resultado:           ", f"f(x) = {fx_val:.8f}", GREEN)
        except Exception as e:
            _linea(c, "Error al evaluar f(x): ", str(e), RED)
            return
        _espacio(c)

        # ── PASO 3: evaluar P(x)
        _seccion(si, "PASO 3 — Valor del polinomio P(x)", PURPLE)
        c = _card(si, PURPLE)
        _linea(c, f"P({x_eval:.6f}) = ", f"{px:.8f}", PURPLE)
        _espacio(c)

        # ── PASO 4: calcular error
        _seccion(si, "PASO 4 — Calculo del error local", RED)
        c = _card(si, RED)
        err = abs(fx_val - px)
        _linea(c, "Formula:       ", "|E(x)| = |f(x) - P(x)|")
        _linea(c, "Reemplazando:  ",
               f"|E| = |{fx_val:.8f} - {px:.8f}|")
        _linea(c, "               ",
               f"     = |{fx_val - px:.8f}|", MUTED)
        _linea(c, "ERROR LOCAL =  ", f"{err:.2e}   =   {err:.10f}", RED)
        _espacio(c)

        # ── PASO 5: interpretacion
        _seccion(si, "PASO 5 — Interpretacion", YELLOW)
        c = _card(si, YELLOW)
        if err < 1e-6:
            calidad = "MUY BUENA  (error < 1e-6)"
            color_k = GREEN
        elif err < 1e-3:
            calidad = "BUENA  (error < 1e-3)"
            color_k = ACCENT
        elif err < 1e-1:
            calidad = "ACEPTABLE  (error < 0.1)"
            color_k = YELLOW
        else:
            calidad = "ALTA  — revisar el polinomio o agregar mas puntos"
            color_k = RED
        _linea(c, "Precision:     ", calidad, color_k)
        _linea(c, "Error relativo:", f"{abs(err/fx_val)*100:.4f} %  (si f(x) != 0)" if fx_val != 0 else "f(x) = 0, no se puede calcular relativo", MUTED)
        _espacio(c)

    # ══════════════════════════════════════
    # RENDER: ERROR GLOBAL (cota teorica)
    # ══════════════════════════════════════
    def _render_error_global(self, xs, ys, x_eval, px):
        si = self._si_eg
        for w in si.winfo_children():
            w.destroy()

        n     = len(xs)
        grado = n - 1
        fexpr = self._get_fexpr()

        # ── DEFINICION
        _seccion(si, "ERROR GLOBAL — Cota teorica", ORANGE)
        c = _card(si, ORANGE)
        _linea(c, "Formula (Caceres pag. 21):", "")
        _linea(c, "  |E(x)| <=", " M_(n+1) / (n+1)!  *  |prod(x - x_i)|", ORANGE)
        _linea(c, "Donde:", "")
        _linea(c, "  M_(n+1) = ", "max |f^(n+1)(xi)| en el intervalo", MUTED)
        _linea(c, "  (n+1)!  = ", "factorial de n+1", MUTED)
        _linea(c, "  prod    = ", "producto (x-x_0)(x-x_1)...(x-x_n)", MUTED)
        _espacio(c)

        # ── PASO 1: orden de la derivada
        _seccion(si, "PASO 1 — Orden de la derivada necesaria", ACCENT)
        c = _card(si, ACCENT)
        orden = n   # f^(n+1), con n+1 puntos → derivada de orden n+1
        _linea(c, "n + 1 puntos:          ", f"{n}")
        _linea(c, "Grado del polinomio:   ", f"{grado}")
        _linea(c, "Derivada necesaria:    ", f"f^({orden})(x)  = derivada de orden {orden}", ACCENT)
        _linea(c, "Factorial (n+1)! = :   ",
               f"{orden}! = {math.factorial(orden)}", GREEN)
        _espacio(c)

        # ── PASO 2: intervalo de analisis
        a = min(xs)
        b = max(xs)
        _seccion(si, "PASO 2 — Intervalo de analisis", PURPLE)
        c = _card(si, PURPLE)
        _linea(c, "Intervalo:  ", f"[{a:.6f},  {b:.6f}]", PURPLE)
        _linea(c, "Basado en:  ", "los nodos x_0, x_1, ..., x_n", MUTED)
        _espacio(c)

        # ── PASO 3: producto omega(x)
        _seccion(si, "PASO 3 — Calculo de  omega(x) = prod(x - x_i)", GREEN)
        c = _card(si, GREEN)
        partes_sym = " · ".join(f"(x - x_{i})" for i in range(n))
        _linea(c, "Formula simbolica: ", f"omega(x) = {partes_sym}")

        partes_num = " · ".join(f"({x_eval:.4f} - {xs[i]:.4f})" for i in range(n))
        vals_omega = [x_eval - float(xs[i]) for i in range(n)]
        omega = 1.0
        for v in vals_omega:
            omega *= v
        vals_str = " · ".join(f"{v:.6f}" for v in vals_omega)
        _linea(c, "Evaluando en x:    ", f"omega({x_eval:.4f}) = {partes_num}")
        _linea(c, "                 = ", f"{vals_str}")
        _linea(c, "|omega(x)| =       ", f"{abs(omega):.8f}", GREEN)
        _espacio(c)

        # ── PASO 4: M_(n+1)
        _seccion(si, "PASO 4 — Estimacion de  M_(n+1) = max |f^(n+1)(x)|", YELLOW)
        c = _card(si, YELLOW)

        if not fexpr:
            _linea(c, "Necesitas ingresar f(x) en el sidebar", "", YELLOW)
            _linea(c, "para calcular M_(n+1) numericamente.", "", MUTED)
            # Igual mostramos la formula con M desconocida
            facto = math.factorial(orden)
            _linea(c, f"Cota = M_{orden} / {facto} * |omega|", "", ORANGE)
            _linea(c, "     = M / ?  (ingresa f(x) para calcular)", "", MUTED)
            return

        # derivada numerica de orden n+1 en grilla
        try:
            _linea(c, "Metodo:        ", f"derivada numerica de orden {orden} en {200} puntos del intervalo", MUTED)
            M = max_derivada_intervalo(fexpr, orden, a, b, puntos=200)
            _linea(c, f"M_{orden} = max |f^({orden})(x)| = ", f"{M:.6f}  (en [{a:.4f}, {b:.4f}])", YELLOW)
        except Exception as e:
            _linea(c, "Error al calcular derivada: ", str(e), RED)
            return
        _espacio(c)

        # ── PASO 5: cota final
        _seccion(si, "PASO 5 — Cota del error global", RED)
        c = _card(si, RED)
        facto   = math.factorial(orden)
        cota    = M / facto * abs(omega)
        _linea(c, "Formula:           ", f"|E(x)| <= M_{orden} / {orden}! * |omega(x)|")
        _linea(c, "Reemplazando:      ", f"|E| <= {M:.4f} / {facto} * {abs(omega):.6f}")
        _linea(c, "                 = ", f"{M:.4f} / {facto}  *  {abs(omega):.6f}")
        _linea(c, "COTA DE ERROR  =   ", f"{cota:.2e}   =   {cota:.10f}", RED)
        _espacio(c)

        # ── PASO 6: comparacion con error real
        fxval = None
        try:
            fxval = evaluar_expr(fexpr, x_eval)
        except Exception:
            pass

        if fxval is not None:
            _seccion(si, "PASO 6 — Verificacion con error real", ACCENT)
            c = _card(si, ACCENT)
            err_real = abs(fxval - px)
            _linea(c, "Error real:        ", f"|f(x) - P(x)| = {err_real:.2e}", ACCENT)
            _linea(c, "Cota teorica:      ", f"{cota:.2e}", ORANGE)
            ok = err_real <= cota * (1 + 1e-6)
            msg = "OK — el error real es menor que la cota" if ok else "REVISAR — el error supera la cota (puede ser por la derivada numerica)"
            color_v = GREEN if ok else RED
            _linea(c, "Verificacion:      ", msg, color_v)
            _espacio(c)

    # ══════════════════════════════════════
    # RENDER: ANALISIS
    # ══════════════════════════════════════
    def _render_analisis(self, xs, ys, x_eval, px, poly):
        ta = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        n = len(xs)
        w("ANALISIS COMPLETO — LAGRANGE\n", "title")
        w("Ref: Caceres, Modelado y Simulacion, 2 ed. 2026, pag. 20-23\n\n", "muted")

        w("DATOS\n", "title")
        w(f"  Puntos:       n+1 = {n}  →  grado = {n-1}\n")
        w(f"  x_i = {[round(float(v),6) for v in xs]}\n", "info")
        w(f"  y_i = {[round(float(v),6) for v in ys]}\n\n", "info")

        w("RESULTADO\n", "title")
        w(f"  x evaluado:   {x_eval:.8f}\n")
        w(f"  P(x) =        "); w(f"{px:.8f}\n", "ok")

        fexpr = self._get_fexpr()
        if fexpr:
            try:
                fx = evaluar_expr(fexpr, x_eval)
                err = abs(fx - px)
                w(f"  f(x) real =   "); w(f"{fx:.8f}\n", "info")
                w(f"  Error local = "); w(f"{err:.2e}\n\n", "warn")
            except Exception:
                pass

        w("POLINOMIO EXPANDIDO\n", "title")
        try:
            coefs = np.round(poly.coefficients, 8)
            terminos = []
            grd = len(coefs) - 1
            for k, ck in enumerate(coefs):
                exp = grd - k
                if abs(ck) < 1e-10:
                    continue
                if exp == 0:
                    terminos.append(f"{ck:+.4f}")
                elif exp == 1:
                    terminos.append(f"{ck:+.4f}x")
                else:
                    terminos.append(f"{ck:+.4f}x^{exp}")
            poly_str = "  ".join(terminos) if terminos else "0"
            w(f"  P(x) = {poly_str}\n\n", "info")
        except Exception:
            pass

        w("TEOREMA (Caceres pag. 20)\n", "title")
        w("  Existencia:  para n+1 puntos distintos siempre\n")
        w("               existe un unico polinomio de grado n.\n", "ok")
        w("  Unicidad:    el polinomio es unico.\n\n", "ok")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = LagrangeApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()