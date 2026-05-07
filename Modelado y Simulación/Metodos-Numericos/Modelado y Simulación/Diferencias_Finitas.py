"""
Diferencias Finitas
====================
Referencia: Caceres, O. J. — Fundamentos de Modelado y Simulacion, 2 ed. 2026
            Cap. I — Diferencias Finitas (pag. 24-25)

Formulas (Caceres, pag. 24):

  PROGRESIVAS (forward):
    f'(x_i)  = [f(x_{i+1}) - f(x_i)] / h
    f''(x_i) = [f(x_{i+2}) - 2f(x_{i+1}) + f(x_i)] / h^2

  REGRESIVAS (backward):
    f'(x_i)  = [f(x_i) - f(x_{i-1})] / h
    f''(x_i) = [f(x_i) - 2f(x_{i-1}) + f(x_{i-2})] / h^2

  CENTRALES:
    f'(x_i)  = [f(x_{i+1}) - f(x_{i-1})] / (2h)
    f''(x_i) = [f(x_{i+1}) - 2f(x_i) + f(x_{i-1})] / h^2
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ══════════════════════════════════════
# PALETA — dark terminal
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
# ENTORNO SEGURO
# ══════════════════════════════════════
def _env(x_val):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x_val
    return e

def evaluar(expr, x_val):
    return eval(expr, {"__builtins__": {}}, _env(x_val))


# ══════════════════════════════════════
# LOGICA — DIFERENCIAS FINITAS
# Ref: Caceres pag. 24
# ══════════════════════════════════════

# ── PROGRESIVAS ──────────────────────
def df_progresiva_1(f, x, h):
    """f'(x) progresiva orden 1  (Caceres, pag. 24)."""
    return (f(x + h) - f(x)) / h

def df_progresiva_2(f, x, h):
    """f''(x) progresiva orden 1  (Caceres, pag. 24)."""
    return (f(x + 2*h) - 2*f(x + h) + f(x)) / h**2

# ── REGRESIVAS ───────────────────────
def df_regresiva_1(f, x, h):
    """f'(x) regresiva orden 1  (Caceres, pag. 24)."""
    return (f(x) - f(x - h)) / h

def df_regresiva_2(f, x, h):
    """f''(x) regresiva orden 1  (Caceres, pag. 24)."""
    return (f(x) - 2*f(x - h) + f(x - 2*h)) / h**2

# ── CENTRALES ────────────────────────
def df_central_1(f, x, h):
    """f'(x) central orden 2  (Caceres, pag. 24)."""
    return (f(x + h) - f(x - h)) / (2 * h)

def df_central_2(f, x, h):
    """f''(x) central orden 2  (Caceres, pag. 24)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# ── TABLA COMPLETA ───────────────────
def calcular_tabla_completa(fexpr, xs_list, h):
    """
    Genera tabla con f'(x) y f''(x) por los tres metodos
    en cada punto dado, usando diferencias finitas.
    Ref: Caceres pag. 24-25.
    """
    def f(x): return evaluar(fexpr, x)

    rows = []
    for x in xs_list:
        row = {
            "x":   x,
            "fx":  f(x),
            # progresivas
            "fp1": df_progresiva_1(f, x, h),
            "fp2": df_progresiva_2(f, x, h),
            # regresivas
            "fr1": df_regresiva_1(f, x, h),
            "fr2": df_regresiva_2(f, x, h),
            # centrales
            "fc1": df_central_1(f, x, h),
            "fc2": df_central_2(f, x, h),
        }
        rows.append(row)
    return rows

def calcular_punto_unico(fexpr, x, h):
    """Calcula todos los metodos en un solo punto con detalle."""
    def f(v): return evaluar(fexpr, v)

    vals = {
        "fx":    f(x),
        "fxph":  f(x + h),
        "fx2ph": f(x + 2*h),
        "fxmh":  f(x - h),
        "fx2mh": f(x - 2*h),
    }

    return {
        **vals,
        "fp1":  df_progresiva_1(f, x, h),
        "fp2":  df_progresiva_2(f, x, h),
        "fr1":  df_regresiva_1(f, x, h),
        "fr2":  df_regresiva_2(f, x, h),
        "fc1":  df_central_1(f, x, h),
        "fc2":  df_central_2(f, x, h),
    }


# ══════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════
def _lbl(parent, text, bg=BG2, fg=MUTED, font=("Consolas", 10)):
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
                 font=("Segoe UI", 15, "bold"),
                 padx=12, pady=7, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h):
    r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0,int(r*.8)), max(0,int(g*.8)), max(0,int(b*.8)))


# ══════════════════════════════════════
# CLASE PRINCIPAL — DIFERENCIAS FINITAS
# ══════════════════════════════════════
class DiferenciasFinitasApp(tk.Frame):

    TABS = [
        ("📊", "Grafico"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🧠", "Analisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)
        if standalone:
            master.title("Diferencias Finitas — Metodos Numericos")
            master.configure(bg=BG)
            master.geometry("1280x720")
            master.minsize(980, 580)
        self._rows  = []
        self._build()

    # ──────────── LAYOUT ────────────
    def _build(self):
        self._topbar()
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        self._sidebar(body)
        self._main_area(body)

    def _topbar(self):
        bar = tk.Frame(self, bg=BG2, height=44)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="  Diferencias Finitas",
                 bg=BG2, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar,
                 text="Progresivas  |  Regresivas  |  Centrales  |  Ref: Caceres 2026 pag.24",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=340)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        _lbl(inner, "FUNCION Y PARAMETROS", fg=MUTED,
             font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 6))

        self.e_f = _labeled_entry(inner, "f(x)", "x**3 - 2*x + 1")
        self.e_x = _labeled_entry(inner, "x  (punto a derivar)", "2")
        self.e_h = _labeled_entry(inner, "h  (paso)", "0.1")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=4)
        _lbl(inner, "MULTIPLES PUNTOS", fg=MUTED,
             font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(4, 4))

        self.e_xs = _labeled_entry(inner, "x_i (sep. por coma)", "0, 1, 2, 3")

        # selector de metodo
        _lbl(inner, "Metodo para el grafico:").pack(anchor="w")
        self.metodo_var = tk.StringVar(value="Central")
        frame_radio = tk.Frame(inner, bg=BG2)
        frame_radio.pack(anchor="w", pady=(2, 8))
        for m in ["Progresiva", "Regresiva", "Central"]:
            tk.Radiobutton(frame_radio, text=m, variable=self.metodo_var,
                           value=m, bg=BG2, fg=MUTED,
                           selectcolor=BG3, activebackground=BG2,
                           font=("Segoe UI", 11)).pack(anchor="w")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)
        _btn(inner, "Calcular punto unico", self._calcular_punto).pack(fill=tk.X, pady=3)
        _btn(inner, "Tabla multiples puntos", self._calcular_tabla,
             color=BG3, fg=ACCENT).pack(fill=tk.X, pady=3)
        _btn(inner, "Graficar f(x) y f'(x)", self._graficar,
             color=BG3, fg=GREEN).pack(fill=tk.X, pady=3)

    def _main_area(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tab_bar = tk.Frame(right, bg=BG2, height=40)
        self._tab_bar.pack(fill=tk.X)
        self._tab_bar.pack_propagate(False)

        self._tab_btns   = {}
        self._tab_frames = {}

        for icon, name in self.TABS:
            b = tk.Label(self._tab_bar, text=f"{icon} {name}",
                         bg=BG2, fg=MUTED, font=("Segoe UI", 13),
                         padx=14, pady=12, cursor="hand2")
            b.pack(side=tk.LEFT)
            b.bind("<Button-1>", lambda e, n=name: self._show_tab(n))
            self._tab_btns[name] = b

        self._panels = tk.Frame(right, bg=BG)
        self._panels.pack(fill=tk.BOTH, expand=True)

        self._build_panel_grafico()
        self._build_panel_tabla()
        self._build_panel_steps()
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
        self._ax1 = self._fig.add_subplot(211)
        self._ax2 = self._fig.add_subplot(212)
        self._fig.subplots_adjust(hspace=0.4)
        for ax in (self._ax1, self._ax2):
            self._style_ax(ax)
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
                        font=("Segoe UI", 15, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])

        cols = ("x", "f(x)", "f'prog", "f''prog", "f'reg", "f''reg", "f'cent", "f''cent")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        for col in cols:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=95, anchor="e")

        # scrollbar horizontal
        hsb = ttk.Scrollbar(f, orient="horizontal", command=self._tree.xview)
        self._tree.configure(xscrollcommand=hsb.set)
        vsb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._tree.pack(fill=tk.BOTH, expand=True)

    # ──────────── PANEL: PASO A PASO ────────────
    def _build_panel_steps(self):
        f = self._panel("Paso a paso")
        wrap = tk.Frame(f, bg=BG)
        wrap.pack(fill=tk.BOTH, expand=True)
        self._sc = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(wrap, orient="vertical", command=self._sc.yview)
        self._sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._si = tk.Frame(self._sc, bg=BG)
        self._sw = self._sc.create_window((0,0), window=self._si, anchor="nw")
        self._si.bind("<Configure>",
            lambda e: self._sc.configure(scrollregion=self._sc.bbox("all")))
        self._sc.bind("<Configure>",
            lambda e: self._sc.itemconfig(self._sw, width=e.width))
        self._sc.bind_all("<MouseWheel>",
            lambda e: self._sc.yview_scroll(int(-1*(e.delta/120)), "units"))

    # ──────────── PANEL: ANALISIS ────────────
    def _build_panel_analisis(self):
        f = self._panel("Analisis")
        self._ta = tk.Text(f, bg=BG3, fg=TEXT,
                           font=("Consolas", 12), bd=0, padx=20, pady=16,
                           relief="flat", wrap="word", state="disabled")
        self._ta.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        for tag, col in [("title", ACCENT), ("ok", GREEN), ("warn", YELLOW),
                          ("info", PURPLE), ("muted", MUTED), ("val", ORANGE)]:
            kw = {"foreground": col}
            if tag == "title":
                kw["font"] = ("Consolas", 11, "bold")
            self._ta.tag_config(tag, **kw)

    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)

    # ──────────── CALCULAR PUNTO UNICO ────────────
    def _calcular_punto(self):
        try:
            fexpr = self.e_f.get().strip()
            x     = float(eval(self.e_x.get()))
            h     = float(eval(self.e_h.get()))
            v     = calcular_punto_unico(fexpr, x, h)

            self._render_pasos_punto(fexpr, x, h, v)
            self._render_analisis_punto(fexpr, x, h, v)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── CALCULAR TABLA ────────────
    def _calcular_tabla(self):
        try:
            fexpr = self.e_f.get().strip()
            xs    = [float(v.strip()) for v in self.e_xs.get().split(",")]
            h     = float(eval(self.e_h.get()))
            rows  = calcular_tabla_completa(fexpr, xs, h)
            self._rows = rows
            self._render_tabla(rows)
            self._show_tab("Tabla")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── GRAFICAR ────────────
    def _graficar(self):
        try:
            fexpr  = self.e_f.get().strip()
            x_ref  = float(eval(self.e_x.get()))
            h      = float(eval(self.e_h.get()))
            metodo = self.metodo_var.get()

            def f(x): return evaluar(fexpr, x)

            xs = np.linspace(x_ref - 3, x_ref + 3, 400)
            ys = [f(xi) for xi in xs]

            # derivada numerica en cada punto
            dy = []
            for xi in xs:
                try:
                    if metodo == "Progresiva":
                        dy.append(df_progresiva_1(f, xi, h))
                    elif metodo == "Regresiva":
                        dy.append(df_regresiva_1(f, xi, h))
                    else:
                        dy.append(df_central_1(f, xi, h))
                except Exception:
                    dy.append(float("nan"))

            for ax in (self._ax1, self._ax2):
                ax.clear()
                self._style_ax(ax)

            self._ax1.plot(xs, ys, color=ACCENT, linewidth=2, label="f(x)")
            self._ax1.axhline(0, color=BORDER, linewidth=0.8)
            self._ax1.axvline(x_ref, color=YELLOW, linewidth=1,
                              linestyle="--", alpha=0.6, label=f"x={x_ref}")
            self._ax1.scatter([x_ref], [f(x_ref)], color=ORANGE, s=60, zorder=5)
            self._ax1.set_title("f(x)", color=TEXT, fontsize=9)
            self._ax1.legend(facecolor=BG3, edgecolor=BORDER,
                             labelcolor=TEXT, fontsize=8)

            self._ax2.plot(xs, dy, color=GREEN, linewidth=2,
                           label=f"f'(x) — {metodo}")
            self._ax2.axhline(0, color=BORDER, linewidth=0.8)
            self._ax2.axvline(x_ref, color=YELLOW, linewidth=1,
                              linestyle="--", alpha=0.6)
            self._ax2.set_title("f'(x) aproximada", color=TEXT, fontsize=9)
            self._ax2.legend(facecolor=BG3, edgecolor=BORDER,
                             labelcolor=TEXT, fontsize=8)

            self._canvas.draw()
            self._show_tab("Grafico")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── RENDER: TABLA ────────────
    def _render_tabla(self, rows):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for r in rows:
            self._tree.insert("", "end", values=(
                f"{r['x']:.4f}",
                f"{r['fx']:.6f}",
                f"{r['fp1']:.6f}",
                f"{r['fp2']:.6f}",
                f"{r['fr1']:.6f}",
                f"{r['fr2']:.6f}",
                f"{r['fc1']:.6f}",
                f"{r['fc2']:.6f}",
            ))

    # ──────────── RENDER: PASO A PASO PUNTO UNICO ────────────
    def _render_pasos_punto(self, fexpr, x, h, v):
        for w in self._si.winfo_children():
            w.destroy()

        # bloque config
        cfg = tk.Frame(self._si, bg=BG3,
                       highlightthickness=1, highlightbackground=BORDER)
        cfg.pack(fill=tk.X, padx=12, pady=(12, 8))
        tk.Label(cfg, text="Configuracion inicial", bg=BG3, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=14, pady=(10,4))
        for label, val, col in [
            ("f(x)",      fexpr,     ACCENT),
            ("x",         str(x),    ACCENT),
            ("h (paso)",  str(h),    YELLOW),
            ("f(x)",      f"{v['fx']:.8f}",    GREEN),
            ("f(x+h)",    f"{v['fxph']:.8f}",  MUTED),
            ("f(x+2h)",   f"{v['fx2ph']:.8f}", MUTED),
            ("f(x-h)",    f"{v['fxmh']:.8f}",  MUTED),
            ("f(x-2h)",   f"{v['fx2mh']:.8f}", MUTED),
        ]:
            row = tk.Frame(cfg, bg=BG3)
            row.pack(anchor="w", padx=14, pady=1)
            tk.Label(row, text=f"-- {label} = ", bg=BG3, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG3, fg=col,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
        tk.Frame(cfg, bg=BG3, height=8).pack()

        # ── los 3 metodos como bloques
        metodos = [
            ("PROGRESIVA", GREEN,
             [
                ("f'(x)",  f"[f(x+h) - f(x)] / h",
                 f"[{v['fxph']:.6f} - {v['fx']:.6f}] / {h}",
                 f"{v['fp1']:.8f}"),
                ("f''(x)", f"[f(x+2h) - 2f(x+h) + f(x)] / h^2",
                 f"[{v['fx2ph']:.6f} - 2*{v['fxph']:.6f} + {v['fx']:.6f}] / {h}^2",
                 f"{v['fp2']:.8f}"),
             ]),
            ("REGRESIVA", PURPLE,
             [
                ("f'(x)",  f"[f(x) - f(x-h)] / h",
                 f"[{v['fx']:.6f} - {v['fxmh']:.6f}] / {h}",
                 f"{v['fr1']:.8f}"),
                ("f''(x)", f"[f(x) - 2f(x-h) + f(x-2h)] / h^2",
                 f"[{v['fx']:.6f} - 2*{v['fxmh']:.6f} + {v['fx2mh']:.6f}] / {h}^2",
                 f"{v['fr2']:.8f}"),
             ]),
            ("CENTRAL", ACCENT,
             [
                ("f'(x)",  f"[f(x+h) - f(x-h)] / (2h)",
                 f"[{v['fxph']:.6f} - {v['fxmh']:.6f}] / (2*{h})",
                 f"{v['fc1']:.8f}"),
                ("f''(x)", f"[f(x+h) - 2f(x) + f(x-h)] / h^2",
                 f"[{v['fxph']:.6f} - 2*{v['fx']:.6f} + {v['fxmh']:.6f}] / {h}^2",
                 f"{v['fc2']:.8f}"),
             ]),
        ]

        for nombre, color, lineas in metodos:
            outer = tk.Frame(self._si, bg=BG)
            outer.pack(fill=tk.X, padx=12, pady=4)
            tk.Frame(outer, bg=color, width=3).pack(side=tk.LEFT, fill=tk.Y)
            inner = tk.Frame(outer, bg=BG2)
            inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # header
            hdr = tk.Frame(inner, bg=BG2)
            hdr.pack(anchor="w", padx=12, pady=(8,4))
            tk.Label(hdr, text=f" {nombre} ", bg=color, fg="#000",
                     font=("Segoe UI", 11, "bold"), padx=6, pady=2).pack(side=tk.LEFT)
            tk.Label(hdr, text=f"  Ref: Caceres pag. 24",
                     bg=BG2, fg=MUTED, font=("Consolas", 10)).pack(side=tk.LEFT)

            for deriv, formula, sustit, resultado in lineas:
                # formula
                fr = tk.Frame(inner, bg=BG2)
                fr.pack(anchor="w", padx=12, pady=(3,0))
                tk.Label(fr, text=f"-- {deriv} = ", bg=BG2, fg=MUTED,
                         font=("Consolas", 11)).pack(side=tk.LEFT)
                tk.Label(fr, text=formula, bg=BG2, fg=MUTED,
                         font=("Consolas", 11)).pack(side=tk.LEFT)
                # sustitucion
                fs = tk.Frame(inner, bg=BG2)
                fs.pack(anchor="w", padx=24, pady=0)
                tk.Label(fs, text="= ", bg=BG2, fg=TEXT,
                         font=("Consolas", 11)).pack(side=tk.LEFT)
                tk.Label(fs, text=sustit, bg=BG2, fg=MUTED,
                         font=("Consolas", 10)).pack(side=tk.LEFT)
                # resultado
                fres = tk.Frame(inner, bg=BG2)
                fres.pack(anchor="w", padx=24, pady=(0,4))
                tk.Label(fres, text="= ", bg=BG2, fg=TEXT,
                         font=("Consolas", 11)).pack(side=tk.LEFT)
                tk.Label(fres, text=resultado, bg=BG2, fg=color,
                         font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

            tk.Frame(inner, bg=BG2, height=6).pack()

    # ──────────── RENDER: ANALISIS ────────────
    def _render_analisis_punto(self, fexpr, x, h, v):
        ta = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANALISIS — DIFERENCIAS FINITAS\n", "title")
        w("Ref: Caceres, Modelado y Simulacion, 2 ed. 2026, pag. 24\n\n", "muted")

        w("PARAMETROS\n", "title")
        w(f"  f(x) = {fexpr}\n", "info")
        w(f"  x    = {x}\n", "info")
        w(f"  h    = {h}\n\n", "info")

        w("VALORES CALCULADOS\n", "title")
        w(f"  f(x)    = {v['fx']:.10f}\n")
        w(f"  f(x+h)  = {v['fxph']:.10f}\n")
        w(f"  f(x-h)  = {v['fxmh']:.10f}\n")
        w(f"  f(x+2h) = {v['fx2ph']:.10f}\n")
        w(f"  f(x-2h) = {v['fx2mh']:.10f}\n\n")

        w("PRIMERA DERIVADA f'(x)\n", "title")
        w(f"  Progresiva:  "); w(f"{v['fp1']:.8f}\n", "ok")
        w(f"  Regresiva:   "); w(f"{v['fr1']:.8f}\n", "info")
        w(f"  Central:     "); w(f"{v['fc1']:.8f}\n", "val")

        w("\nSEGUNDA DERIVADA f''(x)\n", "title")
        w(f"  Progresiva:  "); w(f"{v['fp2']:.8f}\n", "ok")
        w(f"  Regresiva:   "); w(f"{v['fr2']:.8f}\n", "info")
        w(f"  Central:     "); w(f"{v['fc2']:.8f}\n", "val")

        # comparar errores entre metodos usando central como referencia
        w("\nCOMPARACION (central como referencia)\n", "title")
        err_p = abs(v['fp1'] - v['fc1'])
        err_r = abs(v['fr1'] - v['fc1'])
        w(f"  |f'_prog - f'_cent| = "); w(f"{err_p:.2e}\n", "warn")
        w(f"  |f'_reg  - f'_cent| = "); w(f"{err_r:.2e}\n", "warn")
        w("\n  El metodo central tiene error O(h^2),\n", "ok")
        w("  prog/reg tienen error O(h).\n", "muted")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = DiferenciasFinitasApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()