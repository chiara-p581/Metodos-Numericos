import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ══════════════════════════════════════
# PALETA — dark terminal
# ══════════════════════════════════════
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


# ══════════════════════════════════════
# LÓGICA — PUNTO FIJO
# ══════════════════════════════════════
def _env(x):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x
    return e

def evaluar(expr, x):
    return eval(expr, {"__builtins__": {}}, _env(x))

def derivada_num(expr, x):
    h = 1e-7
    return (evaluar(expr, x + h) - evaluar(expr, x - h)) / (2 * h)

def punto_fijo(fexpr, gexpr, x0, tol, max_iter):
    hist = []
    x    = x0

    for i in range(1, max_iter + 1):
        fx   = evaluar(fexpr, x)
        xnew = evaluar(gexpr, x)
        err  = abs(xnew - x)

        hist.append({"i": i, "xn": x, "xn1": xnew,
                     "fx": fx, "gx": xnew, "error": err})

        if err < tol:
            return xnew, hist, True
        x = xnew

    return x, hist, False

def analisis_punto_fijo(hist, raiz, tol, fexpr, gexpr, x0):
    errores = [r["error"] for r in hist if r["error"] > 0]
    ratios  = [errores[i]/errores[i-1] for i in range(1, len(errores))
               if errores[i-1] != 0]
    factor  = sum(ratios)/len(ratios) if ratios else 0
    tipo    = "rápida" if factor < 0.1 else ("moderada" if factor < 0.5 else "lenta")
    ue      = errores[-1] if errores else 0
    fval    = evaluar(fexpr, raiz)

    try:
        gp = abs(derivada_num(gexpr, x0))
        if gp < 1:
            conv_txt = f"|g'(x₀)| ≈ {gp:.4f} < 1  →  converge"
            conv_ok  = True
        else:
            conv_txt = f"|g'(x₀)| ≈ {gp:.4f} ≥ 1  →  puede diverger"
            conv_ok  = False
    except Exception:
        conv_txt = "No se pudo evaluar g'(x₀)"
        conv_ok  = None

    return {
        "iters":    len(hist),
        "raiz":     raiz,
        "fval":     fval,
        "ue":       ue,
        "factor":   factor,
        "tipo":     tipo,
        "tol":      tol,
        "ok":       ue < tol,
        "conv_txt": conv_txt,
        "conv_ok":  conv_ok,
    }

def sugerencias_g(fexpr):
    return [
        (f"x - 0.5*({fexpr})",  "Relajación media"),
        (f"x - 0.1*({fexpr})",  "Relajación suave"),
        (f"x - 0.01*({fexpr})", "Relajación muy suave"),
    ]


# ══════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════
def _labeled_entry(parent, label, default, bg=BG2):
    tk.Label(parent, text=label, bg=bg, fg=MUTED,
             font=("Consolas", 10)).pack(anchor="w")
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 font=("Consolas", 12), bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, relief="flat")
    e.insert(0, default)
    e.pack(fill=tk.X, ipady=7, pady=(2, 8))
    return e

def _btn(parent, text, cmd, color=ACCENT, fg="#000000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 15, "bold"),
                 padx=12, pady=7, cursor="hand2")
    b.bind("<Button-1>", lambda e: cmd())
    b.bind("<Enter>",    lambda e: b.config(bg=_darken(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _darken(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0, int(r*0.8)), max(0, int(g*0.8)), max(0, int(b*0.8)))


# ══════════════════════════════════════
# CLASE PRINCIPAL — PUNTO FIJO
# ══════════════════════════════════════
class PuntoFijoApp(tk.Frame):

    TABS = [
        ("📉", "Convergencia"),
        ("📊", "Función f(x)"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🧠", "Análisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)

        if standalone:
            master.title("Punto Fijo — Métodos Numéricos")
            master.configure(bg=BG)
            master.geometry("1200x700")
            master.minsize(900, 580)

        self._hist  = []
        self._raiz  = None
        self._fexpr = ""
        self._gexpr = ""

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
        tk.Label(bar, text="⚙  Punto Fijo", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar, text="Método de Punto Fijo  |  x = g(x)", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=280)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(inner, text="PARÁMETROS", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 8))

        self.e_f   = _labeled_entry(inner, "f(x)",           "2**(-x) - x")

        # g(x) con botón sugerir
        tk.Label(inner, text="g(x)  — función de iteración",
                 bg=BG2, fg=MUTED, font=("Consolas", 10)).pack(anchor="w")
        g_row = tk.Frame(inner, bg=BG2)
        g_row.pack(fill=tk.X, pady=(2, 8))

        self.e_g = tk.Entry(g_row, bg=BG3, fg=TEXT, insertbackground=TEXT,
                            font=("Consolas", 12), bd=0,
                            highlightthickness=1, highlightbackground=BORDER,
                            highlightcolor=ACCENT, relief="flat")
        self.e_g.insert(0, "2**(-x)")
        self.e_g.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7)

        sug_btn = tk.Label(g_row, text=" ✨ ", bg=BG3, fg=MUTED,
                           font=("Segoe UI", 11), cursor="hand2",
                           highlightthickness=1, highlightbackground=BORDER)
        sug_btn.pack(side=tk.LEFT, padx=(4, 0), ipady=4)
        sug_btn.bind("<Button-1>", self._show_suggest)
        sug_btn.bind("<Enter>",    lambda e: sug_btn.config(fg=PURPLE))
        sug_btn.bind("<Leave>",    lambda e: sug_btn.config(fg=MUTED))

        self.e_x0  = _labeled_entry(inner, "x₀  (punto inicial)", "0.5")
        self.e_tol = _labeled_entry(inner, "Tolerancia",           "1e-6")
        self.e_it  = _labeled_entry(inner, "Max iteraciones",      "100")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        _btn(inner, "▶  Calcular", self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "📈  Graficar f(x) y g(x)", self._graficar,
             color=BG3, fg=ACCENT).pack(fill=tk.X, pady=3)

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
                         bg=BG2, fg=MUTED,
                         font=("Segoe UI", 13),
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
        self._build_panel_analisis()

        self._show_tab("Paso a paso")

    # ──────────── TAB SWITCHING ────────────
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

    # ──────────── PANEL: CONVERGENCIA ────────────
    def _build_panel_conv(self):
        f = self._panel("Convergencia")
        self._fig_conv = Figure(figsize=(7, 4), facecolor=BG)
        self._ax_conv  = self._fig_conv.add_subplot(111)
        self._style_ax(self._ax_conv)
        self._canvas_conv = FigureCanvasTkAgg(self._fig_conv, master=f)
        self._canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────── PANEL: FUNCIÓN ────────────
    def _build_panel_func(self):
        f = self._panel("Función f(x)")
        self._fig_func = Figure(figsize=(7, 4), facecolor=BG)
        self._ax_func  = self._fig_func.add_subplot(111)
        self._style_ax(self._ax_func)
        self._canvas_func = FigureCanvasTkAgg(self._fig_func, master=f)
        self._canvas_func.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

        cols = ("i", "xₙ", "g(xₙ)", "f(xₙ)", "error")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        widths = [40, 130, 130, 130, 110]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="e")

        sb = ttk.Scrollbar(f, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)

    # ──────────── PANEL: PASO A PASO ────────────
    def _build_panel_steps(self):
        f = self._panel("Paso a paso")

        frame_scroll = tk.Frame(f, bg=BG)
        frame_scroll.pack(fill=tk.BOTH, expand=True)

        self._steps_canvas = tk.Canvas(frame_scroll, bg=BG,
                                        highlightthickness=0)
        vsb = tk.Scrollbar(frame_scroll, orient="vertical",
                            command=self._steps_canvas.yview)
        self._steps_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._steps_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._steps_inner = tk.Frame(self._steps_canvas, bg=BG)
        self._steps_win   = self._steps_canvas.create_window(
            (0, 0), window=self._steps_inner, anchor="nw")

        self._steps_inner.bind("<Configure>", self._on_steps_resize)
        self._steps_canvas.bind("<Configure>", self._on_canvas_resize)
        self._steps_canvas.bind_all("<MouseWheel>",
            lambda e: self._steps_canvas.yview_scroll(
                int(-1*(e.delta/120)), "units"))

    def _on_steps_resize(self, e):
        self._steps_canvas.configure(
            scrollregion=self._steps_canvas.bbox("all"))

    def _on_canvas_resize(self, e):
        self._steps_canvas.itemconfig(self._steps_win, width=e.width)

    # ──────────── PANEL: ANÁLISIS ────────────
    def _build_panel_analisis(self):
        f = self._panel("Análisis")
        self._txt_analisis = tk.Text(
            f, bg=BG3, fg=TEXT,
            font=("Consolas", 12), bd=0, padx=20, pady=16,
            relief="flat", wrap="word",
            state="disabled")
        self._txt_analisis.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        ta = self._txt_analisis
        ta.tag_config("title",  foreground=ACCENT,  font=("Consolas", 11, "bold"))
        ta.tag_config("ok",     foreground=GREEN)
        ta.tag_config("warn",   foreground=YELLOW)
        ta.tag_config("err",    foreground=RED)
        ta.tag_config("info",   foreground=PURPLE)
        ta.tag_config("muted",  foreground=MUTED)

    # ──────────── MATPLOTLIB STYLE ────────────
    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)

    # ──────────── CALCULAR ────────────
    def _calcular(self):
        try:
            fexpr = self.e_f.get().strip()
            gexpr = self.e_g.get().strip()
            x0    = float(eval(self.e_x0.get()))
            tol   = float(eval(self.e_tol.get()))
            it    = int(self.e_it.get())

            raiz, hist, converged = punto_fijo(fexpr, gexpr, x0, tol, it)
            self._hist  = hist
            self._raiz  = raiz
            self._fexpr = fexpr
            self._gexpr = gexpr
            self._x0    = x0

            self._render_convergencia(hist)
            self._render_tabla(hist)
            self._render_pasos(hist, fexpr, gexpr, x0, tol)
            self._render_analisis(hist, raiz, tol, fexpr, gexpr, x0, converged)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── GRAFICAR ────────────
    def _graficar(self):
        try:
            fexpr = self.e_f.get().strip()
            gexpr = self.e_g.get().strip()
            x0    = float(eval(self.e_x0.get()))
            margin = 2.0
            xs = np.linspace(x0 - margin, x0 + margin, 500)

            def safe(expr, x):
                try:    return evaluar(expr, x)
                except: return float("nan")

            ys  = [safe(fexpr, x) for x in xs]
            gys = [safe(gexpr, x) for x in xs]

            ax = self._ax_func
            ax.clear()
            self._style_ax(ax)
            ax.plot(xs, ys,  color=ACCENT,  linewidth=2, label="f(x)")
            ax.plot(xs, gys, color=PURPLE,  linewidth=2, linestyle="--", label="g(x)")
            ax.plot(xs, xs,  color=GREEN,   linewidth=1, linestyle=":",  label="y = x")
            ax.axhline(0, color=BORDER, linewidth=0.8)

            if self._raiz is not None:
                ax.scatter([self._raiz], [0], color=ORANGE,
                           zorder=5, s=70, label=f"raíz ≈ {self._raiz:.6f}")

            ax.legend(facecolor=BG3, edgecolor=BORDER,
                      labelcolor=TEXT, fontsize=8)
            self._canvas_func.draw()
            self._show_tab("Función f(x)")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── RENDER: CONVERGENCIA ────────────
    def _render_convergencia(self, hist):
        iters  = [r["i"]     for r in hist]
        errors = [r["error"] for r in hist]

        ax = self._ax_conv
        ax.clear()
        self._style_ax(ax)
        ax.semilogy(iters, errors, color=ACCENT, linewidth=2, marker="o",
                    markersize=4, markerfacecolor=ACCENT)
        ax.fill_between(iters, errors, alpha=0.08, color=ACCENT)
        ax.set_xlabel("Iteración", color=MUTED, fontsize=9)
        ax.set_ylabel("Error (log)", color=MUTED, fontsize=9)
        ax.set_title("Convergencia del error", color=TEXT, fontsize=10, pad=10)
        self._canvas_conv.draw()

    # ──────────── RENDER: TABLA ────────────
    def _render_tabla(self, hist):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for r in hist:
            self._tree.insert("", "end", values=(
                r["i"],
                f"{r['xn']:.8f}",
                f"{r['gx']:.8f}",
                f"{r['fx']:.8f}",
                f"{r['error']:.2e}",
            ))

    # ──────────── RENDER: PASO A PASO ────────────
    def _render_pasos(self, hist, fexpr, gexpr, x0, tol):
        for w in self._steps_inner.winfo_children():
            w.destroy()

        # derivada para mostrar en config
        try:
            gp = abs(derivada_num(gexpr, x0))
            gp_txt   = f"|g'(x₀)| ≈ {gp:.4f} {'< 1  →  converge' if gp < 1 else '≥ 1  →  puede diverger'}"
            gp_color = ACCENT if gp < 1 else YELLOW
        except Exception:
            gp_txt   = "No se pudo evaluar g'(x₀)"
            gp_color = YELLOW

        # ── bloque config
        cfg = tk.Frame(self._steps_inner, bg=BG3,
                       highlightthickness=1, highlightbackground=BORDER)
        cfg.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(cfg, text="Configuración inicial", bg=BG3, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=14, pady=(10, 4))

        for label, val, col in [
            ("f(x)",           fexpr,                    ACCENT),
            ("g(x)",           gexpr,                    PURPLE),
            ("Punto inicial",  f"x₀ = {x0}",             ACCENT),
            ("Tolerancia",     str(tol),                  ACCENT),
            ("Fórmula",        "x_{n+1} = g(x_n)",        ACCENT),
            ("Condición",      gp_txt,                    gp_color),
        ]:
            row = tk.Frame(cfg, bg=BG3)
            row.pack(anchor="w", padx=14, pady=1)
            tk.Label(row, text=f"— {label} = ", bg=BG3, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG3, fg=col,
                     font=("Consolas", 11)).pack(side=tk.LEFT)

        tk.Frame(cfg, bg=BG3, height=8).pack()

        # ── iteraciones
        for r in hist:
            converged = r["error"] < tol
            self._step_block(r, converged, tol)

    def _step_block(self, r, converged, tol):
        bar_color = GREEN if converged else ACCENT

        outer = tk.Frame(self._steps_inner, bg=BG)
        outer.pack(fill=tk.X, padx=12, pady=3)

        bar = tk.Frame(outer, bg=bar_color, width=3)
        bar.pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(outer, bg=BG2)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # header
        hdr = tk.Frame(inner, bg=BG2)
        hdr.pack(anchor="w", padx=12, pady=(8, 4))

        num_lbl = tk.Label(hdr, text=f" {r['i']} ", bg=bar_color,
                           fg="#000", font=("Segoe UI", 15, "bold"),
                           padx=4, pady=1)
        num_lbl.pack(side=tk.LEFT)

        tk.Label(hdr,
                 text=f"  ·  x_n = {r['xn']:.8f}",
                 bg=BG2, fg=MUTED,
                 font=("Consolas", 10)).pack(side=tk.LEFT)

        # líneas de cálculo
        lines = [
            (f"— f(x_n) = f({r['xn']:.6f})", "=", f" {r['fx']:.8f}",  PURPLE),
            (f"— g(x_n) = g({r['xn']:.6f})", "=", f" {r['gx']:.8f}",  PURPLE),
            (f"— x_{{n+1}}",                  "=", f" {r['xn1']:.8f}", ACCENT),
        ]

        for pre, eq, val, col in lines:
            row = tk.Frame(inner, bg=BG2)
            row.pack(anchor="w", padx=12, pady=1)
            tk.Label(row, text=pre, bg=BG2, fg=MUTED,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=eq,  bg=BG2, fg=TEXT,
                     font=("Consolas", 11)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG2, fg=col,
                     font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        # error + estado
        err_row = tk.Frame(inner, bg=BG2)
        err_row.pack(anchor="w", padx=12, pady=(1, 8))
        tk.Label(err_row, text="— Error = |x_{n+1} − x_n| = ", bg=BG2, fg=MUTED,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        tk.Label(err_row, text=f"{r['error']:.2e}", bg=BG2, fg=ORANGE,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        estado_text  = "  ✔ convergido" if converged else "  → continuar"
        estado_color = GREEN           if converged else YELLOW
        tk.Label(err_row, text=estado_text, bg=BG2, fg=estado_color,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

    # ──────────── RENDER: ANÁLISIS ────────────
    def _render_analisis(self, hist, raiz, tol, fexpr, gexpr, x0, converged):
        info = analisis_punto_fijo(hist, raiz, tol, fexpr, gexpr, x0)
        ta   = self._txt_analisis
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANÁLISIS DEL RESULTADO — PUNTO FIJO\n\n", "title")
        w("✔", "ok");  w(f" Convergió en ");    w(str(info["iters"]), "info"); w(" iteraciones\n")
        w("✔", "ok");  w(f" Raíz ≈ ");          w(f"{info['raiz']:.8f}", "info"); w("\n")
        w("✔", "ok");  w(f" f(raíz) = ");        w(f"{info['fval']:.10f}", "info"); w("\n\n")
        w("✔", "ok");  w(f" Error final:         "); w(f"{info['ue']:.2e}", "info"); w("\n")
        w("✔", "ok");  w(f" Factor de reducción: "); w(f"{info['factor']:.4f}", "info")
        w(f"  →  convergencia "); w(info["tipo"], "ok"); w("\n\n")

        # condición |g'(x)|
        if info["conv_ok"] is True:
            w("✔", "ok");   w(f" {info['conv_txt']}\n")
        elif info["conv_ok"] is False:
            w("⚠", "warn"); w(f" {info['conv_txt']}\n")
        else:
            w("?", "muted"); w(f" {info['conv_txt']}\n")

        w("\nCRITERIO DE PARADA\n", "title")
        w(f"  {info['ue']:.2e} < {info['tol']}  →  ")
        if info["ok"]:
            w("✔ cumplido\n", "ok")
        else:
            w("✗ no cumplido (max iter alcanzado)\n", "warn")

        ta.config(state="disabled")

    # ──────────── SUGERENCIAS g(x) ────────────
    def _show_suggest(self, event):
        fexpr = self.e_f.get().strip()
        sugs  = sugerencias_g(fexpr)

        top = tk.Toplevel(self)
        top.title("")
        top.configure(bg=BG2)
        top.resizable(False, False)
        top.geometry(f"+{event.x_root}+{event.y_root + 8}")

        tk.Label(top, text="  Sugerencias de g(x)", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=8, pady=(8, 4))

        for expr, desc in sugs:
            row = tk.Frame(top, bg=BG2, cursor="hand2")
            row.pack(fill=tk.X, padx=6, pady=2)
            row.bind("<Enter>", lambda e, r=row: r.config(bg=BG3))
            row.bind("<Leave>", lambda e, r=row: r.config(bg=BG2))

            tk.Label(row, text=expr, bg=BG2, fg=ACCENT,
                     font=("Consolas", 11),
                     padx=8, pady=4).pack(anchor="w")
            tk.Label(row, text=f"  {desc}", bg=BG2, fg=MUTED,
                     font=("Segoe UI", 13), pady=0).pack(anchor="w", padx=8)

            def on_click(e=expr, w=top):
                self.e_g.delete(0, tk.END)
                self.e_g.insert(0, e)
                w.destroy()

            for widget in (row,) + row.winfo_children():
                widget.bind("<Button-1>", lambda ev, fn=on_click: fn())

        tk.Frame(top, bg=BG2, height=6).pack()


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = PuntoFijoApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()