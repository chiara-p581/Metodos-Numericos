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

MONO  = ("JetBrains Mono", 10)
MONO_S= ("JetBrains Mono",  9)
SANS  = ("Segoe UI", 10)
SANS_B= ("Segoe UI", 11, "bold")
SANS_T= ("Segoe UI", 13, "bold")


# ══════════════════════════════════════
# LÓGICA — BISECCIÓN
# ══════════════════════════════════════
def _env(x):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x
    return e

def evaluar(expr, x):
    return eval(expr, {"__builtins__": {}}, _env(x))

def biseccion(expr, a, b, tol, max_iter):
    fa = evaluar(expr, a)
    fb = evaluar(expr, b)
    if fa * fb >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")

    hist = []
    c    = a

    for i in range(1, max_iter + 1):
        c      = (a + b) / 2
        fc     = evaluar(expr, c)
        error  = abs(b - a) / 2

        hist.append({"i": i, "a": a, "b": b, "c": c,
                     "fa": fa, "fb": fb, "fc": fc, "error": error})

        if abs(fc) < tol or error < tol:
            return c, hist, True

        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc

    return c, hist, False

def analisis_biseccion(hist, raiz, tol, fexpr):
    errores = [r["error"] for r in hist if r["error"] > 0]
    ratios  = [errores[i]/errores[i-1] for i in range(1, len(errores))
               if errores[i-1] != 0]
    factor  = sum(ratios)/len(ratios) if ratios else 0
    tipo    = "rápida" if factor < 0.1 else ("moderada" if factor < 0.5 else "lenta")
    ue      = errores[-1] if errores else 0
    fval    = evaluar(fexpr, raiz)

    return {
        "iters":  len(hist),
        "raiz":   raiz,
        "fval":   fval,
        "ue":     ue,
        "factor": factor,
        "tipo":   tipo,
        "tol":    tol,
        "ok":     ue < tol,
    }


# ══════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════
def _labeled_entry(parent, label, default, bg=BG2):
    tk.Label(parent, text=label, bg=bg, fg=MUTED,
             font=("JetBrains Mono", 8)).pack(anchor="w")
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 font=("JetBrains Mono", 10), bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, relief="flat")
    e.insert(0, default)
    e.pack(fill=tk.X, ipady=5, pady=(2, 8))
    return e

def _btn(parent, text, cmd, color=ACCENT, fg="#000000"):
    b = tk.Label(parent, text=text, bg=color, fg=fg,
                 font=("Segoe UI", 10, "bold"),
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
# CLASE PRINCIPAL — BISECCIÓN
# ══════════════════════════════════════
class BiseccionApp(tk.Frame):

    TABS = [
        ("📉", "Convergencia"),
        ("📊", "Función f(x)"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🧠", "Análisis"),
    ]

    def __init__(self, master=None, standalone=True):
        bg = BG if standalone else master.cget("bg")
        super().__init__(master, bg=BG)

        if standalone:
            master.title("Bisección — Métodos Numéricos")
            master.configure(bg=BG)
            master.geometry("1200x700")
            master.minsize(900, 580)

        self._hist  = []
        self._raiz  = None
        self._fexpr = ""

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
        tk.Label(bar, text="⚙  Bisección", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar, text="Método de Bisección", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=260)
        sb.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(inner, text="PARÁMETROS", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 8))

        self.e_f   = _labeled_entry(inner, "f(x)",          "x**2 - 2")
        self.e_a   = _labeled_entry(inner, "a  (extremo izq)", "0")
        self.e_b   = _labeled_entry(inner, "b  (extremo der)", "2")
        self.e_tol = _labeled_entry(inner, "Tolerancia",    "1e-6")
        self.e_it  = _labeled_entry(inner, "Max iteraciones", "50")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        _btn(inner, "▶  Calcular", self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "📈  Graficar f(x)", self._graficar,
             color=BG3, fg=ACCENT).pack(fill=tk.X, pady=3)

    def _main_area(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── tab bar
        self._tab_bar  = tk.Frame(right, bg=BG2, height=40)
        self._tab_bar.pack(fill=tk.X)
        self._tab_bar.pack_propagate(False)

        self._tab_btns   = {}
        self._tab_frames = {}
        self._active_tab = tk.StringVar(value="Paso a paso")

        for icon, name in self.TABS:
            b = tk.Label(self._tab_bar, text=f"{icon} {name}",
                         bg=BG2, fg=MUTED,
                         font=("Segoe UI", 10),
                         padx=14, pady=10, cursor="hand2")
            b.pack(side=tk.LEFT)
            b.bind("<Button-1>", lambda e, n=name: self._show_tab(n))
            self._tab_btns[name] = b

        # ── panels container
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
        self._active_tab.set(name)
        for n, b in self._tab_btns.items():
            if n == name:
                b.config(fg=TEXT)
                # underline via a tiny frame trick
            else:
                b.config(fg=MUTED)
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
                        foreground=TEXT, rowheight=26,
                        font=("JetBrains Mono", 9))
        style.configure("Dark.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Segoe UI", 8, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])

        cols = ("i", "a", "b", "c", "f(c)", "error")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        widths = [40, 110, 110, 110, 110, 110]
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
            font=("JetBrains Mono", 10),
            bd=0, padx=20, pady=16,
            relief="flat", wrap="word",
            state="disabled")
        self._txt_analisis.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # tags de color
        ta = self._txt_analisis
        ta.tag_config("title",  foreground=ACCENT,  font=("JetBrains Mono", 10, "bold"))
        ta.tag_config("ok",     foreground=GREEN)
        ta.tag_config("warn",   foreground=YELLOW)
        ta.tag_config("info",   foreground=PURPLE)
        ta.tag_config("muted",  foreground=MUTED)
        ta.tag_config("bold",   font=("JetBrains Mono", 10, "bold"))

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
            a     = float(eval(self.e_a.get()))
            b     = float(eval(self.e_b.get()))
            tol   = float(eval(self.e_tol.get()))
            it    = int(self.e_it.get())

            raiz, hist, converged = biseccion(fexpr, a, b, tol, it)
            self._hist  = hist
            self._raiz  = raiz
            self._fexpr = fexpr

            self._render_convergencia(hist)
            self._render_tabla(hist)
            self._render_pasos(hist, fexpr, a, b, tol)
            self._render_analisis(hist, raiz, tol, fexpr, converged)
            self._show_tab("Paso a paso")

        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ──────────── GRAFICAR ────────────
    def _graficar(self):
        try:
            fexpr = self.e_f.get().strip()
            a     = float(eval(self.e_a.get()))
            b     = float(eval(self.e_b.get()))
            margin = abs(b - a) * 0.6
            xs = np.linspace(a - margin, b + margin, 500)
            ys = []
            for x in xs:
                try:    ys.append(evaluar(fexpr, x))
                except: ys.append(float("nan"))

            ax = self._ax_func
            ax.clear()
            self._style_ax(ax)
            ax.plot(xs, ys, color=ACCENT, linewidth=2, label="f(x)")
            ax.axhline(0, color=BORDER, linewidth=1)
            ax.axvline(a, color=YELLOW, linewidth=1, linestyle="--", alpha=0.6, label="a")
            ax.axvline(b, color=YELLOW, linewidth=1, linestyle="--", alpha=0.6, label="b")

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
                f"{r['a']:.8f}", f"{r['b']:.8f}",
                f"{r['c']:.8f}", f"{r['fc']:.8f}",
                f"{r['error']:.2e}",
            ))

    # ──────────── RENDER: PASO A PASO ────────────
    def _render_pasos(self, hist, fexpr, a, b, tol):
        # limpiar
        for w in self._steps_inner.winfo_children():
            w.destroy()

        pad = dict(padx=16, pady=4)

        # ── bloque config
        cfg = tk.Frame(self._steps_inner, bg=BG3,
                       highlightthickness=1, highlightbackground=BORDER)
        cfg.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(cfg, text="Configuración inicial", bg=BG3, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(10, 4))

        for line, val in [
            ("f(x)", fexpr),
            ("Intervalo", f"[{a}, {b}]"),
            ("Tolerancia", str(tol)),
            ("Fórmula", "c = (a + b) / 2"),
        ]:
            row = tk.Frame(cfg, bg=BG3)
            row.pack(anchor="w", padx=14, pady=1)
            tk.Label(row, text=f"— {line} = ", bg=BG3, fg=MUTED,
                     font=("JetBrains Mono", 9)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG3, fg=ACCENT,
                     font=("JetBrains Mono", 9)).pack(side=tk.LEFT)

        tk.Frame(cfg, bg=BG3, height=8).pack()

        # ── iteraciones
        for r in hist:
            converged = r["error"] < tol or abs(r["fc"]) < tol
            self._step_block(r, converged, tol)

    def _step_block(self, r, converged, tol):
        bar_color = GREEN if converged else ACCENT

        outer = tk.Frame(self._steps_inner, bg=BG)
        outer.pack(fill=tk.X, padx=12, pady=3)

        # barra lateral de color
        bar = tk.Frame(outer, bg=bar_color, width=3)
        bar.pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(outer, bg=BG2)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # header
        hdr = tk.Frame(inner, bg=BG2)
        hdr.pack(anchor="w", padx=12, pady=(8, 4))

        num_lbl = tk.Label(hdr, text=f" {r['i']} ", bg=bar_color,
                           fg="#000", font=("Segoe UI", 8, "bold"),
                           padx=4, pady=1)
        num_lbl.pack(side=tk.LEFT)

        tk.Label(hdr,
                 text=f"  ·  a = {r['a']:.6f}   |   b = {r['b']:.6f}",
                 bg=BG2, fg=MUTED,
                 font=("JetBrains Mono", 8)).pack(side=tk.LEFT)

        lines = [
            (f"— c = ({r['a']:.6f} + {r['b']:.6f}) / 2", "=",
             f" {r['c']:.8f}", PURPLE),
            (f"— f(c) = f({r['c']:.6f})", "=",
             f" {r['fc']:.8f}", PURPLE),
        ]

        for pre, eq, val, col in lines:
            row = tk.Frame(inner, bg=BG2)
            row.pack(anchor="w", padx=12, pady=1)
            tk.Label(row, text=pre, bg=BG2, fg=MUTED,
                     font=("JetBrains Mono", 9)).pack(side=tk.LEFT)
            tk.Label(row, text=eq,  bg=BG2, fg=TEXT,
                     font=("JetBrains Mono", 9)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG2, fg=col,
                     font=("JetBrains Mono", 9, "bold")).pack(side=tk.LEFT)

        # error + estado
        err_row = tk.Frame(inner, bg=BG2)
        err_row.pack(anchor="w", padx=12, pady=(1, 4))
        tk.Label(err_row, text="— Error = |b − a| / 2 = ", bg=BG2, fg=MUTED,
                 font=("JetBrains Mono", 9)).pack(side=tk.LEFT)
        tk.Label(err_row, text=f"{r['error']:.2e}", bg=BG2, fg=ORANGE,
                 font=("JetBrains Mono", 9, "bold")).pack(side=tk.LEFT)

        estado_text  = "  ✔ convergido" if converged else "  → continuar"
        estado_color = GREEN           if converged else YELLOW
        tk.Label(err_row, text=estado_text, bg=BG2, fg=estado_color,
                 font=("JetBrains Mono", 9, "bold")).pack(side=tk.LEFT)

        # subintervalo siguiente
        next_txt = "→ b ← c  (raíz en [a, c])" if r["fa"]*r["fc"] < 0 \
                   else "→ a ← c  (raíz en [c, b])"
        tk.Label(inner, text=next_txt, bg=BG2, fg=MUTED,
                 font=("JetBrains Mono", 8)).pack(anchor="w", padx=12, pady=(0, 8))

    # ──────────── RENDER: ANÁLISIS ────────────
    def _render_analisis(self, hist, raiz, tol, fexpr, converged):
        info = analisis_biseccion(hist, raiz, tol, fexpr)
        ta   = self._txt_analisis
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANÁLISIS DEL RESULTADO — BISECCIÓN\n\n", "title")
        w("✔", "ok");   w(f" Convergió en ");    w(str(info["iters"]), "info"); w(" iteraciones\n")
        w("✔", "ok");   w(f" Raíz ≈ ");          w(f"{info['raiz']:.8f}", "info"); w("\n")
        w("✔", "ok");   w(f" f(raíz) = ");        w(f"{info['fval']:.10f}", "info"); w("\n\n")
        w("✔", "ok");   w(f" Error final:         "); w(f"{info['ue']:.2e}", "info"); w("\n")
        w("✔", "ok");   w(f" Factor de reducción: "); w(f"{info['factor']:.4f}", "info")
        w(f"  →  convergencia "); w(info["tipo"], "ok"); w("\n")
        w("✔", "ok");   w(" Bisección garantiza convergencia (cambio de signo).\n\n")
        w("CRITERIO DE PARADA\n", "title")
        w(f"  {info['ue']:.2e} < {info['tol']}  →  ")
        if info["ok"]:
            w("✔ cumplido\n", "ok")
        else:
            w("✗ no cumplido (max iter alcanzado)\n", "warn")

        ta.config(state="disabled")


# ══════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = BiseccionApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()