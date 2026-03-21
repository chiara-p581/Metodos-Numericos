import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
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
# LÓGICA — NEWTON-RAPHSON
# ══════════════════════════════════════
def _env(x):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x
    return e

def evaluar(expr, x):
    return eval(expr, {"__builtins__": {}}, _env(x))

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

            hist.append({
                "i":    i + 1,
                "xn":   x,
                "xn1":  xnew,
                "fx":   fx,
                "dfx":  dfx,
                "paso": fx / dfx,
                "error": error,
            })

            if error < tol:
                return xnew, hist, "convergencia"

            x = xnew

        except Exception as exc:
            return None, hist, f"error: {exc}"

    return x, hist, "max_iter"

def analisis_newton(hist, raiz, tol, fexpr, dfexpr):
    errores = [r["error"] for r in hist if r["error"] > 0]
    ratios  = [errores[i]/errores[i-1]**2 for i in range(1, len(errores))
               if errores[i-1] != 0]
    factor_cuad = sum(ratios)/len(ratios) if ratios else 0
    ue   = errores[-1] if errores else 0

    try:
        fval = evaluar(fexpr, raiz)
    except Exception:
        fval = float("nan")

    return {
        "iters":       len(hist),
        "raiz":        raiz,
        "fval":        fval,
        "ue":          ue,
        "factor_cuad": factor_cuad,
        "tol":         tol,
        "ok":          ue < tol,
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
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h):
    r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0,int(r*.8)), max(0,int(g*.8)), max(0,int(b*.8)))


# ══════════════════════════════════════
# CLASE PRINCIPAL — NEWTON-RAPHSON
# ══════════════════════════════════════
class NewtonApp(tk.Frame):

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
            master.title("Newton-Raphson — Métodos Numéricos")
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
        tk.Label(bar, text="⚙  Newton-Raphson", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar, text="x_{n+1} = x_n − f(x_n) / f'(x_n)",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=270)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(inner, text="PARÁMETROS", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 8))

        self.e_f   = _labeled_entry(inner, "f(x)",            "x**3 - x - 4")
        self.e_df  = _labeled_entry(inner, "f'(x)  — derivada", "3*x**2 - 1")
        self.e_x0  = _labeled_entry(inner, "x₀  (punto inicial)", "1")
        self.e_tol = _labeled_entry(inner, "Tolerancia",          "1e-6")
        self.e_it  = _labeled_entry(inner, "Max iteraciones",     "100")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        _btn(inner, "▶  Calcular",       self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "📈  Graficar f(x)", self._graficar,
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
                         font=("Segoe UI", 10), padx=14, pady=10,
                         cursor="hand2")
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
            (f.pack if n == name else f.pack_forget)(
                **({} if n != name else {"fill": tk.BOTH, "expand": True}))

    def _panel(self, name):
        f = tk.Frame(self._panels, bg=BG)
        self._tab_frames[name] = f
        return f

    # ──────────── PANEL: CONVERGENCIA ────────────
    def _build_panel_conv(self):
        f = self._panel("Convergencia")
        self._fig_conv = Figure(figsize=(7,4), facecolor=BG)
        self._ax_conv  = self._fig_conv.add_subplot(111)
        self._style_ax(self._ax_conv)
        self._canvas_conv = FigureCanvasTkAgg(self._fig_conv, master=f)
        self._canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────── PANEL: FUNCIÓN ────────────
    def _build_panel_func(self):
        f = self._panel("Función f(x)")
        self._fig_func = Figure(figsize=(7,4), facecolor=BG)
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

        cols = ("i", "xₙ", "f(xₙ)", "f'(xₙ)", "paso", "error")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        widths = [35, 120, 100, 100, 100, 100]
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

    # ──────────── PANEL: ANÁLISIS ────────────
    def _build_panel_analisis(self):
        f = self._panel("Análisis")
        self._ta = tk.Text(f, bg=BG3, fg=TEXT,
                           font=("JetBrains Mono", 10),
                           bd=0, padx=20, pady=16,
                           relief="flat", wrap="word", state="disabled")
        self._ta.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        self._ta.tag_config("title", foreground=ACCENT,
                             font=("JetBrains Mono", 10, "bold"))
        self._ta.tag_config("ok",   foreground=GREEN)
        self._ta.tag_config("warn", foreground=YELLOW)
        self._ta.tag_config("info", foreground=PURPLE)
        self._ta.tag_config("muted",foreground=MUTED)

    # ──────────── MATPLOTLIB STYLE ────────────
    def _style_ax(self, ax):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)

    # ──────────── CALCULAR ────────────
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

    # ──────────── GRAFICAR ────────────
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
            ax.plot(xs, ys,  color=ACCENT,  linewidth=2, label="f(x)")
            ax.plot(xs, dys, color=PURPLE,  linewidth=1.5,
                    linestyle="--", alpha=0.7, label="f'(x)")
            ax.axhline(0, color=BORDER, linewidth=0.8)

            # tangentes en cada iteración
            if self._hist:
                for r in self._hist[:6]:   # max 6 tangentes visibles
                    xn  = r["xn"]
                    fx  = r["fx"]
                    dfx = r["dfx"]
                    if dfx != 0:
                        x_tang = np.array([xn - 1.5, xn + 1.5])
                        y_tang = fx + dfx * (x_tang - xn)
                        ax.plot(x_tang, y_tang, color=YELLOW,
                                linewidth=0.8, alpha=0.5)
                        ax.scatter([xn], [fx], color=ORANGE, s=35, zorder=4)

            if self._raiz is not None:
                ax.scatter([self._raiz], [0], color=GREEN,
                           zorder=5, s=80, label=f"raíz ≈ {self._raiz:.6f}")

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
        ax.semilogy(iters, errors, color=ACCENT, linewidth=2,
                    marker="o", markersize=4, markerfacecolor=ACCENT)
        ax.fill_between(iters, errors, alpha=0.08, color=ACCENT)
        ax.set_xlabel("Iteración", color=MUTED, fontsize=9)
        ax.set_ylabel("Error (log)", color=MUTED, fontsize=9)
        ax.set_title("Convergencia del error — Newton-Raphson",
                     color=TEXT, fontsize=10, pad=10)
        self._canvas_conv.draw()

    # ──────────── RENDER: TABLA ────────────
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

    # ──────────── RENDER: PASO A PASO ────────────
    def _render_pasos(self, hist, fexpr, dfexpr, x0, tol):
        for w in self._si.winfo_children():
            w.destroy()

        # ── bloque config
        cfg = tk.Frame(self._si, bg=BG3,
                       highlightthickness=1, highlightbackground=BORDER)
        cfg.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(cfg, text="Configuración inicial", bg=BG3, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(
                     anchor="w", padx=14, pady=(10, 4))

        for label, val, col in [
            ("f(x)",        fexpr,                    ACCENT),
            ("f'(x)",       dfexpr,                   PURPLE),
            ("x₀",          str(x0),                  ACCENT),
            ("Tolerancia",  str(tol),                  ACCENT),
            ("Fórmula",     "x_{n+1} = x_n − f(x_n) / f'(x_n)", ACCENT),
        ]:
            row = tk.Frame(cfg, bg=BG3)
            row.pack(anchor="w", padx=14, pady=1)
            tk.Label(row, text=f"— {label} = ", bg=BG3, fg=MUTED,
                     font=("JetBrains Mono", 9)).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG3, fg=col,
                     font=("JetBrains Mono", 9)).pack(side=tk.LEFT)

        tk.Frame(cfg, bg=BG3, height=8).pack()

        # ── iteraciones
        for r in hist:
            self._step_block(r, r["error"] < tol)

    def _step_block(self, r, converged):
        bar_color = GREEN if converged else ACCENT

        outer = tk.Frame(self._si, bg=BG)
        outer.pack(fill=tk.X, padx=12, pady=3)

        tk.Frame(outer, bg=bar_color, width=3).pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(outer, bg=BG2)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # header
        hdr = tk.Frame(inner, bg=BG2)
        hdr.pack(anchor="w", padx=12, pady=(8, 4))
        tk.Label(hdr, text=f" {r['i']} ", bg=bar_color, fg="#000",
                 font=("Segoe UI", 8, "bold"), padx=4, pady=1).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"  ·  x_n = {r['xn']:.8f}",
                 bg=BG2, fg=MUTED,
                 font=("JetBrains Mono", 8)).pack(side=tk.LEFT)

        # pasos del método
        steps = [
            (f"— f(x_n)  = f({r['xn']:.6f})",  "=", f" {r['fx']:.8f}",   PURPLE),
            (f"— f'(x_n) = f'({r['xn']:.6f})", "=", f" {r['dfx']:.8f}",  PURPLE),
            ( "— paso    = f(x_n) / f'(x_n)",   "=", f" {r['paso']:.8f}", MUTED),
            (f"— x_{{n+1}} = {r['xn']:.6f} − ({r['paso']:.6f})",
             "=", f" {r['xn1']:.8f}", ACCENT),
        ]

        for pre, eq, val, col in steps:
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
        err_row.pack(anchor="w", padx=12, pady=(1, 8))
        tk.Label(err_row, text="— Error = |x_{n+1} − x_n| = ",
                 bg=BG2, fg=MUTED,
                 font=("JetBrains Mono", 9)).pack(side=tk.LEFT)
        tk.Label(err_row, text=f"{r['error']:.2e}",
                 bg=BG2, fg=ORANGE,
                 font=("JetBrains Mono", 9, "bold")).pack(side=tk.LEFT)
        est_txt = "  ✔ convergido" if converged else "  → continuar"
        tk.Label(err_row, text=est_txt,
                 bg=BG2, fg=GREEN if converged else YELLOW,
                 font=("JetBrains Mono", 9, "bold")).pack(side=tk.LEFT)

    # ──────────── RENDER: ANÁLISIS ────────────
    def _render_analisis(self, hist, raiz, tol, fexpr, dfexpr, estado):
        info = analisis_newton(hist, raiz, tol, fexpr, dfexpr)
        ta   = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANÁLISIS DEL RESULTADO — NEWTON-RAPHSON\n\n", "title")
        w("✔", "ok");  w(f" Convergió en ");    w(str(info["iters"]), "info"); w(" iteraciones\n")
        w("✔", "ok");  w(f" Raíz ≈ ");          w(f"{info['raiz']:.8f}", "info"); w("\n")
        w("✔", "ok");  w(f" f(raíz) = ");        w(f"{info['fval']:.10f}", "info"); w("\n\n")
        w("✔", "ok");  w(f" Error final:              ");
        w(f"{info['ue']:.2e}", "info"); w("\n")
        w("✔", "ok");  w(f" Factor cuadrático (C≈):  ");
        w(f"{info['factor_cuad']:.4f}", "info"); w("\n")
        w("✔", "ok");  w(" Newton-Raphson tiene convergencia ")
        w("cuadrática", "ok"); w(" (orden 2).\n\n")

        w("ESTADO\n", "title")
        est_map = {
            "convergencia":   ("✔ convergencia alcanzada", "ok"),
            "derivada_cero":  ("⚠ derivada = 0 — no se puede continuar", "warn"),
            "max_iter":       ("⚠ máximo de iteraciones alcanzado", "warn"),
        }
        txt, tag = est_map.get(estado, (f"? {estado}", "muted"))
        w(f"  {txt}\n", tag)

        w("\nCRITERIO DE PARADA\n", "title")
        w(f"  {info['ue']:.2e} < {info['tol']}  →  ")
        w("✔ cumplido\n" if info["ok"] else "✗ no cumplido\n",
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