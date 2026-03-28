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
# LÓGICA — AITKEN Δ²
# ══════════════════════════════════════
def _env(x):
    e = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    e["np"] = np
    e["x"]  = x
    return e

def evaluar(expr, x):
    return eval(expr, {"__builtins__": {}}, _env(x))

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

            hist.append({
                "i":    i + 1,
                "xn":   x,
                "x1":   x1,
                "x2":   x2,
                "xhat": xhat,
                "error": error,
            })

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
    tipo    = "rápida" if factor < 0.1 else ("moderada" if factor < 0.5 else "lenta")
    ue      = errores[-1] if errores else 0
    return {
        "iters": len(hist), "raiz": raiz, "ue": ue,
        "factor": factor,   "tipo": tipo,  "tol": tol,
        "ok": ue < tol,
    }


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
    b.bind("<Enter>",    lambda e: b.config(bg=_dk(color)))
    b.bind("<Leave>",    lambda e: b.config(bg=color))
    return b

def _dk(h):
    r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return "#{:02x}{:02x}{:02x}".format(
        max(0,int(r*.8)), max(0,int(g*.8)), max(0,int(b*.8)))


# ══════════════════════════════════════
# CLASE PRINCIPAL — AITKEN
# ══════════════════════════════════════
class AitkenApp(tk.Frame):

    TABS = [
        ("📉", "Convergencia"),
        ("📊", "Función g(x)"),
        ("🗂",  "Tabla"),
        ("🔍", "Paso a paso"),
        ("🧠", "Análisis"),
    ]

    def __init__(self, master=None, standalone=True):
        super().__init__(master, bg=BG)

        if standalone:
            master.title("Aitken Δ² — Métodos Numéricos")
            master.configure(bg=BG)
            master.geometry("1200x700")
            master.minsize(900, 580)

        self._hist  = []
        self._raiz  = None
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
        tk.Label(bar, text="⚙  Aitken Δ²", bg=BG2, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(side=tk.LEFT, padx=16)
        tk.Label(bar, text="Aceleración de Aitken — x̂ = xₙ − (x₁−xₙ)² / (x₂−2x₁+xₙ)",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 11)).pack(side=tk.RIGHT, padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=BG2, width=260)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        inner = tk.Frame(sb, bg=BG2)
        inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(inner, text="PARÁMETROS", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 8))

        self.e_g   = _labeled_entry(inner, "g(x)  — función de iteración",
                                    "sqrt((2*(x+2))/pi)")
        self.e_x0  = _labeled_entry(inner, "x₀  (punto inicial)", "1.4")
        self.e_tol = _labeled_entry(inner, "Tolerancia",          "1e-6")
        self.e_it  = _labeled_entry(inner, "Max iteraciones",     "100")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

        _btn(inner, "▶  Calcular",       self._calcular).pack(fill=tk.X, pady=3)
        _btn(inner, "📈  Graficar g(x)", self._graficar,
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
                         font=("Segoe UI", 13), padx=18, pady=12,
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
        f = self._panel("Función g(x)")
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
                        foreground=TEXT, rowheight=32,
                        font=("Consolas", 11))
        style.configure("Dark.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Segoe UI", 15, "bold"), relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#000")])

        cols = ("i", "xₙ", "x₁=g(xₙ)", "x₂=g(x₁)", "x̂ (Aitken)", "error")
        self._tree = ttk.Treeview(f, columns=cols, show="headings",
                                   style="Dark.Treeview")
        widths = [35, 110, 110, 110, 110, 100]
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
                           font=("Consolas", 12), bd=0, padx=20, pady=16,
                           relief="flat", wrap="word", state="disabled")
        self._ta.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
        self._ta.tag_config("title", foreground=ACCENT,
                             font=("Consolas", 11, "bold"))
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
            gexpr = self.e_g.get().strip()
            x0    = float(eval(self.e_x0.get()))
            tol   = float(eval(self.e_tol.get()))
            it    = int(self.e_it.get())

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

    # ──────────── GRAFICAR ────────────
    def _graficar(self):
        try:
            gexpr = self.e_g.get().strip()
            x0    = float(eval(self.e_x0.get()))
            xs    = np.linspace(x0 - 2, x0 + 2, 500)

            def safe(x):
                try:    return evaluar(gexpr, x)
                except: return float("nan")

            gys = [safe(x) for x in xs]

            ax = self._ax_func
            ax.clear()
            self._style_ax(ax)
            ax.plot(xs, gys, color=PURPLE, linewidth=2, label="g(x)")
            ax.plot(xs, xs,  color=GREEN,  linewidth=1,
                    linestyle=":", label="y = x")
            ax.axhline(0, color=BORDER, linewidth=0.8)

            if self._raiz is not None:
                ax.scatter([self._raiz], [self._raiz], color=ORANGE,
                           zorder=5, s=70, label=f"punto fijo ≈ {self._raiz:.6f}")

            ax.legend(facecolor=BG3, edgecolor=BORDER,
                      labelcolor=TEXT, fontsize=8)
            self._canvas_func.draw()
            self._show_tab("Función g(x)")

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
        ax.set_title("Convergencia del error — Aitken Δ²",
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
                f"{r['x1']:.8f}",
                f"{r['x2']:.8f}",
                f"{r['xhat']:.8f}",
                f"{r['error']:.2e}",
            ))

    # ──────────── RENDER: PASO A PASO ────────────
    def _render_pasos(self, hist, gexpr, x0, tol):
        for w in self._si.winfo_children():
            w.destroy()

        # ── bloque config
        cfg = tk.Frame(self._si, bg=BG3,
                       highlightthickness=1, highlightbackground=BORDER)
        cfg.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(cfg, text="Configuración inicial", bg=BG3, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(
                     anchor="w", padx=14, pady=(10, 4))

        for label, val, col in [
            ("g(x)",       gexpr,                               PURPLE),
            ("x₀",         str(x0),                             ACCENT),
            ("Tolerancia", str(tol),                             ACCENT),
            ("Fórmula",    "x̂ = xₙ − (x₁−xₙ)² / (x₂−2x₁+xₙ)", ACCENT),
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
                 font=("Segoe UI", 15, "bold"), padx=4, pady=1).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"  ·  xₙ = {r['xn']:.8f}",
                 bg=BG2, fg=MUTED,
                 font=("Consolas", 10)).pack(side=tk.LEFT)

        # pasos intermedios del método
        steps = [
            (f"— x₁ = g(xₙ) = g({r['xn']:.6f})",          "=", f" {r['x1']:.8f}",   PURPLE),
            (f"— x₂ = g(x₁) = g({r['x1']:.6f})",          "=", f" {r['x2']:.8f}",   PURPLE),
            ( "— num = (x₁ − xₙ)²",                         "=",
              f" {(r['x1']-r['xn'])**2:.8f}",               MUTED),
            ( "— den = x₂ − 2·x₁ + xₙ",                    "=",
              f" {r['x2']-2*r['x1']+r['xn']:.8f}",         MUTED),
            (f"— x̂ = xₙ − num/den",                         "=", f" {r['xhat']:.8f}", ACCENT),
        ]

        for pre, eq, val, col in steps:
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
        tk.Label(err_row, text="— Error = |x̂ − xₙ| = ",
                 bg=BG2, fg=MUTED,
                 font=("Consolas", 11)).pack(side=tk.LEFT)
        tk.Label(err_row, text=f"{r['error']:.2e}",
                 bg=BG2, fg=ORANGE,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        est_txt = "  ✔ convergido" if converged else "  → continuar"
        tk.Label(err_row, text=est_txt,
                 bg=BG2, fg=GREEN if converged else YELLOW,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

    # ──────────── RENDER: ANÁLISIS ────────────
    def _render_analisis(self, hist, raiz, tol, gexpr, estado):
        info = analisis_aitken(hist, raiz, tol, gexpr)
        ta   = self._ta
        ta.config(state="normal")
        ta.delete("1.0", tk.END)

        def w(text, tag=None):
            ta.insert(tk.END, text, tag)

        w("ANÁLISIS DEL RESULTADO — AITKEN Δ²\n\n", "title")
        w("✔", "ok");  w(f" Convergió en ");   w(str(info["iters"]), "info"); w(" iteraciones\n")
        w("✔", "ok");  w(f" Punto fijo ≈ ");   w(f"{info['raiz']:.8f}", "info"); w("\n\n")
        w("✔", "ok");  w(f" Error final:         "); w(f"{info['ue']:.2e}", "info"); w("\n")
        w("✔", "ok");  w(f" Factor de reducción: "); w(f"{info['factor']:.4f}", "info")
        w(f"  →  convergencia "); w(info["tipo"], "ok"); w("\n")
        w("✔", "ok");  w(" Aitken Δ² acelera la convergencia de punto fijo.\n\n")
        w("ESTADO\n", "title")
        est_map = {
            "convergencia":  ("✔ convergencia alcanzada", "ok"),
            "division_cero": ("⚠ división por cero — denominador = 0", "warn"),
            "max_iter":      ("⚠ máximo de iteraciones alcanzado", "warn"),
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
    app  = AitkenApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()