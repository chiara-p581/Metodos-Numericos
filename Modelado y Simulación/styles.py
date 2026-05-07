"""
Estilos compartidos para los 4 módulos de Modelado y Simulación.
Paleta, fuentes, helpers de widget, sidebar scrollable, inner tabs.
"""
import tkinter as tk
from tkinter import font as tkfont
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ══════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════
BG           = "#0d1117"
BG2          = "#161b22"
BG3          = "#1c2128"
BORDER       = "#30363d"
TEXT         = "#e6edf3"
MUTED        = "#8b949e"
PILL_BG      = "#2d4a7a"
PILL_FG      = "#a5d4ff"
PILL_IDLE    = "#1c2128"
PILL_IDLE_FG = "#4a5a72"
HOVER_BG     = "#243044"
HOVER_FG     = "#c9dfff"
INDICATOR    = "#58a6ff"
TOPBAR_BG    = "#10161e"
ACCENT       = "#58a6ff"
GREEN        = "#4ec97b"
RED          = "#e05252"
YELLOW       = "#f7c948"
PURPLE       = "#c084fc"
ORANGE       = "#ff7b54"

COLORS_TRAJ  = ["#f7c948","#ff7b54","#a5d4ff","#c3f584",
                "#e06ef0","#54d6ff","#ffaa5c","#ff6b9d"]

# ── Estilo matplotlib oscuro ──────────────────────────
MPL = {
    "figure.facecolor":  "#161b22",
    "axes.facecolor":    "#1c2128",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "axes.titlecolor":   "#e6edf3",
    "axes.grid":         True,
    "grid.color":        "#30363d",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "legend.facecolor":  "#1c2128",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#e6edf3",
    "lines.linewidth":   2.2,
    "font.size":         10,
}


# ══════════════════════════════════════════════════════
# HELPERS DE WIDGET
# ══════════════════════════════════════════════════════

def lbl(parent, text, fg=MUTED, font=("Segoe UI", 9), **kw):
    w = tk.Label(parent, text=text, bg=BG2, fg=fg, font=font,
                 anchor="w", **kw)
    w.pack(fill=tk.X, pady=(3, 0))
    return w

def section_title(parent, text):
    tk.Label(parent, text=text, bg=BG2, fg=PILL_FG,
             font=("Segoe UI", 9, "bold"), anchor="w"
             ).pack(fill=tk.X, pady=(12, 1))
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 5))

def entry(parent, default="", width=None, font=("Consolas", 10)):
    kw = {}
    if width: kw["width"] = width
    e = tk.Entry(parent, bg=BG3, fg=TEXT, insertbackground=TEXT,
                 relief=tk.FLAT, font=font,
                 highlightthickness=1,
                 highlightbackground=BORDER,
                 highlightcolor=INDICATOR, **kw)
    e.pack(fill=tk.X, pady=(2, 0), ipady=4)
    e.insert(0, default)
    return e

def labeled_entry(parent, label_text, default=""):
    tk.Label(parent, text=label_text, bg=BG2, fg=MUTED,
             font=("Segoe UI", 8), anchor="w").pack(fill=tk.X, pady=(4,0))
    return entry(parent, default)

def btn(parent, text, cmd, bg=PILL_BG, fg=PILL_FG,
        font=("Segoe UI", 10, "bold")):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, font=font,
                  relief=tk.FLAT, pady=7,
                  activebackground=INDICATOR,
                  activeforeground=BG,
                  cursor="hand2")
    b.pack(fill=tk.X, pady=(3, 0))
    return b

def separator(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=6)

def header_banner(parent, title, subtitle, accent=PILL_BG):
    hdr = tk.Frame(parent, bg=accent)
    hdr.pack(fill=tk.X)
    tk.Label(hdr, text=title, bg=accent, fg=PILL_FG,
             font=("Segoe UI", 13, "bold"), pady=8
             ).pack(anchor="w", padx=14)
    tk.Label(hdr, text=subtitle, bg=accent, fg="#7aa4c8",
             font=("Segoe UI", 8), pady=0
             ).pack(anchor="w", padx=14, pady=(0, 7))
    tk.Frame(hdr, bg="#1c3a5e", height=2).pack(fill=tk.X)


# ══════════════════════════════════════════════════════
# SIDEBAR SCROLLABLE (con scroll de rueda)
# ══════════════════════════════════════════════════════

def build_sidebar(parent, width=310):
    """
    Devuelve el frame 'inner' donde colocar los controles.
    Tiene scroll con rueda de ratón activado solo cuando
    el mouse está sobre el sidebar.
    """
    sb = tk.Frame(parent, bg=BG2, width=width)
    sb.pack(side=tk.LEFT, fill=tk.Y)
    sb.pack_propagate(False)

    container = tk.Frame(sb, bg=BG2)
    container.pack(fill=tk.BOTH, expand=True)

    canvas    = tk.Canvas(container, bg=BG2, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical",
                             command=canvas.yview,
                             bg=BG2, troughcolor=BG3,
                             highlightthickness=0, bd=0)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    inner  = tk.Frame(canvas, bg=BG2)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")

    inner.bind("<Configure>",
               lambda e: canvas.configure(
                   scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>",
                lambda e: canvas.itemconfig(window, width=e.width))

    def _scroll(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    def _bind(e):   canvas.bind_all("<MouseWheel>", _scroll)
    def _unbind(e): canvas.unbind_all("<MouseWheel>")

    for w in (canvas, inner):
        w.bind("<Enter>", _bind)
        w.bind("<Leave>", _unbind)

    # forzar ajuste inicial
    canvas.after(150, lambda: canvas.configure(
        scrollregion=canvas.bbox("all")))

    inner.configure(padx=12, pady=12)
    return inner


# ══════════════════════════════════════════════════════
# INNER TAB BAR (Gráfico / Paso a paso)
# ══════════════════════════════════════════════════════

class InnerTabBar(tk.Frame):
    PAD_X = 18; PAD_Y = 5; R = 8; GAP = 6; M = 10; H = 40

    def __init__(self, parent, tabs, on_select, **kw):
        super().__init__(parent, bg=TOPBAR_BG, height=self.H, **kw)
        self.pack_propagate(False)
        self._tabs = tabs; self._sel = on_select
        self._active = 0;  self._hover = -1

        self._cv = tk.Canvas(self, bg=TOPBAR_BG,
                             highlightthickness=0, height=self.H)
        self._cv.pack(fill=tk.BOTH, expand=True)
        self._cv.bind("<Button-1>", self._click)
        self._cv.bind("<Motion>",   self._motion)
        self._cv.bind("<Leave>",    self._leave)
        self.bind("<Configure>",    lambda e: self._draw())
        self._draw()

    def _coords(self):
        f = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        res, x = [], self.M
        for tab in self._tabs:
            w  = f.measure(tab) + self.PAD_X * 2
            y1 = (self.H - (f.metrics("linespace") + self.PAD_Y*2))//2
            y2 = self.H - y1
            res.append((x, y1, x+w, y2)); x += w + self.GAP
        return res

    def _draw(self):
        c = self._cv; c.delete("all")
        c.create_line(0, self.H-1, c.winfo_width(), self.H-1,
                      fill=BORDER, width=1)
        for i,(tab,(x1,y1,x2,y2)) in enumerate(
                zip(self._tabs, self._coords())):
            act = i==self._active; hov = i==self._hover and not act
            bg = PILL_BG if act else (HOVER_BG if hov else PILL_IDLE)
            fg = PILL_FG if act else (HOVER_FG if hov else PILL_IDLE_FG)
            self._rr(c,x1,y1,x2,y2,self.R,fill=bg,outline="")
            if not act:
                self._rr(c,x1,y1,x2,y2,self.R,fill="",
                         outline="#2a3548" if hov else "#232c3a",width=1)
            if act:
                mx=(x1+x2)//2; iw=min(x2-x1-12,34)
                c.create_line(mx-iw//2,y2+1,mx+iw//2,y2+1,
                              fill=INDICATOR,width=2,capstyle="round")
            c.create_text((x1+x2)//2,(y1+y2)//2,text=tab,fill=fg,
                          font=("Segoe UI",10,"bold"),anchor="center")

    @staticmethod
    def _rr(cv,x1,y1,x2,y2,r,**kw):
        pts=[x1+r,y1,x2-r,y1,x2,y1,x2,y1+r,x2,y2-r,x2,y2,
             x2-r,y2,x1+r,y2,x1,y2,x1,y2-r,x1,y1+r,x1,y1,x1+r,y1]
        cv.create_polygon(pts,smooth=True,**kw)

    def _at(self,x,y):
        for i,(x1,y1,x2,y2) in enumerate(self._coords()):
            if x1<=x<=x2 and y1<=y<=y2: return i
        return -1

    def _click(self,e):
        idx=self._at(e.x,e.y)
        if idx!=-1 and idx!=self._active:
            self._active=idx; self._draw(); self._sel(idx)

    def _motion(self,e):
        idx=self._at(e.x,e.y)
        if idx!=self._hover: self._hover=idx; self._draw()

    def _leave(self,e):
        if self._hover!=-1: self._hover=-1; self._draw()

    def set_active(self,idx):
        self._active=idx; self._draw()


# ══════════════════════════════════════════════════════
# BASE APP — dos pestañas internas (Gráfico | Paso a paso)
# ══════════════════════════════════════════════════════

class BaseApp(tk.Frame):
    """
    Marco base para cada módulo.
    Subclases deben implementar:
      _build_controls(inner)  — controles en el sidebar
      _run()                  — cálculo + actualiza gráfico y log
    """
    TITLE    = "Módulo"
    SUBTITLE = "dx/dt = f(x)"
    FIG_SIZE = (10, 7)

    def __init__(self, parent, standalone=True):
        super().__init__(parent, bg=BG)
        self._standalone = standalone
        self._build_layout()

    def _build_layout(self):
        # Sidebar
        inner = build_sidebar(self, width=310)
        header_banner(inner, self.TITLE, self.SUBTITLE)
        tk.Frame(inner, bg=BG2, height=4).pack()
        self._build_controls(inner)

        # Separador
        tk.Frame(self, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # Panel derecho
        right = tk.Frame(self, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner tabs
        self._itab = InnerTabBar(
            right,
            tabs=["   📈  Gráfico   ", "   📋  Paso a paso   "],
            on_select=self._switch_tab)
        self._itab.pack(fill=tk.X)
        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X)

        pages = tk.Frame(right, bg=BG)
        pages.pack(fill=tk.BOTH, expand=True)

        # Página gráfico
        self._pg_graph = tk.Frame(pages, bg=BG)
        with plt.rc_context(MPL):
            self._fig = Figure(figsize=self.FIG_SIZE, facecolor="#161b22")
        self._canvas_mpl = FigureCanvasTkAgg(self._fig,
                                              master=self._pg_graph)
        self._canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Página paso a paso
        self._pg_log = tk.Frame(pages, bg=BG)
        self._build_log_panel(self._pg_log)

        self._cur_tab = 0
        self._pg_graph.pack(fill=tk.BOTH, expand=True)

    def _build_log_panel(self, parent):
        top = tk.Frame(parent, bg=BG2)
        top.pack(fill=tk.X)
        tk.Label(top, text="Resolución paso a paso",
                 bg=BG2, fg=PILL_FG,
                 font=("Segoe UI", 11, "bold"), pady=7
                 ).pack(side=tk.LEFT, padx=14)
        leg = tk.Frame(top, bg=BG2)
        leg.pack(side=tk.RIGHT, padx=12, pady=5)
        for txt, col in [("● Estable", GREEN),
                          ("○ Inestable", RED),
                          ("◆ Semi", YELLOW),
                          ("· Info", MUTED)]:
            tk.Label(leg, text=txt, bg=BG2, fg=col,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=4)
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X)

        wrap = tk.Frame(parent, bg=BG3)
        wrap.pack(fill=tk.BOTH, expand=True)
        sb2 = tk.Scrollbar(wrap, bg=BG3, troughcolor=BG2,
                           highlightthickness=0, bd=0)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self._log = tk.Text(wrap, bg=BG3, fg=TEXT,
                            font=("Consolas", 10),
                            relief=tk.FLAT, wrap=tk.WORD,
                            state=tk.DISABLED, highlightthickness=0,
                            padx=20, pady=14, spacing1=2, spacing3=2,
                            yscrollcommand=sb2.set)
        sb2.config(command=self._log.yview)
        self._log.pack(fill=tk.BOTH, expand=True)

        # Tags
        T = self._log
        T.tag_config("h1", foreground=BG, background=PILL_BG,
                     font=("Segoe UI", 10, "bold"),
                     spacing1=8, spacing3=4)
        T.tag_config("h2", foreground=PILL_FG,
                     font=("Consolas", 10, "bold"), spacing1=6)
        T.tag_config("dim", foreground=MUTED, font=("Consolas", 10))
        T.tag_config("val", foreground="#c9dfff", font=("Consolas", 10))
        T.tag_config("ok",  foreground=GREEN,
                     font=("Consolas", 10, "bold"))
        T.tag_config("ok2", foreground=GREEN, font=("Consolas", 10))
        T.tag_config("err", foreground=RED,
                     font=("Consolas", 10, "bold"))
        T.tag_config("err2",foreground=RED,  font=("Consolas", 10))
        T.tag_config("semi",foreground=YELLOW,font=("Consolas",10,"bold"))
        T.tag_config("rule",foreground="#8b7fff",
                     font=("Consolas", 9, "italic"))
        T.tag_config("box", foreground=PILL_FG,
                     font=("Consolas", 10, "bold"),
                     background="#1c2e4a",
                     spacing1=3, spacing3=3,
                     lmargin1=10, lmargin2=10)
        T.tag_config("eq",  foreground="#f7c948",
                     font=("Consolas", 11, "bold"))

    def _switch_tab(self, idx):
        self._cur_tab = idx
        self._pg_graph.pack_forget()
        self._pg_log.pack_forget()
        if idx == 0:
            self._pg_graph.pack(fill=tk.BOTH, expand=True)
        else:
            self._pg_log.pack(fill=tk.BOTH, expand=True)
        self._itab.set_active(idx)

    # ── Log helpers ──────────────────────────────────

    def log_clear(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    def w(self, text, tag="dim"):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text, tag)
        self._log.config(state=tk.DISABLED)

    def nl(self, n=1):
        self.w("\n" * n)

    def section(self, num, title):
        self.nl()
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, f"  PASO {num} — {title}  ", "h1")
        self._log.config(state=tk.DISABLED)
        self.nl(2)

    def rule(self, text):
        self.w(f"  {text}\n", "rule")

    def box(self, text):
        self.w(f"  {text}\n", "box")

    def _build_controls(self, inner):
        pass  # override

    def _run(self):
        pass  # override