"""
Bifurcaciones 1D
================
ẋ = F(x, μ)   — μ es el parámetro de control

El script detecta automáticamente el tipo de bifurcación:
  • Silla-Nodo:      ẋ = μ + x²   (dos eq. nacen/desaparecen)
  • Tridente Supercrítica: ẋ = μx - x³
  • Tridente Subcrítica:   ẋ = μx + x³
  • Transcrítica:    ẋ = μx - x²

Basado en apuntes Cáceres 2026 — Clases 8 y 9.
"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import warnings
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from styles import *


# ──────────────────────────────────────────────────────
# Evaluación simbólica segura  f(x, mu)
# ──────────────────────────────────────────────────────
def eval_fxm(expr, x_arr, mu_val):
    x  = x_arr
    mu = mu_val
    ns = {"__builtins__":{}, "x":x, "mu":mu, "μ":mu, "r":mu,
          "np":np, "sin":np.sin,"cos":np.cos,"exp":np.exp,
          "sqrt":np.sqrt,"abs":np.abs,"log":np.log,"pi":np.pi}
    return eval(expr, ns)  # noqa

def eval_scalar(expr, xv, muv):
    try:
        r = float(eval_fxm(expr, np.array([xv]), muv)[0])
        return r if np.isfinite(r) else 0.0
    except: return 0.0

def deriv_x(expr, xv, muv, h=1e-5):
    return (eval_scalar(expr, xv+h, muv)
            - eval_scalar(expr, xv-h, muv)) / (2*h)

def find_eq_for_mu(expr, muv, xmin=-5, xmax=5, n=2000):
    xs = np.linspace(xmin, xmax, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = eval_fxm(expr, xs, muv)
    roots = []
    for i in range(len(xs)-1):
        if np.isnan(fs[i]) or np.isnan(fs[i+1]): continue
        if fs[i]*fs[i+1] < 0:
            try:
                r = brentq(lambda v: eval_scalar(expr, v, muv),
                           xs[i], xs[i+1], xtol=1e-10)
                roots.append(round(r, 8))
            except: pass
        elif abs(fs[i]) < 1e-9:
            roots.append(round(xs[i], 8))
    uniq = []
    for r in roots:
        if not any(abs(r-u)<1e-5 for u in uniq): uniq.append(r)
    return sorted(uniq)


# ──────────────────────────────────────────────────────
# Detección automática del tipo de bifurcación
# ──────────────────────────────────────────────────────
def detect_bifurcation(expr, mu_range, xmin, xmax):
    """
    Analiza cómo cambia el número y tipo de equilibrios
    al variar μ. Retorna una descripción del tipo detectado.
    """
    mu_arr  = np.linspace(mu_range[0], mu_range[1], 200)
    eq_counts = []
    for mu in mu_arr:
        eqs = find_eq_for_mu(expr, mu, xmin, xmax)
        eq_counts.append(len(eqs))

    changes = np.diff(eq_counts)
    has_creation     = any(c > 0 for c in changes)
    has_annihilation = any(c < 0 for c in changes)
    max_eq = max(eq_counts)
    min_eq = min(eq_counts)

    # Evaluar en mu cercano a 0
    eqs_neg = find_eq_for_mu(expr, mu_range[0]/4, xmin, xmax)
    eqs_zero= find_eq_for_mu(expr, 0, xmin, xmax)
    eqs_pos = find_eq_for_mu(expr, mu_range[1]/4, xmin, xmax)

    n_neg = len(eqs_neg); n_zero=len(eqs_zero); n_pos=len(eqs_pos)

    # Transcrítica: siempre 2 eq, intercambian estabilidad
    if n_neg==2 and n_zero>=1 and n_pos==2 and max_eq-min_eq<=1:
        return "Transcrítica"
    # Silla-Nodo: 0 → 1 → 2 o 2 → 1 → 0
    if (n_neg==0 and n_pos==2) or (n_neg==2 and n_pos==0):
        return "Silla-Nodo"
    if (n_neg==2 and n_pos==0) or (n_neg==0 and n_pos>=2):
        return "Silla-Nodo"
    # Tridente: 1 → 3
    if n_neg==1 and n_pos==3:
        return "Tridente Supercrítica"
    if n_neg==3 and n_pos==1:
        return "Tridente Subcrítica"
    if n_neg==1 and n_pos==1 and n_zero==1:
        return "Silla-Nodo (degenerada)"
    # fallback
    if has_creation and not has_annihilation:
        return "Silla-Nodo"
    return "No identificada claramente"


# ──────────────────────────────────────────────────────
# Cálculo del diagrama de bifurcación
# ──────────────────────────────────────────────────────
def bifurcation_diagram(expr, mu_range, xmin, xmax, n_mu=400):
    """
    Para cada μ calcula los equilibrios y su estabilidad.
    Retorna listas: (mu, x*, estabilidad)
    """
    mus   = np.linspace(mu_range[0], mu_range[1], n_mu)
    stable_pts = []
    unstable_pts = []

    for mu in mus:
        eqs = find_eq_for_mu(expr, mu, xmin, xmax)
        for xeq in eqs:
            fp = deriv_x(expr, xeq, mu)
            if fp < -1e-8:
                stable_pts.append((mu, xeq))
            else:
                unstable_pts.append((mu, xeq))

    return np.array(stable_pts), np.array(unstable_pts)


EXAMPLES = [
    ("Silla-Nodo: μ + x²",         "mu + x**2",          "-3",  "1",  "-3","3"),
    ("Silla-Nodo: μ - x²",         "mu - x**2",          "-1",  "3",  "-3","3"),
    ("Transcrítica: μx - x²",      "mu*x - x**2",        "-2",  "2",  "-3","3"),
    ("Transcrítica: (r-2)x - x²",  "(mu-2)*x - x**2",    "-1",  "5",  "-3","3"),
    ("Tridente Super: μx - x³",    "mu*x - x**3",        "-2",  "2",  "-3","3"),
    ("Tridente Sub: μx + x³",      "mu*x + x**3",        "-2",  "2",  "-3","3"),
    ("Silla-nodo: rx+x³-x⁵ [0.3]","mu*x + x**3 - x**5", "-2",  "2",  "-2.2","2.2"),
    ("Personalizado…",              "",                   "-3",  "3",  "-4","4"),
]


class BifurcacionesApp(BaseApp):
    TITLE    = "Bifurcaciones 1D"
    SUBTITLE = "ẋ = F(x, μ)   •   Detección automática del tipo"
    FIG_SIZE = (10, 8)

    def _build_controls(self, inner):
        section_title(inner, "Ejemplos predefinidos")
        self._ex = tk.IntVar(value=0)
        for i,(lbl_text,*_) in enumerate(EXAMPLES):
            tk.Radiobutton(inner, text=lbl_text,
                           variable=self._ex, value=i,
                           bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, activeforeground=TEXT,
                           font=("Segoe UI", 8),
                           command=lambda idx=i: self._load(idx)
                           ).pack(anchor="w", pady=1)

        section_title(inner, "Ecuación  F(x, μ)")
        tk.Label(inner, text="Variables: x  y  mu  (o μ, o r)",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 8)
                 ).pack(anchor="w", fill=tk.X)
        self._ef = labeled_entry(inner, "F(x, mu)", "mu + x**2")

        section_title(inner, "Rango del parámetro μ")
        row = tk.Frame(inner, bg=BG2); row.pack(fill=tk.X, pady=2)
        for lbl_t, attr, default in [("μ min","_emumin","-3"),
                                      ("μ max","_emumax","1")]:
            tk.Label(row,text=lbl_t,bg=BG2,fg=MUTED,
                     font=("Segoe UI",8),width=5).pack(side=tk.LEFT)
            e = tk.Entry(row,bg=BG3,fg=TEXT,insertbackground=TEXT,
                         relief=tk.FLAT,font=("Consolas",9),width=7,
                         highlightthickness=1,highlightbackground=BORDER)
            e.insert(0,default); e.pack(side=tk.LEFT,ipady=3,padx=(2,8))
            setattr(self,attr,e)

        section_title(inner, "Rango de x (espacio de estados)")
        row2 = tk.Frame(inner, bg=BG2); row2.pack(fill=tk.X, pady=2)
        for lbl_t, attr, default in [("x min","_exmin","-3"),
                                      ("x max","_exmax","3")]:
            tk.Label(row2,text=lbl_t,bg=BG2,fg=MUTED,
                     font=("Segoe UI",8),width=5).pack(side=tk.LEFT)
            e = tk.Entry(row2,bg=BG3,fg=TEXT,insertbackground=TEXT,
                         relief=tk.FLAT,font=("Consolas",9),width=7,
                         highlightthickness=1,highlightbackground=BORDER)
            e.insert(0,default); e.pack(side=tk.LEFT,ipady=3,padx=(2,8))
            setattr(self,attr,e)

        section_title(inner, "Diagrama de fase — μ fijo")
        self._emu_fase = labeled_entry(inner, "μ para diagrama de fase", "-1")
        self._eCIs = labeled_entry(inner, "Condiciones iniciales x(0) (sep. coma)",
                                   "-2, -1, 0, 1, 2")
        self._eTend = labeled_entry(inner, "t_end", "10")

        separator(inner)
        btn(inner, "▶  ANALIZAR", self._run)

    def _load(self, idx):
        _, expr, mmin, mmax, xmin, xmax = EXAMPLES[idx]
        for e, v in [(self._ef,expr),(self._emumin,mmin),(self._emumax,mmax),
                     (self._exmin,xmin),(self._exmax,xmax)]:
            e.delete(0, tk.END); e.insert(0, v)

    def _run(self):
        expr = self._ef.get().strip()
        if not expr:
            messagebox.showerror("Error","Ingresá F(x, mu)."); return
        try:
            mu_min = float(self._emumin.get())
            mu_max = float(self._emumax.get())
            xmin   = float(self._exmin.get())
            xmax   = float(self._exmax.get())
            mu_fase= float(self._emu_fase.get())
            x0s    = [float(v.strip())
                      for v in self._eCIs.get().split(",") if v.strip()]
            tend   = float(self._eTend.get())
        except ValueError:
            messagebox.showerror("Error","Revisá los valores."); return

        # Detectar tipo
        bif_type = detect_bifurcation(expr, (mu_min, mu_max), xmin, xmax)

        # Diagrama de bifurcación
        s_pts, u_pts = bifurcation_diagram(expr, (mu_min, mu_max), xmin, xmax)

        # Diagrama de fase para mu_fase
        eqs_fase = find_eq_for_mu(expr, mu_fase, xmin, xmax)
        stabs_fase = [("Estable" if deriv_x(expr, e, mu_fase) < -1e-8
                       else ("Inestable" if deriv_x(expr, e, mu_fase) > 1e-8
                             else "Semiestable"))
                      for e in eqs_fase]

        self._plot(expr, mu_min, mu_max, xmin, xmax,
                   s_pts, u_pts, bif_type,
                   mu_fase, eqs_fase, stabs_fase, x0s, tend)
        self._do_log(expr, mu_min, mu_max, xmin, xmax,
                     s_pts, u_pts, bif_type,
                     mu_fase, eqs_fase, stabs_fase, x0s, tend)

    def _plot(self, expr, mu_min, mu_max, xmin, xmax,
              s_pts, u_pts, bif_type,
              mu_fase, eqs_fase, stabs_fase, x0s, tend):

        self._fig.clear()
        with plt.rc_context(MPL):
            ax1 = self._fig.add_subplot(1,2,1)   # Diagrama bifurcación
            ax2 = self._fig.add_subplot(2,2,2)   # Diagrama de fase
            ax3 = self._fig.add_subplot(2,2,4)   # Soluciones en el tiempo

        # ── Diagrama de bifurcación ────────────────────
        with plt.rc_context(MPL):
            if len(s_pts) > 0:
                ax1.scatter(s_pts[:,0], s_pts[:,1], color=ACCENT,
                            s=6, label="Estable", zorder=3)
            if len(u_pts) > 0:
                ax1.scatter(u_pts[:,0], u_pts[:,1], color=RED,
                            s=6, alpha=0.85, label="Inestable", zorder=3)

            ax1.axvline(0, color=BORDER, lw=0.8, ls=":")
            ax1.axhline(0, color=BORDER, lw=0.6)
            ax1.set_xlabel("Parámetro  μ", fontsize=11)
            ax1.set_ylabel("Posición de equilibrio  x*", fontsize=11)
            ax1.set_title(f"Diagrama de bifurcación\n({bif_type})",
                          fontsize=11, fontweight="bold")
            ax1.legend(fontsize=8,
                       handles=[
                           mpatches.Patch(color=ACCENT, label="Estable  ──"),
                           mpatches.Patch(color=RED,    label="Inestable  - -")
                       ])

            # Marcar línea mu_fase
            ax1.axvline(mu_fase, color=YELLOW, lw=1.2, ls="--", alpha=0.8,
                        label=f"μ = {mu_fase}")

        # ── Diagrama de fase (x̊ vs x) para mu_fase ────
        xs   = np.linspace(xmin, xmax, 600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ys = eval_fxm(expr, xs, mu_fase)

        with plt.rc_context(MPL):
            ax2.plot(xs, ys, color=ORANGE, lw=2.2,
                     label=f"F(x, μ={mu_fase})")
            ax2.axhline(0, color=MUTED, lw=1)
            yr = np.nanmax(np.abs(ys[np.isfinite(ys)])) if np.any(np.isfinite(ys)) else 1

            # Flechas de flujo
            bounds = [xmin]+eqs_fase+[xmax]
            for i in range(len(bounds)-1):
                mid  = (bounds[i]+bounds[i+1])/2
                fval = eval_scalar(expr, mid, mu_fase)
                if abs(fval)<1e-10: continue
                d   = 1 if fval>0 else -1
                col = GREEN if fval>0 else RED
                ax2.annotate("", xy=(mid+d*(xmax-xmin)*0.03, 0),
                             xytext=(mid-d*(xmax-xmin)*0.03, 0),
                             arrowprops=dict(arrowstyle="-|>",color=col,
                                            lw=2,mutation_scale=13))

            for xeq, stab in zip(eqs_fase, stabs_fase):
                if stab=="Estable":
                    ax2.scatter(xeq,0,color=GREEN,s=130,zorder=7)
                elif stab=="Inestable":
                    ax2.scatter(xeq,0,facecolors="none",
                                edgecolors=RED,s=130,lw=2.5,zorder=7)
                else:
                    ax2.scatter(xeq,0,color=YELLOW,s=130,marker="D",zorder=7)

            ax2.set_xlabel("x", fontsize=10)
            ax2.set_ylabel("ẋ", fontsize=10)
            ax2.set_title(f"Diagrama de fase  μ = {mu_fase}",
                          fontsize=10, fontweight="bold")
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(-yr*1.3, yr*1.3)
            ax2.legend(fontsize=7)

        # ── Soluciones en el tiempo ────────────────────
        from scipy.integrate import solve_ivp
        t_eval = np.linspace(0, tend, 500)
        with plt.rc_context(MPL):
            for i, x0 in enumerate(x0s):
                def rhs(t, y, ex=expr, mu=mu_fase):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return [float(eval_fxm(ex, np.array([y[0]]), mu)[0])]
                try:
                    sol = solve_ivp(rhs,(0,tend),[x0],t_eval=t_eval,
                                    method="RK45",rtol=1e-8,atol=1e-10,
                                    max_step=0.05)
                    ax3.plot(sol.t, sol.y[0],
                             color=COLORS_TRAJ[i%len(COLORS_TRAJ)],
                             lw=1.8, label=f"x(0)={x0}")
                except: pass

            for xeq, stab in zip(eqs_fase, stabs_fase):
                ls  = "--" if stab=="Estable" else ":"
                col = GREEN if stab=="Estable" else RED
                ax3.axhline(xeq, color=col, ls=ls, lw=1.1, alpha=0.7)
            ax3.set_xlabel("Tiempo t", fontsize=10)
            ax3.set_ylabel("x(t)",     fontsize=10)
            ax3.set_title(f"Soluciones  μ = {mu_fase}",
                          fontsize=10, fontweight="bold")
            ax3.legend(fontsize=7, ncol=2)

        self._fig.tight_layout(pad=2.0)
        self._canvas_mpl.draw()

    def _do_log(self, expr, mu_min, mu_max, xmin, xmax,
                s_pts, u_pts, bif_type,
                mu_fase, eqs_fase, stabs_fase, x0s, tend):
        self.log_clear()
        w = self.w; nl = self.nl

        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END,
            "  BIFURCACIONES 1D — ANÁLISIS COMPLETO  ", "h1")
        self._log.config(state=tk.DISABLED)
        nl(2)
        w("  Sistema:  ", "dim"); w(f"ẋ = {expr}\n", "eq")
        w(f"  μ ∈ [{mu_min}, {mu_max}]   x ∈ [{xmin}, {xmax}]\n","dim"); nl()

        # PASO 1
        self.section(1, "¿Qué es una bifurcación?")
        nl()
        w("  ẋ = F(x, μ)   →   μ es el PARÁMETRO DE CONTROL\n","dim")
        nl()
        w("  Una bifurcación ocurre cuando, al variar μ, el sistema\n","dim")
        w("  cambia cualitativamente: nacen, desaparecen o cambian\n","dim")
        w("  de estabilidad los puntos de equilibrio.\n","dim"); nl()
        w("  Condiciones para bifurcación en (x*, μ*):\n","dim")
        w("    •  ẋ(x*) = 0         →  x* es equilibrio\n","val")
        w("    •  F(x*, μ*) = 0     →  existe en ese μ*\n","val")
        nl()

        # PASO 2
        self.section(2, "Tipo de bifurcación detectado")
        nl()
        self.box(f"  Tipo identificado:  {bif_type}")
        nl()

        # Descripción según tipo
        descriptions = {
            "Silla-Nodo": [
                "  Dos puntos de equilibrio (uno estable, uno inestable)\n",
                "  COLISIONAN y desaparecen al cruzar μ*.\n",
                "\n",
                "  Antes del punto de bifurcación:  2 equilibrios\n",
                "  En la bifurcación:               1 semiestable\n",
                "  Después de la bifurcación:       0 equilibrios\n",
            ],
            "Transcrítica": [
                "  Dos equilibrios existen para TODO valor de μ,\n",
                "  pero INTERCAMBIAN su estabilidad en μ*.\n",
                "\n",
                "  x*=0 pasa de estable a inestable (o viceversa)\n",
                "  x*=μ pasa de inestable a estable (o viceversa)\n",
            ],
            "Tridente Supercrítica": [
                "  Para r < 0: un único equilibrio estable en x=0\n",
                "  Para r > 0: x=0 se vuelve inestable y aparecen\n",
                "              dos nuevos equilibrios estables  x* = ±√r\n",
                "\n",
                "  Es una bifurcación 'suave' o supercrítica.\n",
            ],
            "Tridente Subcrítica": [
                "  Para μ < 0: coexisten x=0 estable y dos\n",
                "             equilibrios inestables x* = ±√(-μ)\n",
                "  En μ=0: los inestables colisionan con el origen\n",
                "  Para μ > 0: solo x=0 permanece (inestable)\n",
                "\n",
                "  Es una bifurcación 'brusca' o subcrítica.\n",
            ],
        }
        desc = descriptions.get(bif_type, ["  Ver diagrama para más detalles.\n"])
        for line in desc:
            tag = "dim" if not line.startswith("  x*") else "val"
            w(line, "dim")
        nl()

        # PASO 3
        self.section(3, "Puntos de equilibrio  ( F(x, μ) = 0 )")
        self.rule("Para cada μ resolver F(x, μ) = 0")
        nl()
        # Mostrar equilibrios en 5 valores de mu representativos
        mu_samples = np.linspace(mu_min, mu_max, 5)
        for mu_s in mu_samples:
            eqs_s = find_eq_for_mu(expr, mu_s, xmin, xmax)
            w(f"  μ = {mu_s:+.3f}   →   ", "dim")
            if eqs_s:
                for xeq in eqs_s:
                    fp = deriv_x(expr, xeq, mu_s)
                    stab = "Estable" if fp<-1e-8 else ("Inestable" if fp>1e-8 else "Semi")
                    tag  = "ok2" if stab=="Estable" else ("err2" if stab=="Inestable" else "semi")
                    w(f"x*={xeq:.3f}({stab})  ", tag)
            else:
                w("sin equilibrios", "semi")
            nl()
        nl()

        # PASO 4
        self.section(4, f"Análisis de estabilidad para μ = {mu_fase}")
        self.rule("F'(x*, μ) = ∂F/∂x evaluada en el equilibrio")
        nl()
        w(f"  F(x, {mu_fase}) = {expr.replace('mu', str(mu_fase))}\n","val")
        nl()
        if not eqs_fase:
            w(f"  → No hay equilibrios para μ = {mu_fase}\n","semi")
        else:
            for xeq, stab in zip(eqs_fase, stabs_fase):
                fp  = deriv_x(expr, xeq, mu_fase)
                tag = "ok"  if stab=="Estable" else ("err" if stab=="Inestable" else "semi")
                dtg = "ok2" if stab=="Estable" else ("err2"if stab=="Inestable" else "semi")
                w(f"  ┌─── x* = {xeq:.6f}\n","dim")
                w(f"  │   F'(x*, μ) ≈ {fp:+.6f}\n","val")
                w(f"  │   → {stab}\n",tag)
                if stab=="Estable":
                    w(f"  │   lim t→∞  x(t) = {xeq:.4f}\n",dtg)
                elif stab=="Inestable":
                    w("  │   Las trayectorias se alejan de x*\n",dtg)
                w("  └"+"─"*38+"\n","dim"); nl()

        # PASO 5
        self.section(5, "Diagrama de bifurcación — interpretación")
        nl()
        n_stable   = len(s_pts)
        n_unstable = len(u_pts)
        w(f"  Puntos estables graficados   :  {n_stable}\n","ok2")
        w(f"  Puntos inestables graficados :  {n_unstable}\n","err2")
        nl()

        # Hallar punto de bifurcación aproximado
        if bif_type == "Silla-Nodo":
            w("  Punto de bifurcación (Silla-Nodo):\n","dim")
            w("  Condición:  F(x*,μ) = 0   y   F'(x*,μ) = 0\n","rule")
            # buscar mu donde solo hay 1 equilibrio
            for mu_s in np.linspace(mu_min, mu_max, 500):
                eqs_s = find_eq_for_mu(expr, mu_s, xmin, xmax)
                if len(eqs_s) == 1:
                    fp_b = deriv_x(expr, eqs_s[0], mu_s)
                    w(f"  μ* ≈ {mu_s:.4f},  x* ≈ {eqs_s[0]:.4f}\n","val")
                    break
            nl()
        elif bif_type in ("Transcrítica","Tridente Supercrítica","Tridente Subcrítica"):
            w("  Bifurcación en μ* = 0  (x* = 0)\n","val"); nl()

        # PASO 6
        self.section(6, "Dirección del flujo — diagrama de fase")
        self.rule(f"Evaluando F(x, μ={mu_fase}) en cada intervalo")
        nl()
        bounds = [xmin] + eqs_fase + [xmax]
        for i in range(len(bounds)-1):
            a=bounds[i]; b=bounds[i+1]; mid=(a+b)/2
            fv = eval_scalar(expr, mid, mu_fase)
            arr = "→  (crece)" if fv>0 else "←  (decrece)"
            col = "ok2" if fv>0 else "err2"
            w(f"  x ∈ ({a:.2f}, {b:.2f})   F≈{fv:+.4f}   ","dim")
            w(f"{arr}\n",col)
        nl()

        # RESUMEN
        nl()
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, "  RESUMEN  ", "h1")
        self._log.config(state=tk.DISABLED)
        nl(2)
        w(f"  ẋ = {expr}\n","eq"); nl()
        self.box(f"  Tipo de bifurcación: {bif_type}")
        nl()
        for xeq, stab in zip(eqs_fase, stabs_fase):
            tag = "ok" if stab=="Estable" else ("err" if stab=="Inestable" else "semi")
            sym = "●" if stab=="Estable" else ("○" if stab=="Inestable" else "◆")
            w(f"  {sym}  x* = {xeq:.5f}  ({stab})  para μ = {mu_fase}\n",tag)
        nl()

        self._log.config(state=tk.NORMAL)
        self._log.see("1.0")
        self._log.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bifurcaciones 1D")
    root.geometry("1400x820")
    root.configure(bg=BG)
    app = BifurcacionesApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()