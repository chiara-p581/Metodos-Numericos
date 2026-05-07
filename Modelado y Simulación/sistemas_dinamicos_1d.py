"""
Sistemas Dinámicos 1D — Autónomos / Desacoplados
=================================================
ẋ = F(x)   — el tiempo no aparece en el lado derecho

Temas: puntos de equilibrio, estabilidad, diagrama de fases,
       diagrama de tiempo (soluciones), espacio de estados.

Basado en apuntes Cáceres 2026 — Clases 7 y 8.
"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize  import brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from styles import *


# ── eval seguro ──────────────────────────────────────
def eval_f(expr, x_arr):
    x  = x_arr
    ns = {"__builtins__":{}, "x":x, "np":np,
          "sin":np.sin,"cos":np.cos,"tan":np.tan,
          "exp":np.exp,"log":np.log,"sqrt":np.sqrt,
          "abs":np.abs,"pi":np.pi,"e":np.e,
          "sinh":np.sinh,"cosh":np.cosh,"tanh":np.tanh,
          "arctan":np.arctan,"arcsin":np.arcsin,"arccos":np.arccos}
    return eval(expr, ns)   # noqa

def find_eq(expr, xmin, xmax, n=2000):
    xs = np.linspace(xmin, xmax, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = eval_f(expr, xs)
    roots = []
    for i in range(len(xs)-1):
        if np.isnan(fs[i]) or np.isnan(fs[i+1]): continue
        if fs[i]*fs[i+1] < 0:
            try:
                r = brentq(lambda v: float(eval_f(expr, np.array([v]))[0]),
                           xs[i], xs[i+1], xtol=1e-10)
                roots.append(round(r, 8))
            except: pass
        elif abs(fs[i]) < 1e-9:
            roots.append(round(xs[i], 8))
    uniq = []
    for r in roots:
        if not any(abs(r-u)<1e-6 for u in uniq): uniq.append(r)
    return sorted(uniq)

def deriv(expr, xstar, h=1e-5):
    return (eval_f(expr, np.array([xstar+h]))[0]
            - eval_f(expr, np.array([xstar-h]))[0]) / (2*h)

def classify(expr, xstar):
    fp = deriv(expr, xstar)
    if fp < -1e-8: return "Estable"
    elif fp > 1e-8: return "Inestable"
    return "Semiestable"

def integrate(expr, x0s, tend, npts=600):
    t_eval = np.linspace(0, tend, npts)
    res = []
    def rhs(t, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [float(eval_f(expr, np.array([y[0]]))[0])]
    for x0 in x0s:
        try:
            sol = solve_ivp(rhs,(0,tend),[x0],t_eval=t_eval,
                            method="RK45",rtol=1e-8,atol=1e-10,max_step=0.05)
            res.append((sol.t, sol.y[0]))
        except: pass
    return res


EXAMPLES = [
    ("ẋ = 3y - 9  (lineal)",          "3*x - 9",          "-1",  "6"),
    ("ẋ = y² - 6y + 5  (cuadrático)", "x**2 - 6*x + 5",   "-1",  "7"),
    ("ẋ = sen(x)",                     "sin(x)",           "-4",  "8"),
    ("ẋ = x(1-x)  (logístico)",        "x*(1-x)",          "-0.5","2.5"),
    ("ẋ = x(1-x)(x-2)  (triple)",      "x*(1-x)*(x-2)",    "-1",  "3"),
    ("ẋ = y² - 1  (cuadrático)",       "x**2 - 1",         "-2.5","2.5"),
    ("ẋ = y(y-1)(y+2)",               "x*(x-1)*(x+2)",    "-3",  "2"),
    ("Personalizado…",                 "",                  "-5",  "5"),
]


class SistemasDinamicos1DApp(BaseApp):
    TITLE    = "Sistemas Dinámicos 1D"
    SUBTITLE = "ẋ = F(x)   •   Autónomo / Desacoplado"
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

        section_title(inner, "Ecuación")
        self._ef  = labeled_entry(inner, "F(x)  (variable: x)", "3*x - 9")

        section_title(inner, "Rango")
        row = tk.Frame(inner, bg=BG2); row.pack(fill=tk.X, pady=2)
        for lbl_t, attr, default in [("x min","_exmin","-1"),
                                      ("x max","_exmax","6")]:
            tk.Label(row,text=lbl_t,bg=BG2,fg=MUTED,
                     font=("Segoe UI",8),width=5).pack(side=tk.LEFT)
            e = tk.Entry(row,bg=BG3,fg=TEXT,insertbackground=TEXT,
                         relief=tk.FLAT,font=("Consolas",9),width=7,
                         highlightthickness=1,highlightbackground=BORDER)
            e.insert(0,default); e.pack(side=tk.LEFT,ipady=3,padx=(2,8))
            setattr(self,attr,e)

        section_title(inner, "Simulación temporal")
        self._eCIs = labeled_entry(inner, "Condiciones iniciales (sep. coma)",
                                   "-1, 0, 2, 5, 7")
        self._eTend = labeled_entry(inner, "t_end", "10")

        separator(inner)
        btn(inner, "▶  ANALIZAR", self._run)

    def _load(self, idx):
        _, expr, xmin, xmax = EXAMPLES[idx]
        for e, v in [(self._ef, expr),(self._exmin, xmin),(self._exmax, xmax)]:
            e.delete(0, tk.END); e.insert(0, v)

    def _run(self):
        expr = self._ef.get().strip()
        if not expr:
            messagebox.showerror("Error","Ingresá F(x)."); return
        try:
            xmin = float(self._exmin.get())
            xmax = float(self._exmax.get())
            tend = float(self._eTend.get())
            x0s  = [float(v.strip())
                    for v in self._eCIs.get().split(",") if v.strip()]
        except ValueError:
            messagebox.showerror("Error","Revisá los valores."); return
        try:
            test = eval_f(expr, np.linspace(xmin,xmax,5))
            if np.all(np.isnan(test)): raise ValueError
        except:
            messagebox.showerror("Error",
                f"Expresión inválida: '{expr}'\nEj: 3*x - 9"); return

        eqs   = find_eq(expr, xmin, xmax)
        stabs = [classify(expr, e) for e in eqs]
        sd    = dict(zip(eqs, stabs))
        trajs = integrate(expr, x0s, tend)

        self._plot(expr, xmin, xmax, eqs, stabs, sd, trajs, tend)
        self._do_log(expr, xmin, xmax, eqs, stabs, sd, x0s, tend, trajs)

    def _plot(self, expr, xmin, xmax, eqs, stabs, sd, trajs, tend):
        self._fig.clear()
        xs = np.linspace(xmin, xmax, 800)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ys = eval_f(expr, xs)

        with plt.rc_context(MPL):
            ax1 = self._fig.add_subplot(2,1,1)
            ax2 = self._fig.add_subplot(2,1,2)

        # ── Diagrama de fases ──────────────────────────
        with plt.rc_context(MPL):
            ax1.plot(xs, ys, color=ACCENT, lw=2.4,
                     label=f"F(x) = {expr}", zorder=3)
            ax1.axhline(0, color=MUTED, lw=1.0, zorder=2)
            ax1.axvline(0, color=BORDER, lw=0.8, zorder=1)

            yr = np.nanmax(np.abs(ys[np.isfinite(ys)])) if np.any(np.isfinite(ys)) else 1

            # flechas de flujo
            bounds = [xmin] + eqs + [xmax]
            for i in range(len(bounds)-1):
                mid  = (bounds[i]+bounds[i+1])/2
                fval = float(eval_f(expr, np.array([mid]))[0])
                if abs(fval) < 1e-10: continue
                d   = 1 if fval>0 else -1
                col = GREEN if fval>0 else RED
                ax1.annotate("", xy=(mid+d*(xmax-xmin)*0.025, 0),
                             xytext=(mid-d*(xmax-xmin)*0.025, 0),
                             arrowprops=dict(arrowstyle="-|>",color=col,
                                            lw=2,mutation_scale=14))

            # puntos de equilibrio
            offset_dir = 1
            for xeq, stab in zip(eqs, stabs):
                if stab == "Estable":
                    ax1.scatter(xeq,0,color=GREEN,s=140,zorder=7)
                    ax1.annotate(f"x*={xeq:.3f}\n(Estable)",
                                 xy=(xeq,0),xytext=(xeq, yr*0.45*offset_dir),
                                 color=GREEN, fontsize=8.5, ha="center",
                                 arrowprops=dict(arrowstyle="->",
                                                 color=GREEN,lw=1.2))
                elif stab == "Inestable":
                    ax1.scatter(xeq,0,facecolors="none",
                                edgecolors=RED,s=140,lw=2.5,zorder=7)
                    ax1.annotate(f"x*={xeq:.3f}\n(Inestable)",
                                 xy=(xeq,0),xytext=(xeq,-yr*0.45*offset_dir),
                                 color=RED, fontsize=8.5, ha="center",
                                 arrowprops=dict(arrowstyle="->",
                                                 color=RED, lw=1.2))
                else:
                    ax1.scatter(xeq,0,color=YELLOW,s=140,marker="D",zorder=7)
                offset_dir *= -1

            ax1.set_xlabel("x", fontsize=11)
            ax1.set_ylabel("ẋ = F(x)", fontsize=11)
            ax1.set_title(f"Diagrama de fases 1D  —  ẋ = {expr}",
                          fontsize=12, fontweight="bold")
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(-yr*1.35, yr*1.35)
            ax1.legend(fontsize=8)

        # ── Diagrama de tiempo ─────────────────────────
        with plt.rc_context(MPL):
            for i,(t_arr,x_arr) in enumerate(trajs):
                ax2.plot(t_arr, x_arr, color=COLORS_TRAJ[i%len(COLORS_TRAJ)],
                         lw=1.9, label=f"x(0)={x_arr[0]:.2f}")
            for xeq, stab in sd.items():
                ls  = "--" if stab=="Estable" else ":"
                col = GREEN if stab=="Estable" else RED
                ax2.axhline(xeq, color=col, ls=ls, lw=1.1, alpha=0.7,
                            label=f"x*={xeq:.3f} ({stab})")
            ax2.set_xlabel("Tiempo t", fontsize=11)
            ax2.set_ylabel("x(t)", fontsize=11)
            ax2.set_title(f"Diagrama de tiempo  —  ẋ = {expr}",
                          fontsize=12, fontweight="bold")
            ax2.legend(fontsize=8, ncol=2)

        self._fig.tight_layout(pad=2.2, h_pad=3.0)
        self._canvas_mpl.draw()

    def _do_log(self, expr, xmin, xmax, eqs, stabs, sd,
                x0s, tend, trajs):
        self.log_clear()
        w = self.w; nl = self.nl

        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END,
            "  SISTEMA DINÁMICO 1D — ANÁLISIS COMPLETO  ", "h1")
        self._log.config(state=tk.DISABLED)
        nl(2)
        w("  Sistema:  ", "dim"); w(f"ẋ = {expr}\n", "eq")
        w(f"  Rango:    x ∈ [{xmin}, {xmax}]\n", "dim"); nl()

        # PASO 1
        self.section(1, "¿Qué es un sistema autónomo/desacoplado?")
        nl()
        w("  ẋ = F(x)   →   el tiempo NO aparece explícitamente\n","dim")
        w("  en el lado derecho. Por eso es 'autónomo' y 'desacoplado'.\n","dim")
        nl()
        w("  Diferencia con no-autónomo: si fuera ẋ = F(x, λ) con λ\n","dim")
        w("  dependiente del tiempo, ya no sería autónomo.\n","dim"); nl()

        # PASO 2
        self.section(2, "Puntos de equilibrio")
        self.rule("Resolver F(x) = 0  →  ẋ = 0  →  sin cambio")
        nl()
        w("  Método: muestreo de cambios de signo + Brentq (tol 1e-10)\n","dim")
        nl()
        if not eqs:
            w("  → No se encontraron equilibrios en el rango.\n","semi")
        else:
            w(f"  Se encontraron {len(eqs)} punto(s):\n","dim"); nl()
            for i,xeq in enumerate(eqs):
                fv = float(eval_f(expr, np.array([xeq]))[0])
                w(f"  x*{i+1} = ", "dim"); w(f"{xeq:.8f}", "val")
                w(f"   F(x*) = {fv:.2e} ≈ 0  ✓\n","dim")

        # PASO 3
        self.section(3, "Clasificación de equilibrios")
        self.rule("Criterio de la derivada  →  linealización local")
        nl()
        w("  Si  F'(x*) < 0   →  Estable    (atractor)\n","ok2")
        w("  Si  F'(x*) > 0   →  Inestable  (repulsor)\n","err2")
        w("  Si  F'(x*) ≈ 0   →  Semiestable\n","semi"); nl()

        for xeq, stab in zip(eqs, stabs):
            fp  = deriv(expr, xeq)
            tag = "ok" if stab=="Estable" else ("err" if stab=="Inestable" else "semi")
            dtg = "ok2"if stab=="Estable" else ("err2"if stab=="Inestable" else "semi")

            w(f"  ┌─── x* = {xeq:.6f} ","dim"); w("─"*32+"\n","dim")
            w( "  │   F'(x*)  ≈  ","dim"); w(f"{fp:+.6f}\n","val")
            w( "  │   Tipo    :  ","dim"); w(f"{stab}\n",tag)

            if stab == "Estable":
                w("  │   Trayectorias cercanas convergen → x*\n",dtg)
                w(f"  │   lim t→∞  x(t) = {xeq:.5f}\n",dtg)
            elif stab == "Inestable":
                w("  │   Trayectorias cercanas se alejan de x*\n",dtg)
                w("  │   Actúa como separatriz entre cuencas\n",dtg)
            else:
                w("  │   Comportamiento distinto a cada lado\n","semi")
            w("  └"+"─"*40+"\n","dim"); nl()

        # PASO 4
        self.section(4, "Dirección del flujo  ( signo de F(x) )")
        self.rule("F > 0 → x crece  |  F < 0 → x decrece")
        nl()
        bounds = [xmin] + eqs + [xmax]
        for i in range(len(bounds)-1):
            a=bounds[i]; b=bounds[i+1]; mid=(a+b)/2
            fv = float(eval_f(expr, np.array([mid]))[0])
            arr = "→  (crece)" if fv>0 else "←  (decrece)"
            col = "ok2" if fv>0 else "err2"
            w(f"  x ∈ ({a:.3f},  {b:.3f})   F≈{fv:+.4f}   ","dim")
            w(f"{arr}\n",col)
        nl()

        # PASO 5
        self.section(5, "Comportamiento asintótico  ( t → ∞ )")
        self.rule("¿Hacia dónde converge x(t) según la condición inicial?")
        nl()
        stable_eq = [x for x,s in sd.items() if s=="Estable"]
        if stable_eq:
            for xeq in stable_eq:
                self.box(f"  lim t→∞  x(t)  =  {xeq:.5f}")
                nl()
                w(f"  (para x₀ en la cuenca de atracción de x*={xeq:.4f})\n","ok2"); nl()
        else:
            w("  No hay equilibrios estables → las trayectorias divergen.\n","semi"); nl()

        # PASO 6
        self.section(6, f"Conjuntos límite ω(x₀)   [t_end = {tend}]")
        self.rule("Integración numérica RK45  →  x(t_end) ≈ ω-límite")
        nl()
        for i,(t_arr,x_arr) in enumerate(trajs):
            x0 = x0s[i] if i<len(x0s) else x_arr[0]
            xf = x_arr[-1]
            near = (min(sd.keys(), key=lambda e:abs(e-xf)) if sd else None)
            conv = (near is not None and abs(near-xf)<0.5)
            w(f"  x(0)={x0:+.3f}  →  x({tend:.0f})≈","dim"); w(f"{xf:.5f}","val")
            if conv:
                tag = "ok2" if sd[near]=="Estable" else "err2"
                w(f"  →  x*={near:.4f} ({sd[near]})\n",tag)
            else:
                w("  →  diverge\n","err2")
        nl()

        # RESUMEN
        nl()
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, "  RESUMEN  ", "h1")
        self._log.config(state=tk.DISABLED)
        nl(2)
        w(f"  ẋ = {expr}\n","eq"); nl()
        for xeq, stab in sd.items():
            tag = "ok"  if stab=="Estable" else ("err" if stab=="Inestable" else "semi")
            sym = "●"   if stab=="Estable" else ("○"   if stab=="Inestable" else "◆")
            w(f"  {sym}  x* = {xeq:.5f}   →   {stab}\n", tag)
        nl()

        self._log.config(state=tk.NORMAL)
        self._log.see("1.0")
        self._log.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sistemas Dinámicos 1D")
    root.geometry("1400x820")
    root.configure(bg=BG)
    app = SistemasDinamicos1DApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()