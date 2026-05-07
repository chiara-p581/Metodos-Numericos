"""
Sistema Logístico (Verhulst)
============================
dx/dt = r·x·(1 - x/K)
Solución analítica: P(t) = N / (1 + A·e^(-rt))

Basado en apuntes Cáceres 2026 — Clase 7 y 8.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from styles import *


class SistemaLogisticoApp(BaseApp):
    TITLE    = "Sistema Logístico"
    SUBTITLE = "dP/dt = rP(1 - P/K)   •   Modelo de Verhulst"
    FIG_SIZE = (10, 8)

    def _build_controls(self, inner):
        section_title(inner, "Parámetros del modelo")

        self._eN = labeled_entry(inner, "N  — capacidad de carga  (K)", "800")
        self._er = labeled_entry(inner, "r  — tasa de crecimiento", "0.9343")
        self._eP0= labeled_entry(inner, "P(0) — población inicial",  "1")

        section_title(inner, "Simulación temporal")
        self._eT = labeled_entry(inner, "t_end  (días)", "15")
        self._eCIs = labeled_entry(inner,
            "Condiciones iniciales extra  (sep. por coma)", "1, 5, 50, 200")

        section_title(inner, "Datos para estimar r  (opcional)")
        tk.Label(inner, text="Si tenés dos mediciones podés calcular r:",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 8),
                 wraplength=260, justify=tk.LEFT, anchor="w"
                 ).pack(fill=tk.X, pady=(0,2))

        row1 = tk.Frame(inner, bg=BG2); row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="t₁", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 8), width=4).pack(side=tk.LEFT)
        self._et1 = tk.Entry(row1, bg=BG3, fg=TEXT, insertbackground=TEXT,
                              relief=tk.FLAT, font=("Consolas", 9), width=6,
                              highlightthickness=1, highlightbackground=BORDER)
        self._et1.insert(0, "4"); self._et1.pack(side=tk.LEFT, ipady=3, padx=3)
        tk.Label(row1, text="P(t₁)", bg=BG2, fg=MUTED,
                 font=("Segoe UI", 8), width=6).pack(side=tk.LEFT)
        self._eP1 = tk.Entry(row1, bg=BG3, fg=TEXT, insertbackground=TEXT,
                              relief=tk.FLAT, font=("Consolas", 9), width=6,
                              highlightthickness=1, highlightbackground=BORDER)
        self._eP1.insert(0, "40"); self._eP1.pack(side=tk.LEFT, ipady=3, padx=3)

        separator(inner)
        btn(inner, "▶  CALCULAR", self._run)
        btn(inner, "↺  Estimar r desde datos", self._estimar_r,
            bg=BG3, fg=ACCENT)

    # ── CÁLCULO ──────────────────────────────────────

    def _parse(self):
        N  = float(self._eN.get())
        r  = float(self._er.get())
        P0 = float(self._eP0.get())
        T  = float(self._eT.get())
        cis= [float(v.strip()) for v in self._eCIs.get().split(",") if v.strip()]
        return N, r, P0, T, cis

    def _estimar_r(self):
        try:
            N  = float(self._eN.get())
            P0 = float(self._eP0.get())
            t1 = float(self._et1.get())
            P1 = float(self._eP1.get())
        except ValueError:
            messagebox.showerror("Error", "Revisá los valores."); return
        A   = (N - P0) / P0
        r   = -np.log((N/P1 - 1) / A) / t1
        self._er.delete(0, tk.END)
        self._er.insert(0, f"{r:.6f}")
        messagebox.showinfo("r estimado",
                            f"r ≈ {r:.6f}\n(actualizado en el campo)")

    def _run(self):
        try:
            N, r, P0, T, cis = self._parse()
        except ValueError:
            messagebox.showerror("Error", "Revisá los valores numéricos.")
            return

        A = (N - P0) / P0          # constante de integración

        def P_analitica(t):
            return N / (1 + A * np.exp(-r * t))

        t_arr = np.linspace(0, T, 600)
        P_sol = P_analitica(t_arr)

        # ── Gráfico ───────────────────────────────────
        self._fig.clear()
        with plt.rc_context(MPL):
            ax1 = self._fig.add_subplot(2, 1, 1)
            ax2 = self._fig.add_subplot(2, 1, 2)

        # ── Ax1: Diagrama de fase (dP/dt vs P) ───────
        P_range = np.linspace(0, N * 1.1, 600)
        dP      = r * P_range * (1 - P_range / N)
        with plt.rc_context(MPL):
            ax1.plot(P_range, dP, color=ACCENT, lw=2.2,
                     label=r"$\frac{dP}{dt} = rP\left(1-\frac{P}{K}\right)$")
            ax1.axhline(0, color=MUTED, lw=1)
            ax1.axvline(0, color=BORDER, lw=0.8)

            # puntos de equilibrio
            ax1.scatter(0, 0, color=RED, s=120, zorder=6, facecolors="none",
                        edgecolors=RED, linewidths=2.5, label="P*=0 (Inestable)")
            ax1.scatter(N, 0, color=GREEN, s=130, zorder=6, label=f"P*=K={N} (Estable)")

            # flechas de flujo
            for mid, fp in [(N*0.3, r*0.3*N*(1-0.3)), (N*0.7, r*0.7*N*(1-0.7/1))]:
                d   = 1 if fp > 0 else -1
                col = GREEN if fp > 0 else RED
                ax1.annotate("", xy=(mid+d*N*0.04, 0),
                             xytext=(mid-d*N*0.04, 0),
                             arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=2, mutation_scale=14))
            ax1.annotate(f"K = {N}", xy=(N, 0), xytext=(N, max(dP)*0.4),
                         color=GREEN, fontsize=9, ha="center",
                         arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
            ax1.set_xlabel("P (población)", fontsize=11)
            ax1.set_ylabel(r"$\dot{P}$", fontsize=11)
            ax1.set_title("Diagrama de fases — Sistema Logístico",
                          fontsize=12, fontweight="bold")
            ax1.legend(fontsize=8)

        # ── Ax2: Soluciones en el tiempo ──────────────
        all_cis = sorted(set([P0] + cis))
        with plt.rc_context(MPL):
            for i, p0 in enumerate(all_cis):
                Ai  = (N - p0) / p0 if p0 != 0 else 1e9
                sol = N / (1 + Ai * np.exp(-r * t_arr))
                lw  = 2.5 if p0 == P0 else 1.6
                ax2.plot(t_arr, sol, color=COLORS_TRAJ[i % len(COLORS_TRAJ)],
                         lw=lw, label=f"P(0)={p0}")
            ax2.axhline(N, color=GREEN, ls="--", lw=1.2, alpha=0.7,
                        label=f"K = {N}")
            ax2.axhline(N/2, color=YELLOW, ls=":", lw=1.0, alpha=0.6,
                        label=f"K/2 = {N/2:.1f}  (punto de inflexión)")
            ax2.set_xlabel("Tiempo t", fontsize=11)
            ax2.set_ylabel("P(t)", fontsize=11)
            ax2.set_title(r"Soluciones en el tiempo  $P(t) = \dfrac{K}{1+Ae^{-rt}}$",
                          fontsize=12, fontweight="bold")
            ax2.legend(fontsize=8, ncol=2)

        self._fig.tight_layout(pad=2.2, h_pad=3.0)
        self._canvas_mpl.draw()

        # ── Log ───────────────────────────────────────
        self._do_log(N, r, P0, T, A, P_analitica)

    def _do_log(self, N, r, P0, T, A, P_func):
        self.log_clear()
        w = self.w; nl = self.nl

        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END,
            "  SISTEMA LOGÍSTICO — RESOLUCIÓN ANALÍTICA  ", "h1")
        self._log.config(state=tk.DISABLED)
        nl(2)

        w("  Modelo:  ", "dim"); w("dP/dt = rP(1 - P/N)\n", "eq")
        w(f"  N = {N}  |  r = {r}  |  P(0) = {P0}\n", "dim")
        nl()

        # PASO 1
        self.section(1, "Separación de variables")
        self.rule("Separar P y t en lados distintos")
        nl()
        w("  dP / [P(1 - P/N)]  =  r dt\n", "val")
        nl()
        w("  Multiplicamos por N arriba y abajo:\n", "dim")
        w("  N·dP / [P(N - P)]  =  r dt\n", "val")
        nl()

        # PASO 2
        self.section(2, "Fracciones parciales")
        self.rule("1/[P(N-P)] = A/P + B/(N-P)")
        nl()
        w("  1 = A(N-P) + B·P\n", "val")
        w("  Si P = 0  →  A = 1/N\n", "ok2")
        w("  Si P = N  →  B = 1/N\n", "ok2")
        nl()
        w("  Entonces:\n", "dim")
        w("  1/[P(N-P)] = (1/N)/P + (1/N)/(N-P)\n", "val")
        nl()

        # PASO 3
        self.section(3, "Integración")
        self.rule("Regla de la cadena + logaritmo")
        nl()
        w("  (1/N)·ln|P| - (1/N)·ln|N-P|  =  rt + C₁\n", "val")
        w("  (1/N)·ln|P/(N-P)|  =  rt + C₁\n", "val")
        w("  ln|P/(N-P)|  =  Nrt + C₂\n", "val")
        nl()
        w("  Exponencial a ambos lados:\n", "dim")
        w("  P/(N-P)  =  C·e^(rt)\n", "val")
        nl()

        # PASO 4
        self.section(4, "Despejar P(t)")
        nl()
        w("  P  =  N·C·e^(rt) - P·C·e^(rt)\n", "val")
        w("  P(1 + C·e^(rt))  =  N·C·e^(rt)\n", "val")
        w("  P(t)  =  N / (1 + (1/C)·e^(-rt))\n", "val")
        nl()
        w("  Definiendo  A = 1/C  →\n", "dim")
        self.box(f"  P(t)  =  N / (1 + A·e^(-rt))")
        nl()

        # PASO 5
        self.section(5, "Condiciones iniciales")
        self.rule(f"P(0) = {P0}")
        nl()
        w(f"  P(0) = N / (1 + A·e^0) = {P0}\n", "val")
        w(f"  {P0} = {N} / (1 + A)\n", "val")
        w(f"  1 + A = {N}/{P0} = {N/P0:.4f}\n", "val")
        w(f"  A = {A:.6f}\n", "ok")
        nl()
        self.box(f"  P(t)  =  {N} / (1 + {A:.4f}·e^(-{r}·t))")
        nl()

        # PASO 6
        self.section(6, "Puntos de equilibrio y estabilidad")
        self.rule("Resolver dP/dt = 0  →  rP(1 - P/N) = 0")
        nl()
        w("  Caso 1:  P = 0\n", "dim")
        fp0 = r * (1 - 0/N) - 0  # f'(0) = r
        w(f"  f'(P*=0)  =  r·(1 - 2·0/N)  =  r  =  {r:.4f}  >  0\n", "err2")
        w("  → P* = 0  es INESTABLE\n", "err")
        nl()
        w(f"  Caso 2:  P = N = {N}\n", "dim")
        fpN = r - 2*r*N/N  # f'(N) = r - 2r = -r
        w(f"  f'(P*=N)  =  r·(1 - 2N/N)  =  -r  =  {-r:.4f}  <  0\n", "ok2")
        w(f"  → P* = K = {N}  es ESTABLE  ✓\n", "ok")
        nl()

        # PASO 7
        self.section(7, "Punto de inflexión")
        self.rule("Máximo de dP/dt  →  d²P/dt² = 0")
        nl()
        w(f"  Se produce en  P = K/2 = {N/2:.2f}\n", "val")
        w("  Ahí la población crece más rápido (inflexión de la curva S)\n", "dim")
        nl()

        # PASO 8 — Valores concretos
        self.section(8, "Valores calculados")
        nl()
        for t_val in [1, 2, 4, 6, 8, 10, int(T)]:
            p = P_func(t_val)
            w(f"  P({t_val:3d})  =  ", "dim")
            w(f"{p:.2f}\n", "val")
        nl()
        w(f"  lim t→∞   P(t)  =  K = {N}\n", "box")
        nl()

        self._log.config(state=tk.NORMAL)
        self._log.see("1.0")
        self._log.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sistema Logístico")
    root.geometry("1400x820")
    root.configure(bg=BG)
    app = SistemaLogisticoApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()