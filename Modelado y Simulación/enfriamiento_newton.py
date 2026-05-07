"""
Enfriamiento de Newton
======================
dT/dt = -k(T - Ta)
Solución: T(t) = Ta + (T0 - Ta)·e^(-kt)

Basado en apuntes Cáceres 2026 — Clase 7 y 8.
"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from styles import *


class EnfriamientoNewtonApp(BaseApp):
    TITLE    = "Enfriamiento de Newton"
    SUBTITLE = "dT/dt = -k(T - Tₐ)   •   Decaimiento exponencial"
    FIG_SIZE = (10, 8)

    def _build_controls(self, inner):
        section_title(inner, "Parámetros del modelo")
        self._eT0 = labeled_entry(inner, "T(0)  — temperatura inicial (°C)",  "40")
        self._eTa = labeled_entry(inner, "Tₐ    — temperatura ambiente (°C)", "20")
        self._ek  = labeled_entry(inner, "k     — constante de enfriamiento",  "0.1155")
        self._eT_end = labeled_entry(inner, "t_end  (unidades de tiempo)", "30")

        section_title(inner, "Estimar k desde una medición")
        tk.Label(inner,
                 text="Si T(0) y T(tₘ) son conocidos, se calcula k:",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 8),
                 wraplength=270, justify=tk.LEFT, anchor="w"
                 ).pack(fill=tk.X, pady=(0, 2))
        row = tk.Frame(inner, bg=BG2)
        row.pack(fill=tk.X, pady=2)
        for lbl_text, attr, default, w_size in [
            ("t_m", "_etm", "6",  4),
            ("T(tₘ)", "_ePm", "30", 5)]:
            tk.Label(row, text=lbl_text, bg=BG2, fg=MUTED,
                     font=("Segoe UI", 8), width=w_size
                     ).pack(side=tk.LEFT)
            e = tk.Entry(row, bg=BG3, fg=TEXT, insertbackground=TEXT,
                         relief=tk.FLAT, font=("Consolas", 9), width=7,
                         highlightthickness=1, highlightbackground=BORDER)
            e.insert(0, default)
            e.pack(side=tk.LEFT, ipady=3, padx=3)
            setattr(self, attr, e)

        section_title(inner, "Calcular temperatura en t")
        self._et_eval = labeled_entry(inner, "t  (calcular T en este instante)", "18")

        section_title(inner, "Condiciones iniciales adicionales")
        self._eCIs = labeled_entry(inner, "T(0) extra (sep. coma)", "5, 20, 60, 80")

        separator(inner)
        btn(inner, "▶  CALCULAR", self._run)
        btn(inner, "↺  Estimar k desde medición", self._estimar_k,
            bg=BG3, fg=ACCENT)

    def _estimar_k(self):
        try:
            T0 = float(self._eT0.get())
            Ta = float(self._eTa.get())
            tm = float(self._etm.get())
            Pm = float(self._ePm.get())
        except ValueError:
            messagebox.showerror("Error", "Revisá los valores."); return
        if T0 == Ta:
            messagebox.showerror("Error", "T(0) no puede ser igual a Tₐ.")
            return
        k = -np.log((Pm - Ta) / (T0 - Ta)) / tm
        self._ek.delete(0, tk.END)
        self._ek.insert(0, f"{k:.7f}")
        messagebox.showinfo("k estimado",
                            f"k ≈ {k:.7f}\n(actualizado en el campo)")

    def _run(self):
        try:
            T0    = float(self._eT0.get())
            Ta    = float(self._eTa.get())
            k     = float(self._ek.get())
            t_end = float(self._eT_end.get())
            t_eval= float(self._et_eval.get())
            cis   = [float(v.strip())
                     for v in self._eCIs.get().split(",") if v.strip()]
        except ValueError:
            messagebox.showerror("Error", "Revisá los valores numéricos.")
            return

        A = T0 - Ta  # constante de integración

        def T_func(t, t0=T0):
            return Ta + (t0 - Ta) * np.exp(-k * t)

        t_arr = np.linspace(0, t_end, 600)

        # ── Gráfico ────────────────────────────────────
        self._fig.clear()
        with plt.rc_context(MPL):
            ax1 = self._fig.add_subplot(2, 1, 1)
            ax2 = self._fig.add_subplot(2, 1, 2)

        # Diagrama de fases (dT/dt vs T)
        with plt.rc_context(MPL):
            T_range = np.linspace(min(-10, Ta-20), max(T0*1.3, Ta+20), 500)
            dT_dt   = -k * (T_range - Ta)
            ax1.plot(T_range, dT_dt, color=ORANGE, lw=2.2,
                     label=r"$\frac{dT}{dt} = -k(T-T_a)$")
            ax1.axhline(0, color=MUTED, lw=1)
            ax1.axvline(Ta, color=GREEN, lw=1.2, ls="--", alpha=0.8)
            ax1.scatter(Ta, 0, color=GREEN, s=130, zorder=6,
                        label=f"T* = Tₐ = {Ta} (Estable)")
            # flecha de flujo
            for T_test in [Ta + 15, Ta - 15]:
                fv  = -k*(T_test - Ta)
                d   = 1 if fv > 0 else -1
                col = GREEN if fv > 0 else RED
                ax1.annotate("", xy=(T_test + d*2, 0),
                             xytext=(T_test - d*2, 0),
                             arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=2, mutation_scale=13))
            ax1.annotate(f"Equilibrio\nT* = Tₐ = {Ta}°C",
                         xy=(Ta, 0), xytext=(Ta + (max(T_range)-Ta)*0.25, max(dT_dt)*0.5),
                         color=GREEN, fontsize=9,
                         arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
            ax1.set_xlabel("T (temperatura)", fontsize=11)
            ax1.set_ylabel(r"$\dot{T}$", fontsize=11)
            ax1.set_title("Diagrama de fases — Enfriamiento de Newton",
                          fontsize=12, fontweight="bold")
            ax1.legend(fontsize=8)

        # Soluciones T(t)
        with plt.rc_context(MPL):
            all_cis = sorted(set([T0] + cis))
            for i, t0 in enumerate(all_cis):
                sol = T_func(t_arr, t0=t0)
                lw  = 2.5 if t0 == T0 else 1.6
                ax2.plot(t_arr, sol, color=COLORS_TRAJ[i%len(COLORS_TRAJ)],
                         lw=lw, label=f"T(0)={t0}°C")
            ax2.axhline(Ta, color=GREEN, ls="--", lw=1.2, alpha=0.7,
                        label=f"Tₐ = {Ta}°C")

            # marcar t_eval
            Tv = T_func(t_eval)
            ax2.scatter([t_eval], [Tv], color=YELLOW, s=120, zorder=8)
            ax2.annotate(f"T({t_eval}) ≈ {Tv:.2f}°C",
                         xy=(t_eval, Tv),
                         xytext=(t_eval + t_end*0.05, Tv + (T0-Ta)*0.1),
                         color=YELLOW, fontsize=9,
                         arrowprops=dict(arrowstyle="->",
                                         color=YELLOW, lw=1.2))
            ax2.set_xlabel("Tiempo t", fontsize=11)
            ax2.set_ylabel("T(t)  [°C]", fontsize=11)
            ax2.set_title(r"Soluciones  $T(t) = T_a + (T_0-T_a)\,e^{-kt}$",
                          fontsize=12, fontweight="bold")
            ax2.legend(fontsize=8, ncol=2)

        self._fig.tight_layout(pad=2.2, h_pad=3.0)
        self._canvas_mpl.draw()

        self._do_log(T0, Ta, k, A, t_eval, T_func)

    def _do_log(self, T0, Ta, k, A, t_eval, T_func):
        self.log_clear()
        w = self.w; nl = self.nl

        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END,
            "  ENFRIAMIENTO DE NEWTON — RESOLUCIÓN ANALÍTICA  ", "h1")
        self._log.config(state=tk.DISABLED)
        nl(2)
        w("  Modelo:  ", "dim"); w("dT/dt = -k(T - Tₐ)\n", "eq")
        w(f"  T(0) = {T0}°C  |  Tₐ = {Ta}°C  |  k = {k:.6f}\n", "dim"); nl()

        # PASO 1
        self.section(1, "Análisis del modelo")
        nl()
        w("  La ecuación describe el cambio de temperatura de un objeto\n", "dim")
        w("  rodeado por un ambiente a temperatura constante Tₐ.\n", "dim")
        nl()
        w("  k > 0  →  constante de enfriamiento (k > 0  siempre)\n", "dim")
        w("  Cuanto mayor k, más rápido se acerca T al equilibrio.\n", "dim"); nl()

        # PASO 2
        self.section(2, "Separación de variables")
        self.rule("Separar T y t")
        nl()
        w("  dT / (T - Tₐ)  =  -k dt\n", "val"); nl()

        # PASO 3
        self.section(3, "Integración")
        self.rule("∫ dT/(T-Tₐ) = -∫k dt")
        nl()
        w("  ln|T - Tₐ|  =  -kt + C₁\n", "val")
        w("  Exponencial a ambos lados:\n", "dim")
        w("  T - Tₐ  =  A·e^(-kt)\n", "val"); nl()
        self.box("  T(t)  =  Tₐ + A·e^(-kt)")
        nl()

        # PASO 4
        self.section(4, "Condición inicial  T(0) = T₀")
        nl()
        w(f"  T(0) = Tₐ + A·e^0 = Tₐ + A = {T0}\n", "val")
        w(f"  A = T₀ - Tₐ = {T0} - {Ta} = {A:.4f}\n", "ok"); nl()
        self.box(f"  T(t)  =  {Ta} + {A:.4f}·e^(-{k:.6f}·t)")
        nl()

        # PASO 5
        self.section(5, "Punto de equilibrio y estabilidad")
        self.rule("Resolver dT/dt = 0  →  -k(T - Tₐ) = 0")
        nl()
        w(f"  T* = Tₐ = {Ta}°C\n", "ok")
        w( "  f'(T*) = -k < 0   →   T* es ESTABLE\n", "ok")
        nl()
        w( "  Interpretación física:\n", "dim")
        w(f"  Si T > Tₐ  →  dT/dt < 0  →  objeto se ENFRÍA\n", "err2")
        w(f"  Si T < Tₐ  →  dT/dt > 0  →  objeto se CALIENTA\n", "ok2")
        w(f"  En ambos casos tiende a T* = {Ta}°C\n", "dim"); nl()
        self.box(f"  lim t→∞  T(t)  =  Tₐ = {Ta}°C")
        nl()

        # PASO 6
        self.section(6, "Comportamiento asintótico")
        nl()
        w(f"  La distancia al equilibrio: |T(t) - Tₐ| = {abs(A):.4f}·e^(-{k:.4f}t)\n",
          "val")
        w( "  Tiende a 0 exponencialmente cuando t → ∞\n", "dim")
        w(f"  (el tiempo para llegar al 50% del equilibrio ≈ ln2/k = {np.log(2)/k:.2f})\n",
          "dim"); nl()

        # PASO 7
        self.section(7, "Evaluación en puntos concretos")
        nl()
        for tv in [1, 2, 5, 10, 15, 20, int(self._eT_end.get())]:
            Tv = T_func(tv)
            w(f"  T({tv:3d})  =  ", "dim"); w(f"{Tv:.4f}°C\n", "val")
        nl()
        Tv_eval = T_func(t_eval)
        w(f"  ► T({t_eval}) = {Ta} + {A:.4f}·e^(-{k:.6f}·{t_eval})\n", "dim")
        self.box(f"  T({t_eval})  ≈  {Tv_eval:.4f}°C")
        nl()

        self._log.config(state=tk.NORMAL)
        self._log.see("1.0")
        self._log.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Enfriamiento de Newton")
    root.geometry("1400x820")
    root.configure(bg=BG)
    app = EnfriamientoNewtonApp(root, standalone=True)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()