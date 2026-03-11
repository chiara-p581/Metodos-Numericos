"""
Módulo: Método de Punto Fijo
Busca raíces de f(x) iterando x_{n+1} = g(x_n) hasta convergencia.
"""

import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def _evaluar(expr: str, x_val) -> float:
    """Evalúa una expresión matemática en x = x_val (escalar o array numpy)."""
    entorno = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    entorno["x"] = x_val
    entorno["np"] = np
    entorno["sin"]  = np.sin
    entorno["cos"]  = np.cos
    entorno["tan"]  = np.tan
    entorno["exp"]  = np.exp
    entorno["log"]  = np.log
    entorno["sqrt"] = np.sqrt
    return eval(expr, {"__builtins__": {}}, entorno)


def _evaluar_escalar(expr: str) -> float:
    """Evalúa una expresión que devuelve un número (para x0, tolerancia, etc.)."""
    entorno = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    entorno["np"] = np
    return float(eval(str(expr), {"__builtins__": {}}, entorno))


# ─────────────────────────────────────────────
#  LÓGICA NUMÉRICA (pura, sin GUI)
# ─────────────────────────────────────────────

def punto_fijo(expr_f: str, expr_g: str, x0: float, tolerancia: float, max_iter: int):
    """
    Ejecuta el método de Punto Fijo.

    Parameters
    ----------
    expr_f : str   — función original f(x) (solo para mostrar f(x_n) en tabla)
    expr_g : str   — función despejada g(x) tal que x = g(x)
    x0     : float — punto inicial
    tolerancia : float
    max_iter   : int

    Returns
    -------
    raiz : float
    iteraciones : list[dict]  — cada dict: {i, x, fx, error}
    estado : str              — "convergencia" | "max_iter"
    """
    iteraciones = []
    x_n = x0

    for i in range(1, max_iter + 1):
        x_next = float(_evaluar(expr_g, x_n))
        error = abs(x_next - x_n)

        try:
            fx = float(_evaluar(expr_f, x_next))
        except Exception:
            fx = float("nan")

        iteraciones.append({"i": i, "x": x_next, "fx": fx, "error": error})

        x_n = x_next

        if error < tolerancia:
            return x_n, iteraciones, "convergencia"

    return x_n, iteraciones, "max_iter"


# ─────────────────────────────────────────────
#  CLASE GUI
# ─────────────────────────────────────────────

class PuntoFijoFrame(tk.Frame):
    """
    Panel completo del Método de Punto Fijo.
    Se puede embeber en cualquier ventana Tkinter.
    """

    FONT_MONO  = ("Consolas", 9)
    FONT_LABEL = ("Segoe UI", 10)
    COLOR_BTN_CALC  = "#4527A0"
    COLOR_BTN_GRAPH = "#00695C"

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._raiz = None
        self._historial = []
        self._build_ui()

    # ── Construcción de la interfaz ──────────────────────────────────────────

    def _build_ui(self):
        panel_ctrl = tk.Frame(self, padx=12, pady=12, bg="#F3F0FF")
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)

        panel_graf = tk.Frame(self)
        panel_graf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controles(panel_ctrl)
        self._build_grafico(panel_graf)

    def _lbl(self, parent, texto, pady_top=8):
        tk.Label(
            parent, text=texto, anchor="w",
            bg="#F3F0FF", font=self.FONT_LABEL
        ).pack(anchor="w", pady=(pady_top, 0))

    def _entry(self, parent, default):
        e = tk.Entry(parent, width=32, font=self.FONT_LABEL)
        e.insert(0, default)
        e.pack(pady=2)
        return e

    def _build_controles(self, parent):
        parent.config(bg="#F3F0FF")

        tk.Label(
            parent, text="Método de Punto Fijo",
            font=("Segoe UI", 13, "bold"), bg="#F3F0FF", fg="#1A237E"
        ).pack(anchor="w", pady=(0, 10))

        self._lbl(parent, "Función original f(x):", pady_top=0)
        self.e_f = self._entry(parent, "2**(-x) - x")

        self._lbl(parent, "Función iterativa g(x)  [x = g(x)]:")
        self.e_g = self._entry(parent, "2**(-x)")

        self._lbl(parent, "Punto inicial x₀:")
        self.e_x0 = self._entry(parent, "2/3")

        self._lbl(parent, "Tolerancia (error):")
        self.e_tol = self._entry(parent, "1e-6")

        self._lbl(parent, "Máx. iteraciones:")
        self.e_iter = self._entry(parent, "100")

        # Botones
        frame_btn = tk.Frame(parent, bg="#F3F0FF")
        frame_btn.pack(fill=tk.X, pady=14)

        tk.Button(
            frame_btn, text="▶  Calcular raíz",
            bg=self.COLOR_BTN_CALC, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6,
            command=self._calcular
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            frame_btn, text="📈  Ver gráfica",
            bg=self.COLOR_BTN_GRAPH, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6,
            command=self._graficar
        ).pack(side=tk.LEFT)

        # Consola
        tk.Label(
            parent, text="Tabla de iteraciones:",
            bg="#F3F0FF", font=self.FONT_LABEL
        ).pack(anchor="w", pady=(6, 2))

        frame_txt = tk.Frame(parent)
        frame_txt.pack(fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(frame_txt)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt = tk.Text(
            frame_txt, height=18, width=58,
            font=self.FONT_MONO,
            yscrollcommand=scroll.set,
            bg="#1E1E1E", fg="#D4D4D4",
            insertbackground="white"
        )
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.txt.yview)

    def _build_grafico(self, parent):
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._ax_placeholder()

    def _ax_placeholder(self):
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            "Ingresá los datos y presioná\n'Calcular raíz' o 'Ver gráfica'",
            ha="center", va="center", fontsize=12,
            color="#888", transform=self.ax.transAxes
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    # ── Lógica de eventos ────────────────────────────────────────────────────

    def _log(self, texto):
        self.txt.insert(tk.END, texto + "\n")
        self.txt.see(tk.END)

    def _calcular(self):
        self.txt.delete(1.0, tk.END)
        try:
            expr_f = self.e_f.get().strip()
            expr_g = self.e_g.get().strip()
            x0 = _evaluar_escalar(self.e_x0.get())
            tol = _evaluar_escalar(self.e_tol.get())
            max_iter = int(self.e_iter.get())

            raiz, historial, estado = punto_fijo(expr_f, expr_g, x0, tol, max_iter)
            self._raiz = raiz
            self._historial = historial

            # Encabezado tabla
            self._log(f"  Punto Fijo: f(x) = {expr_f}")
            self._log(f"  g(x) = {expr_g}  |  x₀ = {x0}  |  Tolerancia: {tol}\n")
            self._log(f"{'Iter':<6} | {'x (aprox.)':<15} | {'f(x)':<16} | {'Error Est.'}")
            self._log("─" * 60)

            for r in historial:
                self._log(f"{r['i']:<6} | {r['x']:<15.8f} | {r['fx']:<16.6e} | {r['error']:.6e}")

            self._log("─" * 60)
            if estado == "max_iter":
                self._log("⚠  Máximas iteraciones alcanzadas sin convergencia.")
            else:
                self._log(f"✔  Raíz aproximada : {raiz:.10f}")
                self._log(f"   Error final     : {historial[-1]['error']:.2e}")
                self._log(f"   Iteraciones     : {len(historial)}")

            self._graficar()

        except Exception as exc:
            messagebox.showerror("Error", f"Verificá los datos ingresados.\n{exc}")

    def _graficar(self):
        try:
            expr_f = self.e_f.get().strip()
            expr_g = self.e_g.get().strip()
            x0 = _evaluar_escalar(self.e_x0.get())
        except Exception as exc:
            messagebox.showerror("Error", f"Valores inválidos para graficar.\n{exc}")
            return

        self.ax.clear()

        # Rango dinámico centrado en x0 o en la raíz si ya se calculó
        centro = self._raiz if self._raiz is not None else x0
        margen = max(abs(centro) * 0.6, 1.5)
        x_vals = np.linspace(centro - margen, centro + margen, 600)

        def safe_eval_array(expr):
            ys = []
            for xv in x_vals:
                try:
                    ys.append(float(_evaluar(expr, float(xv))))
                except Exception:
                    ys.append(np.nan)
            return np.array(ys, dtype=float)

        y_f = safe_eval_array(expr_f)
        y_g = safe_eval_array(expr_g)

        self.ax.plot(x_vals, y_f, color="#1565C0", linewidth=2.2, label=f"f(x) = {expr_f}")
        self.ax.plot(x_vals, y_g, color="#2E7D32", linewidth=2.0, linestyle="--",
                     label=f"g(x) = {expr_g}")
        self.ax.plot(x_vals, x_vals, color="#B71C1C", linewidth=1.4, linestyle="-.",
                     alpha=0.7, label="Identidad y = x")

        self.ax.axhline(0, color="#333", linewidth=0.8, alpha=0.55)
        self.ax.axvline(0, color="#333", linewidth=0.8, alpha=0.55)

        if self._raiz is not None:
            # Marcar el punto fijo (intersección g(x) = x)
            self.ax.scatter(
                [self._raiz], [self._raiz], color="#FF6F00",
                zorder=6, s=90, label=f"Punto fijo ≈ {self._raiz:.6f}"
            )
            # Marcar la raíz en f(x) = 0
            self.ax.scatter(
                [self._raiz], [0], color="#FF6F00",
                zorder=6, s=90, marker="X"
            )
            self.ax.annotate(
                f"  Raíz ≈ {self._raiz:.6f}",
                xy=(self._raiz, 0), fontsize=9,
                color="#FF6F00", va="bottom"
            )

        # Límites Y razonables (ignorar outliers)
        all_finite = np.concatenate([
            y_f[np.isfinite(y_f)],
            y_g[np.isfinite(y_g)]
        ])
        if len(all_finite):
            ymin, ymax = np.percentile(all_finite, 2), np.percentile(all_finite, 98)
            rng = ymax - ymin or 1
            self.ax.set_ylim(ymin - rng * 0.15, ymax + rng * 0.15)

        self.ax.set_title("Método de Punto Fijo", fontsize=13, fontweight="bold")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend(fontsize=9, loc="best")
        self.ax.grid(True, linestyle=":", alpha=0.55)
        self.fig.tight_layout()
        self.canvas.draw()
