"""
Módulo: Método de Bisección (Búsqueda Binaria)
Busca raíces de funciones continuas en un intervalo [a, b].
"""

import tkinter as tk
from tkinter import messagebox
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def _evaluar(expr: str, x_val: float) -> float:
    """Evalúa una expresión matemática en x = x_val."""
    entorno = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    entorno["x"] = x_val
    entorno["np"] = np
    return eval(expr, {"__builtins__": {}}, entorno)


def _evaluar_escalar(expr: str) -> float:
    """Evalúa una expresión que devuelve un número (para a, b, tolerancia)."""
    entorno = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    entorno["np"] = np
    return float(eval(str(expr), {"__builtins__": {}}, entorno))


# ─────────────────────────────────────────────
#  LÓGICA NUMÉRICA (pura, sin GUI)
# ─────────────────────────────────────────────

def biseccion(expr: str, a: float, b: float, tolerancia: float, max_iter: int):
    """
    Ejecuta el método de bisección.

    Returns
    -------
    raiz : float
    iteraciones : list[dict]   — cada dict: {i, c, fc, error}
    mensaje : str              — resultado final
    """
    fa = _evaluar(expr, a)
    fb = _evaluar(expr, b)

    if fa * fb >= 0:
        raise ValueError(
            f"Bolzano no se cumple: f({a:.4g})={fa:.4g}, f({b:.4g})={fb:.4g} "
            f"tienen el mismo signo."
        )

    iteraciones = []
    c = a

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = _evaluar(expr, c)
        error = (b - a) / 2
        iteraciones.append({"i": i, "c": c, "fc": fc, "error": error})

        if fc == 0 or error < tolerancia:
            return c, iteraciones, "convergencia"

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

    return c, iteraciones, "max_iter"


# ─────────────────────────────────────────────
#  CLASE GUI
# ─────────────────────────────────────────────

class BiseccionFrame(tk.Frame):
    """
    Panel completo del Método de Bisección.
    Se puede embeber en cualquier ventana Tkinter.
    """

    FONT_MONO = ("Consolas", 9)
    FONT_LABEL = ("Segoe UI", 10)
    COLOR_BTN_CALC = "#2E7D32"
    COLOR_BTN_GRAPH = "#1565C0"

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._raiz = None
        self._historial = []
        self._build_ui()

    # ── Construcción de la interfaz ──────────────────────────────────────────

    def _build_ui(self):
        # Dividir en panel izquierdo (controles) y derecho (gráfico)
        panel_ctrl = tk.Frame(self, padx=12, pady=12, bg="#F5F5F5")
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)

        panel_graf = tk.Frame(self)
        panel_graf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controles(panel_ctrl)
        self._build_grafico(panel_graf)

    def _lbl(self, parent, texto, pady_top=8):
        tk.Label(
            parent, text=texto, anchor="w",
            bg="#F5F5F5", font=self.FONT_LABEL
        ).pack(anchor="w", pady=(pady_top, 0))

    def _entry(self, parent, default):
        e = tk.Entry(parent, width=32, font=self.FONT_LABEL)
        e.insert(0, default)
        e.pack(pady=2)
        return e

    def _build_controles(self, parent):
        parent.config(bg="#F5F5F5")

        tk.Label(
            parent, text="Método de Bisección",
            font=("Segoe UI", 13, "bold"), bg="#F5F5F5", fg="#1A237E"
        ).pack(anchor="w", pady=(0, 10))

        self._lbl(parent, "Función f(x):", pady_top=0)
        self.e_f = self._entry(parent, "x**2 - 2")

        self._lbl(parent, "Límite inferior a:")
        self.e_a = self._entry(parent, "0.1")

        self._lbl(parent, "Límite superior b:")
        self.e_b = self._entry(parent, "2")

        self._lbl(parent, "Tolerancia (error):")
        self.e_tol = self._entry(parent, "1e-6")

        self._lbl(parent, "Máx. iteraciones:")
        self.e_iter = self._entry(parent, "100")

        # Botones
        frame_btn = tk.Frame(parent, bg="#F5F5F5")
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

        # Consola de resultados
        tk.Label(
            parent, text="Tabla de iteraciones:",
            bg="#F5F5F5", font=self.FONT_LABEL
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
            0.5, 0.5, "Ingresá los datos y presioná\n'Calcular raíz' o 'Ver gráfica'",
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
            expr = self.e_f.get().strip()
            a = _evaluar_escalar(self.e_a.get())
            b = _evaluar_escalar(self.e_b.get())
            tol = _evaluar_escalar(self.e_tol.get())
            max_iter = int(self.e_iter.get())

            if a >= b:
                messagebox.showwarning("Advertencia", "'a' debe ser estrictamente menor que 'b'.")
                return

            raiz, historial, estado = biseccion(expr, a, b, tol, max_iter)
            self._raiz = raiz
            self._historial = historial

            # Encabezado tabla
            self._log(f"  Bisección en f(x) = {expr}")
            self._log(f"  Intervalo: [{a}, {b}]  |  Tolerancia: {tol}\n")
            self._log(f"{'Iter':<6} | {'Centro c':<14} | {'f(c)':<16} | {'Error Est.'}")
            self._log("─" * 58)

            for r in historial:
                self._log(f"{r['i']:<6} | {r['c']:<14.8f} | {r['fc']:<16.6e} | {r['error']:.6e}")

            self._log("─" * 58)
            if estado == "max_iter":
                self._log("⚠  Máximas iteraciones alcanzadas sin convergencia.")
            else:
                self._log(f"✔  Raíz aproximada : {raiz:.10f}")
                self._log(f"   Error final     : {historial[-1]['error']:.2e}")
                self._log(f"   Iteraciones     : {len(historial)}")

            self._graficar()

        except ValueError as exc:
            messagebox.showerror("Error numérico", str(exc))
        except Exception as exc:
            messagebox.showerror("Error", f"Verificá los datos ingresados.\n{exc}")

    def _graficar(self):
        try:
            expr = self.e_f.get().strip()
            a = _evaluar_escalar(self.e_a.get())
            b = _evaluar_escalar(self.e_b.get())
        except Exception as exc:
            messagebox.showerror("Error", f"Valores inválidos para graficar.\n{exc}")
            return

        self.ax.clear()
        margen = abs(b - a) * 0.4 or 1.0
        x_vals = np.linspace(a - margen, b + margen, 600)

        y_vals = []
        for xv in x_vals:
            try:
                y_vals.append(_evaluar(expr, float(xv)))
            except Exception:
                y_vals.append(np.nan)
        y_vals = np.array(y_vals, dtype=float)

        self.ax.plot(x_vals, y_vals, color="#1565C0", linewidth=2.2, label=f"f(x) = {expr}")
        self.ax.axhline(0, color="#333", linewidth=0.9, alpha=0.6)
        self.ax.axvline(0, color="#333", linewidth=0.9, alpha=0.6)
        self.ax.axvline(a, color="#E53935", linestyle=":", linewidth=1.8, label=f"a = {a:.4g}")
        self.ax.axvline(b, color="#7B1FA2", linestyle=":", linewidth=1.8, label=f"b = {b:.4g}")

        if self._raiz is not None:
            try:
                fy = _evaluar(expr, self._raiz)
            except Exception:
                fy = 0
            self.ax.scatter(
                [self._raiz], [0], color="#E65100", zorder=6,
                s=90, label=f"Raíz ≈ {self._raiz:.6f}"
            )
            self.ax.annotate(
                f"  Raíz ≈ {self._raiz:.6f}",
                xy=(self._raiz, 0), fontsize=9,
                color="#E65100", va="bottom"
            )

        # Límites Y razonables
        finite = y_vals[np.isfinite(y_vals)]
        if len(finite):
            ymin, ymax = finite.min(), finite.max()
            rng = ymax - ymin or 1
            self.ax.set_ylim(ymin - rng * 0.15, ymax + rng * 0.15)

        self.ax.set_title("Método de Bisección", fontsize=13, fontweight="bold")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.legend(fontsize=9, loc="best")
        self.ax.grid(True, linestyle=":", alpha=0.55)
        self.fig.tight_layout()
        self.canvas.draw()
