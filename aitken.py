import math
import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
 
 
# ─────────────────────────────
# ENTORNO SEGURO PARA eval
# ─────────────────────────────
 
def _entorno():
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env["np"] = np
    return env
 
 
def evaluar(expr, x):
    env = _entorno()
    env["x"] = x
    return eval(expr, {"__builtins__": {}}, env)
 
 
# ─────────────────────────────
# MÉTODO DE AITKEN
# ─────────────────────────────
 
def aitken(expr_g, x0, tol=1e-6, max_iter=100):
    historial = []
    x_n = x0
 
    for i in range(max_iter):
        try:
            x1 = evaluar(expr_g, x_n)
            x2 = evaluar(expr_g, x1)
 
            numerador = (x1 - x_n) ** 2
            denominador = x2 - 2 * x1 + x_n
 
            if denominador == 0:
                return x_n, historial, "division_cero"
 
            x_hat = x_n - numerador / denominador
            error = abs(x_hat - x_n)
 
            historial.append({
                "i": i + 1,
                "x": x_hat,
                "error": error
            })
 
            if error < tol:
                return x_hat, historial, "convergencia"
 
            x_n = x_hat
 
        except Exception as e:
            return None, historial, f"error: {e}"
 
    return x_n, historial, "max_iter"
 
 
# ─────────────────────────────
# INTERFAZ GRÁFICA
# ─────────────────────────────
 
class AitkenFrame(tk.Frame):
 
    FONT_MONO  = ("Consolas", 9)
    FONT_LABEL = ("Segoe UI", 10)
    COLOR_BTN  = "#C62828"
    COLOR_GRAPH = "#1565C0"
 
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._raiz = None
        self._historial = []
        self._build_ui()
 
    # UI
    def _build_ui(self):
        panel_ctrl = tk.Frame(self, padx=12, pady=12, bg="#FFEBEE")
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y)
 
        panel_graf = tk.Frame(self)
        panel_graf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
 
        self._build_controles(panel_ctrl)
        self._build_grafico(panel_graf)
 
    def _lbl(self, parent, texto, pady_top=8):
        tk.Label(parent, text=texto, bg="#FFEBEE",
                 font=self.FONT_LABEL).pack(anchor="w", pady=(pady_top, 0))
 
    def _entry(self, parent, default):
        e = tk.Entry(parent, width=32, font=self.FONT_LABEL)
        e.insert(0, default)
        e.pack(pady=2)
        return e
 
    def _build_controles(self, parent):
        tk.Label(
            parent, text="Método de Aitken (Δ²)",
            font=("Segoe UI", 13, "bold"),
            bg="#FFEBEE", fg="#B71C1C"
        ).pack(anchor="w", pady=(0, 10))
 
        self._lbl(parent, "Función g(x):", 0)
        self.e_g = self._entry(parent, "sqrt((2*(x+2))/pi)")
 
        self._lbl(parent, "Valor inicial x₀:")
        self.e_x0 = self._entry(parent, "1.4")
 
        self._lbl(parent, "Tolerancia:")
        self.e_tol = self._entry(parent, "1e-6")
 
        self._lbl(parent, "Máx. iteraciones:")
        self.e_iter = self._entry(parent, "100")
 
        # Botones
        frame_btn = tk.Frame(parent, bg="#FFEBEE")
        frame_btn.pack(fill=tk.X, pady=12)
 
        tk.Button(
            frame_btn, text="▶ Calcular",
            bg=self.COLOR_BTN, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6,
            command=self._calcular
        ).pack(side=tk.LEFT, padx=4)
 
        tk.Button(
            frame_btn, text="📈 Graficar",
            bg=self.COLOR_GRAPH, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6,
            command=self._graficar
        ).pack(side=tk.LEFT)
 
        # Consola
        tk.Label(parent, text="Iteraciones:",
                 bg="#FFEBEE").pack(anchor="w")
 
        frame_txt = tk.Frame(parent)
        frame_txt.pack(fill=tk.BOTH, expand=True)
 
        scroll = tk.Scrollbar(frame_txt)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
 
        self.txt = tk.Text(
            frame_txt, height=18, width=55,
            font=self.FONT_MONO,
            yscrollcommand=scroll.set,
            bg="#1E1E1E", fg="#D4D4D4"
        )
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.txt.yview)
 
    # GRÁFICO
    def _build_grafico(self, parent):
        self.fig = Figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._placeholder()
 
    def _placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Ejecutá el método",
                     ha="center", va="center")
        self.canvas.draw()
 
    # LÓGICA UI
    def _log(self, t):
        self.txt.insert(tk.END, t + "\n")
        self.txt.see(tk.END)
 
    def _calcular(self):
        self.txt.delete(1.0, tk.END)
        try:
            g = self.e_g.get()
            x0 = float(eval(self.e_x0.get()))
            tol = float(eval(self.e_tol.get()))
            it = int(self.e_iter.get())
 
            raiz, hist, estado = aitken(g, x0, tol, it)
 
            self._raiz = raiz
            self._historial = hist
 
            self._log(f"Aitken con g(x) = {g}\n")
            self._log(f"{'Iter':<6} | {'x':<15} | Error")
            self._log("-" * 40)
 
            for r in hist:
                self._log(f"{r['i']:<6} | {r['x']:<15.8f} | {r['error']:.6e}")
 
            self._log("-" * 40)
 
            if estado == "convergencia":
                self._log(f"✔ Raíz ≈ {raiz:.10f}")
            else:
                self._log(f"⚠ Estado: {estado}")
 
        except Exception as e:
            messagebox.showerror("Error", str(e))
 
    def _graficar(self):
        try:
            g = self.e_g.get()
            x0 = float(eval(self.e_x0.get()))
 
            xs = np.linspace(x0 - 2, x0 + 2, 500)
            ys = []
 
            for x in xs:
                try:
                    ys.append(evaluar(g, x))
                except:
                    ys.append(np.nan)
 
            self.ax.clear()
            self.ax.plot(xs, ys, label="g(x)")
            self.ax.plot(xs, xs, linestyle="--", label="y = x")
 
            if self._raiz:
                self.ax.scatter([self._raiz], [self._raiz],
                                color="red", label=f"≈ {self._raiz:.4f}")
 
            self.ax.legend()
            self.ax.grid()
            self.canvas.draw()
 
        except Exception as e:
            messagebox.showerror("Error gráfico", str(e))
 