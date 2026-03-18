"""
Módulo: Método de Punto Fijo
Busca raíces de f(x) iterando x_{n+1} = g(x_n) hasta convergencia.

Incluye sugerencia automática de funciones g(x) válidas (|g'(x)| < 1)
usando la API de Anthropic.
"""

import tkinter as tk
from tkinter import messagebox
import math
import json
import urllib.request
import threading
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ─────────────────────────────────────────────
#  HELPERS DE EVALUACIÓN
# ─────────────────────────────────────────────

def _entorno_base():
    env = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    env.update({
        "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs,
    })
    return env


def _evaluar(expr: str, x_val) -> float:
    env = _entorno_base()
    env["x"] = x_val
    return eval(expr, {"__builtins__": {}}, env)


def _evaluar_escalar(expr: str) -> float:
    env = _entorno_base()
    return float(eval(str(expr), {"__builtins__": {}}, env))


# ─────────────────────────────────────────────
#  VALIDACIÓN |g'(x)| < 1
# ─────────────────────────────────────────────

def _derivada_numerica(expr_g: str, x: float, h: float = 1e-7) -> float:
    """Derivada centrada de g en x."""
    return (float(_evaluar(expr_g, x + h)) - float(_evaluar(expr_g, x - h))) / (2 * h)


def validar_convergencia(expr_g: str, x0: float, radio: float = 0.5, n_puntos: int = 40):
    """
    Verifica |g'(x)| < 1 en un entorno de x0.
    Devuelve (es_valida: bool, max_deriv: float, x_max: float).
    """
    xs = np.linspace(x0 - radio, x0 + radio, n_puntos)
    max_d, x_max = 0.0, x0
    for xv in xs:
        try:
            d = abs(_derivada_numerica(expr_g, float(xv)))
            if d > max_d:
                max_d = d
                x_max = float(xv)
        except Exception:
            pass
    return max_d < 1.0, max_d, x_max


# ─────────────────────────────────────────────
#  LÓGICA NUMÉRICA (pura, sin GUI)
# ─────────────────────────────────────────────

def punto_fijo(expr_f: str, expr_g: str, x0: float, tolerancia: float, max_iter: int):
    """
    Ejecuta el método de Punto Fijo.

    Returns
    -------
    raiz        : float
    iteraciones : list[dict]  — cada dict: {i, x, fx, error}
    estado      : str         — "convergencia" | "max_iter"
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
#  SUGERENCIAS VÍA API ANTHROPIC
# ─────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

_SYSTEM_PROMPT = """Eres un asistente de métodos numéricos.
El usuario te da una función f(x) en sintaxis Python/numpy.
Tu tarea: proponer exactamente 3 formas distintas de despejar x = g(x) a partir de f(x) = 0,
de modo que el método de Punto Fijo pueda converger (idealmente |g'(x)| < 1 cerca de la raíz).

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código, sin backticks.
Formato exacto:
{
  "candidatas": [
    {"expr": "<expresión Python>", "descripcion": "<cómo se obtuvo>"},
    {"expr": "<expresión Python>", "descripcion": "<cómo se obtuvo>"},
    {"expr": "<expresión Python>", "descripcion": "<cómo se obtuvo>"}
  ]
}

Reglas para las expresiones:
- Usar sintaxis Python válida (** para potencia, np.exp, np.log, np.sqrt, np.sin, np.cos)
- La variable es siempre x
- No incluir "g(x) =" ni "x =", solo la expresión del lado derecho
- Variar los despejes: algebraico, con exponencial, con raíz, etc. según corresponda
"""


def _llamar_api(expr_f: str) -> list:
    """
    Llama a la API de Anthropic y devuelve lista de candidatas.
    Lanza excepción si hay error.
    """
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"f(x) = {expr_f}"}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw = "".join(
        blk.get("text", "") for blk in data.get("content", [])
        if blk.get("type") == "text"
    ).strip()

    # Limpiar posibles backticks residuales
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)
    return parsed["candidatas"]


# ─────────────────────────────────────────────
#  VENTANA DE SUGERENCIAS
# ─────────────────────────────────────────────

class _VentanaSugerencias(tk.Toplevel):
    """
    Popup que muestra las candidatas g(x) con su validación |g'(x)| < 1
    y permite seleccionar una para cargarla en el campo principal.
    """

    def __init__(self, master, candidatas, x0: float, callback):
        super().__init__(master)
        self.title("Sugerencias de g(x)")
        self.geometry("640x440")
        self.resizable(False, False)
        self.grab_set()  # modal
        self.configure(bg="#F3F0FF")

        self._callback = callback
        self._seleccion = tk.StringVar()

        # Título
        tk.Label(
            self, text="Candidatas g(x) sugeridas por IA",
            font=("Segoe UI", 12, "bold"), bg="#F3F0FF", fg="#1A237E"
        ).pack(pady=(14, 4), padx=16, anchor="w")

        tk.Label(
            self,
            text=f"Validación numérica de |g'(x)| evaluada en entorno de x₀ = {x0:.4g}",
            font=("Segoe UI", 9), bg="#F3F0FF", fg="#555"
        ).pack(padx=16, anchor="w")

        # Frame con las opciones
        frame_opts = tk.Frame(self, bg="#F3F0FF")
        frame_opts.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

        self._poblar(frame_opts, candidatas, x0)

        # Botones
        frame_btn = tk.Frame(self, bg="#F3F0FF")
        frame_btn.pack(fill=tk.X, padx=16, pady=(0, 14))

        tk.Button(
            frame_btn, text="✔  Usar seleccionada",
            bg="#4527A0", fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, padx=10, pady=6,
            command=self._usar
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            frame_btn, text="Cancelar",
            bg="#757575", fg="white",
            font=("Segoe UI", 10),
            relief=tk.FLAT, padx=10, pady=6,
            command=self.destroy
        ).pack(side=tk.LEFT)

    def _poblar(self, parent, candidatas, x0):
        for c in candidatas:
            expr = c.get("expr", "")
            desc = c.get("descripcion", "")

            # Validar convergencia numéricamente
            try:
                valida, max_d, x_max = validar_convergencia(expr, x0)
                if valida:
                    badge_txt   = f"✔  |g'(x)| ≤ {max_d:.3f}  →  CONVERGE"
                    badge_color = "#1B5E20"
                    badge_bg    = "#E8F5E9"
                else:
                    badge_txt   = f"✘  |g'(x)| = {max_d:.3f} en x ≈ {x_max:.3f}  →  PUEDE DIVERGIR"
                    badge_color = "#B71C1C"
                    badge_bg    = "#FFEBEE"
            except Exception:
                badge_txt   = "⚠  No se pudo validar"
                badge_color = "#E65100"
                badge_bg    = "#FFF3E0"

            # Card de la candidata
            card = tk.Frame(parent, bg="white", relief=tk.RIDGE, bd=1)
            card.pack(fill=tk.X, pady=5)

            row_top = tk.Frame(card, bg="white")
            row_top.pack(fill=tk.X, padx=10, pady=(8, 2))

            tk.Radiobutton(
                row_top, variable=self._seleccion, value=expr,
                bg="white", activebackground="white"
            ).pack(side=tk.LEFT)

            tk.Label(
                row_top, text=f"g(x) = {expr}",
                font=("Consolas", 10, "bold"), bg="white", fg="#1A237E"
            ).pack(side=tk.LEFT, padx=4)

            tk.Label(
                card, text=badge_txt,
                font=("Segoe UI", 9), bg=badge_bg, fg=badge_color,
                padx=8, pady=2
            ).pack(anchor="w", padx=10, pady=(0, 2))

            tk.Label(
                card, text=desc,
                font=("Segoe UI", 9), bg="white", fg="#555",
                wraplength=580, justify=tk.LEFT
            ).pack(anchor="w", padx=10, pady=(0, 8))

    def _usar(self):
        sel = self._seleccion.get()
        if not sel:
            messagebox.showwarning("Aviso", "Seleccioná una opción primero.", parent=self)
            return
        self._callback(sel)
        self.destroy()


# ─────────────────────────────────────────────
#  CLASE GUI PRINCIPAL
# ─────────────────────────────────────────────

class PuntoFijoFrame(tk.Frame):
    """
    Panel completo del Método de Punto Fijo.
    Se puede embeber en cualquier ventana Tkinter.
    """

    FONT_MONO         = ("Consolas", 9)
    FONT_LABEL        = ("Segoe UI", 10)
    COLOR_BTN_CALC    = "#4527A0"
    COLOR_BTN_GRAPH   = "#00695C"
    COLOR_BTN_SUGGEST = "#E65100"

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

        # f(x) + botón sugerir en la misma fila
        self._lbl(parent, "Función original f(x):", pady_top=0)
        frame_f = tk.Frame(parent, bg="#F3F0FF")
        frame_f.pack(fill=tk.X, pady=2)

        self.e_f = tk.Entry(frame_f, width=23, font=self.FONT_LABEL)
        self.e_f.insert(0, "2**(-x) - x")
        self.e_f.pack(side=tk.LEFT)

        self._btn_sugerir = tk.Button(
            frame_f, text="✨ Sugerir g(x)",
            bg=self.COLOR_BTN_SUGGEST, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, padx=8, pady=3,
            command=self._sugerir_g
        )
        self._btn_sugerir.pack(side=tk.LEFT, padx=(6, 0))

        # g(x)
        self._lbl(parent, "Función iterativa g(x)  [x = g(x)]:")
        self.e_g = self._entry(parent, "2**(-x)")

        # Indicador de validez en tiempo real
        self._lbl_validez = tk.Label(
            parent, text="", bg="#F3F0FF",
            font=("Segoe UI", 8, "italic"), anchor="w"
        )
        self._lbl_validez.pack(anchor="w")
        self.e_g.bind("<FocusOut>", lambda e: self._actualizar_validez())
        self.e_g.bind("<Return>",   lambda e: self._actualizar_validez())

        self._lbl(parent, "Punto inicial x₀:")
        self.e_x0 = self._entry(parent, "2/3")
        self.e_x0.bind("<FocusOut>", lambda e: self._actualizar_validez())

        self._lbl(parent, "Tolerancia (error):")
        self.e_tol = self._entry(parent, "1e-6")

        self._lbl(parent, "Máx. iteraciones:")
        self.e_iter = self._entry(parent, "100")

        # Botones principales
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

    # ── Validez en tiempo real ───────────────────────────────────────────────

    def _actualizar_validez(self):
        expr_g = self.e_g.get().strip()
        if not expr_g:
            self._lbl_validez.config(text="")
            return
        try:
            x0 = _evaluar_escalar(self.e_x0.get())
            valida, max_d, _ = validar_convergencia(expr_g, x0)
            if valida:
                self._lbl_validez.config(
                    text=f"✔ |g'(x)| ≤ {max_d:.3f} cerca de x₀  → converge",
                    fg="#1B5E20"
                )
            else:
                self._lbl_validez.config(
                    text=f"✘ |g'(x)| = {max_d:.3f} cerca de x₀  → puede divergir",
                    fg="#B71C1C"
                )
        except Exception:
            self._lbl_validez.config(text="⚠ No se pudo validar g(x)", fg="#E65100")

    # ── Sugerencias IA ───────────────────────────────────────────────────────

    def _sugerir_g(self):
        expr_f = self.e_f.get().strip()
        if not expr_f:
            messagebox.showwarning("Aviso", "Ingresá primero la función f(x).")
            return

        self._btn_sugerir.config(state=tk.DISABLED, text="⏳ Consultando IA...")

        def _tarea():
            try:
                candidatas = _llamar_api(expr_f)
                self.after(0, lambda: self._mostrar_sugerencias(candidatas))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror(
                    "Error de API",
                    f"No se pudieron obtener sugerencias.\n\n{exc}"
                ))
            finally:
                self.after(0, lambda: self._btn_sugerir.config(
                    state=tk.NORMAL, text="✨ Sugerir g(x)"
                ))

        threading.Thread(target=_tarea, daemon=True).start()

    def _mostrar_sugerencias(self, candidatas):
        try:
            x0 = _evaluar_escalar(self.e_x0.get())
        except Exception:
            x0 = 0.0

        _VentanaSugerencias(
            self, candidatas, x0,
            callback=self._cargar_g_seleccionada
        )

    def _cargar_g_seleccionada(self, expr_g: str):
        self.e_g.delete(0, tk.END)
        self.e_g.insert(0, expr_g)
        self._actualizar_validez()

    # ── Lógica de cálculo ────────────────────────────────────────────────────

    def _log(self, texto):
        self.txt.insert(tk.END, texto + "\n")
        self.txt.see(tk.END)

    def _calcular(self):
        self.txt.delete(1.0, tk.END)
        try:
            expr_f   = self.e_f.get().strip()
            expr_g   = self.e_g.get().strip()
            x0       = _evaluar_escalar(self.e_x0.get())
            tol      = _evaluar_escalar(self.e_tol.get())
            max_iter = int(self.e_iter.get())

            # Advertencia si g(x) puede divergir
            valida, max_d, _ = validar_convergencia(expr_g, x0)
            if not valida:
                continuar = messagebox.askyesno(
                    "Advertencia de convergencia",
                    f"|g'(x)| ≈ {max_d:.3f} cerca de x₀.\n"
                    "El método puede divergir.\n\n¿Querés continuar de todas formas?"
                )
                if not continuar:
                    return

            raiz, historial, estado = punto_fijo(expr_f, expr_g, x0, tol, max_iter)
            self._raiz = raiz
            self._historial = historial

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

    # ── Gráfico ──────────────────────────────────────────────────────────────

    def _graficar(self):
        try:
            expr_f = self.e_f.get().strip()
            expr_g = self.e_g.get().strip()
            x0     = _evaluar_escalar(self.e_x0.get())
        except Exception as exc:
            messagebox.showerror("Error", f"Valores inválidos para graficar.\n{exc}")
            return

        self.ax.clear()

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
            self.ax.scatter(
                [self._raiz], [self._raiz], color="#FF6F00",
                zorder=6, s=90, label=f"Punto fijo ≈ {self._raiz:.6f}"
            )
            self.ax.scatter([self._raiz], [0], color="#FF6F00", zorder=6, s=90, marker="X")
            self.ax.annotate(
                f"  Raíz ≈ {self._raiz:.6f}",
                xy=(self._raiz, 0), fontsize=9, color="#FF6F00", va="bottom"
            )

        all_finite = np.concatenate([y_f[np.isfinite(y_f)], y_g[np.isfinite(y_g)]])
        if len(all_finite):
            ymin = np.percentile(all_finite, 2)
            ymax = np.percentile(all_finite, 98)
            rng  = ymax - ymin or 1
            self.ax.set_ylim(ymin - rng * 0.15, ymax + rng * 0.15)

        self.ax.set_title("Método de Punto Fijo", fontsize=13, fontweight="bold")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend(fontsize=9, loc="best")
        self.ax.grid(True, linestyle=":", alpha=0.55)
        self.fig.tight_layout()
        self.canvas.draw()
