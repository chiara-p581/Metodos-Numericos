"""
Aplicación: Buscador de Raíces Numéricas
=========================================
Módulos disponibles:
  - Bisección    (biseccion.py)
  - Punto Fijo   (punto_fijo.py)
  - Aitken       (aitken.py)
 
Uso:
    python main.py
"""
 
import tkinter as tk
from tkinter import ttk
 
from biseccion import BiseccionFrame
from punto_fijo import PuntoFijoFrame
from aitken import AitkenFrame  
 
 
class RootFinderApp:
    """Ventana principal que aloja los métodos numéricos en pestañas."""
 
    APP_TITLE  = "Buscador de Raíces Numéricas"
    WIN_SIZE   = "1200x700"
    FONT_TAB   = ("Segoe UI", 11, "bold")
 
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.APP_TITLE)
        self.root.geometry(self.WIN_SIZE)
        self.root.resizable(True, True)
        self._apply_theme()
        self._build_ui()
 
    def _apply_theme(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TNotebook",
            background="#ECEFF1",
            borderwidth=0
        )
        style.configure(
            "TNotebook.Tab",
            font=self.FONT_TAB,
            padding=[18, 8],
            background="#CFD8DC",
            foreground="#37474F"
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1A237E")],
            foreground=[("selected", "white")]
        )
 
    def _build_ui(self):
        # Barra superior
        header = tk.Frame(self.root, bg="#1A237E", height=48)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
 
        tk.Label(
            header,
            text="⚙ Métodos Numéricos — Búsqueda de Raíces",
            font=("Segoe UI", 14, "bold"),
            bg="#1A237E", fg="white"
        ).pack(side=tk.LEFT, padx=18, pady=10)
 
        # Notebook (pestañas)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
 
        # Tabs
        tab_biseccion  = tk.Frame(notebook, bg="white")
        tab_punto_fijo = tk.Frame(notebook, bg="white")
        tab_aitken     = tk.Frame(notebook, bg="white")  # 👈 NUEVO
 
        notebook.add(tab_biseccion,  text="  Bisección  ")
        notebook.add(tab_punto_fijo, text="  Punto Fijo  ")
        notebook.add(tab_aitken,     text="  Aitken  ")   # 👈 NUEVO
 
        # Frames
        BiseccionFrame(tab_biseccion).pack(fill=tk.BOTH, expand=True)
        PuntoFijoFrame(tab_punto_fijo).pack(fill=tk.BOTH, expand=True)
        AitkenFrame(tab_aitken).pack(fill=tk.BOTH, expand=True)  # 👈 NUEVO
 
    def run(self):
        self.root.mainloop()
 
 
if __name__ == "__main__":
    app = RootFinderApp()
    app.run()
 