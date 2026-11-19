"""
A* Pathfinding Visualizer - Graphical User Interface.
Built with Python Tkinter and connected to astar.py.
"""

from __future__ import annotations
import math
import random
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Set, Tuple

from astar import Coordinate, Grid, astar, manhattan_distance


# --- Theme & Colors ---
BG_MAIN = "#1a1b26"         # Dark background
BG_PANEL = "#24283b"        # Card / Panel background
BG_SURFACE = "#2f3549"      # Elevated control surface
FG_PRIMARY = "#c0caf5"     # Primary text
FG_SECONDARY = "#7aa2f7"   # Accent text
FG_MUTED = "#565f89"       # Muted text
BORDER_COLOR = "#414868"   # Border color

# Cell Colors
COLOR_EMPTY = "#1f2335"
COLOR_GRID_LINE = "#292e42"
COLOR_START = "#9ece6a"       # Emerald green
COLOR_GOAL = "#f7768e"        # Coral red
COLOR_OBSTACLE = "#15161e"    # Dark obsidian
COLOR_WEIGHT = "#e0af68"      # Amber / Gold
COLOR_EXPLORED = "#7dcfff"    # Light cyan / blue
COLOR_PATH = "#bb9af7"        # Vivid purple / violet
COLOR_PATH_BORDER = "#ff9e64" # Accent border for path


class AStarGUI:
    """
    Tkinter Graphical User Interface for A* Path Finding.
    """

    def __init__(self, root: tk.Tk, rows: int = 15, cols: int = 15) -> None:
        self.root = root
        self.root.title("A* Pathfinding Visualizer")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(False, False)

        self.rows: int = rows
        self.cols: int = cols
        self.cell_size: int = 36

        # State
        self.start_node: Coordinate = (2, 2)
        self.goal_node: Coordinate = (12, 12)
        self.obstacles: Set[Coordinate] = set()
        self.weights: Dict[Coordinate, float] = {}
        self.current_weight_val: float = 5.0

        # Tool Mode: "start", "goal", "obstacle", "erase", "weight"
        self.mode: str = "obstacle"

        # Search & Animation state
        self.is_animating: bool = False
        self.animation_job: Optional[str] = None
        self.path_cells: List[Coordinate] = []
        self.explored_cells: List[Coordinate] = []

        # Build UI layout
        self._setup_styles()
        self._build_layout()
        self._bind_mouse_events()
        self.redraw_grid()

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "TFrame",
            background=BG_MAIN,
        )
        style.configure(
            "Card.TFrame",
            background=BG_PANEL,
            relief="flat",
        )
        style.configure(
            "TLabel",
            background=BG_PANEL,
            foreground=FG_PRIMARY,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel",
            background=BG_PANEL,
            foreground=FG_SECONDARY,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Title.TLabel",
            background=BG_MAIN,
            foreground="#7aa2f7",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=BG_MAIN,
            foreground=FG_MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background=BG_PANEL,
            foreground="#9ece6a",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Value.TLabel",
            background=BG_PANEL,
            foreground="#bb9af7",
            font=("Segoe UI", 10, "bold"),
        )

    def _build_layout(self) -> None:
        # Title bar
        title_frame = tk.Frame(self.root, bg=BG_MAIN, padx=16, pady=10)
        title_frame.pack(fill=tk.X)

        tk.Label(
            title_frame,
            text="Optimal Path Finding Using A* Search Algorithm",
            bg=BG_MAIN,
            fg="#7aa2f7",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Interactive 15×15 Grid • Manhattan Distance Heuristic • 4-Direction Movement • Weighted Cells",
            bg=BG_MAIN,
            fg="#7982a9",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        # Main container: Canvas on left, Controls on right
        content_frame = tk.Frame(self.root, bg=BG_MAIN, padx=14, pady=6)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Canvas
        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size

        canvas_border = tk.Frame(content_frame, bg=BORDER_COLOR, padx=2, pady=2)
        canvas_border.pack(side=tk.LEFT, padx=(0, 16), pady=4)

        self.canvas = tk.Canvas(
            canvas_border,
            width=canvas_width,
            height=canvas_height,
            bg=COLOR_EMPTY,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack()

        # Right: Control Panel (Scrollable if needed, structured with cards)
        control_panel = tk.Frame(content_frame, bg=BG_MAIN, width=380)
        control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Tools Card
        self._build_tools_card(control_panel)

        # 2. Actions Card
        self._build_actions_card(control_panel)

        # 3. Test Scenarios Card
        self._build_scenarios_card(control_panel)

        # 4. Metrics Card
        self._build_metrics_card(control_panel)

        # 5. Legend Card
        self._build_legend_card(control_panel)

    def _build_tools_card(self, parent: tk.Widget) -> None:
        card = tk.LabelFrame(
            parent,
            text=" Editor Tools ",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            relief="groove",
            bd=1,
            padx=10,
            pady=8,
        )
        card.pack(fill=tk.X, pady=(0, 8))

        tools_frame = tk.Frame(card, bg=BG_PANEL)
        tools_frame.pack(fill=tk.X)

        self.tool_buttons: Dict[str, tk.Button] = {}
        tool_defs = [
            ("start", "🟢 Start", "#22c55e"),
            ("goal", "🔴 Goal", "#ef4444"),
            ("obstacle", "⬛ Obstacle", "#475569"),
            ("weight", "🔶 Weight", "#d97706"),
            ("erase", "🧹 Erase", "#64748b"),
        ]

        for mode_key, text, active_col in tool_defs:
            btn = tk.Button(
                tools_frame,
                text=text,
                font=("Segoe UI", 9, "bold"),
                bg=BG_SURFACE if self.mode != mode_key else active_col,
                fg="white",
                activebackground=active_col,
                activeforeground="white",
                relief="flat",
                bd=0,
                padx=8,
                pady=4,
                cursor="hand2",
                command=lambda m=mode_key: self.set_mode(m),
            )
            btn.pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
            self.tool_buttons[mode_key] = btn

        # Weight slider row
        weight_row = tk.Frame(card, bg=BG_PANEL, pady=6)
        weight_row.pack(fill=tk.X)

        tk.Label(weight_row, text="Weight Cost:", bg=BG_PANEL, fg=FG_PRIMARY, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.weight_label = tk.Label(weight_row, text=f"{int(self.current_weight_val)}", bg=BG_PANEL, fg="#e0af68", font=("Segoe UI", 9, "bold"), width=3)
        self.weight_label.pack(side=tk.LEFT, padx=4)

        self.weight_slider = tk.Scale(
            weight_row,
            from_=2,
            to=20,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=BG_PANEL,
            troughcolor=BG_SURFACE,
            highlightthickness=0,
            command=self._on_weight_slider_change,
        )
        self.weight_slider.set(int(self.current_weight_val))
        self.weight_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    def _build_actions_card(self, parent: tk.Widget) -> None:
        card = tk.LabelFrame(
            parent,
            text=" Actions ",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            relief="groove",
            bd=1,
            padx=10,
            pady=8,
        )
        card.pack(fill=tk.X, pady=(0, 8))

        # Top row: Run A* (Green accent) and Random Maze
        row1 = tk.Frame(card, bg=BG_PANEL)
        row1.pack(fill=tk.X, pady=(0, 4))

        self.btn_run = tk.Button(
            row1,
            text="▶ Run A* Search",
            font=("Segoe UI", 10, "bold"),
            bg="#22c55e",
            fg="white",
            activebackground="#16a34a",
            relief="flat",
            bd=0,
            pady=5,
            cursor="hand2",
            command=self.run_astar_search,
        )
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_maze = tk.Button(
            row1,
            text="🎲 Random Maze",
            font=("Segoe UI", 9, "bold"),
            bg="#6366f1",
            fg="white",
            activebackground="#4f46e5",
            relief="flat",
            bd=0,
            pady=5,
            cursor="hand2",
            command=self.generate_random_maze,
        )
        self.btn_maze.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Second row: Clear Path and Reset Grid
        row2 = tk.Frame(card, bg=BG_PANEL)
        row2.pack(fill=tk.X, pady=(4, 0))

        self.btn_clear_path = tk.Button(
            row2,
            text="🧹 Clear Path",
            font=("Segoe UI", 9),
            bg=BG_SURFACE,
            fg=FG_PRIMARY,
            activebackground=BORDER_COLOR,
            relief="flat",
            bd=0,
            pady=4,
            cursor="hand2",
            command=self.clear_search_path,
        )
        self.btn_clear_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_reset = tk.Button(
            row2,
            text="🔄 Reset Grid",
            font=("Segoe UI", 9),
            bg=BG_SURFACE,
            fg=FG_PRIMARY,
            activebackground=BORDER_COLOR,
            relief="flat",
            bd=0,
            pady=4,
            cursor="hand2",
            command=self.reset_entire_grid,
        )
        self.btn_reset.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Animation Speed row
        speed_row = tk.Frame(card, bg=BG_PANEL, pady=4)
        speed_row.pack(fill=tk.X)

        tk.Label(speed_row, text="Speed:", bg=BG_PANEL, fg=FG_PRIMARY, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="Normal")
        for sp in ["Slow", "Normal", "Fast", "Instant"]:
            rb = tk.Radiobutton(
                speed_row,
                text=sp,
                value=sp,
                variable=self.speed_var,
                bg=BG_PANEL,
                fg=FG_PRIMARY,
                selectcolor=BG_SURFACE,
                activebackground=BG_PANEL,
                font=("Segoe UI", 8),
            )
            rb.pack(side=tk.LEFT, padx=2)

    def _build_scenarios_card(self, parent: tk.Widget) -> None:
        card = tk.LabelFrame(
            parent,
            text=" Test Scenarios ",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            relief="groove",
            bd=1,
            padx=10,
            pady=6,
        )
        card.pack(fill=tk.X, pady=(0, 8))

        desc = tk.Label(
            card,
            text="Load preset benchmark scenarios:",
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=("Segoe UI", 8),
        )
        desc.pack(anchor="w", pady=(0, 4))

        btn_row1 = tk.Frame(card, bg=BG_PANEL)
        btn_row1.pack(fill=tk.X, pady=2)

        tk.Button(
            btn_row1,
            text="Cheap Detour vs Expensive",
            font=("Segoe UI", 8),
            bg=BG_SURFACE,
            fg=FG_PRIMARY,
            relief="flat",
            cursor="hand2",
            command=self.load_expensive_vs_cheap_scenario,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Button(
            btn_row1,
            text="Equal-Cost Alternative",
            font=("Segoe UI", 8),
            bg=BG_SURFACE,
            fg=FG_PRIMARY,
            relief="flat",
            cursor="hand2",
            command=self.load_equal_cost_scenario,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        btn_row2 = tk.Frame(card, bg=BG_PANEL)
        btn_row2.pack(fill=tk.X, pady=2)

        tk.Button(
            btn_row2,
            text="Unreachable Goal",
            font=("Segoe UI", 8),
            bg=BG_SURFACE,
            fg=FG_PRIMARY,
            relief="flat",
            cursor="hand2",
            command=self.load_unreachable_scenario,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Button(
            btn_row2,
            text="Start = Goal",
            font=("Segoe UI", 8),
            bg=BG_SURFACE,
            fg=FG_PRIMARY,
            relief="flat",
            cursor="hand2",
            command=self.load_start_equals_goal_scenario,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def _build_metrics_card(self, parent: tk.Widget) -> None:
        card = tk.LabelFrame(
            parent,
            text=" Execution Metrics ",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            relief="groove",
            bd=1,
            padx=10,
            pady=6,
        )
        card.pack(fill=tk.X, pady=(0, 8))

        grid_frame = tk.Frame(card, bg=BG_PANEL)
        grid_frame.pack(fill=tk.X)

        # Status
        tk.Label(grid_frame, text="Status:", bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_status = tk.Label(grid_frame, text="Ready", bg=BG_PANEL, fg="#9ece6a", font=("Segoe UI", 9, "bold"))
        self.lbl_status.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        # Path Cost
        tk.Label(grid_frame, text="Total Cost:", bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_cost = tk.Label(grid_frame, text="--", bg=BG_PANEL, fg="#bb9af7", font=("Segoe UI", 9, "bold"))
        self.lbl_cost.grid(row=1, column=1, sticky="w", padx=6, pady=2)

        # Path Steps
        tk.Label(grid_frame, text="Path Steps:", bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w", padx=(12, 0), pady=2)
        self.lbl_steps = tk.Label(grid_frame, text="--", bg=BG_PANEL, fg="#bb9af7", font=("Segoe UI", 9, "bold"))
        self.lbl_steps.grid(row=0, column=3, sticky="w", padx=6, pady=2)

        # Explored Nodes
        tk.Label(grid_frame, text="Explored Nodes:", bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9)).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=2)
        self.lbl_explored = tk.Label(grid_frame, text="--", bg=BG_PANEL, fg="#7dcfff", font=("Segoe UI", 9, "bold"))
        self.lbl_explored.grid(row=1, column=3, sticky="w", padx=6, pady=2)

        # Execution Time
        tk.Label(grid_frame, text="Execution Time:", bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", pady=2)
        self.lbl_time = tk.Label(grid_frame, text="--", bg=BG_PANEL, fg="#ff9e64", font=("Segoe UI", 9, "bold"))
        self.lbl_time.grid(row=2, column=1, columnspan=3, sticky="w", padx=6, pady=2)

    def _build_legend_card(self, parent: tk.Widget) -> None:
        card = tk.LabelFrame(
            parent,
            text=" Color Legend ",
            bg=BG_PANEL,
            fg=FG_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            relief="groove",
            bd=1,
            padx=10,
            pady=6,
        )
        card.pack(fill=tk.X)

        items = [
            (COLOR_START, "Start Node"),
            (COLOR_GOAL, "Goal Node"),
            (COLOR_OBSTACLE, "Obstacle / Wall"),
            (COLOR_WEIGHT, "Weighted Cell"),
            (COLOR_EXPLORED, "Explored Node"),
            (COLOR_PATH, "Optimal Path"),
        ]

        legend_grid = tk.Frame(card, bg=BG_PANEL)
        legend_grid.pack(fill=tk.X)

        for idx, (col, label) in enumerate(items):
            r = idx // 2
            c = (idx % 2) * 2

            box = tk.Canvas(legend_grid, width=14, height=14, bg=col, highlightthickness=1, highlightbackground=BORDER_COLOR)
            box.grid(row=r, column=c, padx=(0, 4), pady=2)

            lbl = tk.Label(legend_grid, text=label, bg=BG_PANEL, fg=FG_PRIMARY, font=("Segoe UI", 8))
            lbl.grid(row=r, column=c + 1, sticky="w", padx=(0, 10), pady=2)

    # --- Mouse Event Handlers ---
    def _bind_mouse_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<B3-Motion>", self._on_canvas_right_click)

    def _get_cell_at_coords(self, event_x: int, event_y: int) -> Optional[Coordinate]:
        c = event_x // self.cell_size
        r = event_y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def _on_canvas_click(self, event) -> None:
        if self.is_animating:
            return
        cell = self._get_cell_at_coords(event.x, event.y)
        if cell:
            self._apply_tool_to_cell(cell)

    def _on_canvas_drag(self, event) -> None:
        if self.is_animating:
            return
        cell = self._get_cell_at_coords(event.x, event.y)
        if cell and self.mode in ("obstacle", "erase", "weight"):
            self._apply_tool_to_cell(cell)

    def _on_canvas_right_click(self, event) -> None:
        if self.is_animating:
            return
        cell = self._get_cell_at_coords(event.x, event.y)
        if cell:
            self._erase_cell(cell)

    def _apply_tool_to_cell(self, cell: Coordinate) -> None:
        if self.mode == "start":
            if cell != self.goal_node:
                self.obstacles.discard(cell)
                self.weights.pop(cell, None)
                self.start_node = cell
                self.clear_search_path()
        elif self.mode == "goal":
            if cell != self.start_node:
                self.obstacles.discard(cell)
                self.weights.pop(cell, None)
                self.goal_node = cell
                self.clear_search_path()
        elif self.mode == "obstacle":
            if cell != self.start_node and cell != self.goal_node:
                self.weights.pop(cell, None)
                self.obstacles.add(cell)
                self.clear_search_path()
        elif self.mode == "weight":
            if cell != self.start_node and cell != self.goal_node:
                self.obstacles.discard(cell)
                self.weights[cell] = float(self.current_weight_val)
                self.clear_search_path()
        elif self.mode == "erase":
            self._erase_cell(cell)

    def _erase_cell(self, cell: Coordinate) -> None:
        self.obstacles.discard(cell)
        self.weights.pop(cell, None)
        self.clear_search_path()

    def _on_weight_slider_change(self, val: str) -> None:
        self.current_weight_val = float(val)
        self.weight_label.config(text=f"{int(self.current_weight_val)}")

    def set_mode(self, mode_key: str) -> None:
        self.mode = mode_key
        for m, btn in self.tool_buttons.items():
            if m == mode_key:
                active_colors = {
                    "start": "#22c55e",
                    "goal": "#ef4444",
                    "obstacle": "#475569",
                    "weight": "#d97706",
                    "erase": "#64748b",
                }
                btn.config(bg=active_colors.get(m, BG_SURFACE))
            else:
                btn.config(bg=BG_SURFACE)

    # --- Drawing Logic ---
    def redraw_grid(self) -> None:
        self.canvas.delete("all")
        cs = self.cell_size

        # Draw grid cells
        for r in range(self.rows):
            for c in range(self.cols):
                coord = (r, c)
                x1 = c * cs
                y1 = r * cs
                x2 = x1 + cs
                y2 = y1 + cs

                # Determine cell color & label
                fill_color = COLOR_EMPTY
                text_label = ""
                text_color = "white"

                if coord == self.start_node:
                    fill_color = COLOR_START
                    text_label = "S"
                    text_color = "#15161e"
                elif coord == self.goal_node:
                    fill_color = COLOR_GOAL
                    text_label = "G"
                    text_color = "#ffffff"
                elif coord in self.obstacles:
                    fill_color = COLOR_OBSTACLE
                elif coord in self.path_cells:
                    fill_color = COLOR_PATH
                    if coord in self.weights:
                        text_label = str(int(self.weights[coord]))
                        text_color = "#ffffff"
                elif coord in self.explored_cells:
                    fill_color = COLOR_EXPLORED
                    if coord in self.weights:
                        text_label = str(int(self.weights[coord]))
                        text_color = "#15161e"
                elif coord in self.weights:
                    fill_color = COLOR_WEIGHT
                    text_label = str(int(self.weights[coord]))
                    text_color = "#15161e"

                # Draw rectangle
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill_color,
                    outline=COLOR_GRID_LINE,
                    width=1,
                    tags=f"cell_{r}_{c}",
                )

                # Draw text badge if any
                if text_label:
                    font_spec = ("Segoe UI", 10, "bold") if coord in (self.start_node, self.goal_node) else ("Segoe UI", 8, "bold")
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=text_label,
                        fill=text_color,
                        font=font_spec,
                        tags=f"text_{r}_{c}",
                    )

    def _draw_single_cell(self, coord: Coordinate, fill_color: str, text_label: str = "", text_color: str = "white") -> None:
        r, c = coord
        cs = self.cell_size
        x1 = c * cs
        y1 = r * cs
        x2 = x1 + cs
        y2 = y1 + cs

        self.canvas.delete(f"cell_{r}_{c}")
        self.canvas.delete(f"text_{r}_{c}")

        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=fill_color,
            outline=COLOR_GRID_LINE,
            width=1,
            tags=f"cell_{r}_{c}",
        )
        if text_label:
            font_spec = ("Segoe UI", 10, "bold") if coord in (self.start_node, self.goal_node) else ("Segoe UI", 8, "bold")
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=text_label,
                fill=text_color,
                font=font_spec,
                tags=f"text_{r}_{c}",
            )

    # --- Actions ---
    def clear_search_path(self) -> None:
        """Clear only the visualized path and explored nodes."""
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        self.is_animating = False
        self.path_cells = []
        self.explored_cells = []
        self.redraw_grid()
        self.lbl_status.config(text="Ready", fg="#9ece6a")
        self.lbl_cost.config(text="--")
        self.lbl_steps.config(text="--")
        self.lbl_explored.config(text="--")
        self.lbl_time.config(text="--")

    def reset_entire_grid(self) -> None:
        """Reset all obstacles, weights, path, and reset start and goal."""
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        self.is_animating = False
        self.obstacles.clear()
        self.weights.clear()
        self.start_node = (2, 2)
        self.goal_node = (12, 12)
        self.clear_search_path()

    def generate_random_maze(self) -> None:
        """Generate a random obstacle layout preserving start and goal."""
        self.clear_search_path()
        self.obstacles.clear()
        self.weights.clear()

        # Place obstacles with ~28% density
        for r in range(self.rows):
            for c in range(self.cols):
                coord = (r, c)
                if coord != self.start_node and coord != self.goal_node:
                    if random.random() < 0.28:
                        self.obstacles.add(coord)

        # Ensure immediate neighbors of start and goal have at least one passage
        for endpoint in (self.start_node, self.goal_node):
            r, c = endpoint
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if random.random() < 0.5:
                        self.obstacles.discard((nr, nc))

        self.redraw_grid()
        self.lbl_status.config(text="Random Maze Generated", fg="#6366f1")

    # --- Preset Scenarios for Verification ---
    def load_expensive_vs_cheap_scenario(self) -> None:
        """
        Preset: Expensive Direct Path vs Cheap Indirect Detour.
        Start at (7, 2), Goal at (7, 12).
        Direct path has high weight (15), while detour along row 5 is cost 1.
        Configures grid only without auto-executing search.
        """
        self.clear_search_path()
        self.obstacles.clear()
        self.weights.clear()

        self.start_node = (7, 2)
        self.goal_node = (7, 12)

        # Place expensive weights along the direct path on row 7
        for c in range(4, 11):
            self.weights[(7, c)] = 15.0

        # Place wall barriers forcing the detour to row 5 or 9
        for c in range(4, 11):
            self.obstacles.add((6, c))
            self.obstacles.add((8, c))

        # Open up entry and exit for the detour at row 5
        self.obstacles.discard((6, 4))
        self.obstacles.discard((6, 10))

        self.redraw_grid()
        self.lbl_status.config(text="Scenario Loaded — Ready to Search", fg="#7aa2f7")

    def load_equal_cost_scenario(self) -> None:
        """
        Preset: Symmetrical grid showcasing deterministic Manhattan 4-direction choice.
        Configures grid only without auto-executing search.
        """
        self.clear_search_path()
        self.obstacles.clear()
        self.weights.clear()

        self.start_node = (4, 4)
        self.goal_node = (10, 10)

        # Central obstacle forcing equal top/right or bottom/left routes
        for r in range(6, 9):
            for c in range(6, 9):
                self.obstacles.add((r, c))

        self.redraw_grid()
        self.lbl_status.config(text="Scenario Loaded — Ready to Search", fg="#7aa2f7")

    def load_unreachable_scenario(self) -> None:
        """
        Preset: Goal completely enclosed by a wall.
        Configures grid only without auto-executing search.
        """
        self.clear_search_path()
        self.obstacles.clear()
        self.weights.clear()

        self.start_node = (2, 2)
        self.goal_node = (10, 10)

        # Enclose goal with a wall box
        gr, gc = self.goal_node
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr != 0 or dc != 0:
                    self.obstacles.add((gr + dr, gc + dc))

        self.redraw_grid()
        self.lbl_status.config(text="Scenario Loaded — Ready to Search", fg="#7aa2f7")

    def load_start_equals_goal_scenario(self) -> None:
        """
        Preset: Start is set to the same node as Goal.
        Configures grid only without auto-executing search.
        """
        self.clear_search_path()
        self.obstacles.clear()
        self.weights.clear()

        self.start_node = (7, 7)
        self.goal_node = (7, 7)

        self.redraw_grid()
        self.lbl_status.config(text="Scenario Loaded — Ready to Search", fg="#7aa2f7")

    # --- Run A* Algorithm & Animation ---
    def run_astar_search(self) -> None:
        """
        Execute actual astar algorithm from astar.py and animate the results.
        """
        if self.is_animating:
            return

        # Clear any existing visual path
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        self.path_cells = []
        self.explored_cells = []
        self.redraw_grid()

        # Build Grid object matching current UI state
        grid = Grid(
            rows=self.rows,
            cols=self.cols,
            default_cost=1.0,
            obstacles=self.obstacles,
            weights=self.weights,
        )

        # Execute search and measure real execution time
        start_time = time.perf_counter()
        result = astar(
            grid=grid,
            start=self.start_node,
            goal=self.goal_node,
            heuristic=manhattan_distance,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics card
        self.lbl_time.config(text=f"{elapsed_ms:.2f} ms")
        self.lbl_explored.config(text=f"{result.nodes_explored}")

        speed = self.speed_var.get()
        delay_map = {
            "Slow": 40,
            "Normal": 15,
            "Fast": 5,
            "Instant": 0,
        }
        delay = delay_map.get(speed, 15)

        # Handle Start == Goal
        if self.start_node == self.goal_node:
            self.lbl_status.config(text="Start equals Goal!", fg="#9ece6a")
            self.lbl_cost.config(text="0.0")
            self.lbl_steps.config(text="1")
            self.path_cells = [self.start_node]
            self.redraw_grid()
            return

        # Handle Unreachable Goal
        if result.path is None:
            self.lbl_status.config(text="Goal Unreachable!", fg="#f7768e")
            self.lbl_cost.config(text="∞ (No path)")
            self.lbl_steps.config(text="0")

            # Animate whatever nodes were explored before exhaustion
            explored_to_animate = [
                n for n in result.explored_order
                if n != self.start_node and n != self.goal_node
            ]
            if delay == 0:
                self.explored_cells = explored_to_animate
                self.redraw_grid()
            else:
                self._animate_explored(explored_to_animate, [], delay, 0)
            return

        # Path Found
        self.lbl_status.config(text="Optimal Path Found!", fg="#9ece6a")
        self.lbl_cost.config(text=f"{result.cost:.1f}")
        self.lbl_steps.config(text=f"{len(result.path)}")

        explored_to_animate = [
            n for n in result.explored_order
            if n != self.start_node and n != self.goal_node
        ]
        path_to_animate = [
            n for n in result.path
            if n != self.start_node and n != self.goal_node
        ]

        if delay == 0:
            self.explored_cells = explored_to_animate
            self.path_cells = result.path
            self.redraw_grid()
        else:
            self.is_animating = True
            self._set_controls_state(tk.DISABLED)
            self._animate_explored(explored_to_animate, path_to_animate, delay, 0)

    def _set_controls_state(self, state) -> None:
        self.btn_run.config(state=state)
        self.btn_maze.config(state=state)
        self.btn_reset.config(state=state)
        self.btn_clear_path.config(state=state)

    def _animate_explored(
        self,
        explored_nodes: List[Coordinate],
        path_nodes: List[Coordinate],
        delay: int,
        index: int,
    ) -> None:
        if index < len(explored_nodes):
            node = explored_nodes[index]
            self.explored_cells.append(node)

            # Draw cell
            fill_col = COLOR_EXPLORED
            text_val = ""
            text_col = "white"
            if node in self.weights:
                text_val = str(int(self.weights[node]))
                text_col = "#15161e"
            self._draw_single_cell(node, fill_col, text_val, text_col)

            # Schedule next explored node
            self.animation_job = self.root.after(
                delay,
                lambda: self._animate_explored(explored_nodes, path_nodes, delay, index + 1),
            )
        else:
            # Explored animation finished, animate path
            self._animate_path(path_nodes, delay, 0)

    def _animate_path(
        self,
        path_nodes: List[Coordinate],
        delay: int,
        index: int,
    ) -> None:
        if index < len(path_nodes):
            node = path_nodes[index]
            self.path_cells.append(node)

            fill_col = COLOR_PATH
            text_val = ""
            text_col = "white"
            if node in self.weights:
                text_val = str(int(self.weights[node]))
            self._draw_single_cell(node, fill_col, text_val, text_col)

            # Schedule next path node (path animates slightly faster/smoother)
            path_delay = max(delay, 20)
            self.animation_job = self.root.after(
                path_delay,
                lambda: self._animate_path(path_nodes, delay, index + 1),
            )
        else:
            # Animation complete
            self.is_animating = False
            self.animation_job = None
            self._set_controls_state(tk.NORMAL)


def launch() -> None:
    root = tk.Tk()
    app = AStarGUI(root, rows=15, cols=15)
    root.mainloop()


if __name__ == "__main__":
    launch()

