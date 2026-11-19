"""
Automated testing for AStarGUI (gui.py).
Tests all UI components, buttons, and scenarios programmatically.
"""

import time
import tkinter as tk
import unittest
from astar import Coordinate, Grid, astar
from gui import AStarGUI


class TestAStarGUI(unittest.TestCase):
    """Test suite for AStarGUI interface and actions."""

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        # Hide the window during automated headless test run
        cls.root.withdraw()
        cls.app = AStarGUI(cls.root, rows=15, cols=15)
        # Use Instant speed for automated tests so callbacks execute immediately
        cls.app.speed_var.set("Instant")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.app.reset_entire_grid()
        self.app.speed_var.set("Instant")
        self.root.update()

    def test_grid_initial_state(self):
        self.assertEqual(self.app.rows, 15)
        self.assertEqual(self.app.cols, 15)
        self.assertEqual(self.app.start_node, (2, 2))
        self.assertEqual(self.app.goal_node, (12, 12))
        self.assertEqual(len(self.app.obstacles), 0)
        self.assertEqual(len(self.app.weights), 0)

    def test_tool_mode_selection(self):
        for mode in ["start", "goal", "obstacle", "weight", "erase"]:
            self.app.set_mode(mode)
            self.assertEqual(self.app.mode, mode)

    def test_draw_and_erase_obstacle(self):
        self.app.set_mode("obstacle")
        target_cell = (5, 5)
        self.app._apply_tool_to_cell(target_cell)
        self.assertIn(target_cell, self.app.obstacles)

        # Erase using erase tool
        self.app.set_mode("erase")
        self.app._apply_tool_to_cell(target_cell)
        self.assertNotIn(target_cell, self.app.obstacles)

    def test_draw_and_erase_weight(self):
        self.app.set_mode("weight")
        self.app.current_weight_val = 8.0
        target_cell = (4, 4)
        self.app._apply_tool_to_cell(target_cell)
        self.assertIn(target_cell, self.app.weights)
        self.assertEqual(self.app.weights[target_cell], 8.0)

        # Erase weight
        self.app.set_mode("erase")
        self.app._apply_tool_to_cell(target_cell)
        self.assertNotIn(target_cell, self.app.weights)

    def test_move_start_and_goal_nodes(self):
        self.app.set_mode("start")
        self.app._apply_tool_to_cell((1, 1))
        self.assertEqual(self.app.start_node, (1, 1))

        self.app.set_mode("goal")
        self.app._apply_tool_to_cell((13, 13))
        self.assertEqual(self.app.goal_node, (13, 13))

    def test_run_astar_default_grid(self):
        # Default start (2, 2) to goal (12, 12)
        self.app.run_astar_search()
        self.root.update()

        self.assertEqual(self.app.lbl_status.cget("text"), "Optimal Path Found!")
        # Manhattan distance from (2, 2) to (12, 12) is 10 + 10 = 20 steps
        self.assertEqual(self.app.lbl_cost.cget("text"), "20.0")
        self.assertEqual(self.app.lbl_steps.cget("text"), "21")  # 21 nodes (start + 20 steps)
        self.assertGreater(int(self.app.lbl_explored.cget("text")), 0)
        self.assertTrue(self.app.lbl_time.cget("text").endswith("ms"))

    def test_scenario_start_equals_goal(self):
        # 1. Loading scenario must only configure grid and NOT auto-execute search
        self.app.load_start_equals_goal_scenario()
        self.assertEqual(self.app.start_node, self.app.goal_node)
        self.assertEqual(self.app.lbl_status.cget("text"), "Scenario Loaded — Ready to Search")
        self.assertEqual(self.app.lbl_cost.cget("text"), "--")
        self.assertEqual(self.app.lbl_steps.cget("text"), "--")
        self.assertEqual(self.app.lbl_explored.cget("text"), "--")
        self.assertEqual(self.app.lbl_time.cget("text"), "--")
        self.assertEqual(len(self.app.path_cells), 0)
        self.assertEqual(len(self.app.explored_cells), 0)

        # 2. Search only executes upon explicit run action
        self.app.run_astar_search()
        self.root.update()

        self.assertEqual(self.app.lbl_status.cget("text"), "Start equals Goal!")
        self.assertEqual(self.app.lbl_cost.cget("text"), "0.0")
        self.assertEqual(self.app.lbl_steps.cget("text"), "1")
        self.assertEqual(self.app.path_cells, [self.app.start_node])

    def test_scenario_unreachable_goal(self):
        # 1. Loading scenario must only configure grid and NOT auto-execute search
        self.app.load_unreachable_scenario()
        self.assertEqual(self.app.lbl_status.cget("text"), "Scenario Loaded — Ready to Search")
        self.assertEqual(self.app.lbl_cost.cget("text"), "--")
        self.assertEqual(self.app.lbl_steps.cget("text"), "--")
        self.assertEqual(self.app.lbl_explored.cget("text"), "--")
        self.assertEqual(self.app.lbl_time.cget("text"), "--")
        self.assertEqual(len(self.app.path_cells), 0)
        self.assertEqual(len(self.app.explored_cells), 0)
        self.assertGreater(len(self.app.obstacles), 0)

        # 2. Search only executes upon explicit run action
        self.app.run_astar_search()
        self.root.update()

        self.assertEqual(self.app.lbl_status.cget("text"), "Goal Unreachable!")
        self.assertIn("No path", self.app.lbl_cost.cget("text"))
        self.assertEqual(self.app.lbl_steps.cget("text"), "0")

    def test_scenario_expensive_direct_vs_cheap_detour(self):
        # 1. Loading scenario must only configure grid and NOT auto-execute search
        self.app.load_expensive_vs_cheap_scenario()
        self.assertEqual(self.app.lbl_status.cget("text"), "Scenario Loaded — Ready to Search")
        self.assertEqual(self.app.lbl_cost.cget("text"), "--")
        self.assertEqual(self.app.lbl_steps.cget("text"), "--")
        self.assertEqual(self.app.lbl_explored.cget("text"), "--")
        self.assertEqual(self.app.lbl_time.cget("text"), "--")
        self.assertEqual(len(self.app.path_cells), 0)
        self.assertEqual(len(self.app.explored_cells), 0)
        self.assertGreater(len(self.app.weights), 0)

        # 2. Search only executes upon explicit run action
        self.app.run_astar_search()
        self.root.update()

        self.assertEqual(self.app.lbl_status.cget("text"), "Optimal Path Found!")
        cost = float(self.app.lbl_cost.cget("text"))
        # Detour takes row 5, avoiding row 7's weight 15 cells
        # Ensure the chosen path cost is significantly lower than direct path (>100 cost)
        self.assertLess(cost, 50.0)
        # Verify row 7 high-cost cells were avoided
        for c in range(4, 11):
            self.assertNotIn((7, c), self.app.path_cells)

    def test_scenario_equal_cost(self):
        # 1. Loading scenario must only configure grid and NOT auto-execute search
        self.app.load_equal_cost_scenario()
        self.assertEqual(self.app.lbl_status.cget("text"), "Scenario Loaded — Ready to Search")
        self.assertEqual(self.app.lbl_cost.cget("text"), "--")
        self.assertEqual(self.app.lbl_steps.cget("text"), "--")
        self.assertEqual(self.app.lbl_explored.cget("text"), "--")
        self.assertEqual(self.app.lbl_time.cget("text"), "--")
        self.assertEqual(len(self.app.path_cells), 0)
        self.assertEqual(len(self.app.explored_cells), 0)
        self.assertGreater(len(self.app.obstacles), 0)

        # 2. Search only executes upon explicit run action
        self.app.run_astar_search()
        self.root.update()

        self.assertEqual(self.app.lbl_status.cget("text"), "Optimal Path Found!")
        self.assertIsNotNone(self.app.path_cells)
        self.assertGreater(len(self.app.path_cells), 0)

    def test_all_scenarios_configure_without_auto_execution(self):
        """Explicitly test all 4 scenario presets to ensure none auto-executes."""
        scenarios = [
            ("Cheap vs Expensive", self.app.load_expensive_vs_cheap_scenario),
            ("Equal-Cost", self.app.load_equal_cost_scenario),
            ("Unreachable Goal", self.app.load_unreachable_scenario),
            ("Start = Goal", self.app.load_start_equals_goal_scenario),
        ]

        for name, load_fn in scenarios:
            with self.subTest(scenario=name):
                # Ensure a clean slate first
                self.app.reset_entire_grid()
                self.root.update()

                # Load preset scenario
                load_fn()
                self.root.update()

                # Verify clean configuration without search execution
                self.assertEqual(
                    self.app.lbl_status.cget("text"),
                    "Scenario Loaded — Ready to Search",
                    f"{name} should set status to 'Scenario Loaded — Ready to Search'",
                )
                self.assertEqual(self.app.lbl_cost.cget("text"), "--")
                self.assertEqual(self.app.lbl_steps.cget("text"), "--")
                self.assertEqual(self.app.lbl_explored.cget("text"), "--")
                self.assertEqual(self.app.lbl_time.cget("text"), "--")
                self.assertEqual(len(self.app.path_cells), 0)
                self.assertEqual(len(self.app.explored_cells), 0)
                self.assertFalse(self.app.is_animating)

    def test_random_maze_generation(self):
        self.app.generate_random_maze()
        self.root.update()

        self.assertGreater(len(self.app.obstacles), 10)
        self.assertNotIn(self.app.start_node, self.app.obstacles)
        self.assertNotIn(self.app.goal_node, self.app.obstacles)

    def test_clear_path_button(self):
        self.app.run_astar_search()
        self.root.update()
        self.assertGreater(len(self.app.path_cells), 0)

        self.app.clear_search_path()
        self.root.update()
        self.assertEqual(len(self.app.path_cells), 0)
        self.assertEqual(len(self.app.explored_cells), 0)
        self.assertEqual(self.app.lbl_status.cget("text"), "Ready")


if __name__ == "__main__":
    unittest.main()

