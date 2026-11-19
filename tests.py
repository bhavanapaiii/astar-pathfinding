"""
Unit tests for Optimal Path Finding Using A* Search Algorithm (astar.py).
"""

import math
import unittest
from astar import DIRECTIONS, Grid, astar, manhattan_distance, reconstruct_path


class TestManhattanDistance(unittest.TestCase):
    """Test Manhattan distance heuristic."""

    def test_basic_distance(self):
        self.assertEqual(manhattan_distance((0, 0), (3, 4)), 7)
        self.assertEqual(manhattan_distance((1, 2), (4, 6)), 7)

    def test_same_point(self):
        self.assertEqual(manhattan_distance((5, 5), (5, 5)), 0)

    def test_symmetry(self):
        p1 = (2, 8)
        p2 = (9, 3)
        self.assertEqual(manhattan_distance(p1, p2), manhattan_distance(p2, p1))


class TestAStarGridAndSearch(unittest.TestCase):
    """Test A* search algorithm requirements and edge cases."""

    def test_start_equals_goal(self):
        grid = Grid(rows=5, cols=5)
        start = (2, 2)
        goal = (2, 2)
        result = astar(grid, start, goal)

        self.assertTrue(result)
        self.assertEqual(result.path, [(2, 2)])
        self.assertEqual(result.cost, 0.0)
        self.assertEqual(result.nodes_explored, 0)

        # Verify tuple unpacking
        path, cost = result
        self.assertEqual(path, [(2, 2)])
        self.assertEqual(cost, 0.0)

    def test_simple_unweighted_path(self):
        grid = Grid(rows=4, cols=4, default_cost=1.0)
        start = (0, 0)
        goal = (3, 3)
        path, cost = astar(grid, start, goal)

        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        # Manhattan distance is 6; each step costs 1.0, so 6 steps = cost 6.0
        self.assertEqual(cost, 6.0)
        self.assertEqual(len(path), 7)  # start + 6 steps

    def test_four_direction_movement_only(self):
        """Verify that every step is purely orthogonal (Up, Down, Left, Right)."""
        grid = Grid(rows=5, cols=5)
        start = (0, 0)
        goal = (4, 4)
        path, _ = astar(grid, start, goal)

        self.assertIsNotNone(path)
        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i + 1]
            diff = (nxt[0] - curr[0], nxt[1] - curr[1])
            self.assertIn(diff, DIRECTIONS, f"Step from {curr} to {nxt} is not 4-directional")

    def test_obstacle_avoidance(self):
        """
        Grid 3x3:
        S  #  G
        .  #  .
        .  .  .
        Wall at (0, 1) and (1, 1). A* must go around via row 2.
        """
        grid = Grid(rows=3, cols=3, obstacles=[(0, 1), (1, 1)])
        start = (0, 0)
        goal = (0, 2)
        path, cost = astar(grid, start, goal)

        self.assertIsNotNone(path)
        self.assertNotIn((0, 1), path)
        self.assertNotIn((1, 1), path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(cost, 6.0)

    def test_unreachable_goal_enclosed(self):
        """Goal enclosed by obstacles on all 4 sides."""
        grid = Grid(
            rows=5,
            cols=5,
            obstacles=[(1, 2), (3, 2), (2, 1), (2, 3)],
        )
        start = (0, 0)
        goal = (2, 2)
        result = astar(grid, start, goal)

        self.assertFalse(result)
        self.assertIsNone(result.path)
        self.assertEqual(result.cost, float("inf"))

    def test_start_or_goal_is_obstacle(self):
        grid = Grid(rows=3, cols=3, obstacles=[(0, 0), (2, 2)])

        # Start is obstacle
        res_start = astar(grid, (0, 0), (1, 1))
        self.assertIsNone(res_start.path)
        self.assertEqual(res_start.cost, float("inf"))

        # Goal is obstacle
        res_goal = astar(grid, (1, 1), (2, 2))
        self.assertIsNone(res_goal.path)
        self.assertEqual(res_goal.cost, float("inf"))

    def test_start_or_goal_out_of_bounds(self):
        grid = Grid(rows=3, cols=3)
        res1 = astar(grid, (-1, 0), (2, 2))
        self.assertIsNone(res1.path)
        self.assertEqual(res1.cost, float("inf"))

        res2 = astar(grid, (0, 0), (3, 3))
        self.assertIsNone(res2.path)
        self.assertEqual(res2.cost, float("inf"))

    def test_weighted_cells_optimal_detour(self):
        """
        Test that A* chooses a longer geometric path with lower weight
        over a shorter path with heavy weight.

        3x3 Grid:
        (0, 0)[cost 1] -> (0, 1)[cost 50] -> (0, 2)[cost 1]  (direct cost = 51)
        Detour via:
        (0, 0) -> (1, 0)[1] -> (1, 1)[1] -> (1, 2)[1] -> (0, 2)[1] (detour cost = 4)
        """
        grid = Grid(rows=3, cols=3, default_cost=1.0)
        grid.set_weight(0, 1, 50.0)

        start = (0, 0)
        goal = (0, 2)
        path, cost = astar(grid, start, goal)

        self.assertIsNotNone(path)
        self.assertNotIn((0, 1), path, "A* should have avoided high-cost cell (0, 1)")
        self.assertEqual(cost, 4.0)
        expected_path = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
        self.assertEqual(path, expected_path)

    def test_parent_tracking_integrity(self):
        """Verify that reconstruct_path and came_from maintain exact lineage."""
        grid = Grid(rows=4, cols=4, obstacles=[(1, 1), (1, 2)])
        start = (0, 0)
        goal = (2, 3)
        result = astar(grid, start, goal)

        self.assertIsNotNone(result.path)
        # Trace backwards using came_from
        curr = goal
        traced_path = [curr]
        while curr != start:
            self.assertIn(curr, result.came_from)
            curr = result.came_from[curr]
            traced_path.append(curr)
        traced_path.reverse()

        self.assertEqual(result.path, traced_path)

    def test_matrix_input_with_custom_obstacles(self):
        """Test calling astar directly with a 2D matrix (list of lists)."""
        matrix = [
            [1, 1, 1],
            [1, "#", 1],
            [1, 1, 1],
        ]
        start = (0, 0)
        goal = (2, 2)
        path, cost = astar(matrix, start, goal)

        self.assertIsNotNone(path)
        self.assertNotIn((1, 1), path)
        self.assertEqual(cost, 4.0)


if __name__ == "__main__":
    unittest.main()

