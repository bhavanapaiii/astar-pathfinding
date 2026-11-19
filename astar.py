"""
Optimal Path Finding Using A* Search Algorithm.

This module implements:
- Manhattan distance heuristic.
- 4-direction movement (Up, Down, Left, Right).
- Support for obstacles and weighted cells.
- Open set management with Python heapq.
- Evaluation function: f(n) = g(n) + h(n).
- Parent tracking and path reconstruction.
- Handling of start == goal and unreachable goals.
"""

from __future__ import annotations
import heapq
import itertools
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Coordinate representation: (row, col)
Coordinate = Tuple[int, int]
# Geographic coordinate representation: (latitude, longitude)
GeoCoordinate = Tuple[float, float]

# Four-direction movements: Up, Down, Left, Right
DIRECTIONS: Tuple[Coordinate, ...] = (
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1),   # Right
)


def manhattan_distance(a: Coordinate, b: Coordinate) -> int:
    """
    Calculate the Manhattan distance between two coordinates.
    h(a, b) = |r1 - r2| + |c1 - c2|
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Grid:
    """
    Represents a 2D grid supporting obstacles and weighted traversal costs.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        default_cost: float = 1.0,
        obstacles: Optional[Iterable[Coordinate]] = None,
        weights: Optional[Dict[Coordinate, float]] = None,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive integers.")
        self.rows: int = rows
        self.cols: int = cols
        self.default_cost: float = float(default_cost)
        self.obstacles: Set[Coordinate] = set(obstacles) if obstacles else set()
        self.weights: Dict[Coordinate, float] = dict(weights) if weights else {}

    def in_bounds(self, node: Coordinate) -> bool:
        """Check if a coordinate lies within grid boundaries."""
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_passable(self, node: Coordinate) -> bool:
        """Check if a coordinate is inside bounds and not an obstacle."""
        return self.in_bounds(node) and (node not in self.obstacles)

    def get_cost(self, node: Coordinate) -> float:
        """
        Get the traversal cost to enter the specified node.
        Defaults to default_cost if no custom weight is set.
        """
        return self.weights.get(node, self.default_cost)

    def set_obstacle(self, r: int, c: int) -> None:
        """Mark a cell as an obstacle."""
        if self.in_bounds((r, c)):
            self.obstacles.add((r, c))

    def remove_obstacle(self, r: int, c: int) -> None:
        """Remove obstacle status from a cell."""
        self.obstacles.discard((r, c))

    def set_weight(self, r: int, c: int, weight: float) -> None:
        """Set custom traversal cost for entering a cell."""
        if weight <= 0:
            raise ValueError("Cell traversal weight must be positive.")
        if self.in_bounds((r, c)):
            self.weights[(r, c)] = float(weight)

    def get_neighbors(self, node: Coordinate) -> List[Coordinate]:
        """
        Return all valid 4-direction passable neighboring coordinates.
        """
        r, c = node
        neighbors: List[Coordinate] = []
        for dr, dc in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            if self.is_passable(neighbor):
                neighbors.append(neighbor)
        return neighbors

    @classmethod
    def from_matrix(
        cls,
        matrix: Sequence[Sequence[Any]],
        obstacle_values: Optional[Iterable[Any]] = None,
        default_cost: float = 1.0,
    ) -> "Grid":
        """
        Create a Grid from a 2D matrix / nested list.

        Obstacle detection:
        - If obstacle_values is provided, any cell whose value matches is an obstacle.
        - Otherwise, None, -1, float('inf'), '#', and 'X' are treated as obstacles.
        
        Weight detection:
        - Numeric values (> 0) that are not obstacles are treated as cell traversal weights.
        """
        if not matrix or not matrix[0]:
            raise ValueError("Matrix cannot be empty.")

        rows = len(matrix)
        cols = len(matrix[0])
        obstacles: Set[Coordinate] = set()
        weights: Dict[Coordinate, float] = {}

        obs_set = set(obstacle_values) if obstacle_values is not None else {None, -1, float("inf"), "#", "X"}

        for r in range(rows):
            for c in range(cols):
                val = matrix[r][c]
                coord = (r, c)
                if val in obs_set:
                    obstacles.add(coord)
                elif isinstance(val, (int, float)):
                    if val <= 0:
                        # Values <= 0 not in obs_set can be treated as obstacles
                        obstacles.add(coord)
                    else:
                        weights[coord] = float(val)

        return cls(rows=rows, cols=cols, default_cost=default_cost, obstacles=obstacles, weights=weights)


class AStarResult:
    """
    Result container for A* search.
    Supports tuple unpacking: path, cost = astar(...)
    """

    def __init__(
        self,
        path: Optional[List[Coordinate]],
        cost: float,
        nodes_explored: int = 0,
        came_from: Optional[Dict[Coordinate, Coordinate]] = None,
        g_scores: Optional[Dict[Coordinate, float]] = None,
        explored_order: Optional[List[Coordinate]] = None,
    ) -> None:
        self.path: Optional[List[Coordinate]] = path
        self.cost: float = cost
        self.nodes_explored: int = nodes_explored
        self.came_from: Dict[Coordinate, Coordinate] = came_from if came_from is not None else {}
        self.g_scores: Dict[Coordinate, float] = g_scores if g_scores is not None else {}
        self.explored_order: List[Coordinate] = explored_order if explored_order is not None else []

    def __bool__(self) -> bool:
        """True if a valid path was found."""
        return self.path is not None

    def __iter__(self):
        """Allows unpacking as (path, cost)."""
        return iter((self.path, self.cost))

    def __getitem__(self, index: int):
        return (self.path, self.cost)[index]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return (
            f"AStarResult(path={self.path}, cost={self.cost}, "
            f"nodes_explored={self.nodes_explored})"
        )


def reconstruct_path(
    came_from: Dict[Coordinate, Coordinate],
    current: Coordinate,
) -> List[Coordinate]:
    """
    Reconstruct the path from start to goal by tracing parent pointers.
    """
    path: List[Coordinate] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(
    grid: Union[Grid, Sequence[Sequence[Any]]],
    start: Coordinate,
    goal: Coordinate,
    heuristic: Callable[[Coordinate, Coordinate], float] = manhattan_distance,
    obstacles: Optional[Iterable[Coordinate]] = None,
    weights: Optional[Dict[Coordinate, float]] = None,
    obstacle_values: Optional[Iterable[Any]] = None,
) -> AStarResult:
    """
    Find the optimal path using the A* search algorithm.

    Parameters:
    - grid: A Grid instance or a 2D matrix (nested list).
    - start: Starting coordinate (row, col).
    - goal: Goal coordinate (row, col).
    - heuristic: Heuristic function h(n, goal). Defaults to Manhattan distance.
    - obstacles: Optional extra obstacle coordinates.
    - weights: Optional extra weight mappings {(r, c): weight}.
    - obstacle_values: Optional values in matrix to consider as obstacles.

    Returns:
    - AStarResult containing:
        - path: List of coordinates from start to goal, or None if unreachable.
        - cost: Total g-score cost of the path, or float('inf') if unreachable.
        - nodes_explored: Number of nodes popped/expanded from open set.
        - came_from: Parent tracking dictionary.
        - g_scores: Final g-score map.
    """
    # 1. Normalize grid input
    if isinstance(grid, Grid):
        grid_obj = grid
        if obstacles:
            grid_obj.obstacles.update(obstacles)
        if weights:
            grid_obj.weights.update(weights)
    else:
        grid_obj = Grid.from_matrix(
            grid,
            obstacle_values=obstacle_values,
        )
        if obstacles:
            grid_obj.obstacles.update(obstacles)
        if weights:
            grid_obj.weights.update(weights)

    # 2. Validate start and goal within bounds
    if not grid_obj.in_bounds(start) or not grid_obj.in_bounds(goal):
        return AStarResult(path=None, cost=float("inf"), nodes_explored=0)

    # 3. Check if start or goal is an obstacle
    if not grid_obj.is_passable(start) or not grid_obj.is_passable(goal):
        return AStarResult(path=None, cost=float("inf"), nodes_explored=0)

    # 4. Handle start equals goal
    if start == goal:
        return AStarResult(
            path=[start],
            cost=0.0,
            nodes_explored=0,
            came_from={},
            g_scores={start: 0.0},
        )

    # 5. Initialize A* search data structures
    # Priority queue stores tuples: (f_score, h_score, tie_breaker_id, node)
    counter = itertools.count()
    open_heap: List[Tuple[float, float, int, Coordinate]] = []

    # g(n): exact cost from start to node n
    g_score: Dict[Coordinate, float] = {start: 0.0}

    # h(n): heuristic estimate from node n to goal
    h_start = float(heuristic(start, goal))

    # f(n) = g(n) + h(n)
    f_start = g_score[start] + h_start

    heapq.heappush(open_heap, (f_start, h_start, next(counter), start))

    # Parent tracking: came_from[node] = parent_node
    came_from: Dict[Coordinate, Coordinate] = {}

    # Closed set to avoid re-expanding nodes with suboptimal paths
    closed_set: Set[Coordinate] = set()
    explored_order: List[Coordinate] = []

    nodes_explored = 0

    # 6. Search loop
    while open_heap:
        current_f, current_h, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        closed_set.add(current)
        explored_order.append(current)
        nodes_explored += 1

        # Goal check
        if current == goal:
            path = reconstruct_path(came_from, current)
            return AStarResult(
                path=path,
                cost=g_score[goal],
                nodes_explored=nodes_explored,
                came_from=came_from,
                g_scores=g_score,
                explored_order=explored_order,
            )

        # Explore 4-direction neighbors
        for neighbor in grid_obj.get_neighbors(current):
            if neighbor in closed_set:
                continue

            step_cost = grid_obj.get_cost(neighbor)
            tentative_g = g_score[current] + step_cost

            # Check if this path to neighbor is better than any previous one
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_cost = float(heuristic(neighbor, goal))
                # f(n) = g(n) + h(n)
                f_cost = tentative_g + h_cost
                heapq.heappush(open_heap, (f_cost, h_cost, next(counter), neighbor))

    # 7. Unreachable goal: open set exhausted without reaching goal
    return AStarResult(
        path=None,
        cost=float("inf"),
        nodes_explored=nodes_explored,
        came_from=came_from,
        g_scores=g_score,
        explored_order=explored_order,
    )


# ============================================================================
# Geographic Road Network Graph & Haversine A* Navigation
# ============================================================================

def haversine_distance(coord1: GeoCoordinate, coord2: GeoCoordinate) -> float:
    """
    Calculate the great-circle distance between two geographic coordinates (lat, lon)
    in meters using the Haversine formula.

    This is an admissible and consistent heuristic for road networks on Earth
    (straight-line distance on a sphere never overestimates actual road distance).
    h(n, goal) <= true_cost(n, goal)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000.0  # Earth's mean radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


class RoadNode:
    """Represents an intersection or landmark node in a real road network."""

    def __init__(self, node_id: str, lat: float, lon: float, name: str = "", is_landmark: bool = False):
        self.node_id = str(node_id)
        self.lat = float(lat)
        self.lon = float(lon)
        self.name = name or self.node_id
        self.is_landmark = is_landmark

    @property
    def coord(self) -> GeoCoordinate:
        return (self.lat, self.lon)

    def __repr__(self) -> str:
        return f"RoadNode({self.node_id}, lat={self.lat:.4f}, lon={self.lon:.4f}, name='{self.name}')"


class RoadEdge:
    """Represents a street segment connecting two intersections."""

    def __init__(
        self,
        from_id: str,
        to_id: str,
        street_name: str,
        length_m: float,
        oneway: bool = False,
        speed_limit_kmh: float = 40.0,
        weight_factor: float = 1.0,
        is_blocked: bool = False,
    ):
        self.from_id = str(from_id)
        self.to_id = str(to_id)
        self.street_name = street_name
        self.length_m = float(length_m)
        self.oneway = oneway
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.weight_factor = float(weight_factor)
        self.is_blocked = is_blocked

    @property
    def traversal_cost(self) -> float:
        """Effective traversal cost in meters (length * congestion weight factor)."""
        if self.is_blocked:
            return float("inf")
        return self.length_m * self.weight_factor


class RoadGraph:
    """Represents a real road network graph derived from OpenStreetMap."""

    def __init__(self):
        self.nodes: Dict[str, RoadNode] = {}
        # adjacency: node_id -> list of (neighbor_id, RoadEdge)
        self.adj: Dict[str, List[Tuple[str, RoadEdge]]] = {}
        self.blocked_nodes: Set[str] = set()
        self.blocked_edges: Set[Tuple[str, str]] = set()
        self.weights: Dict[str, float] = {}  # custom congestion weights on nodes

    def add_node(self, node: RoadNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = []

    def add_edge(self, edge: RoadEdge) -> None:
        if edge.from_id not in self.nodes or edge.to_id not in self.nodes:
            raise KeyError(f"Both nodes must exist in graph: {edge.from_id} -> {edge.to_id}")
        self.adj[edge.from_id].append((edge.to_id, edge))
        if not edge.oneway:
            reverse_edge = RoadEdge(
                from_id=edge.to_id,
                to_id=edge.from_id,
                street_name=edge.street_name,
                length_m=edge.length_m,
                oneway=False,
                speed_limit_kmh=edge.speed_limit_kmh,
                weight_factor=edge.weight_factor,
                is_blocked=edge.is_blocked,
            )
            self.adj[edge.to_id].append((edge.from_id, reverse_edge))

    def get_neighbors(self, node_id: str) -> List[Tuple[str, RoadEdge]]:
        """Return list of accessible (neighbor_id, edge) pairs."""
        if node_id in self.blocked_nodes:
            return []
        valid_neighbors = []
        for nbr_id, edge in self.adj.get(node_id, []):
            if nbr_id in self.blocked_nodes:
                continue
            if (node_id, nbr_id) in self.blocked_edges or (nbr_id, node_id) in self.blocked_edges:
                continue
            if edge.is_blocked or edge.traversal_cost == float("inf"):
                continue
            valid_neighbors.append((nbr_id, edge))
        return valid_neighbors

    def find_nearest_node(self, lat: float, lon: float) -> Optional[RoadNode]:
        """Find the closest road intersection to the given geographic coordinate."""
        if not self.nodes:
            return None
        return min(
            self.nodes.values(),
            key=lambda n: haversine_distance((lat, lon), (n.lat, n.lon)),
        )


def astar_road_network(
    graph: RoadGraph,
    start_id: str,
    goal_id: str,
    heuristic: Callable[[GeoCoordinate, GeoCoordinate], float] = haversine_distance,
) -> AStarResult:
    """
    Find the optimal route on a geographic road network graph using the A* algorithm.

    Reuses Python heapq, f(n) = g(n) + h(n), parent pointer tracking,
    exact traversal costs (distance in meters * traffic weights), and the Haversine heuristic.
    """
    if start_id not in graph.nodes or goal_id not in graph.nodes:
        return AStarResult(path=None, cost=float("inf"), nodes_explored=0)

    if start_id in graph.blocked_nodes or goal_id in graph.blocked_nodes:
        return AStarResult(path=None, cost=float("inf"), nodes_explored=0)

    start_node = graph.nodes[start_id]
    goal_node = graph.nodes[goal_id]

    if start_id == goal_id:
        return AStarResult(
            path=[start_id],
            cost=0.0,
            nodes_explored=0,
            came_from={},
            g_scores={start_id: 0.0},
            explored_order=[start_id],
        )

    counter = itertools.count()
    open_heap: List[Tuple[float, float, int, str]] = []

    g_score: Dict[str, float] = {start_id: 0.0}
    h_start = float(heuristic(start_node.coord, goal_node.coord))
    f_start = g_score[start_id] + h_start

    heapq.heappush(open_heap, (f_start, h_start, next(counter), start_id))

    came_from: Dict[str, str] = {}
    closed_set: Set[str] = set()
    explored_order: List[str] = []
    nodes_explored = 0

    while open_heap:
        current_f, current_h, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        closed_set.add(current)
        explored_order.append(current)
        nodes_explored += 1

        if current == goal_id:
            path: List[str] = [current]
            curr = current
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return AStarResult(
                path=path,
                cost=g_score[goal_id],
                nodes_explored=nodes_explored,
                came_from=came_from,
                g_scores=g_score,
                explored_order=explored_order,
            )

        for neighbor_id, edge in graph.get_neighbors(current):
            if neighbor_id in closed_set:
                continue

            step_cost = edge.traversal_cost * graph.weights.get(neighbor_id, 1.0)
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor_id, float("inf")):
                came_from[neighbor_id] = current
                g_score[neighbor_id] = tentative_g
                h_cost = float(heuristic(graph.nodes[neighbor_id].coord, goal_node.coord))
                f_cost = tentative_g + h_cost
                heapq.heappush(open_heap, (f_cost, h_cost, next(counter), neighbor_id))

    return AStarResult(
        path=None,
        cost=float("inf"),
        nodes_explored=nodes_explored,
        came_from=came_from,
        g_scores=g_score,
        explored_order=explored_order,
    )


