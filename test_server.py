"""
Unit and integration tests for the A* Pathfinding Web Server (server.py).
Tests all API endpoints, problem scenarios, and web UI requirements.
"""

import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from astar import DIRECTIONS, Grid, astar, manhattan_distance
from server import AStarRequestHandler, HTML_CONTENT


class TestWebServer(unittest.TestCase):
    """Test suite for server.py HTTP endpoints and API."""

    @classmethod
    def setUpClass(cls):
        # Start a local test server on an ephemeral port
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AStarRequestHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.server.shutdown()
            cls.server.server_close()
        except Exception:
            pass

    def test_frontend_html_served(self):
        """Test GET / serves the full HTML page with dark/light mode and required components."""
        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            # Verify Dark and Light mode theme support
            self.assertIn('data-theme="dark"', html)
            self.assertIn(':root[data-theme="dark"]', html)
            self.assertIn(':root[data-theme="light"]', html)
            self.assertIn("themeToggleBtn", html)
            self.assertIn("localStorage", html)
            # Verify Project title and description
            self.assertIn("Optimal Path Finding Using A* Search Algorithm", html)
            self.assertIn("f(n) = g(n) + h(n)", html)
            self.assertIn("Manhattan Distance Heuristic", html)
            # Verify all four scenarios
            self.assertIn("scenDetour", html)
            self.assertIn("scenEqual", html)
            self.assertIn("scenBlocked", html)
            self.assertIn("scenSame", html)
            # Verify status text requirement
            self.assertIn("Scenario Loaded — Ready to Search", html)
            # Verify metrics elements
            self.assertIn("metricStatus", html)
            self.assertIn("metricCost", html)
            self.assertIn("metricSteps", html)
            self.assertIn("metricExplored", html)
            self.assertIn("metricTime", html)
            # Verify map canvas and rendering components
            self.assertIn('id="mapCanvas"', html)
            self.assertIn('drawRoundRect', html)
            self.assertIn('coordToCanvas', html)
            self.assertIn('drawLocationPin', html)
            self.assertIn('drawMap()', html)

    def test_routing_with_query_params(self):
        """Verify routing works when URL contains query parameters (e.g. utm_source)."""
        # Test /?utm_source=chatgpt.com
        req1 = urllib.request.Request(f"{self.base_url}/?utm_source=chatgpt.com")
        with urllib.request.urlopen(req1) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("Optimal Path Finding Using A* Search Algorithm", html)

        # Test /index?ref=google
        req2 = urllib.request.Request(f"{self.base_url}/index?ref=google")
        with urllib.request.urlopen(req2) as resp:
            self.assertEqual(resp.status, 200)

        # Test /api/health?test=1
        req3 = urllib.request.Request(f"{self.base_url}/api/health?test=1")
        with urllib.request.urlopen(req3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")

    def test_api_health(self):
        """Test GET /api/health."""
        req = urllib.request.Request(f"{self.base_url}/api/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")

    def _post_astar(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/api/astar",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def test_scenario_default_grid(self):
        """Test basic unweighted search (2, 2) to (12, 12)."""
        payload = {
            "start": [2, 2],
            "goal": [12, 12],
            "obstacles": [],
            "weights": {},
        }
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["cost"], 20.0)
        self.assertEqual(len(res["path"]), 21)
        self.assertGreater(res["nodes_explored"], 0)
        self.assertGreater(res["execution_time_ms"], 0)

        # Verify 4-direction movement only (no diagonal steps)
        path = res["path"]
        for i in range(len(path) - 1):
            dr = path[i + 1][0] - path[i][0]
            dc = path[i + 1][1] - path[i][1]
            self.assertIn((dr, dc), DIRECTIONS)

    def test_scenario_cheap_detour_vs_expensive_direct(self):
        """
        Preset Scenario 1: Cheap Detour vs Expensive Direct.
        High weight (15) on direct row 7, detour along row 5 is cost 1.
        """
        weights = {f"7,{c}": 15.0 for c in range(4, 11)}
        obstacles = []
        for c in range(4, 11):
            if c != 4 and c != 10:
                obstacles.append([6, c])
            obstacles.append([8, c])

        payload = {
            "start": [7, 2],
            "goal": [7, 12],
            "obstacles": obstacles,
            "weights": weights,
        }
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        # Detour cost should be significantly less than direct path (7 * 15 = 105)
        self.assertLess(res["cost"], 50.0)
        # Verify direct high-cost cells on row 7 were avoided
        path_tuples = [tuple(p) for p in res["path"]]
        for c in range(4, 11):
            self.assertNotIn((7, c), path_tuples)

    def test_scenario_equal_cost_alternatives(self):
        """Preset Scenario 2: Equal-Cost Alternative with central obstacle."""
        obstacles = [[r, c] for r in range(6, 9) for c in range(6, 9)]
        payload = {
            "start": [4, 4],
            "goal": [10, 10],
            "obstacles": obstacles,
            "weights": {},
        }
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertIsNotNone(res["path"])
        self.assertGreater(len(res["path"]), 0)

    def test_scenario_unreachable_goal(self):
        """Preset Scenario 3: Unreachable Goal completely surrounded by obstacles."""
        obstacles = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr != 0 or dc != 0:
                    obstacles.append([10 + dr, 10 + dc])

        payload = {
            "start": [2, 2],
            "goal": [10, 10],
            "obstacles": obstacles,
            "weights": {},
        }
        res = self._post_astar(payload)
        self.assertFalse(res["found"])
        self.assertIsNone(res["path"])
        self.assertIsNone(res["cost"])
        self.assertGreater(res["nodes_explored"], 0)

    def test_scenario_start_equals_goal(self):
        """Preset Scenario 4: Start equals Goal."""
        payload = {
            "start": [7, 7],
            "goal": [7, 7],
            "obstacles": [],
            "weights": {},
        }
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["path"], [[7, 7]])
        self.assertEqual(res["cost"], 0.0)
        self.assertEqual(res["nodes_explored"], 0)

    def test_real_openstreetmap_leaflet_integration(self):
        """Verify HTML integrates Leaflet.js with authentic OpenStreetMap tiles and Google Maps UI."""
        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            # Verify Leaflet.js library inclusion
            self.assertIn("leaflet.js", html)
            self.assertIn("leaflet.css", html)
            # Verify OpenStreetMap tile endpoint without third-party API key requirements
            self.assertIn("tile.openstreetmap.org", html)
            self.assertNotIn("cartocdn", html)
            # Verify fullscreen map container
            self.assertIn('id="map"', html)
            # Verify Google Maps search inputs
            self.assertIn("startInput", html)
            self.assertIn("destInput", html)
            self.assertIn("btnSwap", html)
            self.assertIn("btnRun", html)
            # Verify roadblock and traffic tools
            self.assertIn('data-tool="roadblock"', html)
            self.assertIn('data-tool="traffic"', html)

    def test_landmarks_api(self):
        """Verify GET /api/landmarks returns authentic landmarks with geographic coordinates."""
        req = urllib.request.Request(f"{self.base_url}/api/landmarks")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("landmarks", data)
            self.assertIn("avenues", data)
            self.assertIn("streets", data)
            landmarks = {l["id"]: l for l in data["landmarks"]}
            self.assertIn("times_square", landmarks)
            self.assertIn("empire_state", landmarks)
            self.assertIn("grand_central", landmarks)
            self.assertIn("central_park_south", landmarks)
            # Verify geographic coordinates are valid
            for l in data["landmarks"]:
                self.assertGreater(l["lat"], 40.0)
                self.assertLess(l["lat"], 41.0)
                self.assertGreater(l["lon"], -75.0)
                self.assertLess(l["lon"], -73.0)

    def test_geo_astar_routing(self):
        """Verify POST /api/astar with mode='geo' calculates real-world route using A*."""
        payload = {
            "mode": "geo",
            "start": "times_square",
            "goal": "empire_state",
            "roadblocks": [],
            "traffic": {},
        }
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["mode"], "geo")
        self.assertGreater(res["distance_m"], 800)
        self.assertLess(res["distance_m"], 3000)
        self.assertGreater(len(res["path_coords"]), 2)
        self.assertGreater(len(res["directions"]), 0)
        self.assertGreater(res["nodes_explored"], 0)
        self.assertGreater(res["execution_time_ms"], 0)
        # Verify first and last coordinates match Times Square and Empire State Building
        first_coord = res["path_coords"][0]
        last_coord = res["path_coords"][-1]
        self.assertAlmostEqual(first_coord[0], 40.7580, places=3)
        self.assertAlmostEqual(last_coord[0], 40.7484, places=3)

    def test_geo_astar_roadblock_recalculation(self):
        """Verify that roadblocks cause A* to calculate an alternative detour route."""
        # 1. Normal route
        payload1 = {
            "mode": "geo",
            "start": "times_square",
            "goal": "empire_state",
            "roadblocks": [],
            "traffic": {},
        }
        res1 = self._post_astar(payload1)
        self.assertTrue(res1["found"])
        orig_path = res1["path"]

        # 2. Block an intermediate node on the optimal path
        block_node = orig_path[1]
        payload2 = {
            "mode": "geo",
            "start": "times_square",
            "goal": "empire_state",
            "roadblocks": [block_node],
            "traffic": {},
        }
        res2 = self._post_astar(payload2)
        self.assertTrue(res2["found"])
        # Path must change to avoid the roadblock
        self.assertNotEqual(res1["path"], res2["path"])
        self.assertNotIn(block_node, res2["path"])

    def test_nominatim_geocode_api(self):
        """Verify GET /api/geocode returns real-world place suggestions with lat/lon."""
        # Test searching for Eiffel Tower
        req = urllib.request.Request(f"{self.base_url}/api/geocode?q=Eiffel+Tower")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            first = data[0]
            self.assertIn("name", first)
            self.assertIn("lat", first)
            self.assertIn("lon", first)
            # Eiffel Tower coordinates in Paris (~48.8584, 2.2945)
            self.assertAlmostEqual(first["lat"], 48.8584, places=2)
            self.assertAlmostEqual(first["lon"], 2.2945, places=2)

    def test_multi_city_astar_routing_paris(self):
        """Verify A* algorithm calculates road route between locations in different cities (e.g. Paris)."""
        # Eiffel Tower to Louvre Museum in Paris
        payload = {
            "mode": "geo",
            "start_coord": [48.8584, 2.2945],
            "goal_coord": [48.8606, 2.3376],
            "start_name": "Eiffel Tower",
            "goal_name": "Louvre Museum",
            "roadblocks": [],
            "traffic": {},
        }
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["mode"], "geo")
        # Distance between Eiffel Tower and Louvre is ~3.1 - 4.5 km by road
        self.assertGreater(res["distance_km"], 2.5)
        self.assertLess(res["distance_km"], 6.0)
        self.assertGreater(len(res["path_coords"]), 2)
        self.assertGreater(res["nodes_explored"], 0)
        # Starting and ending coordinates match Paris
        self.assertAlmostEqual(res["path_coords"][0][0], 48.8584, places=2)
        self.assertAlmostEqual(res["path_coords"][-1][0], 48.8606, places=2)

    def test_academic_scenarios_api_endpoints(self):
        """Verify GET /api/scenario returns independent configurations for all 4 academic scenarios."""
        scenarios = ["detour", "equal", "blocked", "same"]
        configs = {}
        for scen_id in scenarios:
            req = urllib.request.Request(f"{self.base_url}/api/scenario?id={scen_id}")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["scenario_id"], scen_id)
                self.assertIn("start_coord", data)
                self.assertIn("goal_coord", data)
                self.assertIn("start_name", data)
                self.assertIn("goal_name", data)
                configs[scen_id] = data

        # Verify each scenario has an independent start or goal, distinct from default Times Square -> Empire State
        self.assertNotEqual(configs["detour"]["start_coord"], configs["equal"]["start_coord"])
        self.assertNotEqual(configs["detour"]["start_coord"], [40.7580, -73.9855]) # Not default Times Square
        self.assertNotEqual(configs["blocked"]["goal_coord"], [40.7484, -73.9857]) # Not default Empire State

    def test_academic_scenario_1_cheap_detour_vs_expensive(self):
        """
        Verify Academic Benchmark Scenario 1:
        Direct route has higher travel cost (15x congestion) and indirect detour route has lower travel cost.
        A* must choose the cheaper route even though it has more steps.
        """
        payload = {"mode": "geo", "scenario": "detour"}
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["scenario"], "detour")
        self.assertIn("Cheaper Detour Chosen", res["status_message"])
        self.assertGreater(res["nodes_explored"], 0)
        # Detour takes Madison Ave bypass (17 steps) avoiding the high-congestion 5th Ave
        self.assertGreaterEqual(len(res["path"]), 16)
        # Traversed nodes along detour avoid 5th Ave
        detour_nodes = res["path"]
        for nid in detour_nodes[1:-1]:
            self.assertFalse(nid.startswith("osm_") and nid.endswith("_6") and nid != "osm_0_6" and nid != "osm_14_6")

    def test_academic_scenario_2_equal_cost_alternatives(self):
        """
        Verify Academic Benchmark Scenario 2:
        Two different paths with exactly equal total cost.
        A* returns one optimal path and correctly identifies the equal-cost alternative.
        """
        payload = {"mode": "geo", "scenario": "equal"}
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["scenario"], "equal")
        self.assertTrue(res["has_equal_alternative"])
        self.assertIsNotNone(res["alternative_coords"])
        self.assertGreater(len(res["alternative_coords"]), 1)
        # Costs must be exactly equal
        self.assertEqual(res["cost"], res["alternative_cost"])
        self.assertIn("Equal-Cost Alternative Identified", res["status_message"])

    def test_academic_scenario_3_unreachable_goal(self):
        """
        Verify Academic Benchmark Scenario 3:
        Obstacles / roadblocks disconnect the destination.
        A* must report 'No path found' instead of displaying an old route.
        """
        payload = {"mode": "geo", "scenario": "blocked"}
        res = self._post_astar(payload)
        self.assertFalse(res["found"])
        self.assertEqual(res["scenario"], "blocked")
        self.assertIsNone(res["path"])
        self.assertIsNone(res["cost"])
        self.assertEqual(len(res["path_coords"]), 0)
        self.assertIn("No path found", res["status_message"])
        self.assertGreater(res["nodes_explored"], 0)

    def test_academic_scenario_4_start_equals_goal(self):
        """
        Verify Academic Benchmark Scenario 4:
        Start and destination are the exact same node.
        Returns a path containing only that node, with zero cost and zero steps.
        """
        payload = {"mode": "geo", "scenario": "same"}
        res = self._post_astar(payload)
        self.assertTrue(res["found"])
        self.assertEqual(res["scenario"], "same")
        self.assertEqual(len(res["path"]), 1)
        self.assertEqual(res["path_coords"], [[40.7527, -73.9772]])
        self.assertEqual(res["cost"], 0.0)
        self.assertEqual(res["distance_m"], 0.0)
        self.assertEqual(res["nodes_explored"], 0)
        self.assertIn("Start = Goal", res["status_message"])

    def test_dynamic_scenarios_on_user_selected_endpoints(self):
        """
        CRITICAL TEST: Verify all four academic scenarios execute dynamically
        on arbitrary user-selected source and destination locations without resetting
        coordinates or faking results.
        Tested pairs:
          1. Mysore Palace -> Chamundi Hills
          2. Bangalore Palace -> Lalbagh Botanical Garden
          3. Times Square -> Empire State Building
        """
        test_pairs = [
            {
                "start_coord": [12.3051, 76.6551],
                "goal_coord": [12.2741, 76.6705],
                "start_name": "Mysore Palace",
                "goal_name": "Chamundi Hills",
            },
            {
                "start_coord": [12.9988, 77.5921],
                "goal_coord": [12.9507, 77.5848],
                "start_name": "Bangalore Palace",
                "goal_name": "Lalbagh Botanical Garden",
            },
            {
                "start_coord": [40.7580, -73.9855],
                "goal_coord": [40.7484, -73.9857],
                "start_name": "Times Square",
                "goal_name": "Empire State Building",
            },
        ]

        for pair in test_pairs:
            s_coord = pair["start_coord"]
            g_coord = pair["goal_coord"]
            s_name = pair["start_name"]
            g_name = pair["goal_name"]

            # Scenario 1: Cheap Detour vs Expensive
            p1 = {
                "mode": "geo",
                "scenario": "detour",
                "start_coord": s_coord,
                "goal_coord": g_coord,
                "start_name": s_name,
                "goal_name": g_name,
            }
            res1 = self._post_astar(p1)
            self.assertTrue(res1["found"])
            self.assertEqual(res1["scenario"], "detour")
            self.assertGreater(res1["distance_m"], 0.0)
            self.assertGreater(res1["nodes_explored"], 0)
            if res1.get("detour_found"):
                self.assertIn("Cheaper Detour", res1["status_message"])
                self.assertGreater(len(res1.get("traffic", {})), 0)
            else:
                self.assertIn("No Cheaper Detour Exists", res1["status_message"])

            # Scenario 2: Equal-Cost Alternatives
            p2 = {
                "mode": "geo",
                "scenario": "equal",
                "start_coord": s_coord,
                "goal_coord": g_coord,
                "start_name": s_name,
                "goal_name": g_name,
            }
            res2 = self._post_astar(p2)
            self.assertTrue(res2["found"])
            self.assertEqual(res2["scenario"], "equal")
            if res2.get("has_equal_alternative"):
                self.assertIsNotNone(res2["alternative_coords"])
                self.assertGreater(len(res2["alternative_coords"]), 1)
                self.assertIn("Equal-Cost Alternative Identified", res2["status_message"])
            else:
                self.assertIn("No Equal-Cost Alternative Found", res2["status_message"])

            # Scenario 3: Unreachable Goal
            p3 = {
                "mode": "geo",
                "scenario": "blocked",
                "start_coord": s_coord,
                "goal_coord": g_coord,
                "start_name": s_name,
                "goal_name": g_name,
            }
            res3 = self._post_astar(p3)
            self.assertFalse(res3["found"])
            self.assertEqual(res3["scenario"], "blocked")
            self.assertIsNone(res3["path"])
            self.assertIsNone(res3["cost"])
            self.assertEqual(len(res3["path_coords"]), 0)
            self.assertGreater(len(res3.get("roadblocks", [])), 0)
            self.assertIn("Destination Disconnected", res3["status_message"])

            # Scenario 4: Start = Goal (Preserving source, goal = source)
            p4 = {
                "mode": "geo",
                "scenario": "same",
                "start_coord": s_coord,
                "goal_coord": g_coord,
                "start_name": s_name,
                "goal_name": g_name,
            }
            res4 = self._post_astar(p4)
            self.assertTrue(res4["found"])
            self.assertEqual(res4["scenario"], "same")
            self.assertEqual(res4["cost"], 0.0)
            self.assertEqual(res4["distance_m"], 0.0)
            self.assertEqual(res4["steps"], 0)
            self.assertEqual(res4["nodes_explored"], 0)
            self.assertEqual(len(res4["path"]), 1)
            self.assertEqual(res4["path_coords"], [s_coord])
            self.assertIn("Already at Destination", res4["status_message"])


if __name__ == "__main__":
    unittest.main()

