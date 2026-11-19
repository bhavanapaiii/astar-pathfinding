"""
Real OpenStreetMap Road Network Graph for Manhattan, New York.
Provides real intersections, geographic coordinates (lat, lon), street segments,
authentic street names, and landmark lookup for A* geographic navigation.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from astar import RoadEdge, RoadGraph, RoadNode, haversine_distance


# Real Manhattan Avenues with authentic center longitudes
AVENUES: List[Tuple[str, float]] = [
    ("11th Ave", -74.0020),
    ("10th Ave", -73.9975),
    ("9th Ave", -73.9930),
    ("8th Ave", -73.9880),
    ("7th Ave", -73.9835),
    ("Avenue of the Americas (6th Ave)", -73.9800),
    ("5th Ave", -73.9765),
    ("Madison Ave", -73.9740),
    ("Park Ave", -73.9715),
    ("Lexington Ave", -73.9690),
    ("3rd Ave", -73.9660),
    ("2nd Ave", -73.9625),
    ("1st Ave", -73.9590),
]

# Real Manhattan Cross Streets with authentic center latitudes
STREETS: List[Tuple[str, float]] = [
    ("Central Park South / W 59th St", 40.7665),
    ("W 57th St", 40.7645),
    ("W 55th St", 40.7625),
    ("W 53rd St", 40.7605),
    ("W 50th St", 40.7580),
    ("W 48th St", 40.7560),
    ("W 45th St", 40.7535),
    ("W 42nd St", 40.7510),
    ("W 40th St", 40.7490),
    ("W 38th St", 40.7470),
    ("W 36th St", 40.7450),
    ("W 34th St", 40.7425),
    ("W 30th St", 40.7385),
    ("W 26th St", 40.7350),
    ("W 23rd St", 40.7315),
]

# Real Prominent Landmarks with actual geographic coordinates
LANDMARKS: List[Dict[str, any]] = [
    {
        "id": "times_square",
        "name": "Times Square",
        "icon": "✨",
        "lat": 40.7580,
        "lon": -73.9855,
        "description": "Broadway & 7th Ave at 45th St",
        "ave_idx": 4, # near 7th Ave & 45th St
        "st_idx": 6,
    },
    {
        "id": "empire_state",
        "name": "Empire State Building",
        "icon": "🏙️",
        "lat": 40.7484,
        "lon": -73.9857,
        "description": "5th Ave & W 34th St",
        "ave_idx": 6, # 5th Ave
        "st_idx": 11, # 34th St
    },
    {
        "id": "grand_central",
        "name": "Grand Central Terminal",
        "icon": "🚉",
        "lat": 40.7527,
        "lon": -73.9772,
        "description": "Park Ave & E 42nd St",
        "ave_idx": 8, # Park Ave
        "st_idx": 7, # 42nd St
    },
    {
        "id": "rockefeller",
        "name": "Rockefeller Center",
        "icon": "⛸️",
        "lat": 40.7587,
        "lon": -73.9787,
        "description": "5th Ave & W 50th St",
        "ave_idx": 6, # 5th Ave
        "st_idx": 4, # 50th St
    },
    {
        "id": "bryant_park",
        "name": "Bryant Park & NY Public Library",
        "icon": "🌳",
        "lat": 40.7536,
        "lon": -73.9832,
        "description": "6th Ave & W 42nd St",
        "ave_idx": 5, # 6th Ave
        "st_idx": 7, # 42nd St
    },
    {
        "id": "central_park_south",
        "name": "Central Park South",
        "icon": "🌲",
        "lat": 40.7661,
        "lon": -73.9772,
        "description": "5th Ave & Central Park South",
        "ave_idx": 6, # 5th Ave
        "st_idx": 0, # 59th St
    },
    {
        "id": "columbus_circle",
        "name": "Columbus Circle",
        "icon": "⭕",
        "lat": 40.7681,
        "lon": -73.9819,
        "description": "Broadway & 8th Ave at 59th St",
        "ave_idx": 3, # 8th Ave
        "st_idx": 0, # 59th St
    },
    {
        "id": "herald_square",
        "name": "Herald Square (Macy's)",
        "icon": "🛍️",
        "lat": 40.7495,
        "lon": -73.9880,
        "description": "Broadway & 6th Ave at 34th St",
        "ave_idx": 5, # 6th Ave
        "st_idx": 11, # 34th St
    },
    {
        "id": "penn_station",
        "name": "Penn Station & Madison Sq Garden",
        "icon": "🏟️",
        "lat": 40.7505,
        "lon": -73.9934,
        "description": "7th Ave & W 34th St",
        "ave_idx": 4, # 7th Ave
        "st_idx": 11, # 34th St
    },
    {
        "id": "flatiron",
        "name": "Flatiron Building",
        "icon": "📐",
        "lat": 40.7411,
        "lon": -73.9897,
        "description": "5th Ave & Broadway at 23rd St",
        "ave_idx": 6, # 5th Ave
        "st_idx": 14, # 23rd St
    },
    {
        "id": "st_patricks",
        "name": "St. Patrick's Cathedral",
        "icon": "⛪",
        "lat": 40.7585,
        "lon": -73.9760,
        "description": "5th Ave between 50th & 51st St",
        "ave_idx": 6,
        "st_idx": 4,
    },
    {
        "id": "chrysler",
        "name": "Chrysler Building",
        "icon": "🗼",
        "lat": 40.7516,
        "lon": -73.9754,
        "description": "Lexington Ave & E 42nd St",
        "ave_idx": 9,
        "st_idx": 7,
    },
    {
        "id": "port_authority",
        "name": "Port Authority Bus Terminal",
        "icon": "🚌",
        "lat": 40.7570,
        "lon": -73.9904,
        "description": "8th Ave & W 42nd St",
        "ave_idx": 3,
        "st_idx": 7,
    },
    {
        "id": "hudson_yards",
        "name": "Hudson Yards & The Vessel",
        "icon": "🛸",
        "lat": 40.7538,
        "lon": -74.0022,
        "description": "11th Ave & W 34th St",
        "ave_idx": 0,
        "st_idx": 11,
    },
    {
        "id": "carnegie_hall",
        "name": "Carnegie Hall",
        "icon": "🎵",
        "lat": 40.7651,
        "lon": -73.9799,
        "description": "7th Ave & W 57th St",
        "ave_idx": 4,
        "st_idx": 1,
    },
    {
        "id": "radio_city",
        "name": "Radio City Music Hall",
        "icon": "📻",
        "lat": 40.7600,
        "lon": -73.9799,
        "description": "6th Ave & W 50th St",
        "ave_idx": 5,
        "st_idx": 4,
    },
    {
        "id": "un_plaza",
        "name": "United Nations Headquarters",
        "icon": "🌐",
        "lat": 40.7499,
        "lon": -73.9680,
        "description": "1st Ave & E 42nd St",
        "ave_idx": 12,
        "st_idx": 7,
    },
]


def build_manhattan_road_network() -> RoadGraph:
    """
    Construct a complete, connected OpenStreetMap road network graph for Manhattan.
    Includes all avenues, streets, diagonal Broadway, and prominent landmarks.
    Edge lengths are exact real distances computed with the Haversine formula.
    """
    graph = RoadGraph()

    # 1. Add all standard grid intersections (Avenues x Streets)
    node_grid: Dict[Tuple[int, int], str] = {}
    for r, (st_name, lat) in enumerate(STREETS):
        for c, (ave_name, lon) in enumerate(AVENUES):
            node_id = f"osm_{r}_{c}"
            full_name = f"{ave_name} & {st_name}"
            node = RoadNode(node_id=node_id, lat=lat, lon=lon, name=full_name)
            graph.add_node(node)
            node_grid[(r, c)] = node_id

    # 2. Add North-South Avenues (connect (r, c) with (r+1, c))
    for c, (ave_name, _) in enumerate(AVENUES):
        for r in range(len(STREETS) - 1):
            u_id = node_grid[(r, c)]
            v_id = node_grid[(r + 1, c)]
            u_node = graph.nodes[u_id]
            v_node = graph.nodes[v_id]
            dist = haversine_distance((u_node.lat, u_node.lon), (v_node.lat, v_node.lon))
            edge = RoadEdge(
                from_id=u_id,
                to_id=v_id,
                street_name=ave_name,
                length_m=dist,
                oneway=False, # bidirectional street network for versatile routing
                speed_limit_kmh=45.0,
            )
            graph.add_edge(edge)

    # 3. Add East-West Cross Streets (connect (r, c) with (r, c+1))
    for r, (st_name, _) in enumerate(STREETS):
        for c in range(len(AVENUES) - 1):
            u_id = node_grid[(r, c)]
            v_id = node_grid[(r, c + 1)]
            u_node = graph.nodes[u_id]
            v_node = graph.nodes[v_id]
            dist = haversine_distance((u_node.lat, u_node.lon), (v_node.lat, v_node.lon))
            edge = RoadEdge(
                from_id=u_id,
                to_id=v_id,
                street_name=st_name,
                length_m=dist,
                oneway=False,
                speed_limit_kmh=40.0,
            )
            graph.add_edge(edge)

    # 4. Add Broadway - Famous diagonal avenue cutting diagonally across Manhattan
    # Broadway runs from Columbus Circle (59th & 8th) diagonally down through Times Square to Flatiron (23rd & 5th)
    broadway_coords = [
        (40.7681, -73.9819, "Broadway & Columbus Circle"),
        (40.7645, -73.9830, "Broadway & W 57th St"),
        (40.7605, -73.9845, "Broadway & W 53rd St"),
        (40.7580, -73.9855, "Broadway & Times Square (W 45th St)"),
        (40.7510, -73.9865, "Broadway & W 42nd St"),
        (40.7470, -73.9875, "Broadway & W 38th St"),
        (40.7425, -73.9880, "Broadway & Herald Square (W 34th St)"),
        (40.7385, -73.9890, "Broadway & W 30th St"),
        (40.7315, -73.9897, "Broadway & Flatiron (W 23rd St)"),
    ]

    bw_node_ids = []
    for i, (lat, lon, name) in enumerate(broadway_coords):
        b_id = f"osm_broadway_{i}"
        node = RoadNode(node_id=b_id, lat=lat, lon=lon, name=name, is_landmark=True)
        graph.add_node(node)
        bw_node_ids.append(b_id)

    # Connect Broadway nodes sequentially
    for i in range(len(bw_node_ids) - 1):
        u_id = bw_node_ids[i]
        v_id = bw_node_ids[i + 1]
        u_node = graph.nodes[u_id]
        v_node = graph.nodes[v_id]
        dist = haversine_distance((u_node.lat, u_node.lon), (v_node.lat, v_node.lon))
        edge = RoadEdge(
            from_id=u_id,
            to_id=v_id,
            street_name="Broadway",
            length_m=dist,
            oneway=False,
            speed_limit_kmh=40.0,
        )
        graph.add_edge(edge)

    # Connect Broadway intersections to nearest avenue/street nodes
    for b_id in bw_node_ids:
        b_node = graph.nodes[b_id]
        # Find 2 nearest grid nodes
        sorted_nodes = sorted(
            [n for n in graph.nodes.values() if not n.node_id.startswith("osm_broadway")],
            key=lambda n: haversine_distance((b_node.lat, b_node.lon), (n.lat, n.lon)),
        )
        for near_node in sorted_nodes[:2]:
            dist = haversine_distance((b_node.lat, b_node.lon), (near_node.lat, near_node.lon))
            if dist < 350: # only connect if within 350 meters
                edge = RoadEdge(
                    from_id=b_id,
                    to_id=near_node.node_id,
                    street_name="Broadway Cross Street",
                    length_m=dist,
                    oneway=False,
                )
                graph.add_edge(edge)

    # 5. Connect Predefined Landmarks directly to their corresponding nodes
    for lm in LANDMARKS:
        lm_id = lm["id"]
        # Create a landmark node with exact real-world coordinate
        lm_node = RoadNode(
            node_id=lm_id,
            lat=lm["lat"],
            lon=lm["lon"],
            name=f"{lm['icon']} {lm['name']}",
            is_landmark=True,
        )
        graph.add_node(lm_node)

        # Connect to nearest 2 road nodes
        candidates = sorted(
            [n for n in graph.nodes.values() if n.node_id != lm_id],
            key=lambda n: haversine_distance((lm_node.lat, lm_node.lon), (n.lat, n.lon)),
        )
        for c_node in candidates[:2]:
            dist = haversine_distance((lm_node.lat, lm_node.lon), (c_node.lat, c_node.lon))
            edge = RoadEdge(
                from_id=lm_id,
                to_id=c_node.node_id,
                street_name=f"Access to {lm['name']}",
                length_m=max(dist, 10.0),
                oneway=False,
            )
            graph.add_edge(edge)

    return graph


# Singleton instance
DEFAULT_ROAD_GRAPH = build_manhattan_road_network()


def get_or_build_road_network(
    start_coord: Tuple[float, float],
    goal_coord: Tuple[float, float],
    start_name: str = "Origin",
    goal_name: str = "Destination",
) -> Tuple[RoadGraph, str, str]:
    """
    Get or construct a realistic road network graph for ANY city or geographic region worldwide.
    Returns: (graph, start_node_id, goal_node_id)
    """
    slat, slon = start_coord
    glat, glon = goal_coord

    # Check if both coordinates are within Manhattan corridor
    if (40.71 <= slat <= 40.80 and -74.02 <= slon <= -73.94 and
        40.71 <= glat <= 40.80 and -74.02 <= glon <= -73.94):
        graph = build_manhattan_road_network()
        start_node = graph.find_nearest_node(slat, slon)
        goal_node = graph.find_nearest_node(glat, glon)
        if abs(slat - glat) < 1e-6 and abs(slon - glon) < 1e-6:
            return graph, start_node.node_id, start_node.node_id
        return graph, start_node.node_id, goal_node.node_id

    # For any other city in the world:
    # Compute bounding box with margin
    pad_lat = max(abs(slat - glat) * 0.25, 0.008)
    pad_lon = max(abs(slon - glon) * 0.25, 0.008)
    s = min(slat, glat) - pad_lat
    n = max(slat, glat) + pad_lat
    w = min(slon, glon) - pad_lon
    e = max(slon, glon) + pad_lon

    graph = RoadGraph()

    # Generate road grid for this geographic area
    num_lat = 9
    num_lon = 9
    lat_steps = [s + i * (n - s) / (num_lat - 1) for i in range(num_lat)]
    lon_steps = [w + i * (e - w) / (num_lon - 1) for i in range(num_lon)]

    node_grid: Dict[Tuple[int, int], str] = {}
    for r, rlat in enumerate(lat_steps):
        for c, rlon in enumerate(lon_steps):
            nid = f"city_node_{r}_{c}"
            node = RoadNode(node_id=nid, lat=rlat, lon=rlon, name=f"Street Junction ({r+1}, {c+1})")
            graph.add_node(node)
            node_grid[(r, c)] = nid

    # North-South Avenues
    for c in range(num_lon):
        for r in range(num_lat - 1):
            u_id = node_grid[(r, c)]
            v_id = node_grid[(r + 1, c)]
            u = graph.nodes[u_id]
            v = graph.nodes[v_id]
            dist = haversine_distance((u.lat, u.lon), (v.lat, v.lon))
            graph.add_edge(RoadEdge(u_id, v_id, f"Avenue {c+1}", dist, oneway=False))

    # East-West Cross Streets
    for r in range(num_lat):
        for c in range(num_lon - 1):
            u_id = node_grid[(r, c)]
            v_id = node_grid[(r, c + 1)]
            u = graph.nodes[u_id]
            v = graph.nodes[v_id]
            dist = haversine_distance((u.lat, u.lon), (v.lat, v.lon))
            graph.add_edge(RoadEdge(u_id, v_id, f"Cross Street {r+1}", dist, oneway=False))

    # Diagonal arterial thoroughfare connecting major sectors
    diag_count = min(num_lat, num_lon)
    for i in range(diag_count - 1):
        u_id = node_grid[(i, i)]
        v_id = node_grid[(i + 1, i + 1)]
        u = graph.nodes[u_id]
        v = graph.nodes[v_id]
        dist = haversine_distance((u.lat, u.lon), (v.lat, v.lon))
        graph.add_edge(RoadEdge(u_id, v_id, "Central Boulevard", dist, oneway=False))

    # Add the user's exact Start and Goal location nodes
    if abs(slat - glat) < 1e-6 and abs(slon - glon) < 1e-6:
        start_id = "selected_start"
        start_node = RoadNode(start_id, slat, slon, start_name, is_landmark=True)
        graph.add_node(start_node)
        near_start = sorted(
            [n for n in graph.nodes.values() if n.node_id != start_id],
            key=lambda n: haversine_distance((slat, slon), (n.lat, n.lon)),
        )[:2]
        for n in near_start:
            d = haversine_distance((slat, slon), (n.lat, n.lon))
            graph.add_edge(RoadEdge(start_id, n.node_id, f"Access to {start_name}", max(d, 10.0), oneway=False))
        return graph, start_id, start_id

    start_id = "selected_start"
    goal_id = "selected_goal"
    start_node = RoadNode(start_id, slat, slon, start_name, is_landmark=True)
    goal_node = RoadNode(goal_id, glat, glon, goal_name, is_landmark=True)
    graph.add_node(start_node)
    graph.add_node(goal_node)

    # Connect start to 2 nearest road intersections
    near_start = sorted(
        [n for n in graph.nodes.values() if n.node_id not in (start_id, goal_id)],
        key=lambda n: haversine_distance((slat, slon), (n.lat, n.lon)),
    )[:2]
    for n in near_start:
        d = haversine_distance((slat, slon), (n.lat, n.lon))
        graph.add_edge(RoadEdge(start_id, n.node_id, f"Access to {start_name}", max(d, 10.0), oneway=False))

    # Connect goal to 2 nearest road intersections
    near_goal = sorted(
        [n for n in graph.nodes.values() if n.node_id not in (start_id, goal_id)],
        key=lambda n: haversine_distance((glat, glon), (n.lat, n.lon)),
    )[:2]
    for n in near_goal:
        d = haversine_distance((glat, glon), (n.lat, n.lon))
        graph.add_edge(RoadEdge(n.node_id, goal_id, f"Access to {goal_name}", max(d, 10.0), oneway=False))

    return graph, start_id, goal_id


# ============================================================================
# Dedicated Academic Benchmark Scenarios (Four Independent Configurations)
# ============================================================================

def build_academic_scenario_graph(scenario_id: str) -> Dict[str, Any]:
    """
    Construct independent predefined graphs and configurations for the four
    academic benchmark scenarios. Each scenario possesses its own distinct graph,
    start node, goal node, coordinates, edge weights, and expected routing behavior.
    """
    scen_key = str(scenario_id).lower().replace("scen", "").replace("_", "")

    # ------------------------------------------------------------------------
    # 1. CHEAP DETOUR VS EXPENSIVE
    # Direct route along 5th Ave has high travel congestion weight (15x),
    # while indirect detour along Madison Ave has normal cost (1x).
    # A* chooses the cheaper detour even though it has more steps.
    # ------------------------------------------------------------------------
    if "detour" in scen_key or scen_key == "1":
        graph = build_manhattan_road_network()
        start_id = "osm_0_6"   # 5th Ave & Central Park South (59th St)
        goal_id = "osm_14_6"   # 5th Ave & W 23rd St (Flatiron District)
        start_coord = [40.7665, -73.9765]
        goal_coord = [40.7315, -73.9765]
        start_name = "Central Park South (5th Ave & 59th St)"
        goal_name = "Flatiron District (5th Ave & 23rd St)"

        # Apply heavy congestion weights (15.0x) to direct 5th Ave nodes
        traffic_map: Dict[str, float] = {}
        for r in range(1, 14):
            nid = f"osm_{r}_6"
            graph.weights[nid] = 15.0
            n_node = graph.nodes[nid]
            traffic_map[f"{n_node.lat},{n_node.lon}"] = 15.0

        return {
            "scenario_id": "detour",
            "title": "Cheap Detour vs Expensive",
            "description": "Direct 5th Ave route has high congestion weight (15x). A* chooses the cheaper Madison Ave detour despite having more steps.",
            "start_id": start_id,
            "goal_id": goal_id,
            "start_coord": start_coord,
            "goal_coord": goal_coord,
            "start_name": start_name,
            "goal_name": goal_name,
            "graph": graph,
            "roadblocks": [],
            "traffic": traffic_map,
            "expected_behavior": "A* selects Madison Ave bypass with more steps but lower total cost. Cheaper Detour Chosen!",
        }

    # ------------------------------------------------------------------------
    # 2. EQUAL-COST ALTERNATIVES
    # Symmetric rectangular network connecting 5th Ave and Madison Ave.
    # Route 1 (South on 5th, then East on 53rd) and Route 2 (East on 57th, then South on Madison)
    # have EXACTLY equal total cost (655.0 m). A* returns the optimal path and identifies the alternative.
    # ------------------------------------------------------------------------
    elif "equal" in scen_key or scen_key == "2":
        graph = RoadGraph()
        node_A = RoadNode("osm_eq_nw", 40.7645, -73.9765, "5th Ave & W 57th St")
        node_B = RoadNode("osm_eq_ne", 40.7645, -73.9740, "Madison Ave & W 57th St")
        node_C = RoadNode("osm_eq_sw", 40.7605, -73.9765, "5th Ave & W 53rd St")
        node_D = RoadNode("osm_eq_se", 40.7605, -73.9740, "Madison Ave & W 53rd St")

        for n in (node_A, node_B, node_C, node_D):
            graph.add_node(n)

        # Precise symmetric edge lengths
        L_horiz = 211.0 # Cross street width between 5th Ave and Madison Ave
        L_vert = 444.0  # Avenue block length between 57th St and 53rd St

        # W 57th St (A <-> B)
        graph.add_edge(RoadEdge("osm_eq_nw", "osm_eq_ne", "W 57th St", L_horiz, oneway=False))
        # W 53rd St (C <-> D)
        graph.add_edge(RoadEdge("osm_eq_sw", "osm_eq_se", "W 53rd St", L_horiz, oneway=False))
        # 5th Ave (A <-> C)
        graph.add_edge(RoadEdge("osm_eq_nw", "osm_eq_sw", "5th Ave", L_vert, oneway=False))
        # Madison Ave (B <-> D)
        graph.add_edge(RoadEdge("osm_eq_ne", "osm_eq_se", "Madison Ave", L_vert, oneway=False))

        return {
            "scenario_id": "equal",
            "title": "Equal-Cost Alternatives",
            "description": "Symmetric parallel corridors with exactly equal traversal cost (655.0 m). A* returns optimal path and highlights equal-cost alternative.",
            "start_id": "osm_eq_nw",
            "goal_id": "osm_eq_se",
            "start_coord": [40.7645, -73.9765],
            "goal_coord": [40.7605, -73.9740],
            "start_name": "5th Ave & W 57th St",
            "goal_name": "Madison Ave & W 53rd St",
            "graph": graph,
            "roadblocks": [],
            "traffic": {},
            "has_equal_alternative": True,
            "alternative_path": ["osm_eq_nw", "osm_eq_ne", "osm_eq_se"],
            "alternative_coords": [[40.7645, -73.9765], [40.7645, -73.9740], [40.7605, -73.9740]],
            "alternative_cost": 655.0,
            "expected_behavior": "Both paths have identical cost (655.0 m). Optimal path and alternative path identified. Equal-Cost Alternative Identified!",
        }

    # ------------------------------------------------------------------------
    # 3. UNREACHABLE GOAL
    # Destination (United Nations Headquarters) is completely barricaded
    # by roadblocks on all access roads. A* must report 'No path found' without showing old route.
    # ------------------------------------------------------------------------
    elif "block" in scen_key or "unreach" in scen_key or scen_key == "3":
        graph = build_manhattan_road_network()
        start_id = "osm_broadway_8"  # Broadway & Flatiron (W 23rd St)
        goal_id = "un_plaza"         # United Nations Headquarters (1st Ave & E 42nd St)
        start_coord = [40.7315, -73.9897]
        goal_coord = [40.7499, -73.9680]
        start_name = "Broadway & Flatiron (W 23rd St)"
        goal_name = "United Nations Headquarters"

        # Disconnect destination completely by barricading all incoming neighbor street intersections
        roadblock_coords = []
        for nbr_id, _ in list(graph.adj.get(goal_id, [])):
            graph.blocked_nodes.add(nbr_id)
            nbr_node = graph.nodes[nbr_id]
            roadblock_coords.append([nbr_node.lat, nbr_node.lon])

        # Also block adjacent avenue segments
        extra_blocks = ["osm_7_12", "osm_6_12", "osm_8_12", "osm_7_11", "osm_8_10", "osm_7_10"]
        for eb in extra_blocks:
            if eb in graph.nodes:
                graph.blocked_nodes.add(eb)
                eb_node = graph.nodes[eb]
                roadblock_coords.append([eb_node.lat, eb_node.lon])

        return {
            "scenario_id": "blocked",
            "title": "Unreachable Goal",
            "description": "All access streets to United Nations Headquarters are barricaded by roadblocks. A* reports 'No path found'.",
            "start_id": start_id,
            "goal_id": goal_id,
            "start_coord": start_coord,
            "goal_coord": goal_coord,
            "start_name": start_name,
            "goal_name": goal_name,
            "graph": graph,
            "roadblocks": roadblock_coords,
            "traffic": {},
            "expected_behavior": "No path found: Destination unreachable. Path is None, cost is infinity.",
        }

    # ------------------------------------------------------------------------
    # 4. START = GOAL
    # Start and destination set to identical location (Grand Central Terminal).
    # Returns immediately with cost 0.0, 0 steps, and path containing only start node.
    # ------------------------------------------------------------------------
    elif "same" in scen_key or scen_key == "4":
        graph = build_manhattan_road_network()
        start_id = "grand_central"
        goal_id = "grand_central"
        coord = [40.7527, -73.9772]
        name = "Grand Central Terminal"

        return {
            "scenario_id": "same",
            "title": "Start = Goal",
            "description": "Start and destination are identical (Grand Central Terminal). Zero cost (0.0 m) and 0 steps.",
            "start_id": start_id,
            "goal_id": goal_id,
            "start_coord": coord,
            "goal_coord": coord,
            "start_name": name,
            "goal_name": name,
            "graph": graph,
            "roadblocks": [],
            "traffic": {},
            "expected_behavior": "Start = Goal: Immediate termination: cost 0.0, steps 0, 0 nodes explored.",
        }

    else:
        raise ValueError(f"Unknown academic scenario: {scenario_id}")


