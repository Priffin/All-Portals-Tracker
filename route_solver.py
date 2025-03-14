from typing import List, Tuple
from abc import abstractmethod
import logging
import time
import itertools
import networkx as nx
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pulp


class APSolver:
    SH_BOUNDS = [
        (1280, 2816),
        (4352, 5888),
        (7424, 8960),
        (10496, 12032),
        (13568, 15104),
        (16640, 18176),
        (19712, 21248),
        (22784, 24320),
    ]
    STRONGHOLDS_PER_RING = [3, 6, 10, 15, 21, 28, 36, 10]

    @classmethod
    @abstractmethod
    def solve(cls, first_8_strongholds: List[Tuple[float, float]]) -> List[List]:
        """
        Solve an All Portals path given a stronghold measured from each ring

        Args:
            first_8_strongholds: List of coordinates for the first 8 strongholds

        Returns:
            Detailed and ordered information for each stronghold on the path
        """

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @classmethod
    def get_stronghold_ring(cls, coords: Tuple[float, float]) -> int:
        """Determine the ring of a stronghold based on its distance from origin."""
        dist = cls.euclidean_distance((0, 0), coords)
        for ring, bounds in enumerate(cls.SH_BOUNDS):
            if bounds[0] < dist < bounds[1]:
                return ring + 1
        return 0

    @classmethod
    def estimate_stronghold_locations(
        cls, first_8_strongholds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Predict locations of additional strongholds using the first 8 points.

        Args:
            first_8_strongholds: List of coordinates for the first 8 strongholds

        Returns:
            List of estimated stronghold coordinates
        """
        # path should start from the 7th ring measured stronghold
        points = [first_8_strongholds[-2]]
        for ring, stronghold_count in enumerate(cls.STRONGHOLDS_PER_RING):
            x, z = first_8_strongholds[ring]
            # estimate each stronghold to be in the center of the ring
            magnitude = sum(cls.SH_BOUNDS[ring]) // 2
            base_angle = np.arctan2(z, x)

            for i in range(1, stronghold_count):
                # strongholds are roughly equally spaced around the ring
                angle = base_angle + (2 * np.pi * i / stronghold_count)
                estimate_x = round(magnitude * np.cos(angle))
                estimate_z = round(magnitude * np.sin(angle))
                points.append((estimate_x, estimate_z))

        return points


class PuLPRingStarSolver(APSolver):
    """AP Solver using MILP solver & ring-star formulation
    https://gist.github.com/Lincoln-LM/d8985c2074861adc4f27357dfcbe21ae"""

    # useful constant indexes
    # dummy nodes
    DUMMY_ROOT = 0
    DUMMY_RING_1_1 = DUMMY_ROOT + 1
    DUMMY_RING_1_2 = DUMMY_RING_1_1 + 1

    # 7th ring measured stronghold
    REAL_ROOT = DUMMY_RING_1_2 + 1

    # first ring strongholds
    REAL_RING_1_1 = REAL_ROOT + 1
    REAL_RING_1_2 = REAL_RING_1_1 + 1

    @staticmethod
    def delta_edges(set_of_points, all_points):
        """Defines the set of all undirected edges between all points within the set
        and all points outside of the set"""
        for point1 in set_of_points:
            for point2 in all_points:
                if point2 not in set_of_points:
                    yield point1, point2

    @classmethod
    def graph_distance(cls, graph, point1_index, point2_index):
        """Euclidean distance between two points in the graph"""
        # dummy points are always free
        if point1_index < cls.REAL_ROOT or point2_index < cls.REAL_ROOT:
            return 0
        return cls.euclidean_distance(
            graph.nodes[point1_index]["pos"], graph.nodes[point2_index]["pos"]
        )

    @classmethod
    def solve(cls, first_8_strongholds: List[Tuple[float, float]]):
        logger = logging.getLogger("milp_solver")
        points = cls.estimate_stronghold_locations(first_8_strongholds)
        graph = nx.Graph()
        for idx, sh in enumerate(points, start=cls.REAL_ROOT):
            graph.add_node(idx, pos=sh)

        NODE_COUNT = len(graph.nodes) + cls.REAL_ROOT
        POINTS_IDX = list(range(NODE_COUNT))
        graph.add_node(0, pos=(0, 0))

        elapsed_time = 0
        iteration = -1
        problem = pulp.LpProblem("RingStarProblem", pulp.LpMinimize)

        # define binary variables x_i_j that equal 1
        # if and only if the edge i <-> j is part of the cycle
        cycle_vars = {
            point1: {
                point2: pulp.LpVariable(f"x_{point1}_{point2}", cat=pulp.LpBinary)
                for point2 in POINTS_IDX[point1 + 1 :]
            }
            for point1 in POINTS_IDX
        }
        # define binary variables y_i_j that equal 1
        # if and only if the node v_i is assigned to the node v_j as an arm
        # for only nodes v_i on the cycle y_i_i == 1 i.e. the node is assigned to itself
        assignment_vars = {
            point1: {
                point2: pulp.LpVariable(f"y_{point1}_{point2}", cat=pulp.LpBinary)
                for point2 in POINTS_IDX
            }
            for point1 in POINTS_IDX
        }

        # enforce that each node has exactly 2 cycle connections
        # if and only if that node is on the cycle
        # and otherwise has 0 cycle connections
        for point1 in POINTS_IDX:
            problem += (
                pulp.lpSum(
                    cycle_vars[min(point1, point2)][max(point1, point2)]
                    for point2 in POINTS_IDX
                    if point2 != point1
                )
                == 2 * assignment_vars[point1][point1]
            )

        # enforce that every real node is assigned to exactly 1 other node
        # this means nodes on the cycle cannot be assigned to any other nodes
        # (they are assigned to themselves)
        # and that nodes external to the cycle are only assigned to 1 other node
        for point1 in POINTS_IDX[cls.REAL_ROOT :]:
            problem += (
                pulp.lpSum(assignment_vars[point1][point2] for point2 in POINTS_IDX)
                == 1
            )

        # enforce that the first node and the dummy nodes are only assigned to themselves
        # (root is always on the cycle)
        problem += assignment_vars[cls.DUMMY_ROOT][cls.DUMMY_ROOT] == 1
        problem += assignment_vars[cls.DUMMY_RING_1_1][cls.DUMMY_RING_1_1] == 1
        problem += assignment_vars[cls.DUMMY_RING_1_2][cls.DUMMY_RING_1_2] == 1

        # dummy nodes & the root are implicitly never going to be assigned to other nodes
        # but this can be an added condition here if it helps solving

        # enforce that nothing is ever assigned to the dummy nodes
        for point in POINTS_IDX[cls.REAL_ROOT :]:
            for dummy_point in (cls.DUMMY_ROOT, cls.DUMMY_RING_1_1, cls.DUMMY_RING_1_2):
                problem += assignment_vars[point][dummy_point] == 0

        # enforce that the first ring nodes cannot connect to each other
        problem += cycle_vars[cls.REAL_RING_1_1][cls.REAL_RING_1_2] == 0
        problem += cycle_vars[cls.DUMMY_RING_1_1][cls.REAL_RING_1_2] == 0
        problem += cycle_vars[cls.DUMMY_RING_1_2][cls.REAL_RING_1_1] == 0

        # enforce that the fake nodes always connect to their real counterpart
        problem += cycle_vars[cls.DUMMY_ROOT][cls.REAL_ROOT] == 1
        problem += cycle_vars[cls.DUMMY_RING_1_1][cls.REAL_RING_1_1] == 1
        problem += cycle_vars[cls.DUMMY_RING_1_2][cls.REAL_RING_1_2] == 1

        # define the objective function
        # minimize the sum of the distances between all edges on the cycle (cycle length)
        # and the sum of the distances between all edges assigned to those on the cycle
        # from their assigned cycle node (total arm length)
        problem += pulp.lpSum(
            cls.graph_distance(graph, point1, point2) * cycle_vars[point1][point2]
            for point1 in POINTS_IDX
            for point2 in POINTS_IDX[point1 + 1 :]
        ) + pulp.lpSum(
            cls.graph_distance(graph, point1, point2) * assignment_vars[point1][point2]
            for point1 in POINTS_IDX
            for point2 in POINTS_IDX
            if point2 != point1
        )
        running = True
        start_time = time.time()
        last_solution = None
        while running:
            problem.solve(pulp.PULP_CBC_CMD(msg=0))
            graph.clear_edges()
            cycle_edges = {}
            assignments = {}
            for point1 in POINTS_IDX:
                for point2 in POINTS_IDX[point1 + 1 :]:
                    if cycle_vars[point1][point2].value() == 1:
                        cycle_edges[point1] = cycle_edges.get(point1, []) + [point2]
                        cycle_edges[point2] = cycle_edges.get(point2, []) + [point1]
                        p1 = point1
                        p2 = point2

                        if p1 == cls.DUMMY_ROOT:
                            p1 = cls.REAL_ROOT
                        elif p1 == cls.DUMMY_RING_1_1:
                            p1 = cls.REAL_RING_1_1
                        elif p1 == cls.DUMMY_RING_1_2:
                            p1 = cls.REAL_RING_1_2
                        if p2 == cls.DUMMY_ROOT:
                            p2 = cls.REAL_ROOT
                        elif p2 == cls.DUMMY_RING_1_1:
                            p2 = cls.REAL_RING_1_1
                        elif p2 == cls.DUMMY_RING_1_2:
                            p2 = cls.REAL_RING_1_2

                        if p1 != p2:
                            graph.add_edge(p1, p2)
            for point1 in POINTS_IDX:
                for point2 in POINTS_IDX:
                    if assignment_vars[point1][point2].value() == 1:
                        if point1 != point2:
                            graph.add_edge(point1, point2)
                        assignments[point1] = point2
            # find all subcycles including the nodes assigned to them
            unvisited_nodes = set(POINTS_IDX)
            cycles = []
            while cycle_edges:
                start = list(cycle_edges.keys())[0]
                cycle = [start]
                cycle_assignments = []
                while True:
                    unvisited_nodes.discard(cycle[-1])
                    for assigned_node, assigned_to_node in assignments.items():
                        if assigned_node == assigned_to_node:
                            continue
                        if assigned_to_node == cycle[-1]:
                            unvisited_nodes.discard(assigned_node)
                            cycle_assignments.append(assigned_node)
                    connected_edges = cycle_edges.pop(cycle[-1])
                    next_point = next(
                        (p for p in connected_edges if p not in cycle), start
                    )
                    cycle.append(next_point)
                    if next_point == start:
                        break
                cycles.append((cycle, cycle_assignments))
            # nodes that arent assigned to a cycle node can be considered on their own
            for node in unvisited_nodes:
                cycles.append(([], [node, assignments[node]]))
            # validate directionality of dummy & real nodes
            invalid_paths = []
            all_paths_valid = True
            for cycle_, _ in cycles:
                for start, end in (
                    (
                        (cls.REAL_RING_1_1, cls.DUMMY_RING_1_1),
                        (cls.DUMMY_RING_1_2, cls.REAL_RING_1_2),
                    ),
                    (
                        (cls.REAL_RING_1_1, cls.DUMMY_RING_1_1),
                        (cls.DUMMY_ROOT, cls.REAL_ROOT),
                    ),
                    (
                        (cls.REAL_RING_1_2, cls.DUMMY_RING_1_2),
                        (cls.DUMMY_ROOT, cls.REAL_ROOT),
                    ),
                ):
                    if start[0] not in cycle_ or end[0] not in cycle_:
                        continue
                    path_is_valid = False
                    cycle = itertools.cycle(cycle_)
                    invalid_path = [start[0]]
                    for point in cycle:
                        if point == start[0]:
                            break
                    search_point = end[0]
                    invalid_point = end[1]
                    # dummy and real nodes must alternate throughout the cycle
                    # D1->R1->...->D2->R2 is valid
                    # D1->R1->...->R2->D2 is invalid
                    # R1->D1->...->D2->R2 is invalid
                    # R1->D1->...->R2->D2 is valid
                    for point in cycle:
                        invalid_path.append(point)
                        if point == start[1]:
                            invalid_path = [start[1]]
                            search_point = end[1]
                            invalid_point = end[0]
                        elif point == invalid_point:
                            path_is_valid = False
                            break
                        elif point == search_point:
                            path_is_valid = True
                            break
                    if not path_is_valid:
                        all_paths_valid = False
                        invalid_paths.append(invalid_path)
                        # lazily apply invalidation constraint
                        # TODO: is this constraint optimal/always correct
                        # it is essentially trying to invalidate the specific group of nodes from
                        # forming a contiguous path by requiring > 2 delta edges
                        problem += (
                            pulp.lpSum(
                                cycle_vars[min(point1, point2)][max(point1, point2)]
                                for point1, point2 in cls.delta_edges(
                                    set(invalid_path), POINTS_IDX
                                )
                            )
                            >= 3
                        )
            # if there is only one cycle & it is valid then an optimal solution has been found
            # and we can stop
            if len(cycles) == 1:
                logger.info("Found single cycle solution")
                if all_paths_valid:
                    logger.info("Found valid single cycle solution")
                    running = False
                else:
                    logger.info("Invalid due to invalid_paths=%s", str(invalid_paths))

            # lazily apply the subcycle elimination constraints on every subcycle found
            for cycle, cycle_assignments in cycles:
                if cls.DUMMY_ROOT in cycle:
                    continue
                subcycle_set = set(cycle) | set(cycle_assignments)
                for v_i in subcycle_set:
                    problem += pulp.lpSum(
                        cycle_vars[min(point1, point2)][max(point1, point2)]
                        for point1, point2 in cls.delta_edges(subcycle_set, POINTS_IDX)
                    ) >= 2 * pulp.lpSum(
                        assignment_vars[v_i][v_j] for v_j in subcycle_set
                    )
            elapsed_time = time.time() - start_time
            iteration += 1
            logger.info(
                "Iteration: %d | %.2f seconds elapsed | State: %s",
                iteration,
                elapsed_time,
                "Solving" if running else "Solved",
            )
            if running and cycles == last_solution:
                logger.error("No new solution despite new constraints")
                running = False
                return []
            last_solution = cycles

        logger.info("Solved in %.2f seconds", time.time() - start_time)
        parsed_strongholds = []
        cycle, cycle_assignments = last_solution[0]
        cycle = [node for node in cycle if node >= cls.REAL_ROOT]
        cycle_assignments = {
            # very inefficient
            cycle_node: [
                assigned_node
                for assigned_node in cycle_assignments
                if assignments[assigned_node] == cycle_node
            ]
            for cycle_node in cycle
        }
        for path_idx, cycle_node in enumerate(cycle):
            last_node = cycle[path_idx - 1]
            next_node = cycle[path_idx % len(cycle)]
            next_is_origin_reset = next_node < cls.REAL_ROOT
            # don't need to draw the root node
            if cycle_node != cls.REAL_ROOT:
                line_start = graph.nodes[last_node]["pos"]
                if cycle_node in (cls.REAL_RING_1_1, cls.REAL_RING_1_2):
                    line_start = (0, 0)
                elif last_node == cls.REAL_ROOT:
                    # center drawing of first node to the ring
                    x, z = line_start
                    angle = np.arctan2(z, x)
                    # should always be 7th ring
                    magnitude = sum(cls.SH_BOUNDS[7 - 1]) // 2
                    line_start = (magnitude * np.cos(angle), magnitude * np.sin(angle))
                parsed_strongholds.append(
                    # filler data since it is not used
                    [
                        graph.nodes[cycle_node]["pos"],
                        None,
                        graph.nodes[cycle_node]["pos"],
                        line_start,
                        None,
                        "black",
                        None,
                        (
                            # leave bed before going to assigned nodes
                            1
                            if cycle_assignments[cycle_node]
                            # if next is an origin reset and there is only one assigned node
                            # then we do not need to leave the spawnpoint
                            and not (
                                next_is_origin_reset
                                and len(cycle_assignments[cycle_node] == 1)
                            )
                            else 0
                        ),
                    ]
                )
            for assigned_node in cycle_assignments[cycle_node]:
                parsed_strongholds.append(
                    [
                        graph.nodes[assigned_node]["pos"],
                        None,
                        graph.nodes[assigned_node]["pos"],
                        graph.nodes[cycle_node]["pos"],
                        None,
                        "black",
                        None,
                        2,
                    ]
                )
            # if the next node is an origin reset then remove the spawnpoint
            # at the cycle node before going to the final assigned node
            if next_is_origin_reset:
                parsed_strongholds[-1][-1] = 3
        return parsed_strongholds


class ORSolver(APSolver):
    """AP Solver utilizing OR-Tools' routing model
    https://gist.github.com/Lincoln-LM/54751387989a02297c0a136b1f02e1e5"""

    # OR-Tools requires integers
    # this is how much to scale by before truncating coordinates
    OR_SCALE_FACTOR = 10000

    # Routing strategies
    STRATEGIES = [
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    ]

    @classmethod
    def solve(cls, first_8_strongholds: List[Tuple[float, float]]) -> List[List]:
        logger = logging.getLogger("ortools_solver")
        # Estimate locations of all strongholds
        points = cls.estimate_stronghold_locations(first_8_strongholds)

        # Prepare distance and reset matrices
        distance_matrix = np.zeros((len(points), len(points)), np.float64)
        origin_reset_matrix = np.zeros((len(points), len(points)), bool)

        # Calculate distances and reset conditions
        for i, (x1, y1) in enumerate(points):
            for j, (x2, y2) in enumerate(points[1:], start=1):
                origin_distance = cls.euclidean_distance((0, 0), (x2, y2))
                real_distance = cls.euclidean_distance((x1, y1), (x2, y2))

                # 8th ring strongholds should never route through the origin
                # this is for convenience when actually following the route
                # and isn't likely to be optimal anyway
                if cls.get_stronghold_ring((x1, y1)) == 8:
                    distance_matrix[i][j] = real_distance
                else:
                    # if O->B is shorter than A->B then B should always be accessed via the origin
                    if origin_distance < real_distance:
                        origin_reset_matrix[i][j] = True
                    distance_matrix[i][j] = min(origin_distance, real_distance)

        # OR-Tools requires integers; scale and truncate
        distance_matrix = (
            np.floor(distance_matrix * cls.OR_SCALE_FACTOR).astype(int).tolist()
        )

        # Set up routing problem
        manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Find best route
        best_path = (float("inf"), None)
        for strategy in cls.STRATEGIES:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.first_solution_strategy = strategy
            search_parameters.time_limit.seconds = 12

            solution = routing.SolveWithParameters(search_parameters)
            if not solution:
                continue

            route_length, route = cls.evaluate_route(
                routing, solution, manager, distance_matrix, origin_reset_matrix
            )
            logger.info("strategy=%s route_length=%d", strategy, route_length)

            if route_length < best_path[0]:
                best_path = (route_length, route)

        route_length, route = best_path
        assert route is not None, "No solution found"

        # Generate detailed stronghold information
        return cls.process_route(points, route, origin_reset_matrix, distance_matrix)

    @staticmethod
    def evaluate_route(
        routing, solution, manager, distance_matrix, origin_reset_matrix
    ):
        """
        Evaluate the total route length and route details.

        Args:
            routing: Routing model
            solution: Routing solution
            manager: Routing index manager
            distance_matrix: Precalculated distance matrix
            origin_reset_matrix: Matrix indicating origin resets

        Returns:
            Tuple of (route_length, route)
        """
        route_length = 0
        index = routing.Start(0)
        route = [manager.IndexToNode(index)]

        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            route_length += routing.GetArcCostForVehicle(route[-1], index, 0)
            route.append(node)

        # Adjust route length based on potential optimizations
        for i, node in enumerate(route[1:-1], start=1):
            last_node = route[i - 1]
            next_node = route[i + 1]
            if 1 < i < len(route) - 2:
                any_reset = (
                    origin_reset_matrix[last_node][node]
                    or origin_reset_matrix[node][next_node]
                )
                if not any_reset:
                    if (
                        distance_matrix[last_node][next_node]
                        < distance_matrix[node][next_node]
                    ):
                        route_length -= (
                            distance_matrix[node][next_node]
                            - distance_matrix[last_node][next_node]
                        )

        return route_length, route

    @classmethod
    def process_route(cls, points, route, origin_reset_matrix, distance_matrix):
        """
        Process the route and generate detailed stronghold information.

        Args:
            points: List of all point coordinates
            route: Calculated route
            origin_reset_matrix: Matrix indicating origin resets
            distance_matrix: Precalculated distance matrix

        Returns:
            List of stronghold route details
        """
        strongholds = []
        spawn_coords = (0, 0)

        for i, node in enumerate(route[1:-1], start=1):
            last_node = route[i - 1]
            next_node = route[i + 1]

            # Determine route characteristics
            is_reset = origin_reset_matrix[last_node][node]
            is_last = next_node == 0
            coords = points[node]
            ring = cls.get_stronghold_ring(coords)

            # Set visual and route properties
            dot_colour = "purple" if is_last else "red" if is_reset else "green"
            line_colour = "green"
            marker = "*" if is_last else "o"
            line_start = points[last_node]
            line_destination = coords
            set_spawn = 0

            # Handle special reset conditions
            if is_reset:
                line_start = spawn_coords
                if strongholds:
                    strongholds[-1][7] = 2
                line_colour = "red"

            # Complex logic for spawn point management
            if len(strongholds) >= 2 and strongholds[-2][6] == "blue":
                line_colour = "blue"
                line_start = strongholds[-2][0]
                last_node = route[i - 2]

            elif 1 < i < len(route) - 2:
                any_reset = is_reset or origin_reset_matrix[node][next_node]
                if not any_reset:
                    if (
                        distance_matrix[last_node][next_node]
                        < distance_matrix[node][next_node]
                    ):
                        if strongholds:
                            strongholds[-1][6] = "blue"
                            strongholds[-1][7] = 1
                        set_spawn = 2

            # Add stronghold to route
            strongholds.append(
                [
                    coords,  # Coordinates
                    ring,  # Ring number
                    line_destination,  # Line destination
                    line_start,  # Line start
                    marker,  # Marker type
                    line_colour,  # Line color
                    dot_colour,  # Dot color
                    set_spawn,  # Spawn point flag
                ]
            )

        return strongholds
