import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import List, Tuple, Optional


class Pathfinding:
    # Constants
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
    OR_SCALE_FACTOR = 10000

    # Routing strategies
    STRATEGIES = [
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    ]

    @staticmethod
    def distance_between_points(
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @classmethod
    def get_stronghold_ring(cls, coords: Tuple[float, float]) -> int:
        """Determine the ring of a stronghold based on its distance from origin."""
        dist = cls.distance_between_points((0, 0), coords)
        for ring, bounds in enumerate(cls.SH_BOUNDS):
            if bounds[0] < dist < bounds[1]:
                return ring + 1
        return 0

    @classmethod
    def estimate_stronghold_locations(
        cls, first8: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Predict locations of additional strongholds using the first 8 points.

        Args:
            first8: List of coordinates for the first 8 strongholds

        Returns:
            List of estimated stronghold coordinates
        """
        points = [first8[-1]]  # Start with the last point from first8

        for ring, stronghold_count in enumerate(cls.STRONGHOLDS_PER_RING):
            ring_strongholds = [
                sh for sh in first8 if cls.get_stronghold_ring(sh) - 1 == ring
            ]

            if not ring_strongholds:
                continue

            x, z = ring_strongholds[0]
            magnitude = sum(cls.SH_BOUNDS[ring]) // 2
            base_angle = np.arctan2(z, x)

            # Generate additional strongholds for this ring
            for i in range(stronghold_count - 1):
                angle = base_angle + (2 * np.pi * (i + 1) / stronghold_count)
                estimate_x = round(magnitude * np.cos(angle))
                estimate_z = round(magnitude * np.sin(angle))
                points.append((estimate_x, estimate_z))

        return points

    def make_stronghold_list(self, first8: List[Tuple[float, float]]) -> List[List]:
        """
        Generate an optimized route through strongholds.

        Args:
            first8: Coordinates of the first 8 strongholds

        Returns:
            List of stronghold route information
        """
        # Estimate locations of all strongholds
        points = self.estimate_stronghold_locations(first8)
        spawn_coords = (0, 0)

        # Prepare distance and reset matrices
        distance_matrix = np.zeros((len(points), len(points)), np.float64)
        origin_reset_matrix = np.zeros((len(points), len(points)), bool)

        # Calculate distances and reset conditions
        for i, (x1, y1) in enumerate(points):
            for j, (x2, y2) in enumerate(points[1:], start=1):
                origin_distance = np.sqrt(
                    (spawn_coords[0] - x2) ** 2 + (spawn_coords[1] - y2) ** 2
                )
                real_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                # Special handling for 8th ring
                if self.get_stronghold_ring((x1, y1)) == 8:
                    distance_matrix[i][j] = real_distance
                else:
                    if origin_distance < real_distance:
                        origin_reset_matrix[i][j] = True
                    distance_matrix[i][j] = min(origin_distance, real_distance)

        # Scale and convert distance matrix
        distance_matrix = (
            np.floor(distance_matrix * self.OR_SCALE_FACTOR).astype(int).tolist()
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
        for strategy in self.STRATEGIES:
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.first_solution_strategy = strategy
            search_parameters.time_limit.seconds = 12

            solution = routing.SolveWithParameters(search_parameters)
            if not solution:
                continue

            route_length, route = self._evaluate_route(
                routing, solution, manager, distance_matrix, origin_reset_matrix
            )
            print(f"{strategy=} {route_length=}")

            if route_length < best_path[0]:
                best_path = (route_length, route)

        route_length, route = best_path
        assert route is not None, "No solution found"

        # Generate detailed stronghold information
        return self._process_route(points, route, origin_reset_matrix, distance_matrix)

    def _evaluate_route(
        self, routing, solution, manager, distance_matrix, origin_reset_matrix
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

    def _process_route(self, points, route, origin_reset_matrix, distance_matrix):
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
            ring = self.get_stronghold_ring(coords)

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
