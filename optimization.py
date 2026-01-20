# Subject       : MECS1033 Advanced Artificial Intelligence
# Task          : Assignment 3 - Optimization
# Script name   : optimization.py
# Description   : Optimization using Ant Colony Optimization
# Author        : MEC255017 - luqmannaim@graduate.utm.my

from __future__ import annotations

import math
import time
import folium
import numpy as np

# ACO solver downloaded from github: https://github.com/OptiNobles/tsp-ant-colony
from aco.AntColonyOptimizer import AntColonyOptimizer

###################
# PLACES TO VISIT #
###################

# Approximate coordiantes for Open Street Map
POINTS_OF_INTEREST: dict[str, tuple[float, float]] = {
    "KLCC": (3.1579, 101.7123),
    "Menara Kuala Lumpur": (3.1528, 101.7037),
    "Merdeka Square": (3.1486, 101.6932),
    "Menara TRX": (3.1420, 101.7199),
    "Stadium Merdeka": (3.1397, 101.7000),
    "National Mosque of Malaysia": (3.1419, 101.6910),
    "Berjaya Times Square": (3.1420, 101.7119),
    "Malaysia National Museum": (3.1379, 101.6870),
}
START_POINT = "KLCC" # Start and end here

########################
# DISTANCE CALCULATION #
########################

def compute_haversine(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Distance between two points in kilometers"""

    # Haversine formula
    r = 6371.0                              # Earth radius in km
    phi1 = math.radians(a_lat)              # Latitude of point A in radians
    phi2 = math.radians(b_lat)              # Latitude of point B in radians
    dphi = math.radians(b_lat - a_lat)      # Delta latitude in radians
    dlambda = math.radians(b_lon - a_lon)   # Delta longitude in radians

    # Haversine calculation
    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    # Return distance in km
    return 2 * r * math.asin(math.sqrt(h))

def compute_distance_matrix(poi_names: list[str]) -> np.ndarray:
    """Create a matrix of straight-line distances in km for points of interest"""

    # Create empty distance matrix
    n = len(poi_names)
    matrix = np.zeros((n, n), dtype=float)

    # Fill in distances for each point
    for i, a in enumerate(poi_names):
        a_lat, a_lon = POINTS_OF_INTEREST[a]
        for j, b in enumerate(poi_names):
            if i == j:
                continue
            b_lat, b_lon = POINTS_OF_INTEREST[b]
            matrix[i, j] = compute_haversine(a_lat, a_lon, b_lat, b_lon)

    # Return distance matrix
    return matrix

def convert_coor_to_dist(city_names: list[str]) -> list[list[float]]:
    """Convert latitude and longitude into xy distance in km"""

    # Extract latitudes and longitudes
    lats = np.array([POINTS_OF_INTEREST[name][0] for name in city_names], dtype=float)
    lons = np.array([POINTS_OF_INTEREST[name][1] for name in city_names], dtype=float)

    # Get the middle latitude
    lat0 = float(np.mean(lats))

    # Latitude to y conversion
    # Approximation of km per latitude -> 1 deg lat = 111.32 km
    km_lat = 111.32
    y = (lats - lat0) * km_lat

    # Longitude to x conversion
    # Approximation of km per longitude -> 1 deg lon = 111.32 * cos(lat0) km
    km_lon = 111.32 * math.cos(math.radians(lat0))
    x = (lons - float(np.mean(lons))) * km_lon
    
    # Return list of xy points
    return np.column_stack([x, y]).tolist()

####################
# HELPER FUNCTIONS #
####################

def rotate_cycle(cycle: list[str], start_point: str) -> list[str]:
    """Rotate a cycle so it begins at start point"""

    # Find index of start point
    idx = cycle.index(start_point)

    # Rotate cycle to start from start point
    return cycle[idx:] + cycle[:idx]

##################################
# RUN ANT COLONY OPTIMIZER (ACO) #
##################################

def run_aco(
    point_names: list[str],
    distance_matrix_km: np.ndarray,
    start_point: str,
    runs: int,
    top_k: int,
    ants: int,
    iterations: int,
    evaporation_rate: float,
    intensification: float,
    alpha: float,
    beta: float,
    beta_evaporation_rate: float,
    choose_best: float,
    conv_crit: int,
) -> tuple[list[list[str]], list[float]]:
    """Find best route using ACO"""

    # Convert lat and lon to xy distances in km
    points = convert_coor_to_dist(point_names)

    # Keep best score for each unique route
    best_by_route: dict[tuple[str, ...], float] = {}

    # Try multiple random runs
    for seed in range(runs):

        # Set random seed
        np.random.seed(seed)

        # Create ACO solver
        aco = AntColonyOptimizer(
            ants=ants,
            evaporation_rate=evaporation_rate,
            intensification=intensification,
            alpha=alpha,
            beta=beta,
            beta_evaporation_rate=beta_evaporation_rate,
            choose_best=choose_best,
        )

        # Fit ACO to points
        aco.fit(
            points,
            iterations=iterations,
            mode='min',
            conv_crit=conv_crit,
            verbose=(seed == 0),
        )

        # Get best route found
        route_idx, _, _, _ = aco.get_result()
        route = [point_names[i] for i in route_idx]

        # Make sure we start and end at the start point
        cycle = route[:-1]
        cycle = rotate_cycle(cycle, start_point)
        route = cycle + [start_point]

        # Calcuate cost of this route
        cost = 0.0
        for i in range(len(route) - 1):
            a = point_names.index(route[i])
            b = point_names.index(route[i + 1])
            cost += float(distance_matrix_km[a, b])

        # Keep best score for each unique route
        key = tuple(route)
        if key not in best_by_route or cost < best_by_route[key]:
            best_by_route[key] = cost

    # Rank best routes found
    ranked = sorted(best_by_route.items(), key=lambda kv: kv[1])
    best_routes = [list(t) for t, _ in ranked[:top_k]]
    best_costs = [c for _, c in ranked[:top_k]]

    # Return best routes and their costs
    return best_routes, best_costs

################
# MAP PLOTTING #
################

def plot_html_map(route: list[str], filename: str = "aco_best_route.html") -> str:
    """Plot HTML map of the route using in OpenStreetMap"""

    # Extract latitudes and longitudes
    lats = [POINTS_OF_INTEREST[name][0] for name in route]
    lons = [POINTS_OF_INTEREST[name][1] for name in route]

    # Create map centered around average location
    center = (float(np.mean(lats)), float(np.mean(lons)))

    # Create open street map using folium
    m = folium.Map(location=center, zoom_start=13, control_scale=True)

    # Add markers for each point
    for i, name in enumerate(route[:-1]):
        lat, lon = POINTS_OF_INTEREST[name]
        folium.Marker(
            location=(lat, lon),
            popup=f"{i}: {name}",
            tooltip=f"{i}: {name}",
        ).add_to(m)

    # Draw a straight line from point to point
    folium.PolyLine(
        locations=[(POINTS_OF_INTEREST[name][0], POINTS_OF_INTEREST[name][1]) for name in route],
        color="blue",
        weight=5,
        opacity=0.85,
    ).add_to(m)

    # Save map to HTML file
    m.save(filename)

    # Return filename
    return filename

#################
# MAIN FUNCTION #
#################

def main() -> None:

    # Step 1: Load points of interest
    point_names = list(POINTS_OF_INTEREST.keys())
    print("\n[Step 1/5] Loading points of interest")
    print(", ".join(point_names))

    # Step 2: Compute distance matrix
    print("\n[Step 2/5] Computing straight line distances")
    t0 = time.time()
    distance_matrix_km = compute_distance_matrix(point_names)
    print(f"Computed in {time.time() - t0:.3f}s")

    # Step 3: Run ACO to find best route
    print("\n[Step 3/5] Running ACO")
    routes, costs = run_aco(
        point_names=point_names,                # Places of interest
        distance_matrix_km=distance_matrix_km,  # Distance matrix table
        start_point=START_POINT,                # Starting point
        runs=8,                                 # 8 runs. Less runs for faster results, more for better quality but slower
        top_k=1,                                # 1 best route. Can be increased to take more best routes
        ants=300,                               # 300 ants. Less ants for faster results, more for better quality but slower
        iterations=200,                         # 200 iters. Lower iters for faster results, higher for better quality but slower
        evaporation_rate=0.20,                  # 20% pheromone evaporation. Lower for faster results, higher for more exploration but slower
        intensification=0.30,                   # 30% pheromone intensification. Lower for more exploration but slower learning, higher for faster learning
        alpha=1.00,                             # 1.0 ant pheromone trust. Higher value puts more trust in pheromone trails, lower value puts more trust in distance
        beta=2.00,                              # 2.0 ant distance trust. Higher value puts more trust in distance, lower value puts more trust in pheromone trails
        beta_evaporation_rate=0.0,              # 0.0 beta evaporation rate. Higher value reduces distance influence, lower value keeps distance influence constant
        choose_best=0.10,                       # 10% of the time. Higher value makes ants choose best path more often, lower value makes ants explore more
        conv_crit=25,                           # 25 iters for convergence. Lower value for faster results, higher for better quality but slower
    )

    # Step 4: Show best route
    print("\n[Step 4/5] Fastest route found!")
    route, sec = routes[0], costs[0]
    print(" -> ".join(route))
    print(f"Total straight line distance: {sec:.2f} km")

    # Step 5: Plot best route on map
    print("\n[Step 5/5] Plotting best route on map")
    filename =plot_html_map(route, filename="aco_best_route.html")
    print(f"Map saved to {filename}")

    print("\nDone!")

if __name__ == "__main__":
    main()
