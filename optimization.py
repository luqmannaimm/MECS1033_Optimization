
import math
import requests
import networkx as nx
import matplotlib.pyplot as plt
# --- Additional imports for map plotting ---
# Requires: pip install contextily geopandas shapely polyline
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString
import polyline


# --- Simple ACO for TSP (self-contained) ---
import random
class ACO_TSP:
    def __init__(self, G, num_ants=50, num_iterations=100, alpha=1.0, beta=3.0, rho=0.1, q=1.0):
        self.G = G
        self.nodes = list(G.nodes)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone = {(i, j): 1.0 for i in self.nodes for j in self.nodes if i != j}

    def run(self, start):
        best_path = None
        best_cost = float('inf')
        best_paths = []
        best_costs = []
        seen = set()
        for it in range(self.num_iterations):
            all_paths = []
            all_costs = []
            for ant in range(self.num_ants):
                path = [start]
                unvisited = set(self.nodes)
                unvisited.remove(start)
                while unvisited:
                    current = path[-1]
                    choices = list(unvisited)
                    probs = []
                    for nxt in choices:
                        tau = self.pheromone[(current, nxt)] ** self.alpha
                        eta = (1.0 / self.G[current][nxt]['cost']) ** self.beta
                        probs.append(tau * eta)
                    s = sum(probs)
                    if s == 0:
                        probs = [1.0 for _ in choices]
                        s = sum(probs)
                    probs = [p / s for p in probs]
                    next_city = random.choices(choices, weights=probs)[0]
                    path.append(next_city)
                    unvisited.remove(next_city)
                path.append(start)  # return to start
                cost = sum(self.G[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_cost and tuple(path) not in seen:
                    best_path = path
                    best_cost = cost
                    seen.add(tuple(path))
                    best_paths.append(path)
                    best_costs.append(cost)
            # Pheromone evaporation
            for k in self.pheromone:
                self.pheromone[k] *= (1 - self.rho)
            # Pheromone update
            for path, cost in zip(all_paths, all_costs):
                for i in range(len(path)-1):
                    self.pheromone[(path[i], path[i+1])] += self.q / cost
        # Return up to 3 best unique tours
        unique = []
        unique_costs = []
        seen = set()
        for p, c in zip(best_paths, best_costs):
            t = tuple(p)
            if t not in seen:
                unique.append(p)
                unique_costs.append(c)
                seen.add(t)
            if len(unique) == 3:
                break
        return unique, unique_costs

# ----------------------------
# 1) Define nodes (major towns) with coordinates (lat, lon)
#    You can add/remove nodes to refine realism.
# ----------------------------

# European roadtrip cities with coordinates (lat, lon)
NODES = {
    "Madrid": (40.4168, -3.7038),
    "Paris": (48.8566, 2.3522),
    "Brussels": (50.8503, 4.3517),
    "Amsterdam": (52.3676, 4.9041),
    "Frankfurt": (50.1109, 8.6821),
    "Zurich": (47.3769, 8.5417),
    "Vienna": (48.2082, 16.3738),
    "Milan": (45.4642, 9.1900),
}

START = "Madrid"
GOAL = "Madrid"  # Roadtrip: start and end at Madrid

# ----------------------------
# 2) Define candidate edges (graph connectivity)
#    This removes the "3 routes only" restriction.
#    ACO will pick any path available in this graph.
# ----------------------------

# Build a complete graph (all pairs, both directions)
EDGES = []
city_list = list(NODES.keys())
for i in range(len(city_list)):
    for j in range(len(city_list)):
        if i != j:
            EDGES.append((city_list[i], city_list[j]))


# ----------------------------
# 3) OSRM route request (fastest route time between 2 coords)
# ----------------------------
OSRM_ROUTE = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
OSRM_ROUTE_GEOM = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=polyline"

def osrm_route_info(a_lat, a_lon, b_lat, b_lon):
    """Returns (duration_seconds, geometry_polyline)"""
    url = OSRM_ROUTE_GEOM.format(lon1=a_lon, lat1=a_lat, lon2=b_lon, lat2=b_lat)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    route = data["routes"][0]
    return float(route["duration"]), route["geometry"]

def osrm_duration_seconds(a_lat, a_lon, b_lat, b_lon) -> float:
    # For backward compatibility (if needed elsewhere)
    url = OSRM_ROUTE.format(lon1=a_lon, lat1=a_lat, lon2=b_lon, lat2=b_lat)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return float(data["routes"][0]["duration"])

# ----------------------------
# 4) Build graph with edge costs from OSRM
# ----------------------------
def build_graph():
    G = nx.DiGraph()
    for name in NODES:
        G.add_node(name)

    for u, v in EDGES:
        lat1, lon1 = NODES[u]
        lat2, lon2 = NODES[v]
        dur, geom = osrm_route_info(lat1, lon1, lat2, lon2)
        # aco_routing expects edge attribute "cost"
        G.add_edge(u, v, cost=dur, geometry=geom)
        # add reverse too (roads are mostly bidirectional at this level)
        dur_rev, geom_rev = osrm_route_info(lat2, lon2, lat1, lon1)
        G.add_edge(v, u, cost=dur_rev, geometry=geom_rev)
    return G


def main():
    global G  # So plot function can access edge geometry
    print("Building graph + fetching edge times from OSRM...")
    G = build_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Find 3 best ACO TSP tours (start and end at Madrid, visit all cities)
    aco = ACO_TSP(G, num_ants=120, num_iterations=200)
    best_paths, best_costs = aco.run(START)

    # If fewer than 3 unique tours, duplicate the best found so that 3 are always plotted
    if len(best_paths) < 3 and len(best_paths) > 0:
        while len(best_paths) < 3:
            best_paths.append(best_paths[0])
            best_costs.append(best_costs[0])

    for idx, (p, c) in enumerate(zip(best_paths, best_costs)):
        print(f"\n[ACO Route {idx+1}] (Tour)")
        print(" -> ".join(p))
        print(f"Total time: {c/3600:.2f} hours")

    # Plot all 3 best tours on the map
    plot_multiple_routes_xy_with_map(best_paths[:3], title="Top 3 ACO Roadtrip Tours (Europe)")
    print("\nSaved graph: best_routes_map_osm.png (OpenStreetMap background)")
# --- Modified function: plot multiple ACO routes on OpenStreetMap background ---
def plot_multiple_routes_xy_with_map(paths, title, colors=None):
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    if colors is None:
        colors = ['blue', 'green', 'orange']

    ax = None
    for idx, path in enumerate(paths):
        lines = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            geom = None
            try:
                geom = G[u][v]["geometry"]
            except Exception:
                continue
            coords = polyline.decode(geom)
            merc_coords = [transformer.transform(lon, lat) for lat, lon in coords]
            lines.append(LineString(merc_coords))
        gdf_lines = gpd.GeoDataFrame(geometry=lines, crs="EPSG:3857")
        ax = gdf_lines.plot(ax=ax, figsize=(12, 10) if ax is None else None, color=colors[idx % len(colors)], linewidth=2, zorder=2+idx)

    # Plot the points (start/end)
    points = [Point(NODES[n][1], NODES[n][0]) for n in NODES]
    gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_points.plot(ax=ax, color='red', marker='o', zorder=10)

    # Annotate points
    for i, n in enumerate(NODES):
        x, y = gdf_points.geometry.iloc[i].x, gdf_points.geometry.iloc[i].y
        ax.text(x, y, n, fontsize=12, zorder=11)

    minx, miny = transformer.transform(99.5, 1.2)
    maxx, maxy = transformer.transform(104.5, 6.5)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.title(title)
    plt.tight_layout()
    plt.savefig("best_routes_map_osm.png", dpi=200)
    plt.show()

def plot_bar_compare(aco_sec, dijkstra_sec):
    plt.figure(figsize=(6, 4))
    plt.bar(["ACO", "Dijkstra"], [aco_sec/60, dijkstra_sec/60])
    plt.ylabel("Total travel time (minutes)")
    plt.title("ACO vs Dijkstra (baseline)")
    plt.tight_layout()
    plt.savefig("aco_vs_dijkstra.png", dpi=200)

def plot_tuning(x_vals, y_vals, x_label, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel(x_label)
    plt.ylabel("Best route time (minutes)")
    plt.title("ACO Parameter Tuning")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)

# ----------------------------
# 7) Main
# ----------------------------

if __name__ == "__main__":
    main()
