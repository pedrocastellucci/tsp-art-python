import math
import sys
from PIL import Image, ImageDraw
from random import randint, random, sample

def usage():
    print("solver.py [image.tsp] [image.jpg]")

def read_data(inputfile):
    """Stores the data for the problem."""
    # Extracts coordinates from IMAGE_TSP and puts them into an array
    list_of_nodes = []
    with open(inputfile) as f:
        for _ in range(6):
            next(f)
        for line in f:
            _,x,y = line.split()
            list_of_nodes.append((int(float(x)),int(float(y))))
    return list_of_nodes


def compute_euclidean_distance_matrix(locations):    
    dist = {}
    for i, p1 in enumerate(locations):
        for j, p2 in enumerate(locations):
            dist[i, j] = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    return dist


def draw_routes(image_file, path, locations):
    """Takes a set of nodes and a path, and outputs an image of the drawn TSP path"""
    tsp_path = []
    for j in path:        
        tsp_path.append((locations[j][0], locations[j][1]))
    tsp_path.append(tsp_path[0])
    original_image = Image.open(image_file)
    width, height = original_image.size

    tsp_image = Image.new("RGBA",(width,height),color='white')
    tsp_image_draw = ImageDraw.Draw(tsp_image)
    #tsp_image_draw.point(tsp_path,fill='black')
    tsp_image_draw.line(tsp_path, fill='black', width=1)
    tsp_image = tsp_image.transpose(Image.FLIP_TOP_BOTTOM)
    final_image = image_file[:-4] + "-tsp.png"
    tsp_image.save(final_image)
    print("TSP solution has been drawn and can be viewed at", final_image)


def get_cost(route, dist):
    cost = dist[route[-1], route[0]]
    for i, n in enumerate(route[:-1]):
        cost += dist[n, route[i+1]]
    return cost


def nearest_neighbor(dist, locations):
    n = len(locations)
    route = [0]
    unvisited = list(range(1, n))  # Zero is already in route
    while len(unvisited) != 0:
        pivot = route[-1]
        min_dist = math.inf
        nearest = None
        for j in unvisited:            
            if dist[pivot, j] < min_dist:
                min_dist = dist[pivot, j]
                nearest = j
        route.append(nearest)
        unvisited.remove(nearest)
    return route


def get_2opt_savings(route, i, j, dist):
    n = len(route)
    i0, i1 = route[i], route[i+1]
    j0, j1 = route[j], route[(j+1) % n]
    savings = dist[i0, i1] + dist[j0, j1]    
    savings = savings - dist[i0, j0] - dist[i1, j1]
    return savings


def get_improved_2opt(nodes, route, dist):
    n = len(nodes)    
    best_i, best_j = None, None
    best_save = 0
    for i in range(n-3):
        for j in range(i+2, n):      
            if (j + 1) % n == i:
                continue
            save = get_2opt_savings(route, i, j, dist)
            if save > best_save:
                best_save = save
                best_i, best_j = i, j

    if best_i is not None:
        i, j = best_i, best_j
        return route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:], best_save
    return route, best_save


def get_random_2opt(n):
    i, j = randint(0, n-3), randint(0, n)
    if i > j:
        i, j = j, i
    while j - i < 2 and (j+1) % n == i:
        i, j = randint(0, n-3), randint(0, n)
        if i > j:
            i, j = j, i
    return i, j


def local_search(nodes, sol, dist):
    cand_sol, save = get_improved_2opt(nodes, sol, dist)
    while save > 1e-4:
        sol = cand_sol
        cand_sol, save = get_improved_2opt(nodes, sol, dist)
    return sol    


def iterated_local_search(nodes, sol, dist, max_iter=100):
    iter = 0
    n = len(nodes)
    best_sol = sol
    best_value = get_cost(sol, dist)
    while iter < max_iter:
        sol = local_search(nodes, sol, dist)        
        cand_value = get_cost(sol, dist)        
        if cand_value < best_value:
            best_value = cand_value
            best_sol = sol
            print(f"Found new best {best_value}")
        i, j = randint(0, n), randint(0, n)
        if i > j:
            i, j = j, i
        sol = best_sol[:i] + sample(best_sol[i:j], j-i) + best_sol[j:]
        iter += 1
    return best_sol    

        

def simulated_annealing(nodes, sol, dist, T0=1000, alpha=0.99, max_iter=1000):
    n = len(nodes)
    T = T0
    iter = 0
    best_sol = sol
    sol_cost = best_cost = get_cost(sol, dist)        
    while iter < max_iter:
        i, j = get_random_2opt(n)
        cand_sol = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
        cand_cost = get_cost(cand_sol, dist)
        print(cand_cost, sol_cost, best_cost, math.exp((best_cost - cand_cost)/T))
        if cand_cost < sol_cost:
            sol = cand_sol
            sol_cost = cand_cost
            if sol_cost < best_cost:
                best_sol = sol
                best_cost = cand_cost
        elif random() <= math.exp((best_cost - cand_cost)/T):            
            sol = cand_sol
            sol_cost = cand_cost
        iter += 1
        T = alpha*T
    return best_sol


if __name__ == "__main__":
    usage()
    tspfile = sys.argv[1]
    originalimage = sys.argv[2]    
    locations = read_data(tspfile)
    nodes = list(range(len(locations)))
    dist = compute_euclidean_distance_matrix(locations)
    route = nearest_neighbor(dist, locations)
    print("Cost is ", get_cost(route, dist))
    #route = local_search(nodes, route, dist)
    #print("After 2-opt ", get_cost(route, dist))
    route = iterated_local_search(nodes, route, dist, max_iter=5)
    print("After ILS", get_cost(route, dist))
    draw_routes(originalimage, route, locations)
    # Test everything and link to draw_path
