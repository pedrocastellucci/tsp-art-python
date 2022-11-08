import numpy as np
import sys
from PIL import Image, ImageDraw

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
            list_of_nodes.append(np.array((int(float(x)),int(float(y)))))
    return list_of_nodes


def compute_euclidean_distance_matrix(locations):    
    n = len(locations)
    dist = np.zeros(shape = (n, n))
    for i, p1 in enumerate(locations):
        for j, p2 in enumerate(locations):
            dist[i, j] = np.linalg.norm(p1 - p2)
    return dist


def draw_routes(image_file, path, locations):
    """Takes a set of nodes and a path, and outputs an image of the drawn TSP path"""
    tsp_path = []
    for j in path:
        # We do not want this to be a numpy array
        tsp_path.append((locations[j][0], locations[j][1]))
    
    print(tsp_path)
    original_image = Image.open(image_file)
    width, height = original_image.size

    tsp_image = Image.new("RGBA",(width,height),color='white')
    tsp_image_draw = ImageDraw.Draw(tsp_image)
    #tsp_image_draw.point(tsp_path,fill='black')
    tsp_image_draw.line(tsp_path, fill='black', width=1)
    tsp_image = tsp_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    final_image = image_file[:-4] + "-tsp.png"
    tsp_image.save(final_image)
    print("TSP solution has been drawn and can be viewed at", final_image)


def nearest_neighbor(dist):
    n = dist.shape[0]
    route = [0]
    unvisited = list(range(1, n))  # Zero is already in route
    while len(unvisited) != 0:
        pivot = route[-1]
        min_dist = np.Inf
        nearest = None
        for j in unvisited:            
            if dist[pivot, j] < min_dist:
                min_dist = dist[pivot, j]
                nearest = j
        route.append(nearest)
        unvisited.remove(nearest)
    return route

if __name__ == "__main__":
    usage()
    tspfile = sys.argv[1]
    originalimage = sys.argv[2]
    locations = read_data(tspfile)
    dist = compute_euclidean_distance_matrix(locations)
    route = nearest_neighbor(dist)
    draw_routes(originalimage, route, locations)
    # Test everything and link to draw_path
