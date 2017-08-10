import numpy as np
from scipy.spatial.distance import pdist, squareform

def main():

    num_neighbours = 10
    filename = "centers.xyz"

    with open(filename, 'r') as input_file:
        num_particles = int(input_file.readline())    # num particles
        input_file.readline()    # comment line
        data = np.zeros((num_particles, 3))
        for i in range(num_particles):
            data[i] = input_file.readline().split()[1:4]

    separation = squareform(pdist(data))

    for particle in range(data.shape[0]):
        nearest_neigbours = np.argpartition(separation[particle], num_neighbours)

main()