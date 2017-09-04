import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.tri import Triangulation
from scipy.linalg import lstsq

def main():

    com_neighbours = 15
    filename = "centers.xyz"
    data = read_xyz(filename)

    #data = np.loadtxt("2ddata.txt")

    separation = squareform(pdist(data))
    coefficients = np.zeros_like(data)
    residuals = np.zeros((data.shape[0]))
    centres_of_mass = np.zeros_like(data)

    # First get centres of mass
    for particle in range(data.shape[0]):
        nearest_neigbour_indicies = np.argpartition(separation[particle], com_neighbours)[:com_neighbours]
        nearest_neigbours = data[nearest_neigbour_indicies, :]
        centres_of_mass[particle] = np.average(nearest_neigbours, axis=0)
    #plot_com(data, centres_of_mass)

    vector_magnitude = np.zeros((data.shape[0]))
    vector_direction =  np.zeros((data.shape[0]))
    displacement_vectors = np.zeros_like(data)
    separation = cdist(centres_of_mass, data)
    for particle in range(centres_of_mass.shape[0]):
        nearest_neigbour_index = np.argmin(separation[particle])
        nearest_neigbour = data[nearest_neigbour_index, :]
        displacement_vectors[particle] = nearest_neigbour - centres_of_mass[particle]
        if displacement_vectors[particle, 0] > 0:
            vector_direction[particle] = 1
            vector_magnitude[particle] = np.sqrt(displacement_vectors[particle, 0] ** 2 + displacement_vectors[particle, 1] ** 2 + displacement_vectors[particle, 2] ** 2)
        else:
            vector_direction[particle] = 0
            vector_magnitude[particle] = - np.sqrt(displacement_vectors[particle, 0] ** 2 + displacement_vectors[particle, 1] ** 2 + displacement_vectors[particle, 2] ** 2)

    particle_diameter = 380/40
    vector_magnitude = vector_magnitude/particle_diameter
    plt.hist(vector_magnitude, bins=np.arange(min(vector_magnitude), max(vector_magnitude) + 0.1, 0.1))
    plt.xlabel("Distance from plane centre (Particle diameters)")
    plt.ylabel("Frequency")
    plt.savefig("density.png")
    plt.show()

    quiver_plot(centres_of_mass, displacement_vectors)

    outputfile = open("vectors.txt", 'a')
    outputfile.close()
    with open("vectors.txt", 'a') as outputfile:
        outputfile.write(str(centres_of_mass.shape[0]) + "\ncomment\n")
        for particle in range(centres_of_mass.shape[0]):
            outputfile.write(str(vector_direction[particle]) + "\t" + str(data[particle, 0]) + "\t" + str(data[particle, 1]) + "\t" + str(data[particle, 2]) + "\n")


def quiver_plot(centres, vectors):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = centres[:, 0]
    y = centres[:, 1]
    z = centres[:, 2]
    ax.quiver(x, y, z, vectors[:, 0], vectors[:, 1], vectors[:, 2])
    # Set up equal sized axes
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()


def read_xyz(filename):
    with open(filename, 'r') as input_file:
        num_particles = int(input_file.readline())    # num particles
        input_file.readline()    # comment line
        data = np.zeros((num_particles, 3))
        for i in range(num_particles):
            data[i] = input_file.readline().split()[1:4]
    return data


def linear_fitting(nearest_neigbours, data):
    # find best-fit linear plane

    A = np.c_[nearest_neigbours[:, 0], nearest_neigbours[:, 1], np.ones(nearest_neigbours.shape[0])]
    coefficients, residual, _, _ = lstsq(A, nearest_neigbours[:, 2])  # coefficients

    return coefficients, residual


def plot_surface(nearest_neigbours, coefficients, data):
    # define a regular grid covering the domain of the data
    mn = np.min(nearest_neigbours, axis=0)
    mx = np.max(nearest_neigbours, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))

    # evaluate it on grid
    Z = coefficients[0] * X + coefficients[1] * Y + coefficients[2]

    # plot points and fitted surface
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(nearest_neigbours[:, 0], nearest_neigbours[:, 1], nearest_neigbours[:, 2], c='r', s=150)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')

    # Set up equal sized axes
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def plot_residuals(data, residuals):
    # plot points on 3d axis
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=residuals, cmap='plasma', s=150)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()


def plot_com(data, com, center=(0, 0, 0)):
    # plot points on 3d axis
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.5, s=10)
    ax.scatter(com[:, 0], com[:, 1], com[:, 2], c='red', s=200)
    if center[0] != 0:
        ax.scatter(center[0], center[1], center[2], c='green', s=300)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    # Set up equal sized axes
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()


main()
