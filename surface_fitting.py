import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.tri import Triangulation
from scipy.linalg import lstsq

def main():

    num_neighbours = 15
    filename = "centers.xyz"
    data = read_xyz(filename)

    #data = np.loadtxt("2ddata.txt")

    separation = squareform(pdist(data))
    coefficients = np.zeros_like(data)
    residuals = np.zeros((data.shape[0]))
    centres_of_mass = np.zeros_like(data)

    # First get centres of mass
    for particle in range(data.shape[0]):
        nearest_neigbour_indicies = np.argpartition(separation[particle], num_neighbours)[:num_neighbours]
        nearest_neigbours = data[nearest_neigbour_indicies, :]
        centres_of_mass[particle] = np.average(nearest_neigbours, axis=0)
    plot_com(data, centres_of_mass)

    # Now fit planes to centres of mass
    separation = squareform(pdist(centres_of_mass))
    for particle in range(centres_of_mass.shape[0]):
        nearest_neigbour_indicies = np.argpartition(separation[particle], 200)[:200]
        nearest_neigbours = data[nearest_neigbour_indicies, :]
        coefficients[particle], residuals[particle] = linear_fitting(nearest_neigbours, data)
        if particle > 200:
            plot_surface(nearest_neigbours, coefficients[particle], data)

    #plt.scatter(data[:, 0], data[:, 1], c ='blue', s=50)
    #plt.scatter(centres_of_mass[:, 0], centres_of_mass[:, 1], c='red', s=50)
    #plt.axes().set_aspect('equal', 'datalim')
    #plt.show()
    #plot_residuals(data, residuals)




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

def plot_com(data, com):
    # plot points on 3d axis
    X = com[:, 0]
    Y = com[:, 1]
    Z = com[:, 2]
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)
    #ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.5, s=50)
    ax.scatter(X, Y, Z, c='red', s=100)
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


main()
