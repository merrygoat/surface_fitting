import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def main():

    num_neighbours = 80
    filename = "centers.xyz"

    with open(filename, 'r') as input_file:
        num_particles = int(input_file.readline())    # num particles
        input_file.readline()    # comment line
        data = np.zeros((num_particles, 3))
        for i in range(num_particles):
            data[i] = input_file.readline().split()[1:4]

    separation = squareform(pdist(data))
    coefficients = np.zeros_like(data)
    residuals = np.zeros((data.shape[0]))

    for particle in range(data.shape[0]):
        nearest_neigbour_indicies = np.argpartition(separation[particle], num_neighbours)[:num_neighbours]
        nearest_neigbours = data[nearest_neigbour_indicies, :]

        coefficients[particle], residuals[particle] = linear_fitting(nearest_neigbours, data)

    plot_residuals(data, residuals)


def linear_fitting(nearest_neigbours, data):
    # find best-fit linear plane
    A = np.c_[nearest_neigbours[:, 0], nearest_neigbours[:, 1], np.ones(nearest_neigbours.shape[0])]
    coefficients, residual, _, _ = np.linalg.lstsq(A, nearest_neigbours[:, 2])  # coefficients

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
    ax.scatter(nearest_neigbours[:, 0], nearest_neigbours[:, 1], nearest_neigbours[:, 2], c='r', s=50)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
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


main()
