"""
This is a program to demonstrate working of basic genetic algorithm which is searching for the least valued points
in the graph of sin(x^2), hope you like it! ^_^
"""

# Made by Lakshya Sharma


# Import Statements:

import numpy
import math
import matplotlib.pyplot as plt
import time
import shutil
import os


# GA Parameters

pc = 0.4
pm = 0.2
population_size = 20
num_chromosomes = 1
Generation = -1
avg = 0.3
parent_size = int(pc*population_size)
children_size = int(pc*population_size)


# Global Lists:

population = numpy.array([[None]*num_chromosomes]*population_size)
fitness_values = numpy.array([None]*population_size)
parents = [None]*parent_size
children = numpy.array([[None]*num_chromosomes]*children_size)
fit_list = []
fittest = population_size


# UDF

def graph_initialization():
    """Initializes a graph of sin(x^2) for better visualization"""

    r = numpy.arange(-1 * math.pi, math.pi, 1e-3)
    f = numpy.vectorize(math.sin)
    plt.plot(r, f(r ** 2), color="BLACK")


def population_initialization():
    """Initializing the population"""

    for i in range(population_size):
        for j in range(num_chromosomes):
            population[i][j] = numpy.random.uniform((-1*math.pi), math.pi)


def fitness_calculation():
    """Calculates Fitness of Each Individual"""

    for i in range(population_size):
        for j in range(num_chromosomes):
            fitness_values[i] = math.sin((population[i][j])**2)
            if fitness_values[i] < -0.999:
                fitness_values[i] = -1


def display():
    """Displays Fittest Value achieved in each generation"""

    global fittest
    min = 2
    fittest = population_size
    for i in range(population_size):
        if fitness_values[i] < min:
            min = fitness_values[i]
            fittest = i
    print("-" * 300 + "\n")
    print("Generation No.", Generation)
    print("Fitness achieved:", fitness_values[fittest])
    print()
    fit_list.append(fitness_values[fittest])


def visualize():
    """A function to help visualize each generation's progress"""

    graph_initialization()
    for i in range(population_size):
        if i != fittest:
            for j in range(num_chromosomes):
                plt.scatter(population[i][j], fitness_values[i], marker=".", color = "RED")
    plt.scatter(population[fittest][0], fitness_values[fittest], marker="*", color="BLUE")

    plt.title("Generation#"+str(Generation))
    plt.xlabel("---x--->")
    plt.ylabel("---sin(x^2)--->")
    plt.grid(color='GREEN', linestyle='-', linewidth=0.5)

    plt.savefig("Output\\Generation#"+str(Generation)+".png")
    plt.show()
    plt.close()

def parent_selection():
    """Selection algorithm for parent selection"""

    for i in range(parent_size):
        min = 2
        index = population_size
        for j in range(population_size):
            if fitness_values[j] < min and j not in parents:
                min = fitness_values[j]
                index = j

        parents[i] = index


def crossover():
    """Performs crossover on selected parents"""

    for i in range(0,parent_size, 2):
        for j in range(num_chromosomes):
            children[i][j] = avg * population[parents[i]][j] + (1-avg)*population[parents[i+ 1]][j]
            children[i+1][j] = (1-avg) * population[parents[i]][j] + avg * population[parents[i + 1]][j]

def mutation():
    """Performs mutation on all new children"""

    for i in range(children_size):
            for j in range(num_chromosomes):
                prob = numpy.random.uniform(0, 1)
                if prob < pm:
                    children[i][j] *= numpy.random.choice([0.8, 1.2])

def survivor_selection():
    """Selection algorithm for survivor selection"""

    visited = []

    for i in range(children_size):
        max = -2
        index = population_size
        for j in range(population_size):
            if fitness_values[j] > max and j not in visited:
                max = fitness_values[j]
                index = j
        visited.append(index)

        for k in range(num_chromosomes):
            population[index][k] = children[i][k]


def termination_conditions():
    """Boolean value returning function which checks the termination conditions"""

    global Generation
    Generation += 1

    if Generation == 100:
        print("\n\n\nMaximum Generation Limit Reached\n")
        return False

    if Generation>20:
        for i in range(len(fit_list)-1, len(fit_list)-15, -1):
            if fit_list[i] != fit_list[i-1]:
                break
        else:
            print("\n\n\nNo Change Detected in past 15 generations\n")
            return False

        if fit_list[len(fit_list)-2] == -1:
            print("\n\n\nMinimum Value Achieved\n")
            return False

    return True


# Main

shutil.rmtree("Output")
time.sleep(1)
os.makedirs("Output")
start = time.time()

#GA

population_initialization()

while termination_conditions():

    fitness_calculation()
    display()
    visualize()
    parent_selection()
    crossover()
    mutation()
    survivor_selection()

end = time.time()
print("Time Taken:",int(end-start),"seconds")