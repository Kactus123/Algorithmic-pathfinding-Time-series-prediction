import numpy as np
import random
import matplotlib.pyplot as plt

# Given values for the problems
locations = ["Central hub", "Hospital", "Warehouse", "Primary school", "Pharmacy", "Football club", "University", "Grocery shop", "Church", "Library"]
n_citys = len(locations)

# Cost of travel
cost_matrix = np.array([[0, 5, 8, 6, 7, 9, 10, 3, 4, 5],  
                        [5, 0, 3, 4, 7, 6, 8, 7, 9, 6],  
                        [8, 3, 0, 5, 3, 6, 4, 7, 6, 8],  
                        [6, 4, 5, 0, 3, 5, 6, 8, 9, 4],  
                        [7, 7, 3, 3, 0, 4, 5, 6, 3, 7],  
                        [9, 6, 6, 5, 4, 0, 7, 5, 6, 3],  
                        [10, 8, 4, 6, 5, 7, 0, 5, 4, 8], 
                        [3, 7, 7, 8, 6, 5, 5, 0, 4, 6],  
                        [4, 9, 6, 9, 3, 6, 4, 4, 0, 3],  
                        [5, 6, 8, 4, 7, 3, 8, 6, 3, 0]])

random.seed(42)

# Fixed coordinates for each location
coordinates = np.array([[100, 400],  # Central hub
                        [200, 100],  # Hospital
                        [300, 600],  # Warehouse
                        [400, 200],  # Primary school
                        [500, 700],  # Pharmacy
                        [600, 400],  # Football club
                        [200, 600],  # University
                        [300, 300],  # Grocery shop
                        [400, 500],  # Church
                        [500, 300]]) # Library

# Initialize other parameters
iteration = 200
n_ants = 10
e = 0.1  # evaporation rate
alpha = 1  # pheromone factor
beta = 2  # visibility factor
exploration = 0.1  # exploration factor

# Initializing pheromone present at the paths to the cities
pheromone = np.ones((n_citys, n_citys)) * 0.9

# the visibility of the next city visibility(i,j)=1/cost(i,j)
visibility = np.where(cost_matrix != 0, 1 / cost_matrix, 0)

# Initialize the best route and its cost
best_route = None
best_cost = float('inf')
repeat_counter = 0
prev_best_cost = None

for ite in range(iteration):
    # Initializing the route of the ants
    route = np.ones((n_ants, n_citys + 1))
    route[:, 0] = 1

    for i in range(n_ants):
        temp_visibility = np.array(visibility)
        for j in range(n_citys - 1):
            cur_loc = int(route[i, j] - 1)
            temp_visibility[:, cur_loc] = 0

            p_feature = np.power(pheromone[cur_loc, :], alpha)
            v_feature = np.power(temp_visibility[cur_loc, :], beta)

            # Introduce exploration
            if random.random() < exploration:
                combine_feature = v_feature
            else:
                combine_feature = np.multiply(p_feature, v_feature)

            # Select the next city based on the probability
            prob = combine_feature / np.sum(combine_feature)
            next_city = np.random.choice(np.arange(n_citys), p=prob)
            route[i, j + 1] = next_city + 1

        left = list(set(range(1, n_citys + 1)) - set(route[i, :-2]))[0]
        route[i, -2] = left

    # Calculate the total distance of each tour
    dist_cost = np.zeros((n_ants, 1))
    for i in range(n_ants):
        for j in range(n_citys - 1):
            dist_cost[i] += cost_matrix[int(route[i, j]) - 1, int(route[i, j + 1]) - 1]

    # Find the best route of this iteration
    dist_min_loc = np.argmin(dist_cost)
    dist_min_cost = dist_cost[dist_min_loc]

    # Update the global best route
    if dist_min_cost < best_cost:
        if dist_min_cost == prev_best_cost:
            repeat_counter += 1
            if repeat_counter == 30:
                print("Best cost repeated 5 times. Terminating...")
                break
        else:
            repeat_counter = 0
        best_route = route[dist_min_loc, :]
        best_cost = dist_min_cost
        prev_best_cost = dist_min_cost
        print(f'Iteration {ite + 1}: Best Cost = {int(best_cost)}')

        # Print the best path for this iteration
        print("Best Path:")
        for city in best_route.astype(int):
            print(locations[int(city) - 1])
        print()

    # Update the pheromone
pheromone *= (1 - e)
for i in range(n_ants):
    for j in range(n_citys - 1):
        dt = 1 / dist_cost[i][0]
        pheromone[int(route[i, j]) - 1, int(route[i, j + 1]) - 1] += dt
    # Update pheromone for the return path to Central hub
    pheromone[int(route[i, -2]) - 1, 0] += dt
    # Update pheromone for the path from Central hub to the last location
    pheromone[0, int(route[i, -2]) - 1] += dt


# Print the best path for the last iteration
print("Last Iteration Best Path:")
for city in best_route.astype(int):
    print(locations[int(city) - 1])
print()

# Visualization
plt.figure(figsize=(8, 8))
for i, loc in enumerate(locations):
    plt.scatter(coordinates[i][0], coordinates[i][1], label=loc, s=100)
    plt.text(coordinates[i][0], coordinates[i][1], loc)

# Plotting paths with annotations representing the ending pheromone levels
for i in range(len(best_route) - 1):
    start_loc = int(best_route[i]) - 1
    end_loc = int(best_route[i + 1]) - 1
    pheromone_level = pheromone[start_loc][end_loc]
    plt.plot([coordinates[start_loc][0], coordinates[end_loc][0]], [coordinates[start_loc][1], coordinates[end_loc][1]], color='blue')
    plt.text((coordinates[start_loc][0] + coordinates[end_loc][0]) / 2, 
             (coordinates[start_loc][1] + coordinates[end_loc][1]) / 2 - 15,
             round(pheromone_level, 2), fontsize=8, horizontalalignment='center')


# Plotting the return path to the Central hub
return_path_x = [coordinates[int(best_route[-1]) - 1][0], coordinates[0][0]]
return_path_y = [coordinates[int(best_route[-1]) - 1][1], coordinates[0][1]]
plt.plot(return_path_x, return_path_y, color='blue')
pheromone_level = pheromone[int(best_route[-1]) - 1][0]
plt.text((return_path_x[0] + return_path_x[1]) / 2, 
         (return_path_y[0] + return_path_y[1]) / 2,
         round(pheromone_level, 2), fontsize=8)

plt.title('Ant Colony Optimization - Best Path with Pheromone Levels')
plt.xlabel('X Position (Pixels)')
plt.ylabel('Y Position (Pixels)')
plt.grid(True)
plt.legend()
plt.show()
