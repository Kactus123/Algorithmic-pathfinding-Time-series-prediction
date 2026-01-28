# Algorithmic-pathfinding-Time-series-prediction

## Overview
Exploration of A, Ant Colony Optimisation, LSTM, and Markov models for optimisation and forecasting problems.
Project is split into two main parts:
1. Route optimisation using pathfinding and heuristic algorithms  
2. Time-series prediction using statistical models and neural networks  

The focus of the project is on comparing approaches, understanding trade-offs, and analysing results rather than building a production system.

---

## Part 1: Pathfinding & Route Optimisation

This section simulates a delivery or navigation system where an the quickest route must be found between multiple locations.

### Algorithms Implemented
- **A\*** pathfinding  
  - Grid-based navigation
  - Considers distance, traffic, and urgency
  - Visualised using Pygame
- **Ant Colony Optimisation (ACO)**  
  - Heuristic optimisation inspired by ant behaviour
  - Uses pheromone levels and probabilistic exploration
  - Finds efficient routes across multiple destinations

---

## Part 2: Time-Series Prediction

This section focuses on predicting temperature in Vancouver using historical weather records.

### Models Implemented
- **LSTM (Long Short-Term Memory)** neural network  
  - Trained on historical temperature data
  - Tested with multiple sequence lengths
  - Evaluated using training and validation loss
- **Markov Chain Model**
  - Discretises temperature values into states
  - Uses transition probabilities for prediction
  - Compared against LSTM performance

---

## Libraries Used
- Python
- NumPy, Pandas
- scikit-learn
- TensorFlow / Keras
- Pygame
- Matplotlib
- Meteostat API

---

