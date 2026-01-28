import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from meteostat import Point, Daily
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

LOCATION = Point(49.2497, -123.1193, 70)
MONTH_START = datetime(2023, 6, 1)
MONTH_END = datetime(2023, 6, 30)
PREDICTION_START = datetime(2024, 6, 1)
PREDICTION_END = datetime(2024, 6, 30)
FEATURES = ['tavg']
NUM_STATES = 5  
SMOOTHING = 1

# temperature data for the specific month
data = Daily(LOCATION, MONTH_START, MONTH_END)
data = data.fetch()
data = data[FEATURES]

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=FEATURES, index=data.index)

# data into states
def discretize_data(data, num_states=NUM_STATES):
    bins = np.linspace(0, 1, num_states)
    states = np.digitize(data, bins) - 1
    states[states == num_states] = num_states - 1
    return states

# state sequences for a given order
def generate_sequences(states, order):
    sequences = []
    for i in range(len(states) - order):
        sequences.append(tuple(states[i:i + order + 1]))
    return sequences

# Calculate transition matrix for a given order with Laplace smoothing
def calculate_transition_matrix(sequences, order, num_states=NUM_STATES, smoothing=SMOOTHING):
    transition_matrix = np.zeros((num_states ** order, num_states))
    for seq in sequences:
        from_state = seq[:-1]
        to_state = seq[-1]
        from_index = sum(s * (num_states ** i) for i, s in enumerate(reversed(from_state)))
        transition_matrix[from_index, to_state] += 1
    
    transition_matrix += smoothing

    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
    return np.nan_to_num(transition_matrix)

# Evaluate model performance using log-likelihood
def evaluate_model(transition_matrix, sequences, order, num_states=NUM_STATES):
    log_likelihood = 0
    for seq in sequences:
        from_state = seq[:-1]
        to_state = seq[-1]
        from_index = sum(s * (num_states ** i) for i, s in enumerate(reversed(from_state)))
        prob = transition_matrix[from_index, to_state]
        log_likelihood += np.log(prob + 1e-10)
    return log_likelihood

# Plot transition probabilities
def plot_transition_probabilities(transition_matrix, order, num_states=NUM_STATES):
    num_plots = min(2, num_states ** order)
    cols = 2
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4), sharex=True, sharey=True)
    fig.suptitle(f'Transition Probabilities for Order {order}', fontsize=16)
    
    for i in range(num_plots):
        ax = axes[i]
        ax.bar(range(num_states), transition_matrix[i], color='blue', alpha=0.7)
        ax.set_title(f'State {i}', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
    
    plt.xlabel('Next State', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Function to input temporal sequences
def input_temporal_sequences(orders):
    states = discretize_data(data_scaled.values.flatten())
    results = {}

    for order in orders:
        sequences = generate_sequences(states, order)
        transition_matrix = calculate_transition_matrix(sequences, order, NUM_STATES)
        log_likelihood = evaluate_model(transition_matrix, sequences, order, NUM_STATES)
        results[order] = log_likelihood
        print(f"Order {order}, Log-Likelihood: {log_likelihood}")

        # Plot transition probabilities for the current order
        plot_transition_probabilities(transition_matrix, order, NUM_STATES)

    # Select the best order based on log-likelihood
    best_order = max(results, key=results.get)
    print(f"Best Order: {best_order}, Log-Likelihood: {results[best_order]}")

    # Calculate transition matrix for the best order
    best_sequences = generate_sequences(states, best_order)
    best_transition_matrix = calculate_transition_matrix(best_sequences, best_order, NUM_STATES)

    # Display the best transition matrix
    print("Best Transition Matrix:")
    print(best_transition_matrix)

    return best_order, best_transition_matrix

# Prediction function
def predict_future(transition_matrix, initial_sequence, num_steps, order, num_states=NUM_STATES):
    current_sequence = initial_sequence
    predictions = []
    
    for _ in range(num_steps):
        from_index = sum(s * (num_states ** i) for i, s in enumerate(reversed(current_sequence)))
        next_state_prob = transition_matrix[from_index]
        next_state = np.random.choice(np.arange(num_states), p=next_state_prob)
        predictions.append(next_state)
        
        current_sequence = current_sequence[1:] + (next_state,)
    
    return predictions

# Train model and get best order and transition matrix
ORDERS = [1, 2, 3]
best_order, best_transition_matrix = input_temporal_sequences(ORDERS)

# Define the initial sequence based on the end of the training data
states = discretize_data(data_scaled.values.flatten())
initial_sequences = [tuple(states[i:i + best_order]) for i in range(len(states) - best_order + 1)]

# Predict future values for June 2024
num_prediction_days = (PREDICTION_END - PREDICTION_START).days + 1
predicted_values_list = []
sequence_limit = 3

for initial_sequence in initial_sequences[:sequence_limit]:
    predicted_states = predict_future(best_transition_matrix, initial_sequence, num_prediction_days, best_order)
    bins = np.linspace(0, 1, NUM_STATES)
    predicted_values = scaler.inverse_transform([[bins[state]] for state in predicted_states])
    predicted_values_list.append(predicted_values.flatten())

# Average the predictions for each day
average_predictions = np.mean(predicted_values_list, axis=0)

# Calculate the average temperature for the month
average_original = np.mean(data.values.flatten())
average_predicted = np.mean(average_predictions[:len(data)])

print("Average predicted values for June 2024:")
print(average_predictions)

# Compare original and average predicted values
comparison = pd.DataFrame({
    'Original': data.values.flatten(),
    'Predicted': average_predictions[:len(data)]
})
print("\nComparison of Original and Average Predicted tavg values:")
print(comparison)

# Plot original and predicted values
plt.figure(figsize=(12, 6))
plt.plot(data.index, data.values.flatten(), label='Original', color='blue')
plt.plot(data.index, average_predictions[:len(data)], label='Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Original vs Predicted Average Temperature for June 2023')
plt.legend()

# Annotate the plot with the average temperatures
plt.text(data.index[-1], average_original, f'Avg Original: {average_original:.2f}°C', color='blue', fontsize=12, verticalalignment='bottom')
plt.text(data.index[-1], average_predicted, f'Avg Predicted: {average_predicted:.2f}°C', color='red', fontsize=12, verticalalignment='bottom')

plt.grid(True)
plt.show()
