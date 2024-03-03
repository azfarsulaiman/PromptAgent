import json
import matplotlib.pyplot as plt
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Load the dataset
with open('datasets.json') as f:
    datasets = json.load(f)

# Define the percentage of dataset to consider
percentage = 100  # Change this value as needed

# Define the number of epochs or queries
epochs = [1, 2, 3, 4, 5]  # Change this list as needed

# Initialize lists to store the number of epochs and performance levels
num_epochs = []
performance = []

# Run the model for each epoch and record the performance
for epoch in epochs:
    # Run the model with the specified percentage of dataset
    # and get the performance level
    performance_level = run_model(datasets, percentage, epoch)

    # Append the number of epochs and performance level to the lists
    num_epochs.append(epoch)
    performance.append(performance_level)

# Plot the convergence curve
plt.plot(num_epochs, performance)
plt.xlabel('Number of Epochs')
plt.ylabel('Performance Level')
plt.title('Convergence Curve')
plt.show()
