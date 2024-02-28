import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('path/to/dataset.csv')

# Set the sample size (number of prompts/tokens)
sample_size = data['Number of Prompts/Tokens']

# Set the loss values
loss = data['Loss']

# Plot the convergence curve
plt.plot(sample_size, loss)
plt.xlabel('Number of Prompts/Tokens')
plt.ylabel('Loss')
plt.title('Convergence Curve')
plt.grid(True)
plt.show()