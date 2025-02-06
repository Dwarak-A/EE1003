import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load the shared library
prob_lib = ctypes.CDLL('./probability.so')

# Define function prototypes for the C functions
prob_lib.compute_probabilities.restype = ctypes.POINTER(ctypes.c_float * 4)
prob_lib.compute_probabilities.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]

# Simulated function to plot the theoretical and simulated probabilities
def plot_probabilities(p_A, p_B, p_AB, n):
    # Call the compute_probabilities function from the C shared library
    result = prob_lib.compute_probabilities(ctypes.c_float(p_A), ctypes.c_float(p_B), ctypes.c_float(p_AB), ctypes.c_int(n))
    
    # Extract the results from the returned pointer
    simulated_probs = np.array(result.contents)
    
    # Calculate theoretical probabilities
    P_A = p_A
    P_B = p_B
    P_AB = p_AB
    P_A_plus_B = P_A + P_B - P_AB  # P(A + B)
    P_AcBc = 1 - P_A - P_B + P_AB  # P(A'B')

    # Prepare the theoretical probabilities
    theoretical_probs = np.array([P_A, P_B, P_AB, P_AcBc])

    # Labels for the probabilities
    labels = ['P(A)', 'P(B)', 'P(AB)', 'P(A+B)\'']
    
    x = np.arange(4)

    # Create a stem plot for the theoretical and simulated probabilities
    plt.figure(figsize=(8, 6))
    markerline1, stemlines1, baseline1 = plt.stem(x - 0.2, theoretical_probs, label='theory', basefmt=" ", use_line_collection=True)
    markerline2, stemlines2, baseline2 = plt.stem(x + 0.2, simulated_probs, label='sim', basefmt=" ", use_line_collection=True)

    # Customizing the appearance of the stems
    markerline1.set_markerfacecolor('blue')
    markerline1.set_markeredgecolor('blue')
    stemlines1.set_color('blue')
    markerline2.set_markerfacecolor('red')
    markerline2.set_markeredgecolor('red')
    stemlines2.set_color('red')

    # Add annotations above each stem
    for i in range(4):
        # Annotate for the theoretical plot
        plt.text(x[i] - 0.2, theoretical_probs[i] + 0.02, f'{theoretical_probs[i]:.3f}', 
                 horizontalalignment='center', verticalalignment='bottom', color='blue', fontsize=10)
        
        # Annotate for the simulated plot
        plt.text(x[i] + 0.2, simulated_probs[i] + 0.02, f'{simulated_probs[i]:.3f}', 
                 horizontalalignment='center', verticalalignment='bottom', color='red', fontsize=10)

    # Set y-axis limit from 0 to 1
    plt.ylim(0, 1)

    # Add labels and title
    plt.xticks(x, labels)
    plt.ylabel('Probability')
    #plt.title('Theoretical vs Simulated Probabilities')
    plt.legend()
    plt.savefig("../figs/plot.png")
    plt.show()

# Example usage
p_A = 0.25
p_B = 0.50
p_AB = 0.125
n = 100000

plot_probabilities(p_A, p_B, p_AB, n)

