import matplotlib.pyplot as plt
import numpy as np

def plot_regression(pred, lower, upper, true_values, title):
    """
    Create a plot showing predictions with confidence intervals and true values
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(pred))
    
    # Sort everything by predictions for better visualization
    sorted_indices = np.argsort(pred.squeeze())
    pred_sorted = pred[sorted_indices]
    lower_sorted = lower[sorted_indices]
    upper_sorted = upper[sorted_indices]
    true_sorted = true_values[sorted_indices]
    
    # Plot prediction line and confidence intervals
    plt.plot(x, pred_sorted.cpu(), 'b-', label='Predictions')
    plt.fill_between(x, 
                    lower_sorted.cpu().numpy().flatten(), 
                    upper_sorted.cpu().numpy().flatten(), 
                    alpha=0.2, 
                    color='blue', 
                    label='Confidence Interval')
    
    # Plot true values as scatter points
    plt.scatter(x, true_sorted.cpu(), 
               color='red', 
               alpha=0.5, 
               label='True Values')
    
    plt.title(title)
    plt.xlabel('Node Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_classification(pred_sets, true_labels, num_classes, title):
    """
    Create a heatmap visualization of prediction sets vs true labels
    
    Args:
        pred_sets: Binary tensor of shape (n_samples, n_classes) indicating prediction sets
        true_labels: Tensor of shape (n_samples,) containing true class labels
        num_classes: Number of classes in the dataset
    """
    # Convert tensors to numpy arrays
    pred_sets = pred_sets.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 4))
    
    # Calculate average set size for each true class
    avg_set_sizes = np.zeros(num_classes)
    counts = np.zeros(num_classes)
    
    # Count samples for each class
    for i in range(len(true_labels)):
        true_class = true_labels[i]
        set_size = pred_sets[i].sum()
        avg_set_sizes[true_class] += set_size
        counts[true_class] += 1
    
    # Calculate averages, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_set_sizes = np.divide(avg_set_sizes, counts)
        avg_set_sizes = np.nan_to_num(avg_set_sizes, 0)  # Replace NaN with 0
    
    # Create bar plot
    bars = plt.bar(range(num_classes), avg_set_sizes)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.title(title)
    plt.xlabel('Node Class')
    plt.ylabel('Average Prediction Set Size')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    plt.xticks(range(num_classes))
    
    # Add a horizontal line for the overall average
    overall_avg = pred_sets.sum(axis=1).mean()
    plt.axhline(y=overall_avg, color='r', linestyle='--', 
                label=f'Overall Average: {overall_avg:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()