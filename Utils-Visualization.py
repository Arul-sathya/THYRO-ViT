import matplotlib.pyplot as plt
import tensorflow as tf

def plot_sample_images(images, labels, predictions=None, num_samples=5):
    """Plot a few sample images with their labels (and predictions if provided)."""
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(tf.squeeze(images[i]), cmap="gray")
        title = f"Label: {tf.argmax(labels[i])}"
        if predictions is not None:
            title += f"\nPred: {tf.argmax(predictions[i])}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    """Plot the Receiver Operating Characteristic (ROC) curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()
