import numpy as np
import matplotlib.pyplot as plt


def get_positional_encoding(max_len, d_model):
    # Initialize matrix
    pos_enc = np.zeros((max_len, d_model))

    # Compute positional encoding values
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                pos_enc[pos, i] = np.cos(pos / (10000 ** (i / d_model)))

    return pos_enc # [max_len=10, d_model=512]


def plot_heatmap(data, title):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    # Set axis labels
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Add title
    plt.title(title)

    # Show plot
    plt.show()


# Example usage
pos_enc = get_positional_encoding(10, 512)
plot_heatmap(pos_enc, 'Transformer Position Encoding Heatmap')
print(pos_enc.shape)
