import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example correlation data
data = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [-3, -6, -9, -12, -15],
        "D": [1, 3, 6, 0, -4],
    }
)
corr = data.corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
