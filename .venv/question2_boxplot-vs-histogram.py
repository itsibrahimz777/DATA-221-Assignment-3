import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("crime1.csv")

# Extract the column
vc = df["ViolentCrimesPerPop"]

# HISTOGRAM
plt.figure(figsize=(8, 5))
plt.hist(vc, bins=30, edgecolor='black')
plt.title("Distribution of Violent Crimes Per Population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")
plt.show()

# BOX PLOT
plt.figure(figsize=(6, 5))
plt.boxplot(vc, vert=True)
plt.title("Box Plot of Violent Crimes Per Population")
plt.ylabel("ViolentCrimesPerPop")
plt.xlabel("Data")
plt.show()

# EXPLANATORY COMMENTS
# The histogram shows how the values of ViolentCrimesPerPop are spread across the dataset,
# revealing whether most communities fall into low, moderate, or high crime ranges.
# It also helps identify whether the distribution is concentrated in one region or stretched out.
# The box plot highlights the median clearly, showing where the middle value of the data lies.
# It also displays the interquartile range, which helps illustrate how tightly or loosely the data is clustered.
# Any points plotted beyond the whiskers suggest potential outliers in the dataset.
# Together, these plots help reveal whether the distribution is symmetric, skewed, or influenced by extreme values.