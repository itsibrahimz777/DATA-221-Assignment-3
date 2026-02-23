import pandas as pd

# Load the dataset
df = pd.read_csv("crime1.csv")

# Focus on the ViolentCrimesPerPop column
vc = df["ViolentCrimesPerPop"]

# Compute statistical measures
mean_val = vc.mean()
median_val = vc.median()
std_val = vc.std()
min_val = vc.min()
max_val = vc.max()

print("Mean:", mean_val)
print("Median:", median_val)
print("Standard Deviation:", std_val)
print("Minimum:", min_val)
print("Maximum:", max_val)

# --- COMMENTS / EXPLANATIONS ---

# Comparing the mean and median:
# If the mean is noticeably larger than the median, the distribution is right‑skewed.
# If the mean is noticeably smaller, the distribution is left‑skewed.
# If they are close, the distribution is more symmetric.

# Extreme values:
# The mean is more affected by extreme values (outliers) because it incorporates every value directly into the calculation.
# The median is more robust since it depends only on the middle position, not the magnitude of extreme values.