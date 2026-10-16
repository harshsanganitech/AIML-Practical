import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"datasets/01_study_correlation.csv")
print(df.head())
print("\nCorrelation matrix:\n",df.corr(numeric_only=True).round(2))
print("\nIndependent variable: study_hours; dependent variable: exam_score")
plt.scatter(df["study_hours"],df["exam_score"])
plt.xlabel("Study hours"); plt.ylabel("Exam score"); plt.title("Study Hours vs Exam Score")
plt.tight_layout(); plt.savefig(ROOT/"outputs/practical_01_correlation.png"); plt.show()
