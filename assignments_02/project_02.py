import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Task 1

base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/student_performance_math.csv"

# Load the dataset into a pandas DataFrame. The csv file is separated by semicolons (;), so we need to make sure to specify the correct separator when reading the file.
df = pd.read_csv(base_url, sep=';')

print(df.shape)
print(df.head(5))
print(df.dtypes)

plt.hist(df['G3'], bins=21, color='blue', alpha=0.7)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.savefig("outputs/g3_distribution.png")
plt.show()

# Task 2: Preprocess the data

filtered_df = df[df["G3"] != 0]
print(df.shape)
print(filtered_df.shape)

# Keeping G3 = 0 rows would make the model think that more absences lead to low grades, even though those students didn’t actually take the exam, so their grades don’t reflect their true performance.

# Convert categorical variables to numeric using one-hot encoding or label encoding as appropriate.
cols = ["schoolsup", "internet", "higher", "activities"]

for col in cols:
    filtered_df[col] = filtered_df[col].replace({"yes": 1, "no": 0})


filtered_df["sex"] = filtered_df["sex"].replace({"M": 1, "F": 0})

# compare absences with G3
corr_original = df["absences"].corr(df["G3"])
corr_filtered = filtered_df["absences"].corr(filtered_df["G3"])
print(
    f"Correlation between absences and G3 in original dataset: {corr_original:.4f}")
print(
    f"Correlation between absences and G3 in filtered dataset: {corr_filtered:.4f}")

# corr_filtered is more trustworthy because it only includes students who took the exam, so the relationship between absences and grades reflects actual performance instead of missing or misleading data.


# scatter plot of absences vs G3
plt.scatter(filtered_df["absences"],
            filtered_df["G3"], color="blue", alpha=0.7)
plt.title("Absences vs Final Grade (G3)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.grid()
plt.savefig("outputs/absences_vs_g3.png")
plt.show()


# Task 3: Exploratory Data Analysis

g3_corrs = filtered_df.corr()["G3"].sort_values()
print(g3_corrs)

# a) first plot: studytime vs G3
plt.scatter(filtered_df["studytime"], filtered_df["G3"])
plt.title("Study Time vs Final Grade (G3)")
plt.xlabel(
    "Study Time (1=less than 2 hours, 2=2 to 5 hours, 3=5 to 10 hours, 4=more than 10 hours)")
plt.ylabel("Final Grade (G3)")
plt.grid()
plt.savefig("outputs/studytime_vs_g3.png")
plt.show()

# The scatter plot shows a negative relationship because students with more failures tend to have lower final grades, although the pattern is not perfect and the points are somewhat spread out.


# b) second plot: failures vs G3

plt.scatter(filtered_df["failures"], filtered_df["G3"])
plt.title("Failures vs Final Grade (G3)")
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade (G3)")
plt.grid()
plt.savefig("outputs/failures_vs_g3.png")
plt.show()

# The scatter plot shows a negative relationship because students with more failures tend to have lower final grades, and as failures increase, grades tend to decrease.


# Task 4: Baseline Model

x = filtered_df[["failures"]]
y = filtered_df["G3"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Linear Regression Question 3

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(x_test, y_test)

print("RMSE:", rmse)
print("R² on the test set:", r2)

# For each additional failure, the final grade decreases by about 1.43 points
# An RMSE of about 3 means the model’s predictions are somewhat off, since it can miss the actual grade by around 3 points on a 0–20 scale.
# An R² of about 0.20 means that failures explain only about 20% of the variation in final grades, so there are likely many other factors influencing student performance that are not captured by this model.
# The model is weak because using only one feature (failures) is too limited, and it does not capture other important factors that affect student performance.

# Task 5: Final Model

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime", "G1"]

X = filtered_df[feature_cols].values
y = filtered_df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42

)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

model_final = LinearRegression()
model_final.fit(X_train, y_train)
y_pred_final = model_final.predict(X_test)

rmse_final = np.sqrt(np.mean((y_pred_final - y_test) ** 2))
r2_train = model_final.score(X_train, y_train)
r2_final = model_final.score(X_test, y_test)

print("Train R²:", r2_train)
print("Test R²:", r2_final)
print("RMSE:", rmse_final)

for name, coef in zip(feature_cols, model_final.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Interpretation:
# Adding more features slightly improved the model because the R² increased, meaning the model explains more variation in final grades. The negative coefficient for schoolsup may be because students who receive extra support are already struggling, so their grades tend to be lower. Students with internet access tend to have higher grades, possibly because they have better access to learning resources, although this does not prove causation. The train and test R² are close, which suggests the model is not overfitting, but the overall R² is still low, meaning the model does not explain much of the variation in grades.

# Task 6: Evaluate and Summarize

plt.scatter(y_pred_final, y_test, color="blue", alpha=0.7)
plt.plot([0, 20], [0, 20], color="red")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Grade (G3)")
plt.ylabel("Actual Final Grade (G3)")
plt.grid()
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# Interpretation:

# The points are not tightly clustered around the diagonal, showing that the model is not very precise. The model struggles more at the high end, where it tends to underestimate student performance. A point above the diagonal means the actual grade is higher than the predicted grade (underestimation), while a point below the diagonal means the model overestimates the grade.

# The filtered dataset contains 357 rows, and the test set contains 72 rows after the train-test split.

# The model has an RMSE of about 2.86, which means predictions are typically off by around 3 points on a 0–20 scale. This is a moderate error and shows the model is not very precise. The R² value of about 0.15 means the model explains only about 15% of the variation in student grades, so most of the variation is due to other factors not included in the model.

# The feature with the largest negative coefficient is schoolsup, meaning students receiving extra school support tend to have lower predicted grades. This likely reflects that these students are already struggling academically. The feature with the largest positive coefficient is internet, meaning students with internet access tend to have higher predicted grades, possibly due to better access to learning resources.

# One surprising result is that school support has a negative coefficient. This is unexpected because support should help students, but it likely reflects that students receiving support are those who are already performing poorly.


# NEGLECTED G1

# Adding G1 greatly increases the model’s R², showing that it is a very strong predictor of the final grade. However, this does not mean that G1 causes G3. Instead, G1 is simply an earlier measurement of the same academic performance, so it is naturally highly correlated with the final grade. This model is not very useful for early intervention because G1 is only available after the first grading period, meaning students may already be struggling by the time it is known. To intervene earlier, educators would need to build a model using only background and behavioral features, such as study time, absences, and prior failures, which are available before students receive their first grade.
