#!/usr/bin/env python
# coding: utf-8

# ## MSDS 460 Decision Analytics Assignment 4 Python

# In[38]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[39]:


overall_start_time = time.time()
start_time = time.time()
file = "/Users/nikhil/Desktop/drug200.csv" 
drug = pd.read_csv(file)

end_time = time.time()

elapsed_time = end_time - start_time
drug.head()


# In[40]:


print(f"It took {elapsed_time} to pull the data")


# In[41]:


drug.isna().sum()


# In[42]:


start_time = time.time()
sns.pairplot(drug, hue = 'Drug')
end_time = time.time()
elapsed_time = end_time - start_time


# In[43]:


print(f"It took {elapsed_time} to create this graph")


# In[44]:


drug['Drug'].sum


# In[45]:


fig = sns.FacetGrid(drug, hue="Drug", aspect=3, palette="Set2")
fig.map(sns.kdeplot, "Na_to_K", shade=True)
fig.add_legend()


# In[46]:


start_time = time.time()

sex_counts = drug['Sex'].value_counts()
bp_counts = drug['BP'].value_counts()
drug_counts = drug['Drug'].value_counts()


def pie_with_percentages(data, labels=None, colors=None, title="Pie Chart with Percentages"):
    percentages = [f"{label}\n{count / sum(data) * 100:.1f}%" for label, count in zip(labels, data)]
    plt.pie(data, labels=percentages, colors=colors, autopct='%1.1f%%')
    plt.title(title)


sex_colors = ['pink', 'lightblue']
bp_colors = ["#FF6347", "#5B9BD5", "#D3D3D3"]
drug_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700']


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot pie chart for Sex distribution
plt.sca(axs[0])
pie_with_percentages(sex_counts.values, labels=['Female', 'Male'], colors=sex_colors, title="Distribution of Gender")
plt.legend(['Female', 'Male'], loc="upper right")

plt.sca(axs[1])
pie_with_percentages(bp_counts.values, labels=bp_counts.index, colors=bp_colors, title="Distribution of Blood Pressure Levels")
plt.legend(bp_counts.index, loc="upper right")

plt.sca(axs[2])
pie_with_percentages(drug_counts.values, labels=drug_counts.index, colors=drug_colors, title="Distribution of Drugs")
plt.legend(drug_counts.index, loc="upper right")

plt.tight_layout()
plt.show()

end_time = time.time()

elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to run these graphs")


# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)  

start_time = time.time()

simulated_drug = drug.copy()
simulated_drug['Age'] = simulated_drug['Age'] + np.random.normal(0, 2, size=len(drug))  
simulated_drug['Na_to_K'] = simulated_drug['Na_to_K'] + np.random.normal(0, 1, size=len(drug))  

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(drug['Age'], bins=15, alpha=0.6, color='blue', label='Original')
plt.hist(simulated_drug['Age'], bins=15, alpha=0.6, color='green', label='Simulated')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(drug['Na_to_K'], bins=15, alpha=0.6, color='blue', label='Original')
plt.hist(simulated_drug['Na_to_K'], bins=15, alpha=0.6, color='green', label='Simulated')
plt.title('Distribution of Na_to_K')
plt.xlabel('Na_to_K')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")



# In[48]:


start_time = time.time()

# Cross-tabulations
sex_drug_crosstab = pd.crosstab(drug['Sex'], drug['Drug'], margins=True, margins_name="Total")
print("Cross-tabulation between Sex and Drug:\n", sex_drug_crosstab)

bp_drug_crosstab = pd.crosstab(drug['BP'], drug['Drug'], margins=True, margins_name="Total")
print("\nCross-tabulation between BP and Drug:\n", bp_drug_crosstab)

cholesterol_drug_crosstab = pd.crosstab(drug['Cholesterol'], drug['Drug'], margins=True, margins_name="Total")
print("\nCross-tabulation between Cholesterol and Drug:\n", cholesterol_drug_crosstab)

# Heatmap for Sex vs Drug
plt.figure(figsize=(10, 6))
sns.heatmap(sex_drug_crosstab.iloc[:-1, :-1], annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title('Heatmap: Sex vs Drug')
plt.xlabel('Drug Class')
plt.ylabel('Sex')
plt.show()

# Heatmap for BP vs Drug
plt.figure(figsize=(10, 6))
sns.heatmap(bp_drug_crosstab.iloc[:-1, :-1], annot=True, cmap="Oranges", fmt="d", cbar=False)
plt.title('Heatmap: BP vs Drug')
plt.xlabel('Drug Class')
plt.ylabel('Blood Pressure')
plt.show()

# Heatmap for Cholesterol vs Drug
plt.figure(figsize=(10, 6))
sns.heatmap(cholesterol_drug_crosstab.iloc[:-1, :-1], annot=True, cmap="Greens", fmt="d", cbar=False)
plt.title('Heatmap: Cholesterol vs Drug')
plt.xlabel('Drug Class')
plt.ylabel('Cholesterol')
plt.show()

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")


# ## Normal Model Logistic Regression Model

# In[49]:


start_time = time.time()
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()
le_drug = LabelEncoder()

drug['Sex'] = le_sex.fit_transform(drug['Sex'])
drug['BP'] = le_bp.fit_transform(drug['BP'])
drug['Cholesterol'] = le_cholesterol.fit_transform(drug['Cholesterol'])
drug['Drug'] = le_drug.fit_transform(drug['Drug'])

# Define features (X) and target (y)
X = drug[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = drug['Drug']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Generate confusion matrix and classification report
class_report = classification_report(y_test, y_pred, target_names=le_drug.classes_)


print("\nClassification Report:")
print(class_report)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nIt took {round(elapsed_time,2)} seconds to run a Logistic Regression Model as well as a classification report")


# ## Logistic Regression Monte Carlo Experiment 

# In[50]:


label_encoder = LabelEncoder()

drug['Sex'] = label_encoder.fit_transform(drug['Sex'])
drug['BP'] = label_encoder.fit_transform(drug['BP'])
drug['Cholesterol'] = label_encoder.fit_transform(drug['Cholesterol'])
drug['Drug'] = label_encoder.fit_transform(drug['Drug'])


train_times = []
accuracies = []
classification_reports = []


num_simulations = 1000


def simulate_drug_data():
    X = drug.drop(columns=['Drug'])  
    y = drug['Drug']

    return X, y


for i in range(num_simulations):
    X, y = simulate_drug_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)  
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_times.append(end_time - start_time)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    class_report = classification_report(y_test, y_pred)
    classification_reports.append(class_report)

avg_train_time = np.mean(train_times)
avg_accuracy = np.mean(accuracies)



# In[51]:


print(f"Average training time: {avg_train_time:.4f} seconds")
print(f"Average accuracy: {avg_accuracy:.4f} seconds")
print("\nClassification Report (Example of last simulation):")
print(classification_reports[-1])


# ## Random Forest Monte Carlo Experiment 

# In[52]:


X = drug.drop('Drug', axis=1)  
y = drug['Drug']  

def monte_carlo_rf(X, y, n_simulations=100):
    accuracies = []
    training_times = []
    classification_reports = []


    for i in range(n_simulations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        model = RandomForestClassifier(n_estimators=100, random_state=i)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        training_times.append(elapsed_time)
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        accuracies.append(accuracy)

        classification_reports.append(classification_report(y_test, y_pred, output_dict=True))

    return accuracies, training_times, classification_reports
accuracies, training_times, classification_reports = monte_carlo_rf(X, y, n_simulations=100)
avg_accuracy = np.mean(accuracies)
avg_training_time = np.mean(training_times)

print(f"It took {avg_train_time} to train the Random Forest Experiment")
print(f"The accuracy model score is: {avg_accuracy}")
print(f"It took {elapsed_time} to run the Random Forest Experiment")


# In[53]:


plt.figure(figsize=(10, 6))
plt.hist(accuracies, bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Accuracies Across Simulations')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(training_times, bins=20, color='green', alpha=0.7)
plt.title('Distribution of Training Times Across Simulations')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Frequency')
plt.show()

print("Classification Report for Last Simulation:")
print(classification_reports[-1])
overall_end_time = time.time()


# In[54]:


total_elapsed_time = overall_end_time - overall_start_time

print(f"\nTotal execution time for the entire script: {total_elapsed_time:.2f} seconds")

