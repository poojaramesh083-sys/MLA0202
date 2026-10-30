import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=X.columns, class_names=encoders['PlayTennis'].classes_, filled=True, rounded=True)
plt.title('ID3 Decision Tree - Play Tennis Dataset')
plt.show()
new_sample = pd.DataFrame([{
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}])
for col in new_sample.columns:
    new_sample[col] = encoders[col].transform(new_sample[col])
prediction = model.predict(new_sample)[0]
print("New Sample:", {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'})
print("Predicted Class:", encoders['PlayTennis'].inverse_transform(prediction.reshape(1))[0])
print("Training Accuracy:", round(accuracy_score(y, model.predict(X)) * 100, 2), "%")
