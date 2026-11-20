# 1) FIND-S Algorithm with Numerical Output and Colorful Graph
import matplotlib.pyplot as plt
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High',   'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Cool', 'Change', 'Yes']
]
attributes = [
    'Sky', 'AirTemp', 'Humidity',
    'Wind', 'Water', 'Forecast'
]
def find_s(data):
    hypothesis = ['0'] * len(attributes)
    hypothesis_history = []
    for example in data:
        if example[-1] == 'Yes':
            for i in range(len(attributes)):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
        hypothesis_history.append(hypothesis.copy())
    return hypothesis, hypothesis_history
final_hypothesis, history = find_s(training_data)
print("TRAINING DATA")
print("=" * 75)
print("Sky\tAirTemp\tHumidity\tWind\tWater\tForecast\tClass")
for row in training_data:
    print(
        f"{row[0]}\t{row[1]}\t{row[2]}\t\t"
        f"{row[3]}\t{row[4]}\t{row[5]}\t\t{row[6]}"
    )
print("\n\nHYPOTHESIS EVOLUTION")
print("=" * 75)

for i, h in enumerate(history):
    print(f"After Example {i + 1}: {h}")
print("\n\nFINAL MOST SPECIFIC HYPOTHESIS")
print("=" * 75)
for attribute, value in zip(attributes, final_hypothesis):
    print(f"{attribute:12} : {value}")
def numerical_value(value):
    if value == '0':
        return 0
    elif value == '?':
        return 2
    else:
        return 1
numeric_history = []
for h in history:
    numeric_history.append(
        [numerical_value(value) for value in h]
    )
print("\nNUMERICAL HYPOTHESIS VALUES")
print("=" * 75)
print("Example\tSky\tAirTemp\tHumidity\tWind\tWater\tForecast")
for i, values in enumerate(numeric_history):
    print(
        f"{i + 1}\t"
        f"{values[0]}\t"
        f"{values[1]}\t"
        f"{values[2]}\t\t"
        f"{values[3]}\t"
        f"{values[4]}\t"
        f"{values[5]}"
    )
plt.figure(figsize=(12, 7))
colors = [
    'red',
    'blue',
    'green',
    'orange',
    'purple',
    'brown'
]
markers = [
    'o',
    's',
    '^',
    'D',
    'P',
    '*'
]
examples = [1, 2, 3, 4]
for i, attribute in enumerate(attributes):
    values = [
        numeric_history[j][i]
        for j in range(len(numeric_history))
    ]
    plt.plot(
        examples,
        values,
        color=colors[i],
        marker=markers[i],
        linewidth=2.5,
        markersize=8,
        label=attribute
    )
plt.title(
    "FIND-S Algorithm - Hypothesis Evolution",
    fontsize=16,
    fontweight='bold'
)
plt.xlabel(
    "Training Example",
    fontsize=12
)
plt.ylabel(
    "Hypothesis Value",
    fontsize=12
)
plt.xticks(examples)
plt.yticks(
    [0, 1, 2],
    [
        '0 - Initial',
        '1 - Specific Value',
        '2 - Generalized (?)'
    ]
)
plt.grid(
    True,
    linestyle='--',
    alpha=0.4
)
plt.legend(
    title="Attributes",
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)
plt.tight_layout()
plt.show()
