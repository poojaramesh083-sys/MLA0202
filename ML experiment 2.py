import csv

def candidate_elimination(filename):
    with open(filename, "r") as file:
        data = list(csv.reader(file))

    attributes = data[0][:-1]
    examples = data[1:]

    S = [None] * len(attributes)
    G = [["?" for _ in attributes]]

    print("\n" + "=" * 85)
    print("              CANDIDATE-ELIMINATION ALGORITHM")
    print("=" * 85)

    print("\nTRAINING DATA")
    print("-" * 90)
    print(f"{'No.':<5}", end="")
    for attr in attributes:
        print(f"{attr:<13}", end="")
    print("Target")
    print("-" * 90)

    for i, row in enumerate(examples, 1):
        print(f"{i:<5}", end="")
        for value in row[:-1]:
            print(f"{value:<13}", end="")
        print(row[-1])

    for example in examples:
        x = example[:-1]
        target = example[-1].strip().lower()

        if target == "yes":
            G = [
                g for g in G
                if all(g[i] == "?" or g[i] == x[i]
                       for i in range(len(x)))
            ]

            for i in range(len(S)):
                if S[i] is None:
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = "?"

        else:
            new_G = []

            for g in G:
                if all(g[i] == "?" or g[i] == x[i]
                       for i in range(len(x))):
                    for i in range(len(g)):
                        if (S[i] is not None and S[i] != "?"
                                and S[i] != x[i]):
                            h = g.copy()
                            h[i] = S[i]

                            if h not in new_G:
                                new_G.append(h)
                else:
                    new_G.append(g)

            G = new_G

    print("\n" + "=" * 85)
    print("                       FINAL RESULTS")
    print("=" * 85)

    print("\nSPECIFIC BOUNDARY (S)")
    print("-" * 85)
    print("  <" + ", ".join(str(v) for v in S) + ">")

    print("\nGENERAL BOUNDARY (G)")
    print("-" * 85)
    for i, hypothesis in enumerate(G, 1):
        print(f"  G{i}: <" + ", ".join(hypothesis) + ">")

    print("\n" + "=" * 85)
    print("Legend: '?' = Any value")
    print("        S   = Most specific boundary")
    print("        G   = Most general boundary")
    print("=" * 85)


candidate_elimination("training_data.csv")
import csv
import matplotlib.pyplot as plt

with open("training_data.csv", "r") as file:
    data = list(csv.reader(file))

examples = data[1:]
positive = sum(row[-1].strip().lower() == "yes" for row in examples)
negative = sum(row[-1].strip().lower() == "no" for row in examples)

plt.bar(["Positive", "Negative"], [positive, negative])
plt.title("Candidate-Elimination: Training Examples")
plt.xlabel("Example Type")
plt.ylabel("Number of Examples")
plt.tight_layout()
plt.savefig("output.png")
plt.show()
