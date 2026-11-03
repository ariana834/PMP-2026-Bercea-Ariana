import csv
import random

studenti = []
with open("studenti.csv", encoding="utf-8") as f:
    cititor = csv.reader(f)
    for rand in cititor:
        if rand:
            studenti.append(rand[0].strip())
print("Lista studenților:", studenti)

k = int(input("Câți studenți aleg? "))

if k > len(studenti):
    print(f"Nu pot alege {k} studenți, sunt doar {len(studenti)} în listă.")
else:
    alesi = random.sample(studenti, k)
    print(f"\nAu fost aleși {k} studenți:")
    for i, nume in enumerate(alesi, 1):
        print(f"{i}. {nume}")