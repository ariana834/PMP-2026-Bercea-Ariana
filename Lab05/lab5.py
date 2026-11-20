import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# starile ascunse = dificultatea testului (nu o vedem direct)
stari = ["Dificil", "Mediu", "Usor"]

# observatiile = notele studentului (pe astea le vedem)
note = ["FB", "B", "S", "NS"]

# probabilitatile de start: primul test are aceeasi sansa sa fie oricare
start = np.array([1/3, 1/3, 1/3])

# matricea de tranzitie: linia = testul de acum, coloana = testul urmator
# dupa dificil -> doar mediu sau usor, cu 0.5 fiecare
# dupa mediu sau usor -> dificil 0.5, mediu 0.25, usor 0.25
tranzitii = np.array([
    [0.0, 0.5,  0.5],    # din Dificil
    [0.5, 0.25, 0.25],   # din Mediu
    [0.5, 0.25, 0.25],   # din Usor
])

# matricea de emisie: linia = dificultatea, coloana = nota (FB, B, S, NS)
emisii = np.array([
    [0.1,  0.2,  0.4, 0.3],   # Dificil
    [0.15, 0.25, 0.5, 0.1],   # Mediu
    [0.2,  0.3,  0.4, 0.1],   # Usor
])

# secventa observata, transformata in indici (FB=0, B=1, S=2, NS=3)
secventa = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B", "S"]
obs = np.array([note.index(n) for n in secventa])


# a) modelul in hmmlearn

# CategoricalHMM e modelul pentru observatii discrete (categorii)
# (in versiunile vechi se numea MultinomialHMM)
try:
    from hmmlearn import hmm

    model = hmm.CategoricalHMM(n_components=3)
    # nu antrenam modelul, ii dam direct parametrii din enunt
    model.startprob_ = start
    model.transmat_ = tranzitii
    model.emissionprob_ = emisii
    model.n_features = 4
    are_hmmlearn = True
except ImportError:
    print("hmmlearn nu e instalat, folosesc doar Viterbi scris de mana")
    are_hmmlearn = False

# diagrama de stare: un nod pentru fiecare dificultate,
# o sageata pentru fiecare tranzitie cu probabilitate > 0
G = nx.DiGraph()
for i in range(3):
    for j in range(3):
        if tranzitii[i][j] > 0:
            G.add_edge(stari[i], stari[j], weight=tranzitii[i][j])

pozitii = {"Dificil": (0, 1), "Mediu": (-1, 0), "Usor": (1, 0)}
plt.figure(figsize=(6, 5))
nx.draw_networkx_nodes(G, pozitii, node_size=2500, node_color="lightblue")
nx.draw_networkx_labels(G, pozitii)
# connectionstyle cu arc ca sa nu se suprapuna sagetile dus-intors
# buclele (Mediu -> Mediu, Usor -> Usor) le scriem separat ca text,
# altfel networkx le deseneaza ascunse sub nod
fara_bucle = [(u, v) for u, v in G.edges() if u != v]
nx.draw_networkx_edges(G, pozitii, edgelist=fara_bucle, arrowsize=20,
                       node_size=2500, connectionstyle="arc3,rad=0.15")
etichete = {(u, v): G[u][v]["weight"] for u, v in fara_bucle}
nx.draw_networkx_edge_labels(G, pozitii, edge_labels=etichete,
                             connectionstyle="arc3,rad=0.15")
for u, v in G.edges():
    if u == v:
        x, y = pozitii[u]
        plt.text(x, y - 0.3, f"bucla {G[u][v]['weight']}", ha="center")
plt.title("Diagrama de stare")
plt.axis("off")
plt.savefig("diagrama_lab5.png")
# plt.show()   # decomenteaza daca rulezi local si vrei fereastra


# b) cea mai probabila secventa de dificultati -> algoritmul Viterbi

# Viterbi scris de mana, ca sa se vada ce face
# v[t][j] = probabilitatea celui mai bun drum care ajunge in starea j la pasul t
# inapoi[t][j] = din ce stare am venit pe drumul ala (ca sa reconstruim drumul)
def viterbi(obs):
    T = len(obs)
    v = np.zeros((T, 3))
    inapoi = np.zeros((T, 3), dtype=int)

    # pasul 0: start * cat de probabila e prima nota in fiecare stare
    v[0] = start * emisii[:, obs[0]]

    for t in range(1, T):
        for j in range(3):
            # incercam toate starile i din care putem veni in j
            # si pastram doar drumul cel mai bun
            candidati = v[t - 1] * tranzitii[:, j]
            inapoi[t][j] = np.argmax(candidati)
            v[t][j] = candidati.max() * emisii[j, obs[t]]

    # la final alegem starea cu valoarea cea mai mare
    # si mergem inapoi pe sageti ca sa aflam tot drumul
    drum = [int(np.argmax(v[T - 1]))]
    for t in range(T - 1, 0, -1):
        drum.append(int(inapoi[t][drum[-1]]))
    drum.reverse()

    return drum, v[T - 1].max()

drum, prob = viterbi(obs)
print("b) Viterbi de mana:")
print("   secventa:", [stari[s] for s in drum])
print("   probabilitate:", prob)

# acelasi lucru cu hmmlearn
# decode intoarce log-probabilitatea, deci facem exp ca sa avem probabilitatea
if are_hmmlearn:
    logprob, drum_hmm = model.decode(obs.reshape(-1, 1), algorithm="viterbi")
    print("b) hmmlearn:")
    print("   secventa:", [stari[s] for s in drum_hmm])
    print("   probabilitate:", np.exp(logprob))
