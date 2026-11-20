# Partial PMP

#am pus toate importurile o singura data aici sus
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from hmmlearn import hmm
from scipy.stats import beta


# starile ascunse = camerele 1,2,3,4
# 0bservatiile= culorile raportate de senzor (0 = red, 1 = green, 2 = blue)

#fiecare camera are 1/4
start = np.array([0.25, 0.25, 0.25, 0.25])


def trans(stay, move):
    A = np.zeros((4, 4))
    for i in range(4):
        A[i, i] = stay              # ramane pe loc
        A[i, (i + 1) % 4] = move    # vecinul din dreapta
        A[i, (i - 1) % 4] = move    # vecinul din stanga
    # fiecare rand e o distributie de probabilitate, deci suma trebuie sa fie 1
    assert np.allclose(A.sum(axis=1), 1)
    return A


# B: emisiile B[camera, culoare] = P(culoare | camera), red green blue
B = np.array([[0.7, 0.2, 0.1],   # camera 1
              [0.2, 0.6, 0.2],   # camera 2
              [0.1, 0.3, 0.6],   # camera 3
              [0.5, 0.2, 0.3]])  # camera 4

# secventa observata: red, green, blue, blue, red
# hmmlearn vrea observatiile pe o coloana, de aici reshape(-1, 1)
obs = np.array([0, 1, 2, 2, 0]).reshape(-1, 1)


#ruleaza Viterbi si intoarce (cel mai probabil drum, probabilitatea lui), am ales asa ca e mult mai usor de implementat
#+observatiile sunt discrete
def solve(stay, move):
    # init_params="" ca hmmlearn sa nu suprascrie parametrii pusi de mn
    model = hmm.CategoricalHMM(n_components=4, init_params="")
    model.startprob_ = start
    model.transmat_ = trans(stay, move)
    model.emissionprob_ = B

    #viterbi = programare dinamica, am incercat sa explic pe foaie cat de cat ce am inteles, e Forward doar ca nu sum, cu max
    logp, states = model.decode(obs, algorithm="viterbi")

    # +1 ca sa numerotam camerele de la 1, ca in enunt, probabilitatea este P(drum, observatii) = probabilitatea comuna ceruta
    return states + 1, np.exp(logp)


# la a: robotul ramane pe loc cu 0.2 si se muta in fiecare vecin cu 0.4
drum_a, prob_a = solve(stay=0.2, move=0.4)

#la b: robotul NU mai poate ramane pe loc, iar cele doua mutari au 0.5 fiecare
drum_b, prob_b = solve(stay=0.0, move=0.5)
# -> se schimba doar matricea de tranzitie, restul (B, start, obs) e la fel

print("Subiectul 1")
print("(a) drum:", drum_a, " probabilitate:", prob_a)
print("(b) drum:", drum_b, " probabilitate:", prob_b)

# Comparatie:
# - drumul este acelasi (1 -> 2 -> 3 -> 4 -> 1): si la (a) drumul optim se mutala fiecare pas, deci interdictia de a ramane pe loc nu il afecteaza
# - probabilitatea creste la b, pentru ca fiecare din cele 4 mutari valoreaza 0.5 in loc de 0.4
print("raport (b)/(a):", prob_b / prob_a)


#Subiectul 2 - retele Markov
# Regula: fiecare factor este o clica, deci toate variabilele dintr-un factor se leaga intre ele. Asa obtinem graful.
# ideea formulei simplificate: P(X|rest) = produs factori /suma dupa x din produs factori, asa Z si toti factorii care NU il contin pe X apar sus si jos, deci se simplifica.
# raman doar factorii in care apare X(am scris pe foaie cum raman)

rng = np.random.default_rng(0)   # seed fix, ca rezultatele sa fie mereu aceleasi

#deseneaza graful retelei Markox si il salvam png
def deseneaza(muchii, titlu, fisier):
    G = nx.Graph()
    G.add_nodes_from("ABCDE")
    G.add_edges_from(muchii)
    plt.figure(figsize=(4, 3))
    nx.draw(G, nx.spring_layout(G, seed=1), with_labels=True,
            node_color="lightblue", node_size=900, font_size=14)
    plt.title(titlu)
    plt.savefig(fisier)
    plt.close()


#un potential cu valori random strict pozitive (cerinta pentru retele Markov)
def potential(*forma):
    return rng.random(forma) + 0.1

deseneaza([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
          "(a) lant", "retea_a.png")

phi1, phi2, phi3, phi4 = (potential(2, 2) for _ in range(4))

# factorizare: produsul factorilor pentru fiecare combinatie
P = np.zeros((2,) * 5)
for a, b, c, d, e in itertools.product([0, 1], repeat=5):
    P[a, b, c, d, e] = phi1[a, b] * phi2[b, c] * phi3[c, d] * phi4[d, e]
P /= P.sum()   # impartim la Z adica suma tuturor produselor

# P(A|B=b,C=c)marginalizam D si E, apoi normalizam dupa A
b, c = 1, 0
P_ABC = P.sum(axis=(3, 4))#P(A,B,C)
pe_lung = P_ABC[:, b, c] / P_ABC[:, b, c].sum()

# phi2(B,C) si suma dupa d, e nu il contin pe A, deci se simplifica
# c nu mai apare deloc: a e independent de c stiind b (b il separa pe a de rest)
formula = phi1[:, b] / phi1[:, b].sum()

print("(a) P(A | B, C) pe lung:", pe_lung, " formula:", formula)

deseneaza([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")],
          "(b) triunghi + 2 muchii", "retea_b.png")

psi1 = potential(2, 2, 2)
psi2 = potential(2, 2)
psi3 = potential(2, 2)

Q = np.zeros((2,) * 5)
for a, b, c, d, e in itertools.product([0, 1], repeat=5):
    Q[a, b, c, d, e] = psi1[a, b, c] * psi2[b, d] * psi3[c, e]
Q /= Q.sum()


a, b, c, e = 0, 1, 1, 0
pe_lung = Q[a, b, c, :, e] / Q[a, b, c, :, e].sum()

# formula simplificata: P(D | A,B,C,E) = psi2(B, D) / suma_d psi2(B, d)
# psi1 si psi3 nu il contin pe D, deci se simplifica, iar singurul vecin al lui D in graf este B
formula = psi2[b, :] / psi2[b, :].sum()

print("(b) P(D | A,B,C,E) pe lung:", pe_lung, " formula:", formula)
# daca cele doua rezultate coincid, formula simplificata e verificata numeric


#Subiectul 3-Inferenta
# Beta este prior conjugat pentru Binomial, deci posteriorul e tot Beta, am aratat in notite
# ideea e ca alpha si betha numara doua chestii diferite, DECI NU trebuie integrat nimic.

#intoarce parametrii posteriorului
def update(a, b, defecte, total):
    return a + defecte, b + (total - defecte)


prior = (2, 8)

post1 = update(*prior, defecte=5, total=20)   #a dupa primul lot -> Beta(7, 23)
post2 = update(*post1, defecte=3, total=15)   #b posteriorul de la a devine prior -> Beta(10, 35)
doar2 = update(*prior, defecte=3, total=15)   #c doar lotul 2, de la Beta(2, 8) -> Beta(5, 20)


x = np.linspace(0, 1, 500)
plt.figure()
for nume, (a, b) in [("prior", prior), ("(a) lot 1", post1),("(b) lot 1+2", post2), ("(c) doar lot 2", doar2)]:
    print(f"{nume:15} Beta({a},{b})  media = {a / (a + b):.4f}")
    plt.plot(x, beta.pdf(x, a, b), label=f"{nume}: Beta({a},{b})")

plt.xlabel("theta")
plt.ylabel("densitate")
plt.title("Prior vs posterior")
plt.legend()
plt.savefig("subiect3.png")
plt.close()

# Comparatie (b) vs (c):
# - b foloseste ambele loturi (8 defecte din 35). E acelasi rezultat ca si cum
#   am fi actualizat priorul o singura data cu toate datele: actualizarea
#   secventiala e echivalenta cu cea in bloc.
# - c foloseste doar lotul 2 (3 din 15), deci media este 0.2, exact cat priorul.
# - b are media 0.222, pentru ca "tine minte" lotul 1, care avea mai multe defecte,si o curba mai ingusta (mai multe date = mai putina incertitudine despre theta).