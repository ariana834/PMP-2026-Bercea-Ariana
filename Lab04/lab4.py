import itertools
import math
import networkx as nx

# muchiile din enunt (graf neorientat, fara sageti)
muchii = [
    ("A1", "A2"),
    ("A1", "A3"),
    ("A2", "A4"),
    ("A2", "A5"),
    ("A3", "A4"),
    ("A4", "A5"),
]
noduri = ["A1", "A2", "A3", "A4", "A5"]


# a) graful in pyAgrum + clicile

# partea cu pyAgrum e pentru vizualizare, merge in jupyter
# daca nu e instalat sarim peste ea si mergem mai departe
try:
    import pyAgrum as gum
    import pyAgrum.lib.notebook as gnb

    # in fastMRF scriem factorii separati prin ";"
    # "A2--A4--A5" inseamna un factor peste toate trei (clica de 3)
    mrf = gum.fastMRF("A1--A2;A1--A3;A3--A4;A2--A4--A5")
    gnb.showMRF(mrf)
except ImportError:
    print("pyAgrum nu e instalat, sar peste desen")

# clicile maximale le aflam cu networkx
# o clica = un grup de noduri in care oricare doua sunt legate
# maximala = nu mai poti adauga niciun nod si sa ramana clica
G = nx.Graph()
G.add_nodes_from(noduri)
G.add_edges_from(muchii)

clici = [sorted(c) for c in nx.find_cliques(G)]
print("a) clicile maximale:", clici)
# iese: [A2, A4, A5], [A1, A2], [A1, A3], [A3, A4]
# A2-A4-A5 e triunghi (toate 3 muchiile exista)
# A1-A2-A4-A3 formeaza un patrat fara diagonale (nu exista A1-A4 sau A2-A3),
# deci acolo clicile sunt doar muchiile


# b) probabilitatea comuna

beta = 1.0   # in enunt nu e dat, iau 1 (orice valoare > 0 da aceleasi concluzii)

# potentialul unei clici: phi(a_C) = exp(beta * suma a_i * a_j pentru i < j din clica)
def potential(clica, stare):
    suma = 0
    # combinations ia toate perechile i < j din clica
    for x, y in itertools.combinations(clica, 2):
        suma += stare[x] * stare[y]
    return math.exp(beta * suma)

# intr-o retea Markov probabilitatea comuna e produsul potentialelor
# pe clicile maximale, impartit la Z ca sa adune la 1:
#   P(a) = (1 / Z) * produs phi_C(a_C)
#   Z = suma produsului peste TOATE starile posibile
def nenormalizat(stare):
    produs = 1.0
    for c in clici:
        produs *= potential(c, stare)
    return produs

# toate cele 2^5 = 32 de stari
stari = []
for valori in itertools.product([0, 1], repeat=5):
    stare = dict(zip(noduri, valori))
    stari.append(stare)

Z = sum(nenormalizat(s) for s in stari)
print()
print("b) Z =", Z)

# calculam P pentru fiecare stare si le sortam descrescator
rezultate = []
for s in stari:
    p = nenormalizat(s) / Z
    rezultate.append((p, s))
rezultate.sort(key=lambda x: x[0], reverse=True)

print("primele 5 stari ca probabilitate:")
for p, s in rezultate[:5]:
    print("  ", [s[n] for n in noduri], round(p, 4))

print("verificare, suma tuturor:", sum(p for p, s in rezultate))

# starea cea mai probabila e [1, 1, 1, 1, 1]
# motiv: fiecare muchie apare in exact o clica, deci produsul de potentiale e
#   exp(beta * suma peste muchii a_i * a_j)
# a_i * a_j e 1 doar daca ambele sunt 1, deci cu beta > 0 cu cat mai multe
# muchii au ambele capete 1 cu atat mai bine. cand toate sunt 1 toate cele 6
# muchii contribuie -> exp(6) / Z, maximul.


# c) modelul Ising
# in modelul de acum, perechea 0-0 primeste a_i*a_j = 0, la fel ca 0-1.
# deci nu sunt favorizati vecinii egali, doar vecinii care sunt ambii 1.
# pentru Ising vrem ca perechea sa fie rasplatita cand valorile sunt EGALE
# (0-0 sau 1-1). doua variante echivalente:
#   - indicator: phi = exp(beta * suma [a_i == a_j])
#   - transformam in spini s = 2a - 1 (0 -> -1, 1 -> +1) si folosim s_i * s_j,
#     care e +1 daca sunt egale si -1 daca difera

def potential_ising(clica, stare):
    suma = 0
    for x, y in itertools.combinations(clica, 2):
        sx = 2 * stare[x] - 1
        sy = 2 * stare[y] - 1
        suma += sx * sy
    return math.exp(beta * suma)

def nenormalizat_ising(stare):
    produs = 1.0
    for c in clici:
        produs *= potential_ising(c, stare)
    return produs

Z_ising = sum(nenormalizat_ising(s) for s in stari)
rez_ising = sorted(((nenormalizat_ising(s) / Z_ising, s) for s in stari),
                   key=lambda x: x[0], reverse=True)

print()
print("c) Ising, primele 4 stari:")
for p, s in rez_ising[:4]:
    print("  ", [s[n] for n in noduri], round(p, 4))
# acum [0,0,0,0,0] si [1,1,1,1,1] sunt la egalitate pe primul loc,
# pentru ca in ambele toti vecinii sunt egali. modelul e simetric intre 0 si 1.
