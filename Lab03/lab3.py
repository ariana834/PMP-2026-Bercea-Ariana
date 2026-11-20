import itertools

# Structura retelei
# S -> O      (spam influenteaza daca apare "oferta")
# S -> L      (spam influenteaza daca are link-uri)
# S -> M      (spam influenteaza lungimea)
# L -> M      (link-urile influenteaza lungimea)
#
#        S
#      / | \
#     O  L  |
#         \ |
#           M
#
# intr-o retea bayesiana fiecare nod depinde direct doar de parintii lui:
#   S nu are parinti
#   O are parinte pe S
#   L are parinte pe S
#   M are parinti pe S si L
#
# de aici, probabilitatea comuna se descompune asa:
#   P(S, O, L, M) = P(S) * P(O | S) * P(L | S) * P(M | S, L)


# Tabelele de probabilitati (CPD-uri)
# tin doar probabilitatea ca variabila sa fie 1,
# pentru 0 iau 1 - valoarea respectiva

p_S = 0.4                       # P(S=1)

p_O = {1: 0.7, 0: 0.1}          # P(O=1 | S=s), cheia e s
p_L = {1: 0.8, 0: 0.3}          # P(L=1 | S=s)

p_M = {                         # P(M=1 | S=s, L=l), cheia e (s, l)
    (1, 1): 0.9,
    (1, 0): 0.5,
    (0, 1): 0.6,
    (0, 0): 0.2,
}


# functie mica: daca valoarea e 1 intoarce p, daca e 0 intoarce 1 - p
def prob(p, valoare):
    if valoare == 1:
        return p
    return 1 - p


# probabilitatea comuna P(S=s, O=o, L=l, M=m) cu descompunerea de mai sus
def comuna(s, o, l, m):
    return (prob(p_S, s)
            * prob(p_O[s], o)
            * prob(p_L[s], l)
            * prob(p_M[(s, l)], m))


# a) Independentele din retea
# le citim din graf cu d-separare. regulile pe scurt:
#   lant   A -> B -> C : blocat daca B e observat
#   fork   A <- B -> C : blocat daca B e observat
#   collider A -> B <- C : blocat daca B NU e observat (si nici descendentii lui)
#
# 1) O si L sunt independente stiind S   (O ⊥ L | S)
#    O <- S -> L e un fork, daca stim S calea se blocheaza.
#    intuitiv: daca stim deja ca e spam, faptul ca are "oferta"
#    nu ne mai spune nimic nou despre link-uri.
#
# 2) O si M sunt independente stiind S   (O ⊥ M | S)
#    toate caile de la O la M trec prin S (O <- S -> M si O <- S -> L -> M),
#    iar S e fork pe ambele, deci observandu-l le blocam pe toate.
#    (merge si daca stim S si L: O ⊥ M | S, L)
#
# 3) O ⊥ L | S, M  -> tot adevarat, S blocheaza ambele cai
#    (O <- S -> L si O <- S -> M <- L)
#
# 4) fara sa stim nimic, NU avem independente:
#    toate variabilele sunt legate prin S, deci de ex. O si L
#    sunt dependente (daca vezi "oferta" creste sansa sa fie spam,
#    deci creste si sansa sa aiba link-uri).
#
# verificam numeric independenta 1): P(O, L | S) == P(O | S) * P(L | S)

print("a) verificare O ⊥ L | S")
for s in [0, 1]:
    # P(S=s) = suma comunei peste toate celelalte variabile
    p_s = sum(comuna(s, o, l, m) for o in [0, 1] for l in [0, 1] for m in [0, 1])
    for o in [0, 1]:
        for l in [0, 1]:
            # P(O=o, L=l | S=s) = P(S=s, O=o, L=l) / P(S=s), iar M o adunam (marginalizam)
            p_ol = sum(comuna(s, o, l, m) for m in [0, 1]) / p_s
            p_o = prob(p_O[s], o)
            p_l = prob(p_L[s], l)
            print(f"  S={s} O={o} L={l}:  P(O,L|S) = {p_ol:.4f}   P(O|S)*P(L|S) = {p_o * p_l:.4f}")

# si verificam ca fara S nu sunt independente
p_o1 = sum(comuna(s, 1, l, m) for s in [0, 1] for l in [0, 1] for m in [0, 1])
p_l1 = sum(comuna(s, o, 1, m) for s in [0, 1] for o in [0, 1] for m in [0, 1])
p_o1_l1 = sum(comuna(s, 1, 1, m) for s in [0, 1] for m in [0, 1])
print(f"  fara S: P(O=1,L=1) = {p_o1_l1:.4f}  vs  P(O=1)*P(L=1) = {p_o1 * p_l1:.4f}  -> diferite, deci dependente")


# b) Clasificarea
# pentru un email vedem O, L, M si vrem P(S=1 | O, L, M).
# cu Bayes:
#   P(S=1 | o, l, m) = P(S=1, o, l, m) / P(o, l, m)
# iar P(o, l, m) = P(S=0, o, l, m) + P(S=1, o, l, m)
#
# regula de clasificare: spam daca P(S=1 | o, l, m) > 0.5
# (adica alegem clasa cu probabilitatea cea mai mare)
#
# trecem prin toate cele 2^3 = 8 combinatii posibile de (O, L, M)

print()
print("b) clasificare")
print("  O L M   P(spam | O,L,M)   clasa")
for o, l, m in itertools.product([0, 1], repeat=3):
    spam = comuna(1, o, l, m)
    non_spam = comuna(0, o, l, m)
    p_spam = spam / (spam + non_spam)

    if p_spam > 0.5:
        clasa = "SPAM"
    else:
        clasa = "non-spam"

    print(f"  {o} {l} {m}       {p_spam:.4f}        {clasa}")

# concluzie: emailul e spam daca contine "oferta" (O=1) si macar unul din L, M e 1.
# O e atributul cel mai puternic (0.7 vs 0.1, de 7 ori mai probabil la spam).
# L si M doar imping probabilitatea in sus sau in jos:
#   - O=1, L=0, M=0 da 0.45, deci non-spam desi are "oferta"
#   - O=0, L=1, M=1 ajunge la 0.47, aproape, dar tot non-spam
# fara "oferta" un email nu e niciodata clasificat ca spam.
