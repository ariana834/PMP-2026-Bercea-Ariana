import random

# datele problemei
p_boala = 0.01          # P(B) - cat de des apare boala in populatie
sensibilitate = 0.95    # P(pozitiv | bolnav)
specificitate = 0.90    # P(negativ | sanatos)

# daca testul e negativ la sanatosi in 90% din cazuri,
# atunci la sanatosi da pozitiv (fals pozitiv) in restul de 10%
p_fals_pozitiv = 1 - specificitate   # P(pozitiv | sanatos) = 0.10


# a) calculul teoretic cu formula lui Bayes
# vrem P(B | pozitiv) = P(pozitiv | B) * P(B) / P(pozitiv)
#
# P(pozitiv) o aflam cu formula probabilitatii totale:
# un test pozitiv poate veni ori de la un bolnav, ori de la un sanatos
# P(pozitiv) = P(pozitiv | B) * P(B) + P(pozitiv | nonB) * P(nonB)

p_pozitiv = sensibilitate * p_boala + p_fals_pozitiv * (1 - p_boala)
# = 0.95 * 0.01 + 0.10 * 0.99 = 0.0095 + 0.099 = 0.1085

p_boala_daca_pozitiv = sensibilitate * p_boala / p_pozitiv
# = 0.0095 / 0.1085 = aprox 0.0876

print("a) P(pozitiv) =", p_pozitiv)
print("a) P(bolnav | pozitiv) teoretic =", p_boala_daca_pozitiv)

# de ce iese asa mic (doar ~8.8%)?
# pentru ca boala e foarte rara. din 1000 de oameni:
#   - 10 sunt bolnavi -> testul prinde ~9.5 dintre ei (adevarat pozitivi)
#   - 990 sunt sanatosi -> testul greseste la 10% -> ~99 fals pozitivi
# deci din ~108.5 teste pozitive doar ~9.5 sunt bolnavi de fapt.
# sanatosii sunt atat de multi incat chiar si 10% greseli de la ei
# depasesc cu mult numarul de bolnavi reali.


# b) simulare
# generam N persoane la intamplare. pentru fiecare:
#   1. decidem daca e bolnava (cu prob 0.01)
#   2. ii facem testul (pozitiv cu 0.95 daca e bolnava, cu 0.10 daca nu)
# apoi ne uitam DOAR la cei cu test pozitiv si vedem cati sunt bolnavi

N = 1000000
nr_pozitivi = 0
nr_bolnavi_si_pozitivi = 0

for i in range(N):
    # random.random() da un numar uniform in [0, 1)
    # e mai mic decat 0.01 exact cu probabilitatea 0.01
    bolnav = random.random() < p_boala

    if bolnav:
        test_pozitiv = random.random() < sensibilitate
    else:
        test_pozitiv = random.random() < p_fals_pozitiv

    if test_pozitiv:
        nr_pozitivi += 1
        if bolnav:
            nr_bolnavi_si_pozitivi += 1

# probabilitatea conditionata estimata:
# dintre cei pozitivi, ce fractiune e bolnava
p_estimat = nr_bolnavi_si_pozitivi / nr_pozitivi

print("b) P(bolnav | pozitiv) simulat  =", p_estimat)
print("b) diferenta fata de teoretic   =", abs(p_estimat - p_boala_daca_pozitiv))


# c) specificitatea minima ca P(B | pozitiv) sa fie 50%
# notam s = specificitatea, deci P(pozitiv | sanatos) = 1 - s
#
# P(B | pozitiv) = 0.95 * 0.01 / (0.95 * 0.01 + (1 - s) * 0.99) = 0.5
#
# 0.5 inseamna ca numaratorul e jumatate din numitor, adica
# cele doua bucati din numitor sunt egale:
#   0.95 * 0.01 = (1 - s) * 0.99
#   1 - s = 0.0095 / 0.99
#   s = 1 - 0.0095 / 0.99 = aprox 0.9904
#
# adica testul trebuie sa aiba specificitate de ~99.04%,
# sa dea fals pozitiv la mai putin de 1% din sanatosi

s_minim = 1 - (sensibilitate * p_boala) / (1 - p_boala)
print("c) specificitate minima =", s_minim)

# verificare: punem s_minim inapoi in formula si trebuie sa iasa 0.5
verificare = sensibilitate * p_boala / (sensibilitate * p_boala + (1 - s_minim) * (1 - p_boala))
print("c) verificare P(bolnav | pozitiv) =", verificare)
