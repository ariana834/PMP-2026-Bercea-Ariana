import random

# urna de la inceput: 3 rosii, 4 albastre, 2 negre
# o tin ca o lista de stringuri, fiecare element e o bila
def experiment():
    urna = ["rosu"] * 3 + ["albastru"] * 4 + ["negru"] * 2

    # aruncam zarul, randint(1, 6) include si 1 si 6
    zar = random.randint(1, 6)

    # 2, 3, 5 sunt prime (1 nu e prim!) -> adaugam neagra
    if zar in [2, 3, 5]:
        urna.append("negru")
    # 6 -> adaugam rosie
    elif zar == 6:
        urna.append("rosu")
    # au ramas 1 si 4 -> adaugam albastra
    else:
        urna.append("albastru")

    # extragem o bila la intamplare din urna (acum are 10 bile)
    # choice alege uniform un element din lista
    bila = random.choice(urna)
    return bila


# b) repetam experimentul de multe ori si numaram de cate ori iese rosu
# cu cat N e mai mare, cu atat estimarea e mai aproape de valoarea reala
# (legea numerelor mari)
N = 100000
nr_rosii = 0

for i in range(N):
    if experiment() == "rosu":
        nr_rosii += 1

# probabilitatea estimata = de cate ori a iesit rosu / cate incercari am facut
p_estimat = nr_rosii / N
print("Probabilitate estimata:", p_estimat)


# c) probabilitatea teoretica, cu formula probabilitatii totale
# P(rosu) = suma peste cazuri de P(caz) * P(rosu | caz)

# caz 1: prim (2,3,5) -> 3/6 = 1/2, urna devine 3R 4A 3N -> P(rosu) = 3/10
# caz 2: zar = 6      -> 1/6,        urna devine 4R 4A 2N -> P(rosu) = 4/10
# caz 3: 1 sau 4      -> 2/6 = 1/3,  urna devine 3R 5A 2N -> P(rosu) = 3/10
p_teoretic = (1/2) * (3/10) + (1/6) * (4/10) + (1/3) * (3/10)
print("Probabilitate teoretica:", p_teoretic)

# diferenta dintre ele ar trebui sa fie foarte mica
print("Diferenta:", abs(p_estimat - p_teoretic))
