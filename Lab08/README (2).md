# Laborator 8 - Programare probabilista cu PyMC

## Modelul

- n = numarul de clienti din ziua respectiva, nu il stim
- prior: n ~ Poisson(10)
- verosimilitate: Y | n ~ Binomial(n, theta), theta e cunoscut
- vrem posteriorul p(n | Y) pt Y in {0, 5, 10} si theta in {0.2, 0.5}

Codul e in `lab8.py`, iar graficul cu toate cele 6 posterioare e in `posterior_n.png`.

## Ce am obtinut (aproximativ)

| Y  | theta | media posterioara a lui n | sd   |
|----|-------|---------------------------|------|
| 0  | 0.2   | ~8                        | ~2.8 |
| 0  | 0.5   | ~5                        | ~2.2 |
| 5  | 0.2   | ~13                       | ~2.8 |
| 5  | 0.5   | ~10                       | ~2.2 |
| 10 | 0.2   | ~18                       | ~2.8 |
| 10 | 0.5   | ~15                       | ~2.2 |

Valorile din MCMC pot sa difere putin de la o rulare la alta, dar ar trebui sa fie in jurul acestora.

## Efectul lui Y

- n nu poate fi mai mic decat Y (nu pot cumpara mai multi clienti decat au venit), deci posteriorul incepe de la Y
- cu cat Y e mai mare, cu atat posteriorul lui n se muta mai la dreapta: daca au cumparat multi, probabil au venit si multi
- practic n = Y (cei care au cumparat) + clientii care n-au cumparat; partea a doua nu depinde de Y, de asta la acelasi theta forma distributiei ramane la fel si doar se translateaza cu Y
- cand Y = 0 posteriorul e sub media priorului (10): faptul ca nu a cumparat nimeni sugereaza ca au venit mai putini clienti

## Efectul lui theta

- la theta mic (0.2) un client cumpara rar, deci ca sa vedem Y cumparaturi trebuie sa fi venit multi clienti -> posteriorul e mai la dreapta
- la theta mare (0.5) fiecare client cumpara mai des, deci ajung mai putini clienti ca sa explice acelasi Y -> posteriorul e mai la stanga
- theta mic inseamna si mai multa incertitudine (sd ~2.8 fata de ~2.2), pt ca sunt mai multi clienti "nevazuti" (care n-au cumparat) si despre ei stim doar ce zice priorul

## Pe scurt

Se poate arata si exact ca n - Y | Y ~ Poisson(10 * (1 - theta)), deci media posterioara este Y + 10 * (1 - theta). Asta se potriveste cu ce iese din PyMC: Y muta distributia, iar theta decide cati clienti in plus fata de Y ne asteptam sa fi fost.
