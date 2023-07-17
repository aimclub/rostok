from dataclasses import dataclass

@dataclass
class Varezhka:
    size_x : float = 0
    size_y : float = 10
    rezinochka : bool = False


big = Varezhka(50)
small = Varezhka(size_y=1)
na_rezinke = Varezhka(rezinochka=True)

