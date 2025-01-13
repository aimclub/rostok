import streamlit as st

st.header("Манипулируемость")

st.markdown(r"""
Манипулируемость это кинематическая характеристика оценивающая связь между скоростью актуаторов и скоростью концевого эффектора. Данный критерий характеризует полную манипулируемость по всем направлениям. Единичный круг в пространтсве скоростей актуаторов преобразуется Якобианом в эллипс манипулируемости:

$$
v^T(JJ^T)^{-1}v=1
$$

Объём(площадь) эллипса пропорционален определителю Якобиана. Чем больше объём, тем большие скорости концевого эффектора достижимы при "единичной" скорости актуаторов $\|q\|=1$. Критерий в точке равен определителю Якобиана, а полное значение вдоль траектории равно: 

$$
R=\frac{1}{n}\sum_1^n(\det{J}),
$$

где $J$ - якобиан связывающий обобщённую скорость $\dot{q}$ и скорость в операционном пространстве $\dot{x},$ n - число точек на траектории.

---

1. V. Batto, T. Flayols, N. Mansard and M. Vulliez, "Comparative Metrics of Advanced Serial/Parallel Biped Design and Characterization of the Main Contemporary Architectures," _2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids)_, Austin, TX, USA, 2023, pp. 1-7, doi: [10.1109/Humanoids57100.2023.10375224](https://doi.org/10.1109/Humanoids57100.2023.10375224)
2. Spong, M.W.; Hutchinson, Seth; Vidyasagar, M. (2005). [_Robot Modeling and Control_](https://books.google.com/books?id=muCMAAAACAAJ). Wiley. Wiley. [ISBN](https://en.wikipedia.org/wiki/ISBN_(identifier) "ISBN (identifier)") [9780471765790](https://en.wikipedia.org/wiki/Special:BookSources/9780471765790 "Special:BookSources/9780471765790").

""")