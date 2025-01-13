import streamlit as st

st.header("Фактор распределения вертикального удара")

st.markdown(r"""
Данный критерий измеряет нормированною величину инерции концевого эффектора. Для этого используется матрица инерции в операционном пространстве: 

$$
\Lambda(x)=(J(q(x))A(q(x))^{-1}J(q(x))^T)^{-1},
$$  

где $A$- матрица инерции в пространстве актуированных сочленений, $J$ - якобиан связывающий обобщённую скорость $\dot{q}$ и скорость в операционном пространстве $\dot{x}$

Для нормировки используется значение матрицы инерции в операционном пространстве при условии, что каждое сочленение неподвижно $\Lambda_L$. Как критерий нас интересует только проекция на вертикальную ось, поэтому итоговое выражение имеет вид:  

$$  
R=\frac{1}{n}\sum_1^n(1-\frac{z^T\Lambda z}{z^T\Lambda_L z}),
$$  

где $z$ - единичный вектор вдоль оси $Z$, n - число точек на траектории.

---


1. P. M. Wensing, A. Wang, S. Seok, D. Otten, J. Lang and S. Kim, "Proprioceptive Actuator Design in the MIT Cheetah: Impact Mitigation and High-Bandwidth Physical Interaction for Dynamic Legged Robots," in _IEEE Transactions on Robotics_, vol. 33, no. 3, pp. 509-522, June 2017, doi: [10.1109/TRO.2016.2640183](https://doi.org/10.1109/TRO.2016.2640183)  
2. V. Batto, T. Flayols, N. Mansard and M. Vulliez, "Comparative Metrics of Advanced Serial/Parallel Biped Design and Characterization of the Main Contemporary Architectures," _2023 IEEE-RAS 22nd International Conference on Humanoid Robots (Humanoids)_, Austin, TX, USA, 2023, pp. 1-7, doi: [10.1109/Humanoids57100.2023.10375224](https://doi.org/10.1109/Humanoids57100.2023.10375224)

""")