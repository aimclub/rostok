
<p align="center">
    <img src="/docs/images/logo_rostok_long.png" width="600">

</p>

[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)

[![Documentation Status](https://readthedocs.org/projects/rostok/badge/?version=latest)](https://rostok.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/github/license/aimclub/rostok)](https://github.com/aimclub/rostok/blob/master/LICENSE)
[![Eng](https://img.shields.io/badge/lang-en-red.svg)](/README_en.rst)
[![Mirror](https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162)](https://gitlab.actcognitive.org/itmo-sai-code/rostok/)

# Rostok

Rostok - это  open source Python framework для генеративного дизайна рычажных механизмов для роботехнических систем. Обеспечивает основу для описания механизмов в виде графа, настройки окружения, выполнения моделирования для сгенерированных механизмов, вычисление вознаграждения в виде значения критериев сгенерированного дизайна и поиска наилучшего возможного дизайна.

Пользователь может использовать весь framework для генерации суботимальных решений или использовать модули, как независимые части. Framework позволяет реализовать свои собственные правила генерации, модифицировать алгоритмы поиска дизайна и оптимизации.

В настоящее момент framework позволяет выполнять со-дизайн механизмов открытой кинематики. Со-дизайн заключается в одновременном поиске конструкции и траекторий движения робота для получения максимально возможной производительности.


<p align="center">
    <img src="/docs/images/brick_anim.gif" width="700">

</p>

## Описание проекта

Есть четыре основных блока:  

* Graph Grammar -- необходим для создания, модификации и извлечения данных из графов, содержащих всю информацию о сгенерированных механизмах
* Virtual Experiment -- это моделирование, необходимое для количественного анализа производительности сгенерированных механизмов, заданных правилами генерации.
* Trajectory Optimization -- находит субоптимальные траектории сочленений, необходимые для эффективного выполнения желаемого движения
* Search Algorithm -- ищет оптимальный граф для представления топологии механизма

![project_general](/docs/images/general_scheme.jpg)
![project_algorithm](/docs/images/Algorithm_shceme.jpg)

Более подробное описание [алгоритмов и методов](https://rostok.readthedocs.io/en/latest/advanced_usage/algorithm.html).

## Необходимые инструменты

* Anaconda3
* При использовании Docker необходимо установить Х-server для Windows <https://sourceforge.net/projects/vcxsrv/>

## Установка в режиме разработки

Для модификации модулей фреймворка Rostok необходимо установить его в режиме разработки:

* Создайте среду, используя `conda env create -f environment.yml`
* Активируйте окружение `rostok`
* Установите пакет в режиме разработки `pip3 install -e`.

### Известные проблемы

На некоторых ПК можно увидеть проблему с модулем tcl «конфликт версий для пакета «Tcl»: есть 8.6.12, нужна именно 8.6.10», попробуйте установить tk 8.6.10, используя «conda install tk=8.6.10».

После установки пакета может появиться ошибка «Исходная ошибка: Ошибка загрузки DLL при импорте _multiarray_umath: указанный модуль не найден», попробуйте переустановить numpy в среде rostok

## Документация

Описание проекта и туториалы доступны [на сайте проекта](https://rostok.readthedocs.io/en/latest/) .

## Публикации

* I. I. Borisov, E. E. Khornutov, D. V. Ivolga, N. A. Molchanov, I. A. Maksimov and S. A. Kolyubin, "Reconfigurable Underactuated Adaptive Gripper Designed by Morphological Computation," 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 1130-1136, doi: 10.1109/ICRA46639.2022.9811738.


## Примеры
Пример настройки и использования пайплайна генеративного дизайна находится в каталоге `rostok\app`.
Примеры использования независимых модулей находятся в директории `rostok\examples`.


## Поддержка

Исследование проводится при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](<https://sai.itmo.ru/>) [Университета ИТМО](https://itmo.ru) в рамках мероприятия программы центра: Разработка и испытания экспериментального образца библиотеки алгоритмов сильного ИИ в части генеративного и интерактивного дизайна плоских механизмов антропоморфных захватных устройств и роботизированных кистей

![logo_aim](/docs/images/AIM-Strong_Sign_Norm-01_Colors.svg)

### Разработчики

* Иван Борисов - исследователь
* Кирилл Жарков - руководитель группы
* Ефим Осипов - разработчик исследователь
* Дмитрий Иволга - разработчик исследователь
* Кирилл Насонов - разработчик исследователь
* Сергей Колюбин - старший научный сотрудник

## Контакты

* Иван Борисов borisovii@itmo.ru по вопросам работы алгоритмов
* Кирилл Жарков kdzharkov@itmo.ru по техническим вопросам
* Сергей Колюбин s.kolyubin@itmo.ru по вопросам сотрудничества
