# BDML-PS03

Big Data y Machine Learning – Universidad de los Andes – 2026  

En este repositorio encontrará la solución del Problem set 2 correspondiente al mejor modelo para predecir la pobreza.

## Autores

-   Leidy Lorena Dávila Vallejo - COD: 202522776
-   Juan Guillermo Sánchez - COD: 202323123
-   Héctor Steben Barrios Carranza - COD: 202116184

## Replicación

Para reproducir todos los resultados, correr:

`source("01_code/00_rundirectory.R")`

## Estructura de código

-   `01_code/00_rundirectory.R`: Master script. Reproduce todos los códigos y resultados.
-   `01_code/01_setup_packages.R`: Carga e instala los paquetes necesarios.
-   `01_code/02_load_and_preprare_data.R`: Importa los datos por webscrapping y realiza la limpieza necesaria a los datos.
-   `01_code/03_descriptive_statistics.R`: Crea tablas y gráficas de estadística descriptiva de la base.
-   `01_code/04_age_labor_income_profile.R`: Estima la relación edad-ingreso de los trabajadores bajo dos modelos:no condicional y condicional.
-   `01_code/05_gender_labor_income_gap.R`: Estima la relación edad-ingreso de los trabajadores a la luz de la brecha de género.
-   `01_code/06_labor_income_prediction.R`: Realiza la predicción de los ingresos bajo el mejor modelo de predicción.

## Salidas

Todos los outputs se generan automáticamente en `02_outputs/`.

-   Figures (`03_outputs/figures/`): visualizaciones generadas por el código
-   Submissions (`03_outputs/submissions/`): archivos con los resultados de los modelos para subir a Kaggle
-   Tables (`03_outputs/tables/`): resultados de estimaciones en formato `.tex` y `.html`

## Software / entorno

-   R version 4.5.1
-   Required packages: Pacman
