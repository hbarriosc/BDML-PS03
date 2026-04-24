# BDML-PS03

Big Data y Machine Learning – Universidad de los Andes – 2026  

En este repositorio encontrará la solución del Problem set 2 correspondiente al mejor modelo para predecir la pobreza.

## Autores

-   Leidy Lorena Dávila Vallejo - COD: 202522776
-   Juan Guillermo Sánchez - COD: 202323123
-   Héctor Steben Barrios Carranza - COD: 202116184

## Descarga de datos

Los datos no están incluidos en este repositorio por restricciones de tamaño. Es necesario:

1. Descargar las bases de datos de kaggle (https://www.kaggle.com/competitions/uniandes-bdml-2026-10-ps-2/data)
2. Guardar los siguientes archivos dentro de la carpeta `02_data/`:

- train_personas.csv
- train_hogares.csv
- test_personas.csv
- test_hogares.csv

## Replicación

Para reproducir todos los resultados, una vez descargados y ubicados los datos, correr:

`source("01_code/00_rundirectory.R")`

## Estructura de código

-   `01_code/00_rundirectory.R`: Master script. Reproduce todos los códigos y resultados.
-   `01_code/01_setup_packages.R`: Carga e instala los paquetes necesarios.
-   `01_code/02_load_and_preprare_data.R`: Llama, transforma y realiza la limpieza necesaria a los datos.
-   `01_code/03_modelo_1_LPM_v1.R`: Estimación del modelo lineal
-   `01_code/04_modelo_2_LPM_v2.R`: Estimación del modelo lineal mejorado
-   `01_code/05_modelo_3_Logit_v1.R`: Estimación del modelo logit
-   `01_code/06_modelo_4_Logit_v2.R`: Estimación del modelo logit mejorado
-   `01_code/07_modelo_5_EN_v1.R`: Estimación del modelo elastic net
-   `01_code/08_modelo_6_EN_v2.R`: Estimación del modelo elastic net mejorado
-   `01_code/09_modelo_7_RF_v1.R`: Estimación del modelo random forest
-   `01_code/10_modelo_8_NB_v1.R`: Estimación del modelo naive bayes
-   `01_code/11_modelo_9_CART_v1.R`: Estimación del modelo de árbol o CART
-   `01_code/12_modelo_10_EN_v3.R`: Estimación del modelo elastic net con ROSE
-   `01_code/13_modelo_11_EN_v4.R`: Estimación del modelo elastic net con UPSUMPLING
-   `01_code/14_modelo_12_EN_v5.R`: Estimación del modelo elastic net con DOWNSAMPLIN
-   `01_code/15_modelo_13_EN_v6.R`: Estimación del modelo elastic net Tuning de Alpha y Lambda
-   `01_code/16_modelo_14_Logit_v3.R`: Estimación del modelo logit alterno
-   `01_code/17_modelo_15_RF_v2.R`: Estimación del modelo random forest alterno
-   `01_code/18_statistics_of_models.R`: Crea tablas y gráficas sobre los resultados y cálculos de los modelos.

## Salidas

Todos los outputs se generan automáticamente en `02_outputs/`.

-   Figures (`03_outputs/figures/`): visualizaciones generadas por el código
-   Submissions (`03_outputs/submissions/`): archivos con los resultados de los modelos para subir a Kaggle
-   Tables (`03_outputs/tables/`): resultados de estimaciones en formato `.tex` y `.html`

## Software / entorno

-   R version 4.5.1
-   Required packages: Pacman
