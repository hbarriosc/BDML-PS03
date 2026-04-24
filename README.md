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
-   `01_code/03_modelo_1_.R`: Estimación del modelo 
-   `01_code/04_modelo_2_.R`: Estimación del modelo 
-   `01_code/05_modelo_3_.R`: Estimación del modelo 
-   `01_code/06_modelo_4_.R`: Estimación del modelo 
-   `01_code/07_modelo_5_.R`: Estimación del modelo 
-   `01_code/08_modelo_6_.R`: Estimación del modelo 
-   `01_code/09_modelo_7_.R`: Estimación del modelo 
-   `01_code/10_modelo_8_.R`: Estimación del modelo 
-   `01_code/11_modelo_9_.R`: Estimación del modelo 
-   `01_code/12_modelo_10_.R`: Estimación del modelo 
-   `01_code/13_modelo_11_.R`: Estimación del modelo 
-   `01_code/14_modelo_12_.R`: Estimación del modelo 
-   `01_code/15_modelo_13_.R`: Estimación del modelo 
-   `01_code/16_modelo_14_.R`: Estimación del modelo 
-   `01_code/17_modelo_15_.R`: Estimación del modelo 
-   `01_code/18_statistics_of_models.R`: Crea tablas y gráficas sobre los resultados y cálculos de los modelos.

## Salidas

Todos los outputs se generan automáticamente en `02_outputs/`.

-   Figures (`03_outputs/figures/`): visualizaciones generadas por el código
-   Submissions (`03_outputs/submissions/`): archivos con los resultados de los modelos para subir a Kaggle
-   Tables (`03_outputs/tables/`): resultados de estimaciones en formato `.tex` y `.html`

## Software / entorno

-   R version 4.5.1
-   Required packages: Pacman
