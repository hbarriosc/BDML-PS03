##########################################################
# Master script
#
# Running this file reproduces all results in the repository.
#
# To reproduce all results, run:
# from an interactive R session: source("01_code/00_rundirectory.R")   
# or from the command line: R CMD BATCH 01_code/00_rundirectory.R
#
# Authors:
#
# - Leidy Lorena Dávila Vallejo
# - Juan Guillermo Sánchez
# - Héctor Steben Barrios Carranza
##########################################################

# Paso 0: Limpieza espacio de trabajo y creación de la carpeta de outputs.
cat("\014")
rm(list = ls())

for (path in c("02_data")) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

for (path in c("03_output", "03_output/figures", "03_output/tables", "03_output/submissions")) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# Paso 1: Carga e instala los paquetes necesarios.
source("01_code/01_setup_packages.R")

# Paso 2: Llama, transforma y realiza la limpieza necesaria a los datos.
source("01_code/02_load_and_prepare_data.R")

# Paso 3: Estimación del modelo lineal.
source("01_code/03_model_1_LPM_v1.R")

# Paso 4: Estimación del modelo lineal mejorado.
source("01_code/04_model_2_LPM_v2.R")

# Paso 5: Estimación del modelo logit.
source("01_code/05_model_3_Logit_v1.R")

# Paso 6: Estimación del modelo logit mejorado.
source("01_code/06_model_4_Logit_v2.R")

# Paso 7: Estimación del modelo elastic net.
source("01_code/07_model_5_EN_v1.R")

# Paso 8: Estimación del modelo elastic net mejorado.
source("01_code/08_model_6_EN_v2.R")

# Paso 9: Estimación del modelo random forest.
source("01_code/09_model_7_RF_v1.R")

# Paso 10: Estimación del modelo naive bayes.
source("01_code/10_model_8_NB_v1.R")

# Paso 11: Estimación del modelo de árbol o CART.
source("01_code/11_model_9_CART_v1.R")

# Paso 12: Estimación del modelo elastic net con ROSE.
source("01_code/12_model_10_EN_v3.R")

# Paso 13: Estimación del modelo elastic net con UPSUMPLING.
source("01_code/13_model_11_EN_v4.R")

# Paso 14: Estimación del modelo elastic net con DOWNSAMPLIN.
source("01_code/14_model_12_EN_v5.R")

# Paso 15: Estimación del modelo elastic net Tuning de Alpha y Lambda.
source("01_code/15_model_13_EN_v6.R")

# Paso 16: Estimación del modelo logit alterno.
source("01_code/16_model_14_Logit_v3.R")

# Paso 17: Estimación del modelo random forest alterno.
source("01_code/17_model_15_RF_v2.R")

# Paso 18: Crea tablas y gráficas sobre los resultados y cálculos de los modelos.
source("01_code/18_statistics_of_models.R")
