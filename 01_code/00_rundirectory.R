# Paso 0: Limpieza espacio de trabajo y creción de la carpeta de outputs.
cat("\014")
rm(list = ls())

for (path in c("02_output", "02_output/figures", "02_output/tables")) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# Paso 1: Carga e instala los paquetes necesarios.
source("01_code/01_setup_packages.R")

# Paso 2: Importa los datos por webscrapping y realiza la limpieza necesaria a los datos.
source("01_code/02_load_and_prepare_data.R")

# Paso 3: Crea tablas y gráficas de estadística descriptiva de la base.
source("01_code/03_descriptive_statistics.R")

# Paso 4: Estima la relación edad-ingreso de los trabajadores bajo dos modelos:no condicional y condicional.
source("01_code/04_age_labor_income_profile.R")

# Paso 5: Estima la relación edad-ingreso de los trabajadores a la luz de la brecha de género.
source("01_code/05_gender_labor_income_gap.R")

# Paso 6: Realiza la predicción de los ingresos bajo el mejor modelo de predicción.
source("01_code/06_labor_income_prediction.R")
