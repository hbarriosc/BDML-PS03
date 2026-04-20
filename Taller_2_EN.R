#==============================================================================#
# 1. LIBRERÍAS (Asegúrate de tener instalado 'glmnet')
library(tidyverse)
library(caret)
library(glmnet)
library(pROC)

#==============================================================================#
# 2. PROCESAMIENTO Y LIMPIEZA
test_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_hogares.csv")
test_personas <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_personas.csv")
train_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_hogares.csv")
train_personas<- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_personas.csv")

# 2.2. WRANGLING: AGREGACIÓN DE PERSONAS A HOGARES
procesar_base <- function(df_hogares, df_personas) {
  personas_agg <- df_personas %>%
    group_by(id) %>%
    summarise(
      across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
      n_personas = n()
    )
  
  df_hogares %>% left_join(personas_agg, by = "id")
}

train_full <- procesar_base(train_hogares, train_personas)
test_full  <- procesar_base(test_hogares, test_personas)

# Identificar variables con > 30% de NAs en Train para eliminarlas en ambos
cols_con_pocos_nas <- colMeans(is.na(train_full)) <= 0.3
train_clean <- train_full[, cols_con_pocos_nas]

# Sincronizar columnas comunes (excepto la variable objetivo 'Pobre')
vars_comunes <- intersect(names(train_clean), names(test_full))
vars_eliminar <- c("prom_Clase", "prom_Fex_c", "prom_Depto", "prom_Fex_dpto")
vars_comunes <- setdiff(vars_comunes, vars_eliminar)

# Bases filtradas
train_final <- train_clean[, c(vars_comunes, "Pobre")]
test_final  <- test_full[, vars_comunes]

# 2.3. AJUSTE DE VARIABLES
preparar_datos <- function(df) {
  df %>%
    mutate(
      hacinamiento = P5010 / (P5000 + 0.1),
      dependencia  = n_personas / (prom_Oc + 1),
      educacion    = prom_P6210 + (replace_na(prom_P6210s1, 0) / 10),
      horas_total  = prom_P6800 * n_personas,
      Dominio      = as.factor(Dominio)
    )
}

train_fe <- preparar_datos(train_final)
test_fe  <- preparar_datos(test_final)

# Convertir variable objetivo a factor para clasificación
train_fe$Pobre <- factor(train_fe$Pobre, levels = c(0, 1), labels = c("No", "Si"))

# Sincronizar niveles de factores (Dominio) para evitar errores en predict
test_fe$Dominio <- factor(test_fe$Dominio, levels = levels(train_fe$Dominio))

# 2.4. IMPUTACIÓN Y NA
# Crear indicadores de NA (solo para variables que tienen NAs en Train)
na_vars <- names(train_fe)[colSums(is.na(train_fe)) > 0]

for(v in na_vars) {
  train_fe[[paste0(v, "_NA")]] <- as.numeric(is.na(train_fe[[v]]))
  test_fe[[paste0(v, "_NA")]]  <- as.numeric(is.na(test_fe[[v]]))
}

# Imputación por mediana (entrenada en Train, aplicada a ambos)
# Excuir id y Pobre de imputación al ser variables de interés
prep_mediana <- preProcess(train_fe %>% select(-id, -Pobre), method = "medianImpute")

train_imp <- predict(prep_mediana, train_fe)
test_imp  <- predict(prep_mediana, test_fe)

# Escalación de datos
prep_en <- preProcess(train_imp %>% select(-id, -Pobre), method = c("center", "scale", "medianImpute"))

train_en <- predict(prep_en, train_imp)
test_en  <- predict(prep_en, test_imp)

#==============================================================================#
# 3. ENTRENAMIENTO ELASTIC NET (Tuning de Alpha y Lambda)

set.seed(123)

# Definimos la grilla de búsqueda:
# alpha = 0 es Ridge, alpha = 1 es Lasso. Entre 0 y 1 es Elastic Net.
en_grid <- expand.grid(
  alpha = seq(0, 1, length = 5), 
  lambda = exp(seq(-10, -1, length = 20))
)

control_en <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Entrenamos excluyendo el ID
train_en_input <- train_en %>% select(-id)

modelo_en <- train(
  Pobre ~ ., 
  data = train_en_input,
  method = "glmnet", # Especificamos Elastic Net
  family = "binomial",
  trControl = control_en,
  tuneGrid = en_grid,
  metric = "ROC"
)

#==============================================================================#
# 4. EVALUACIÓN Y GRÁFICAS

# A. MEJORES PARÁMETROS
print(modelo_en$bestTune)
plot(modelo_en) # Muestra cómo varió el ROC según Alpha y Lambda

# B. VARIABLES SELECCIONADAS (Coeficientes)
# Elastic Net pone en cero las variables que no aportan (si alpha > 0)
coeficientes <- coef(modelo_en$finalModel, modelo_en$bestTune$lambda)
print(coeficientes)

# C. CURVA ROC
pred_prob_en <- predict(modelo_en, train_en_input, type = "prob")
roc_en <- roc(train_en$Pobre, pred_prob_en$Si)

plot(roc_en, col = "#2980b9", lwd = 4, main = "Curva ROC - Elastic Net")
text(0.4, 0.2, paste("AUC =", round(auc(roc_en), 4)), cex = 1.2, col = "blue")

#==============================================================================#
# 5. PREDICCIÓN FINAL

test_en_input <- test_en %>% select(-id)
final_preds_en <- predict(modelo_en, newdata = test_en_input)

submission_en <- data.frame(
  id = test_en$id, 
  Pobre = ifelse(final_preds_en == "Si", 1, 0)
)

write.csv(submission_en, "predicciones_pobreza_en.csv", row.names = FALSE)