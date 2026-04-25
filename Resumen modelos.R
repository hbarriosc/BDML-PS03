#==============================================================================#
# 1. LIBRERÍAS Y CONFIGURACIÓN
#==============================================================================#
library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(pROC)
library(PRROC)

set.seed(123)

#==============================================================================#
# 2. CARGA Y PROCESAMIENTO DE DATOS (UNIFICADO)
#==============================================================================#
# 2. CARGA DE DATOS
test_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_hogares.csv")
test_personas <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_personas.csv")
train_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_hogares.csv")
train_personas<- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_personas.csv")

procesar_datos_taller <- function(df_hogares, df_personas) {
  # Agregación a nivel hogar
  personas_agg <- df_personas %>%
    group_by(id) %>%
    summarise(
      across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
      n_personas = n(),
      .groups = "drop"
    )
  
  # Unión y Feature Engineering
  df_full <- df_hogares %>%
    left_join(personas_agg, by = "id") %>%
    mutate(
      hacinamiento = P5010 / (P5000 + 0.1),
      dependencia  = n_personas / (prom_Oc + 1),
      educacion    = prom_P6210 + (replace_na(prom_P6210s1, 0) / 10),
      Dominio      = as.factor(Dominio)
    )
  return(df_full)
}

# Ejecutar wrangling (asumiendo carga previa)
train_full <- procesar_datos_taller(train_hogares, train_personas)
test_full  <- procesar_datos_taller(test_hogares, test_personas)

# Limpieza: Mantener variables con < 30% NAs
cols_keep <- colMeans(is.na(train_full)) <= 0.3
train_clean <- train_full[, cols_keep]
vars_comunes <- intersect(names(train_clean), names(test_full))
vars_eliminar <- c("prom_Clase", "prom_Depto") # Redundantes
vars_comunes <- setdiff(vars_comunes, vars_eliminar)

train_df <- train_clean[, c(vars_comunes, "Pobre")]
train_df$Pobre <- factor(train_df$Pobre, levels = c(0, 1), labels = c("No", "Si"))

# Imputación y Escalamiento
prep_proc <- preProcess(train_df %>% select(-Pobre, -id), method = c("medianImpute", "center", "scale"))
train_final <- predict(prep_proc, train_df)

#==============================================================================#
# 3. ENTRENAMIENTO DE MODELOS
#==============================================================================#
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# A. LOGIT
modelo_logit <- train(Pobre ~ . -id, data = train_final, method = "glm", family = "binomial", trControl = ctrl)

# B. ELASTIC NET
en_grid <- expand.grid(alpha = seq(0, 1, 0.2), lambda = seq(0.001, 0.1, length = 10))
modelo_en <- train(Pobre ~ . -id, data = train_final, method = "glmnet", trControl = ctrl, tuneGrid = en_grid)

# C. RANDOM FOREST
modelo_rf <- train(Pobre ~ . -id, data = train_final, method = "ranger", trControl = ctrl, 
                   tuneGrid = expand.grid(mtry = 2, splitrule = "extratrees", min.node.size = 5))

# D. LPM (Manual para pesos)
train_lpm <- train_final %>% mutate(Pobre_num = ifelse(Pobre == "Si", 1, 0))
prop_pobre <- mean(train_lpm$Pobre_num)
pesos <- ifelse(train_lpm$Pobre_num == 1, 1/prop_pobre, 1/(1-prop_pobre))
modelo_lpm <- lm(Pobre_num ~ . -id -Pobre, data = train_lpm, weights = pesos)

#==============================================================================#
# 4. ANÁLISIS DE RENDIMIENTO ESPECÍFICO
#==============================================================================#
analisis_rendimiento <- function(model, data, label) {
  preds <- predict(model, data, type = "prob")
  roc_obj <- roc(data$Pobre, preds$Si)
  return(data.frame(Modelo = label, AUC = as.numeric(roc_obj$auc)))
}

res_logit <- analisis_rendimiento(modelo_logit, train_final, "Logit")
res_en    <- analisis_rendimiento(modelo_en, train_final, "Elastic Net")
res_rf    <- analisis_rendimiento(modelo_rf, train_final, "Random Forest")

# Comparativa final
performance_table <- rbind(res_logit, res_en, res_rf)
print(performance_table)

#==============================================================================#
# 5. GENERACIÓN DE CURVAS ROC (VERSIÓN CORREGIDA)
#==============================================================================#
library(pROC)

# Función auxiliar para asegurar que las predicciones sean comparables
obtener_probs <- function(modelo, datos) {
  if (inherits(modelo, "train")) {
    # Para modelos caret (Logit, EN, RF)
    return(predict(modelo, datos, type = "prob")$Si)
  } else {
    # Para el modelo lineal (LPM)
    return(as.numeric(predict(modelo, datos)))
  }
}

# 1. Extraer probabilidades correctamente
p_logit <- obtener_probs(modelo_logit, train_final)
p_en    <- obtener_probs(modelo_en, train_final)
p_rf    <- obtener_probs(modelo_rf, train_final)
p_lpm   <- obtener_probs(modelo_lpm, train_final)

# 2. Crear objetos ROC asegurando que "Si" es el evento de interés
# (Usa levels para evitar que R confunda el orden alfabético)
target <- train_final$Pobre

r_logit <- roc(target, p_logit, levels = c("No", "Si"))
r_en    <- roc(target, p_en,    levels = c("No", "Si"))
r_rf    <- roc(target, p_rf,    levels = c("No", "Si"))
r_lpm   <- roc(target, p_lpm,   levels = c("No", "Si"))

# 3. Graficar con capas independientes para asegurar visibilidad
plot(r_rf, col = "#2c3e50", lwd = 3, main = "Comparativa Final de Modelos")
lines(r_en, col = "#2980b9", lwd = 3)
lines(r_logit, col = "#27ae60", lwd = 3, lty = 2)
lines(r_lpm, col = "#e67e22", lwd = 2, lty = 3)

legend("bottomright", 
       legend = c(
         paste0("Random Forest (AUC: ", round(auc(r_rf), 3), ")"),
         paste0("Elastic Net (AUC: ", round(auc(r_en), 3), ")"),
         paste0("Logit (AUC: ", round(auc(r_logit), 3), ")"),
         paste0("LPM (AUC: ", round(auc(r_lpm), 3), ")")
       ), 
       col = c("#2c3e50", "#2980b9", "#27ae60", "#e67e22"), 
       lwd = 3, lty = c(1, 1, 2, 3))