#==============================================================================#
# 1. LIBRERÍAS
library(tidyverse)
library(caret)
library(ranger)
library(pROC)

#==============================================================================#
# 2. CARGA DE DATOS
test_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_hogares.csv")
test_personas <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_personas.csv")
train_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_hogares.csv")
train_personas<- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_personas.csv")

#==============================================================================#
# 3. WRANGLING: AGREGACIÓN DE PERSONAS A HOGARES
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

#==============================================================================#
# 4. AJUSTE DE VARIABLES
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

#==============================================================================#
# 5. IMPUTACIÓN Y NA
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

#==============================================================================#
# 6. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
set.seed(123)

# Definir validación cruzada
control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Definir grilla de búsqueda
grid <- expand.grid(
  mtry = c(5, 10), 
  splitrule = "gini", 
  min.node.size = c(1, 5)
)

# Entrenar sin id
train_modelado <- train_imp %>% select(-id)

modelo_rf <- train(
  Pobre ~ ., 
  data = train_modelado,
  method = "ranger",
  trControl = control,
  tuneGrid = grid,
  num.trees = 500,
  importance = "impurity"
)

#==============================================================================#
# 7. PREDICCIÓN Y ENTREGA
# Preparar set de prueba (sin id)
test_modelado <- test_imp %>% select(-id)

# Predicción de probabilidad y clase
pred_prob  <- predict(modelo_rf, newdata = test_modelado, type = "prob")
pred_clase <- predict(modelo_rf, newdata = test_modelado)

# Crear dataframe de submission usando los IDs originales
submission <- data.frame(
  id = test_imp$id,
  Pobre = ifelse(pred_clase == "Si", 1, 0)
)

# Guardar resultados
write.csv(submission, "predicciones_pobreza_final.csv", row.names = FALSE)

#==============================================================================#
# 8. GRAFICAS DEL MODELO
# importancia de variables
print(modelo_rf)
plot(varImp(modelo_rf), top = 10)

# Matriz de confusión
pred_train_clase <- predict(modelo_rf, train_imp)
cm <- confusionMatrix(pred_train_clase, train_imp$Pobre)

# Matriz de Confusión
as.data.frame(cm$table) %>%
  ggplot(aes(Reference, Prediction, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "#3498db", high = "#2c3e50") +
  labs(title = "2. Matriz de Confusión", subtitle = "Evaluación sobre set de entrenamiento") +
  theme_minimal()

# CURVA ROC
pred_train_prob <- predict(modelo_rf, train_imp, type = "prob")
roc_obj <- roc(train_imp$Pobre, pred_train_prob$Si)

plot(roc_obj, col = "#e74c3c", lwd = 4, main = "3. Curva ROC")
text(0.4, 0.2, paste("AUC =", round(auc(roc_obj), 4)), cex = 1.2, col = "red")

# DENSIDAD DE PROBABILIDADES
data.frame(Prob = pred_train_prob$Si, Real = train_imp$Pobre) %>%
  ggplot(aes(x = Prob, fill = Real)) +
  geom_density(alpha = 0.6) +
  labs(title = "4. Separación de Clases", x = "Probabilidad de ser Pobre", y = "Densidad") +
  theme_light()