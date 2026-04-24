###Predicción 1 LPM: El LPM es una regresión lineal donde la variable dependiente toma valores 0 y 1.
#Este modelo asume una relación lineal entre las covariables y la probabilidad de pobreza, 
#lo cual permite una interpretación directa de los coeficientes como cambios marginales en 
#dicha probabilidad.

#Nos basamos en teoria del bienestar, la pobreza depende del ingreso, tamaño de hogar, eduacución, empleo, acceso a servicios
#Referencias: Sen (1981) – Poverty and Famines World Bank (2000) – World Development Report

#Realizamos nuestra semilla y la preparacion de los datos 

set.seed(123)

train_model_lpm <- train %>% select(-id, -Dominio, -Depto)
test_model_lpm  <- test %>% select(-id, -Dominio, -Depto)

# Split train-validation
idx_lpm <- createDataPartition(train_model_lpm$Pobre, p = 0.8, list = FALSE)

train_split_lpm <- train_model_lpm[idx_lpm, ]
valid_split_lpm <- train_model_lpm[-idx_lpm, ]

# Solo variables numéricas
train_num_lpm <- train_split_lpm %>% select(where(is.numeric))
valid_num_lpm <- valid_split_lpm %>% select(where(is.numeric))
test_num_lpm  <- test_model_lpm %>% select(where(is.numeric))

# Variable objetivo como 0/1
train_num_lpm$Pobre <- ifelse(train_split_lpm$Pobre == "Yes", 1, 0)
valid_num_lpm$Pobre <- ifelse(valid_split_lpm$Pobre == "Yes", 1, 0)

# Alinear columnas
vars_modelo_lpm <- intersect(names(train_num_lpm), names(test_num_lpm))
vars_modelo_lpm <- setdiff(vars_modelo_lpm, "Pobre")

train_num_lpm <- train_num_lpm[, c(vars_modelo_lpm, "Pobre")]
valid_num_lpm <- valid_num_lpm[, c(vars_modelo_lpm, "Pobre")]
test_num_lpm  <- test_num_lpm[, vars_modelo_lpm]

# Imputación con mediana
for (col in names(train_num_lpm)) {
  if (col != "Pobre") {
    mediana_col <- median(train_num_lpm[[col]], na.rm = TRUE)
    
    if (is.na(mediana_col)) mediana_col <- 0
    
    train_num_lpm[[col]][is.na(train_num_lpm[[col]])] <- mediana_col
    valid_num_lpm[[col]][is.na(valid_num_lpm[[col]])] <- mediana_col
    test_num_lpm[[col]][is.na(test_num_lpm[[col]])] <- mediana_col
  }
}

# Verificación
sum(is.na(train_num_lpm))
sum(is.na(valid_num_lpm))
sum(is.na(test_num_lpm))

# Entrenamiento LPM
lpm_fit <- lm(Pobre ~ ., data = train_num_lpm)

summary(lpm_fit)

# Predicción en validación
lpm_prob <- predict(lpm_fit, newdata = valid_num_lpm)

# Acotamos entre 0 y 1 porque el LPM puede dar valores fuera del rango
lpm_prob <- pmax(pmin(lpm_prob, 1), 0)

# Función F1 para variable objetivo numérica
f1_eval_num <- function(prob, y_true, cutoff = 0.5) {
  pred <- ifelse(prob >= cutoff, 1, 0)
  
  pred_factor <- factor(pred, levels = c(0, 1), labels = c("No", "Yes"))
  true_factor <- factor(y_true, levels = c(0, 1), labels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred_factor, true_factor, positive = "Yes")
  
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- 2 * precision * recall / (precision + recall)
  
  list(
    confusion = cm,
    precision = precision,
    recall = recall,
    f1 = f1
  )
}

# Evaluación base
res_lpm <- f1_eval_num(lpm_prob, valid_num_lpm$Pobre, cutoff = 0.5)

res_lpm$f1
res_lpm$precision
res_lpm$recall
res_lpm$confusion

# Buscar mejor cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_lpm <- data.frame()

for (cut in cuts) {
  met <- f1_eval_num(lpm_prob, valid_num_lpm$Pobre, cutoff = cut)
  
  resultados_lpm <- rbind(
    resultados_lpm,
    data.frame(
      cutoff = cut,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_lpm
resultados_lpm[which.max(resultados_lpm$f1), ]

mejor_cutoff_lpm <- resultados_lpm$cutoff[which.max(resultados_lpm$f1)]
mejor_cutoff_lpm

# Predicción final en test
test_prob_lpm <- predict(lpm_fit, newdata = test_num_lpm)
test_prob_lpm <- pmax(pmin(test_prob_lpm, 1), 0)

test_pred_lpm <- ifelse(test_prob_lpm >= mejor_cutoff_lpm, 1, 0)
test_pred_lpm[is.na(test_pred_lpm)] <- 0
test_pred_lpm <- as.integer(test_pred_lpm)

# Submission
submission_lpm <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_lpm
)

head(submission_lpm)
dim(submission_lpm)
sum(is.na(submission_lpm$Pobre))
unique(submission_lpm$Pobre)
names(submission_lpm)

write.csv(
  submission_lpm,
  here("03_output/submissions","submission_model1_lpm_v1.csv"),
  row.names = FALSE,
  quote = FALSE
)

res_lpm$f1
res_lpm$precision
res_lpm$recall
