#Prediccion 7  - 4 Modelo RANDOM FOREST


# Cargamos la librería para Random Forest
install.packages("ranger")

# Partimos de las bases limpias
train_rf <- train
test_rf  <- test

# Variables derivadas
train_rf <- train_rf %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

test_rf <- test_rf %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

# Quitar variables que no usaremos
train_rf <- train_rf %>% select(-id, -Dominio, -Depto)
test_rf  <- test_rf %>% select(-id, -Dominio, -Depto)

# Split train-validation
set.seed(123)
idx_rf <- createDataPartition(train_rf$Pobre, p = 0.8, list = FALSE)

train_split_rf <- train_rf[idx_rf, ]
valid_split_rf <- train_rf[-idx_rf, ]

# Solo numéricas
train_num_rf <- train_split_rf %>% select(where(is.numeric))
valid_num_rf <- valid_split_rf %>% select(where(is.numeric))
test_num_rf  <- test_rf %>% select(where(is.numeric))

# Reagregar variable objetivo
train_num_rf$Pobre <- train_split_rf$Pobre
valid_num_rf$Pobre <- valid_split_rf$Pobre

# Alinear columnas
vars_modelo_rf <- intersect(names(train_num_rf), names(test_num_rf))
vars_modelo_rf <- setdiff(vars_modelo_rf, "Pobre")

train_num_rf <- train_num_rf[, c(vars_modelo_rf, "Pobre")]
valid_num_rf <- valid_num_rf[, c(vars_modelo_rf, "Pobre")]
test_num_rf  <- test_num_rf[, vars_modelo_rf]

# Imputación con mediana
for (col in names(train_num_rf)) {
  if (col != "Pobre") {
    mediana_col <- median(train_num_rf[[col]], na.rm = TRUE)
    if (is.na(mediana_col)) mediana_col <- 0
    
    train_num_rf[[col]][is.na(train_num_rf[[col]])] <- mediana_col
    valid_num_rf[[col]][is.na(valid_num_rf[[col]])] <- mediana_col
    test_num_rf[[col]][is.na(test_num_rf[[col]])] <- mediana_col
  }
}

# Verificación
sum(is.na(train_num_rf))
sum(is.na(valid_num_rf))
sum(is.na(test_num_rf))

# Variable objetivo como factor
train_num_rf$Pobre <- factor(train_num_rf$Pobre, levels = c("No", "Yes"))
valid_num_rf$Pobre <- factor(valid_num_rf$Pobre, levels = c("No", "Yes"))

# Quitar varianza casi cero
nzv_cols_rf <- nearZeroVar(train_num_rf %>% select(-Pobre))
if (length(nzv_cols_rf) > 0) {
  cols_quitar_rf <- names(train_num_rf %>% select(-Pobre))[nzv_cols_rf]
  
  train_num_rf <- train_num_rf %>% select(-all_of(cols_quitar_rf))
  valid_num_rf <- valid_num_rf %>% select(-all_of(cols_quitar_rf))
  test_num_rf  <- test_num_rf %>% select(-all_of(cols_quitar_rf))
}

# Control más liviano
ctrl_rf <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Grid más pequeño
grid_rf <- expand.grid(
  mtry = c(5, 10),
  splitrule = "gini",
  min.node.size = c(5, 15)
)

# Entrenamiento más liviano
set.seed(123)
rf_fit <- train(
  Pobre ~ .,
  data = train_num_rf,
  method = "ranger",
  trControl = ctrl_rf,
  metric = "ROC",
  tuneGrid = grid_rf,
  num.trees = 150,
  importance = "impurity"
)

# Revisar modelo
rf_fit
rf_fit$bestTune

# Predicción en validación
rf_prob <- predict(rf_fit, newdata = valid_num_rf, type = "prob")[, "Yes"]

# Evaluación base
res_rf <- f1_eval(rf_prob, valid_num_rf$Pobre, cutoff = 0.5)

res_rf$f1
res_rf$precision
res_rf$recall
res_rf$confusion

# Buscar mejor cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_rf <- data.frame()

for (cut in cuts) {
  met <- f1_eval(rf_prob, valid_num_rf$Pobre, cutoff = cut)
  
  resultados_rf <- rbind(
    resultados_rf,
    data.frame(
      cutoff = cut,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

# Revisar resultados
resultados_rf
resultados_rf[which.max(resultados_rf$f1), ]

# Guardar mejor cutoff
mejor_cutoff_rf <- resultados_rf$cutoff[which.max(resultados_rf$f1)]
mejor_cutoff_rf

# Verificar que sí exista
if (length(mejor_cutoff_rf) == 0 || is.na(mejor_cutoff_rf)) {
  stop("No se pudo calcular mejor_cutoff_rf")
}

# Predicción final en test
test_prob_rf <- predict(rf_fit, newdata = test_num_rf, type = "prob")[, "Yes"]

# Verificar que sí existan probabilidades
length(test_prob_rf)
sum(is.na(test_prob_rf))

test_pred_rf <- ifelse(test_prob_rf >= mejor_cutoff_rf, 1, 0)
test_pred_rf[is.na(test_pred_rf)] <- 0
test_pred_rf <- as.integer(test_pred_rf)

# Verificar longitudes
length(base_modelo_test$id)
length(test_pred_rf)

# Submission
submission_rf <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_rf
)

# Validaciones finales
head(submission_rf)
dim(submission_rf)
sum(is.na(submission_rf$Pobre))
unique(submission_rf$Pobre)
names(submission_rf)

# Guardar CSV
write.csv(submission_rf, "submission_model4_rf_v1.csv",
          row.names = FALSE, quote = FALSE)
