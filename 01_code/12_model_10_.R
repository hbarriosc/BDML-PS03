############# Predicción ELASTIC NET + ROSE
library(caret)
library(ROSE)

set.seed(123)

# 🔹 Base 
train_en <- train %>% select(-id, -Dominio, -Depto)
test_en  <- test  %>% select(-id, -Dominio, -Depto)

# 🔹 Split
idx_en <- createDataPartition(train_en$Pobre, p = 0.8, list = FALSE)

train_split_en <- train_en[idx_en, ]
valid_split_en <- train_en[-idx_en, ]

# 🔹 Solo numéricas
train_num_en <- train_split_en %>% select(where(is.numeric))
valid_num_en <- valid_split_en %>% select(where(is.numeric))
test_num_en  <- test_en %>% select(where(is.numeric))

# 🔹 Agregar target
train_num_en$Pobre <- train_split_en$Pobre
valid_num_en$Pobre <- valid_split_en$Pobre

# 🔹 Alinear columnas
vars_modelo_en <- intersect(names(train_num_en), names(test_num_en))
vars_modelo_en <- setdiff(vars_modelo_en, "Pobre")

train_num_en <- train_num_en[, c(vars_modelo_en, "Pobre")]
valid_num_en <- valid_num_en[, c(vars_modelo_en, "Pobre")]
test_num_en  <- test_num_en[, vars_modelo_en]

# 🔹 Imputación (igual a ustedes)
for (col in names(train_num_en)) {
  if (col != "Pobre") {
    med <- median(train_num_en[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    
    train_num_en[[col]][is.na(train_num_en[[col]])] <- med
    valid_num_en[[col]][is.na(valid_num_en[[col]])] <- med
    test_num_en[[col]][is.na(test_num_en[[col]])]   <- med
  }
}

# 🔹 Control con ROSE + F1
ctrl_en <- trainControl(
  method = "cv",
  number = 5,
  sampling = "rose", 
  classProbs = TRUE,
  summaryFunction = prSummary
)

# 🔹 Grid Elastic Net
grid_en <- expand.grid(
  alpha = seq(0, 1, by = 0.25),
  lambda = seq(0.0001, 0.1, length = 10)
)

# 🔹 Modelo
en_tuned <- train(
  Pobre ~ .,
  data = train_num_en,
  method = "glmnet",
  trControl = ctrl_en,
  tuneGrid = grid_en,
  metric = "F",
  preProcess = c("center", "scale")  
)

# 🔹 Mejores parámetros
print(en_tuned)
en_tuned$bestTune

# 🔹 Probabilidades
en_prob <- predict(en_tuned, newdata = valid_num_en, type = "prob")[, "Yes"]
test_prob_en <- predict(en_tuned, newdata = test_num_en, type = "prob")[, "Yes"]

# 🔹 Evaluación base
res_en <- f1_eval(en_prob, valid_num_en$Pobre, cutoff = 0.5)

# 🔹 Optimizar cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_en <- data.frame()

for (c in cuts) {
  met <- f1_eval(en_prob, valid_num_en$Pobre, cutoff = c)
  
  resultados_en <- rbind(
    resultados_en,
    data.frame(
      cutoff = c,
      precision = met$precision,
      recall = met$recall,
      f1 = met$f1
    )
  )
}

# 🔹 Mejor cutoff
mejor_cutoff_en <- resultados_en$cutoff[which.max(resultados_en$f1)]

# 🔹 Predicción final
test_pred_en <- ifelse(test_prob_en >= mejor_cutoff_en, 1, 0)
test_pred_en[is.na(test_pred_en)] <- 0
test_pred_en <- as.integer(test_pred_en)

# 🔹 Submission
submission_en <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_en
)

write.csv(
  submission_en,
  "submission_elasticnet_rose.csv",
  row.names = FALSE,
  quote = FALSE
)
