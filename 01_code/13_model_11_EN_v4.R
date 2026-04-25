###Predicción 11 ELASTIC NET CON UPSAMPLING

set.seed(123)

#Bases
train_en <- train %>% select(-id, -Dominio, -Depto)
test_en  <- test  %>% select(-id, -Dominio, -Depto)

#Split
idx_en <- createDataPartition(train_en$Pobre, p = 0.8, list = FALSE)

train_split_en <- train_en[idx_en, ]
valid_split_en <- train_en[-idx_en, ]

#Solo numéricas
train_num_en <- train_split_en %>% select(where(is.numeric))
valid_num_en <- valid_split_en %>% select(where(is.numeric))
test_num_en  <- test_en %>% select(where(is.numeric))

train_num_en$Pobre <- train_split_en$Pobre
valid_num_en$Pobre <- valid_split_en$Pobre

vars_modelo_en <- intersect(names(train_num_en), names(test_num_en))
vars_modelo_en <- setdiff(vars_modelo_en, "Pobre")

train_num_en <- train_num_en[, c(vars_modelo_en, "Pobre")]
valid_num_en <- valid_num_en[, c(vars_modelo_en, "Pobre")]
test_num_en  <- test_num_en[, vars_modelo_en]

#Imputación
for (col in names(train_num_en)) {
  if (col != "Pobre") {
    med <- median(train_num_en[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    
    train_num_en[[col]][is.na(train_num_en[[col]])] <- med
    valid_num_en[[col]][is.na(valid_num_en[[col]])] <- med
    test_num_en[[col]][is.na(test_num_en[[col]])]   <- med
  }
}

#Control
ctrl_en_up <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = prSummary,
  sampling = "up"
)

#Grid
grid_en <- expand.grid(
  alpha = seq(0, 1, by = 0.25),
  lambda = seq(0.0001, 0.1, length = 10)
)

#Modelo
en_up <- train(
  Pobre ~ .,
  data = train_num_en,
  method = "glmnet",
  trControl = ctrl_en_up,
  tuneGrid = grid_en,
  metric = "F",
  family = "binomial",
  preProcess = c("center", "scale")
)

print(en_up)
en_up$bestTune

#Probabilidades
en_prob_up <- predict(en_up, newdata = valid_num_en, type = "prob")[, "Yes"]
test_prob_up <- predict(en_up, newdata = test_num_en, type = "prob")[, "Yes"]

#Evaluación base
res_en_up <- f1_eval(en_prob_up, valid_num_en$Pobre, cutoff = 0.5)

#Optimiamos el cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_en_up <- data.frame()

for (c in cuts) {
  met <- f1_eval(en_prob_up, valid_num_en$Pobre, cutoff = c)
  
  resultados_en_up <- rbind(
    resultados_en_up,
    data.frame(
      cutoff = c,
      precision = met$precision,
      recall = met$recall,
      f1 = met$f1
    )
  )
}

mejor_cutoff_en_up <- resultados_en_up$cutoff[which.max(resultados_en_up$f1)]

#Predicción final
test_pred_en_up <- ifelse(test_prob_up >= mejor_cutoff_en_up, 1, 0)
test_pred_en_up[is.na(test_pred_en_up)] <- 0
test_pred_en_up <- as.integer(test_pred_en_up)

#Submission
submission_en_up <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_en_up
)

#Guardamos
write.csv(
  submission_en_up,
  here("03_output/submissions","submission_elasticnet_up.csv"),
  row.names = FALSE,
  quote = FALSE
)
