###################Predicción X - Árbol (CART) con tuning de cp

set.seed(123)

# Base 
train_tree <- train %>% select(-id, -Dominio, -Depto)
test_tree  <- test  %>% select(-id, -Dominio, -Depto)

# Split
idx_tree <- createDataPartition(train_tree$Pobre, p = 0.8, list = FALSE)

train_split_tree <- train_tree[idx_tree, ]
valid_split_tree <- train_tree[-idx_tree, ]

# Solo numéricas
train_num_tree <- train_split_tree %>% select(where(is.numeric))
valid_num_tree <- valid_split_tree %>% select(where(is.numeric))
test_num_tree  <- test_tree %>% select(where(is.numeric))

# Agregar target
train_num_tree$Pobre <- train_split_tree$Pobre
valid_num_tree$Pobre <- valid_split_tree$Pobre

# Alinear columnas
vars_modelo_tree <- intersect(names(train_num_tree), names(test_num_tree))
vars_modelo_tree <- setdiff(vars_modelo_tree, "Pobre")

train_num_tree <- train_num_tree[, c(vars_modelo_tree, "Pobre")]
valid_num_tree <- valid_num_tree[, c(vars_modelo_tree, "Pobre")]
test_num_tree  <- test_num_tree[, vars_modelo_tree]

# 🔹 Imputación
for (col in names(train_num_tree)) {
  if (col != "Pobre") {
    med <- median(train_num_tree[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    
    train_num_tree[[col]][is.na(train_num_tree[[col]])] <- med
    valid_num_tree[[col]][is.na(valid_num_tree[[col]])] <- med
    test_num_tree[[col]][is.na(test_num_tree[[col]])]   <- med
  }
}

# Verificación
sum(is.na(train_num_tree))
sum(is.na(valid_num_tree))
sum(is.na(test_num_tree))

# 🔹 Control de entrenamiento
ctrl_tree <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = prSummary 
)

# 🔹 Grid SOLO para cp
grid_cp <- expand.grid(
  cp = seq(0.0001, 0.05, length = 15)
)

# 🔹 Entrenamiento
tree_tuned <- train(
  Pobre ~ .,
  data = train_num_tree,
  method = "rpart",
  trControl = ctrl_tree,
  tuneGrid = grid_cp,
  metric = "F"
)

# Mejor cp
print(tree_tuned)
tree_tuned$bestTune

# 🔹 Probabilidades
tree_prob <- predict(tree_tuned, newdata = valid_num_tree, type = "prob")[, "Yes"]
test_prob_tree <- predict(tree_tuned, newdata = test_num_tree, type = "prob")[, "Yes"]

# 🔹 Evaluación base
res_tree <- f1_eval(tree_prob, valid_num_tree$Pobre, cutoff = 0.5)

res_tree$f1
res_tree$precision
res_tree$recall
res_tree$confusion

# 🔹 Buscar mejor cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_tree <- data.frame()

for (c in cuts) {
  met <- f1_eval(tree_prob, valid_num_tree$Pobre, cutoff = c)
  
  resultados_tree <- rbind(
    resultados_tree,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

# Mejor cutoff
mejor_cutoff_tree <- resultados_tree$cutoff[which.max(resultados_tree$f1)]
mejor_cutoff_tree

# 🔹 Predicción final
test_pred_tree <- ifelse(test_prob_tree >= mejor_cutoff_tree, 1, 0)

test_pred_tree[is.na(test_pred_tree)] <- 0
test_pred_tree <- as.integer(test_pred_tree)

# 🔹 Submission
submission_tree <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_tree
)

# Validaciones
head(submission_tree)
dim(submission_tree)
sum(is.na(submission_tree$Pobre))
unique(submission_tree$Pobre)

# Guardar
write.csv(
  submission_tree,
  "submission_tree_cp_tuned.csv",
  row.names = FALSE,
  quote = FALSE
)
