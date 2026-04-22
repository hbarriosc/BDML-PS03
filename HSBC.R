############################3 Predicción X - Naive Bayes (base)

set.seed(123)

# Partimos de las bases limpias
train_nb <- train %>% select(-id, -Dominio, -Depto)
test_nb  <- test  %>% select(-id, -Dominio, -Depto)

# Split train-validation
idx_nb <- createDataPartition(train_nb$Pobre, p = 0.8, list = FALSE)

train_split_nb <- train_nb[idx_nb, ]
valid_split_nb <- train_nb[-idx_nb, ]

# Solo variables numéricas
train_num_nb <- train_split_nb %>% select(where(is.numeric))
valid_num_nb <- valid_split_nb %>% select(where(is.numeric))
test_num_nb  <- test_nb %>% select(where(is.numeric))

# Agregar variable objetivo
train_num_nb$Pobre <- train_split_nb$Pobre
valid_num_nb$Pobre <- valid_split_nb$Pobre

# Alinear columnas
vars_modelo_nb <- intersect(names(train_num_nb), names(test_num_nb))
vars_modelo_nb <- setdiff(vars_modelo_nb, "Pobre")

train_num_nb <- train_num_nb[, c(vars_modelo_nb, "Pobre")]
valid_num_nb <- valid_num_nb[, c(vars_modelo_nb, "Pobre")]
test_num_nb  <- test_num_nb[, vars_modelo_nb]

# Imputación con mediana (igual a su estándar)
for (col in names(train_num_nb)) {
  if (col != "Pobre") {
    mediana_col <- median(train_num_nb[[col]], na.rm = TRUE)
    if (is.na(mediana_col)) mediana_col <- 0
    
    train_num_nb[[col]][is.na(train_num_nb[[col]])] <- mediana_col
    valid_num_nb[[col]][is.na(valid_num_nb[[col]])] <- mediana_col
    test_num_nb[[col]][is.na(test_num_nb[[col]])]   <- mediana_col
  }
}

# Verificación
sum(is.na(train_num_nb))
sum(is.na(valid_num_nb))
sum(is.na(test_num_nb))

# 🔹 Modelo Naive Bayes
nb_fit <- naiveBayes(Pobre ~ ., data = train_num_nb)

# Predicción (probabilidades)
nb_prob <- predict(nb_fit, newdata = valid_num_nb, type = "raw")[, "Yes"]
test_prob_nb <- predict(nb_fit, newdata = test_num_nb, type = "raw")[, "Yes"]

# 🔹 Evaluación base (cutoff = 0.5)
res_nb <- f1_eval(nb_prob, valid_num_nb$Pobre, cutoff = 0.5)

res_nb$f1
res_nb$precision
res_nb$recall
res_nb$confusion

# 🔹 Búsqueda del mejor cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_nb <- data.frame()

for (c in cuts) {
  met <- f1_eval(nb_prob, valid_num_nb$Pobre, cutoff = c)
  
  resultados_nb <- rbind(
    resultados_nb,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_nb
resultados_nb[which.max(resultados_nb$f1), ]

# 🔹 Mejor cutoff
mejor_cutoff_nb <- resultados_nb$cutoff[which.max(resultados_nb$f1)]
mejor_cutoff_nb

# 🔹 Predicción final en test
test_pred_nb <- ifelse(test_prob_nb >= mejor_cutoff_nb, 1, 0)

test_pred_nb[is.na(test_pred_nb)] <- 0
test_pred_nb <- as.integer(test_pred_nb)

# 🔹 Submission
submission_nb <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_nb
)

# Validaciones
head(submission_nb)
dim(submission_nb)
sum(is.na(submission_nb$Pobre))
unique(submission_nb$Pobre)

# Guardar CSV
write.csv(
  submission_nb,
  "submission_naive_bayes_v1.csv",
  row.names = FALSE,
  quote = FALSE
)

###################Predicción X - Árbol (CART) con tuning de cp

set.seed(123)

# Base (igual que otros modelos)
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

# 🔹 Imputación (igual que ustedes)
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

############# Predicción ELASTIC NET + ROSE
library(caret)
library(ROSE)

set.seed(123)

# 🔹 Base (igual que ustedes)
train_en <- train %>% select(-id, -Dominio, -Depto)
test_en  <- test  %>% select(-id, -Dominio, -Depto)

# 🔹 Split (idealmente usa el mismo idx global)
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
  sampling = "rose",          # 🔥 AQUÍ el cambio clave
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
  preProcess = c("center", "scale")   # 🔥 recomendable para glmnet
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
