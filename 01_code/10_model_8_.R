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

# Imputación con mediana 
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
