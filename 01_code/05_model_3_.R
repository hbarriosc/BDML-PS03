#Modelo 2 Logit- 3 prediccion 


#Acontinuación despues de limpiar nuestros datos vamos a seguir con nuestra semilla en donde se realizar un modelo
#logit con el fin de estimar la probabilidad de que el hogar sea pobre en donde utilizamos variables demograficas
#socieconomicas -una variable dependiente dicotómica: pobre / no pobre.
#El modelo logit se sustenta en la idea de que la probabilidad de pobreza de un hogar puede explicarse a partir de características observables del hogar y sus miembros,
#como tamaño del hogar, educación, empleo y condiciones habitacionales.
#Banco Mundial, Handbook on Poverty and Inequality. Enfoque aplicado para análisis de pobreza y bienestar.

#UCLA OARC, Logistic Regression. interpretación del modelo logit como modelo para variables dicotómicas
#Gary Becker, Human Capital. Fundamento clásico para variables de educación y productividad.

#Semilla  


set.seed(123)

# Para nuestra primera prediccion, eliminaremos variables que no vamos a usar
train_model <- train %>% select(-id, -Dominio, -Depto)
test_model  <- test %>% select(-id, -Dominio, -Depto)

# Vamos a realizar entrenamiento y validación
idx <- createDataPartition(train_model$Pobre, p = 0.8, list = FALSE)

train_split <- train_model[idx, ]
valid_split <- train_model[-idx, ]

# Control de entrenamiento
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

#Aqui, solo utilizaremos solo variables numéricas
train_num <- train_split %>% select(where(is.numeric))
valid_num <- valid_split %>% select(where(is.numeric))
test_num  <- test_model %>% select(where(is.numeric))

# Validamos la variable pobre, la cual es importante para las prediccions 
train_num$Pobre <- train_split$Pobre
valid_num$Pobre <- valid_split$Pobre

#Aqui validaremos train, valid y test
vars_modelo <- intersect(names(train_num), names(test_num))
vars_modelo <- setdiff(vars_modelo, "Pobre")

train_num <- train_num[, c(vars_modelo, "Pobre")]
valid_num <- valid_num[, c(vars_modelo, "Pobre")]
test_num  <- test_num[, vars_modelo]

# Entrenamos el modelo logit
logit_fit <- glm(Pobre ~ ., data = train_num, family = binomial)

# Predicciones
logit_prob <- predict(logit_fit, newdata = valid_num, type = "response")
test_prob_logit <- predict(logit_fit, newdata = test_num, type = "response")

#Utilizamos función F1
f1_eval <- function(prob, y_true, cutoff = 0.5) {
  pred <- ifelse(prob >= cutoff, "Yes", "No")
  pred <- factor(pred, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred, y_true, positive = "Yes")
  
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

# Empezamos a realziar la evaluación inicial
res_logit <- f1_eval(logit_prob, valid_num$Pobre, cutoff = 0.5)

res_logit$f1
res_logit$precision
res_logit$recall
res_logit$confusion

# Buscamos el  mejor cutoff
cuts <- seq(0.10, 0.90, by = 0.05)
resultados_logit <- data.frame()

for (c in cuts) {
  met <- f1_eval(logit_prob, valid_num$Pobre, cutoff = c)
  
  resultados_logit <- rbind(
    resultados_logit,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_logit
resultados_logit[which.max(resultados_logit$f1), ]

# cutoff
mejor_cutoff_logit <- resultados_logit$cutoff[which.max(resultados_logit$f1)]
mejor_cutoff_logit

# Reemplazar valores nulos

test_pred_logit <- ifelse(test_prob_logit >= mejor_cutoff_logit, 1, 0)

test_pred_logit[is.na(test_pred_logit)] <- 0


#Predicción final en test
test_pred_logit <- as.integer(test_pred_logit)

#Submission
submission_logit <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_logit
)

# Validamos que todo este correctamente 
head(submission_logit)
dim(submission_logit)
sum(is.na(submission_logit$Pobre))
unique(submission_logit$Pobre)
names(submission_logit)

#Guardardamos  CSV
write.csv(submission_logit, "submission_logit_v2.csv", row.names = FALSE, quote = FALSE)
