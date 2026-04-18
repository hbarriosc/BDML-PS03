#Limpiamos el script que trabajaremos 

cat("\014")
rm(list = ls())

#Cargaremos las librerias necesarias para realizar el taller

library(tidyverse)
library(dplyr)
library(caret)
library(glmnet)
library(rpart)
library(e1071)
library(pROC)

#Cargamos los archivos csv que utilizaresmos, por otro lado ajustamos los delim, por motivo que se contraban en una sola columna
#Se realiza el mismo proceso con las 4 bases
train_hogares <- read_delim(
  "C:/Users/David/OneDrive - Universidad de los Andes/Documentos/Bigdata/Taller 2/train_hogares.csv",
  delim = ";",
  locale = locale(encoding = "UTF-8"),
  show_col_types = FALSE
)

train_personas <- read_delim(
  "C:/Users/David/OneDrive - Universidad de los Andes/Documentos/Bigdata/Taller 2/train_personas.csv",
  delim = ";",
  locale = locale(encoding = "UTF-8"),
  show_col_types = FALSE
)

test_hogares <- read_delim(
  "C:/Users/David/OneDrive - Universidad de los Andes/Documentos/Bigdata/Taller 2/test_hogares.csv",
  delim = ";",
  locale = locale(encoding = "UTF-8"),
  show_col_types = FALSE
)

test_personas <- read_delim(
  "C:/Users/David/OneDrive - Universidad de los Andes/Documentos/Bigdata/Taller 2/test_personas.csv",
  delim = ";",
  locale = locale(encoding = "UTF-8"),
  show_col_types = FALSE
)

#Para este paso agrupamos la informacion de cada hogar ya que tiene varias filas

personas_hogar_train <- train_personas %>%
  group_by(id) %>%
  summarise(
    across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
    n_personas = n(),
    .groups = "drop"
  )

personas_hogar_test <- test_personas %>%
  group_by(id) %>%
  summarise(
    across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
    n_personas = n(),
    .groups = "drop"
  )


#Aqui unimos los hogares en una sola base
base_modelo_train <- train_hogares %>%
  left_join(personas_hogar_train, by = "id")

base_modelo_test <- test_hogares %>%
  left_join(personas_hogar_test, by = "id")

#Validamos que los ajustes esten correctamente 
dim(base_modelo_train)
dim(base_modelo_test)

glimpse(base_modelo_train)
glimpse(base_modelo_test)

#Aqui empezamos a depurar las variables que no contienen tanta información, ya que puede afectar nuestros modelos cuando
#empecemos a realizar las preddicciones 


base_modelo_train_clean <- base_modelo_train[, colMeans(is.na(base_modelo_train)) <= 0.30]
base_modelo_test_clean  <- base_modelo_test[, colMeans(is.na(base_modelo_test)) <= 0.30]

#Validamos que los ajustes esten correctamente 
dim(base_modelo_train_clean)
dim(base_modelo_test_clean)

#Aqui valdiamos que las variables si enecuentren a la hora de predecir y no nos genere erroes, por la no existencia 

vars_comunes <- intersect(names(base_modelo_train_clean), names(base_modelo_test_clean))
vars_train <- unique(c(vars_comunes, "Pobre"))

train <- base_modelo_train_clean[, vars_train]
test  <- base_modelo_test_clean[, vars_comunes]

#Validamos que los ajustes esten correctamente 
dim(train)
dim(test)

#Validamos que la variable pobre se encuentre en la base, ya que sera funamental en nuestras predicciones 

"Pobre" %in% names(train)

train$Pobre <- factor(train$Pobre, levels = c(0, 1), labels = c("No", "Yes"))
table(train$Pobre)
prop.table(table(train$Pobre))

dim(base_modelo_train)
dim(base_modelo_test)
dim(train)
dim(test)
#Aqui podemos encontrar un desbalance en las diferentes clases sociales que se ecuentran
table(train$Pobre)


#Entrenamiento

###Prediccion 1 LPM: El LPM es una regresión lineal donde la variable dependiente toma valores 0 y 1.
# En este caso, modelamos la probabilidad de que un hogar sea pobre.Este modelo asume una relación lineal entre las covariables y la probabilidad de pobreza, lo cual permite una interpretación
#directa de los coeficientes como cambios marginales en dicha probabilidad.

#Nos basamos en teoria del bienestar la porbreza depende del ingreso, tamaño de hogar, eduacucion, empleo, acceso a servicios
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
  "submission_model1_lpm_v1.csv",
  row.names = FALSE,
  quote = FALSE
)

res_lpm$f1
res_lpm$precision
res_lpm$recall


##Prediccion 1.2 LPM Version mejorada

set.seed(123)


#Seleccionamos la variables a vivienda, tamaño del hogar, educación,
#ocupación, formalidad laboral, salud y transferencias y unicamente nos quedarmos con ñas que  en train como en test
#Con el fin de mejorar nuestra prediccion
#Se incluyeron variables de educación y empleo porque, desde la teoría del capital humano, mayores niveles educativos y mejores inserciones
#laborales aumentan la productividad y reducen la probabilidad de pobreza.

vars_economicas <- c(
  # Características del hogar
  "P5000",        # Número de cuartos
  "P5010",        # Cuartos para dormir
  "P5090",        # Tenencia de vivienda
  "Nper",         # Número de personas
  "Npersug",      # Personas en unidad de gasto
  
  # Características promedio de personas en el hogar
  "prom_P6040",   # Edad promedio
  "prom_P6210",   # Nivel educativo (MUY importante teóricamente)
  "prom_P6240",   # Actividad semana pasada (ocupación)
  "prom_P6430",   # Tipo de empleo (cuenta propia, empleado, etc.)
  "prom_P6800",   # Horas trabajadas
  "prom_P6920",   # Cotiza pensión (proxy formalidad)
  "prom_P6090",   # Afiliación salud
  "prom_P7495",   # Recibe arriendos
  "prom_P7505",   # Recibe transferencias
  "prom_Oc",      # Ocupados en el hogar
  "n_personas"    # Total personas
)

#Tomamos las que existan en train y test

vars_disponibles <- intersect(
  intersect(vars_economicas, names(train)),
  intersect(vars_economicas, names(test))
)

cat("Variables seleccionadas:", length(vars_disponibles), "\n")
print(vars_disponibles)


#Construimos las bases 

train_lpm2 <- train %>% select(all_of(c(vars_disponibles, "Pobre")))
test_lpm2  <- test  %>% select(all_of(vars_disponibles))

# Imputación con mediana calculada solo en train

medianas <- sapply(train_lpm2 %>% select(-Pobre), 
                   function(x) median(x, na.rm = TRUE))

for (col in names(medianas)) {
  med <- ifelse(is.na(medianas[col]), 0, medianas[col])
  train_lpm2[[col]][is.na(train_lpm2[[col]])] <- med
  test_lpm2[[col]][is.na(test_lpm2[[col]])]   <- med
}

# Variable objetivo como 0/1
train_lpm2$Pobre_num <- ifelse(train_lpm2$Pobre == "Yes", 1, 0)

cat("NAs en train:", sum(is.na(train_lpm2)), "\n")
cat("NAs en test:", sum(is.na(test_lpm2)), "\n")



# Definimos control con CV de 5 folds
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = prSummary,   
  classProbs      = TRUE,
  savePredictions = "final"
)

# Para usar trainControl de caret con LPM usamos lm directamente
# Hacemos CV manual para mayor control

set.seed(123)
folds <- createFolds(train_lpm2$Pobre_num, k = 5, list = TRUE)

f1_folds <- c()

for (i in seq_along(folds)) {
  idx_val   <- folds[[i]]
  fold_train <- train_lpm2[-idx_val, ]
  fold_valid <- train_lpm2[idx_val, ]
  
  # Entrenamos LPM
  formula_lpm <- as.formula(
    paste("Pobre_num ~", paste(vars_disponibles, collapse = " + "))
  )
  
  modelo_fold <- lm(formula_lpm, data = fold_train)
  
  # Predicción
  prob_val <- predict(modelo_fold, newdata = fold_valid)
  prob_val <- pmax(pmin(prob_val, 1), 0)
  
  # F1 con cutoff 0.5
  pred_val <- ifelse(prob_val >= 0.5, 1, 0)
  
  tp <- sum(pred_val == 1 & fold_valid$Pobre_num == 1)
  fp <- sum(pred_val == 1 & fold_valid$Pobre_num == 0)
  fn <- sum(pred_val == 0 & fold_valid$Pobre_num == 1)
  
  prec <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  rec  <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  f1   <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
  
  f1_folds[i] <- f1
  cat(sprintf("Fold %d - F1: %.4f\n", i, f1))
}

cat(sprintf("\nF1 promedio CV: %.4f ± %.4f\n", mean(f1_folds), sd(f1_folds)))


# Manejamos desbalance

prop_pobre <- mean(train_lpm2$Pobre_num)
cat(sprintf("Proporción pobres: %.2f%%\n", prop_pobre * 100))

# Peso inversamente proporcional a la frecuencia de clase
pesos <- ifelse(
  train_lpm2$Pobre_num == 1,
  1 / prop_pobre,         # Peso alto para pobres (minoría)
  1 / (1 - prop_pobre)    # Peso bajo para no pobres (mayoría)
)

# Modelo final con PESOS para corregir desbalance
formula_final <- as.formula(
  paste("Pobre_num ~", paste(vars_disponibles, collapse = " + "))
)

lpm_fit_v2 <- lm(formula_final, data = train_lpm2, weights = pesos)

summary(lpm_fit_v2)


#Se busca el mejor cutoff

cutoffs_cv <- seq(0.10, 0.90, by = 0.05)
f1_por_cutoff <- numeric(length(cutoffs_cv))

for (j in seq_along(cutoffs_cv)) {
  cut <- cutoffs_cv[j]
  f1_temp <- c()
  
  for (i in seq_along(folds)) {
    idx_val    <- folds[[i]]
    fold_train <- train_lpm2[-idx_val, ]
    fold_valid <- train_lpm2[idx_val, ]
    
    pesos_fold <- ifelse(
      fold_train$Pobre_num == 1,
      1 / mean(fold_train$Pobre_num),
      1 / (1 - mean(fold_train$Pobre_num))
    )
    
    mod_temp <- lm(formula_final, data = fold_train, weights = pesos_fold)
    
    prob_temp <- predict(mod_temp, newdata = fold_valid)
    prob_temp <- pmax(pmin(prob_temp, 1), 0)
    pred_temp <- ifelse(prob_temp >= cut, 1, 0)
    
    tp <- sum(pred_temp == 1 & fold_valid$Pobre_num == 1)
    fp <- sum(pred_temp == 1 & fold_valid$Pobre_num == 0)
    fn <- sum(pred_temp == 0 & fold_valid$Pobre_num == 1)
    
    prec <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    rec  <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    f1   <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    
    f1_temp[i] <- f1
  }
  
  f1_por_cutoff[j] <- mean(f1_temp)
}

resultados_cutoff <- data.frame(
  cutoff = cutoffs_cv,
  f1_cv  = f1_por_cutoff
)

print(resultados_cutoff)

mejor_cutoff_v2 <- resultados_cutoff$cutoff[which.max(resultados_cutoff$f1_cv)]
cat(sprintf("\nMejor cutoff (CV): %.2f | F1 CV: %.4f\n", 
            mejor_cutoff_v2, max(f1_por_cutoff)))


#Validacion final 


prob_test_v2 <- predict(lpm_fit_v2, newdata = test_lpm2)
prob_test_v2 <- pmax(pmin(prob_test_v2, 1), 0)
pred_test_v2 <- ifelse(prob_test_v2 >= mejor_cutoff_v2, 1, 0)

# Imputar NAs residuales (si los hay) con la clase mayoritaria
pred_test_v2[is.na(pred_test_v2)] <- 0
pred_test_v2 <- as.integer(pred_test_v2)

# Diagnóstico
cat(sprintf("Predichos como pobres: %d (%.1f%%)\n",
            sum(pred_test_v2), mean(pred_test_v2) * 100))


# SUBMISSION


submission_lpm_v2 <- data.frame(
  id    = test$id,
  Pobre = pred_test_v2
)

# Validaciones finales
cat("Dimensiones:", dim(submission_lpm_v2), "\n")
cat("NAs en Pobre:", sum(is.na(submission_lpm_v2$Pobre)), "\n")
cat("Valores únicos:", unique(submission_lpm_v2$Pobre), "\n")

write.csv(
  submission_lpm_v2,
  "submission_lpm_v2_mejorado_cv.csv",
  row.names = FALSE,
  quote     = FALSE
)

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

#Modelo -2.1 Prediccion 3 

#De acuerdo a nuestro segundo modelo, nuestra idea es realizar un modelo mas robusto, a medida que vamos 
#entrenendo, sin embargo por otro lado nos encontramos en el dilema de la incertidumbre, la cual validaremos
#si enverdad nuestro modelo fue mejor que el anterior

#para esto  vamos a realizar un modelo Elastic Net, lo utilizaremos ya que tenemos muchas variables, con el fin de que nuestro modelo NO sea debil por cuentiones
#de variables  que se correlacionen y generen ruido 

#traemos nuevamente nuestra varribel pobre

train_num$Pobre <- factor(train_num$Pobre, levels = c("No", "Yes"))
valid_num$Pobre <- factor(valid_num$Pobre, levels = c("No", "Yes"))

#utilizaremos un for, ya que se necesitara calcular la mediana ya que aun hay variables que tienen N/A, se necesita robustes
#cuando hay variables que tengan sesgos y asi no perdemos tantas observaciones 
for (col in names(train_num)) {
  if (col != "Pobre") {
    
    mediana_col <- median(train_num[[col]], na.rm = TRUE)
    
    train_num[[col]][is.na(train_num[[col]])] <- mediana_col
    valid_num[[col]][is.na(valid_num[[col]])] <- mediana_col
    
    if (col %in% names(test_num)) {
      test_num[[col]][is.na(test_num[[col]])] <- mediana_col
    }
  }
}

#Verificamos que efectivamente para nuestro modelo no tuvieramos N/A
sum(is.na(train_num))
sum(is.na(valid_num))
sum(is.na(test_num))


#ya cuando ajustamos los datos para nuestro modelo, realizamos el entrenamiento, esto es con el fin
#de predecir que un hogar es pobre apartir de nuestra covariables 


set.seed(123)

enet_fit <- train(
  Pobre ~ .,
  data = train_num,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 10
)

#Validamos que estuviera bien nuestro modelo

enet_fit
enet_fit$bestTune
#predecimos la probabilidad mencionada anteriomente si un hogar es pobre apartir de nuestra covariables 

enet_prob <- predict(enet_fit, newdata = valid_num, type = "prob")[, "Yes"]

#Utilizamos función F1

res_enet <- f1_eval(enet_prob, valid_num$Pobre, cutoff = 0.5)

res_enet$f1
res_enet$precision
res_enet$recall
res_enet$confusion

#Empezamos a realizar la evaluación inicial para escojer el mejor cutoff la cual nos ayudara a definir
#cuales son los hogares  pobres

cuts <- seq(0.10, 0.90, by = 0.05)
resultados_enet <- data.frame()


for (c in cuts) {
  met <- f1_eval(enet_prob, valid_num$Pobre, cutoff = c)
  
  resultados_enet <- rbind(
    resultados_enet,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_enet
resultados_enet[which.max(resultados_enet$f1), ]

mejor_cutoff_enet <- resultados_enet$cutoff[which.max(resultados_enet$f1)]
mejor_cutoff_enet

#despues de limpiar nuestra prediccion convertimo a  enteros

test_prob_enet <- predict(enet_fit, newdata = test_num, type = "prob")[, "Yes"]

test_pred_enet <- ifelse(test_prob_enet >= mejor_cutoff_enet, 1, 0)

test_pred_enet[is.na(test_pred_enet)] <- 0
test_pred_enet <- as.integer(test_pred_enet)

#Validamos 

submission_enet <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_enet
)

head(submission_enet)
dim(submission_enet)
sum(is.na(submission_enet$Pobre))
unique(submission_enet$Pobre)
names(submission_enet)

#Guardamos
write.csv(submission_enet, "submission_model2_v1.csv", row.names = FALSE, quote = FALSE)

#Modelo 2.1

#Acontinuacion realizaremos un ejercicio de transformació/y/o depuracion, hay que tener encuenta que la transformacio puede jugar
#a favor o encuentra. pero el ejercicio se basa con el fin de mejorar nuestro modelo anterior
#Validamos nuestras bases iniciales
train_ref <- train
test_ref  <- test

#Validamos las variables exitentes#

var_ingreso_pc <- grep("^ingpcug$|^Ingpcug$|ingpc", names(train_ref), value = TRUE)
var_ingreso_total <- grep("^ingtotug$|^Ingtotug$|ingtot", names(train_ref), value = TRUE)

#Aca buscamos la variable de ingreso con el fin de realziar la trasformacion 

if (length(var_ingreso_pc) > 0) {
  var_ingreso_pc <- var_ingreso_pc[1]
} else {
  var_ingreso_pc <- NA
}

if (length(var_ingreso_total) > 0) {
  var_ingreso_total <- var_ingreso_total[1]
} else {
  var_ingreso_total <- NA
}


#####Escogimos las siguientes variables, con el fin de validar número de cuartos por separado, ya que es una variable
#que puede incurrir en relación con si el hogar es pobre 

train_ref <- train_ref %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

test_ref <- test_ref %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

####Realizamos logartimo a las variables, con el fin de eliminar ruido y que queden mucho mas limpias a la hora de realizar la prediccion
##

if (!is.na(var_ingreso_pc)) {
train_ref$ingreso_pc_log <- ifelse(train_ref[[var_ingreso_pc]] >= 0,
                                   log1p(train_ref[[var_ingreso_pc]]), NA)
test_ref$ingreso_pc_log  <- ifelse(test_ref[[var_ingreso_pc]] >= 0,
                                   log1p(test_ref[[var_ingreso_pc]]), NA)
}



if (!is.na(var_ingreso_total)) {
  train_ref$ingreso_total_log <- ifelse(train_ref[[var_ingreso_total]] >= 0,
                                        log1p(train_ref[[var_ingreso_total]]), NA)
  test_ref$ingreso_total_log  <- ifelse(test_ref[[var_ingreso_total]] >= 0,
                                        log1p(test_ref[[var_ingreso_total]]), NA)
}
  

##### eliminamos las varibles que no utilizaremos### 

train_ref <- train_ref %>% select(-id, -Dominio, -Depto)
test_ref  <- test_ref %>% select(-id, -Dominio, -Depto)

#Vamos a organizar el modelo para el nuevo entranamiento 

set.seed(123)
idx_ref <- createDataPartition(train_ref$Pobre, p = 0.8, list = FALSE)

train_split_ref <- train_ref[idx_ref, ]
valid_split_ref <- train_ref[-idx_ref, ]

train_num_ref <- train_split_ref %>% select(where(is.numeric))
valid_num_ref <- valid_split_ref %>% select(where(is.numeric))
test_num_ref  <- test_ref %>% select(where(is.numeric))


train_num_ref$Pobre <- train_split_ref$Pobre
valid_num_ref$Pobre <- valid_split_ref$Pobre

vars_modelo_ref <- intersect(names(train_num_ref), names(test_num_ref))
vars_modelo_ref <- setdiff(vars_modelo_ref, "Pobre")

train_num_ref <- train_num_ref[, c(vars_modelo_ref, "Pobre")]
valid_num_ref <- valid_num_ref[, c(vars_modelo_ref, "Pobre")]
test_num_ref  <- test_num_ref[, vars_modelo_ref]

### Aqui vamos a utilizar nuvamente la mediana de las varibales para calcular nuestro modelo

for (col in names(train_num_ref)) {
  if (col != "Pobre") {
    mediana_col <- median(train_num_ref[[col]], na.rm = TRUE)
    
    if (is.na(mediana_col)) mediana_col <- 0
    
    train_num_ref[[col]][is.na(train_num_ref[[col]])] <- mediana_col
    valid_num_ref[[col]][is.na(valid_num_ref[[col]])] <- mediana_col
    
    if (col %in% names(test_num_ref)) {
      test_num_ref[[col]][is.na(test_num_ref[[col]])] <- mediana_col
    }
  }
}

#Verificamos que todo este correcto 

sum(is.na(train_num_ref))
sum(is.na(valid_num_ref))
sum(is.na(test_num_ref))


train_num_ref$Pobre <- factor(train_num_ref$Pobre, levels = c("No", "Yes"))
valid_num_ref$Pobre <- factor(valid_num_ref$Pobre, levels = c("No", "Yes"))


nzv_cols <- nearZeroVar(train_num_ref %>% select(-Pobre))

if (length(nzv_cols) > 0) {
  cols_quitar <- names(train_num_ref %>% select(-Pobre))[nzv_cols]
  
  train_num_ref <- train_num_ref %>% select(-all_of(cols_quitar))
  valid_num_ref <- valid_num_ref %>% select(-all_of(cols_quitar))
  test_num_ref  <- test_num_ref %>% select(-all_of(cols_quitar))
}

#Aqui realizamos la mezcla entre ridge  y lasso , con el fin de buscar la mejor combinancion para nuestro modelo 

grid_enet <- expand.grid(
  alpha = seq(0, 1, by = 0.2),
  lambda = 10^seq(-3, 0, length = 20)
)


#Empezamos a entrenar el modelo 

set.seed(123)

enet_fit_ref <- train(
  Pobre ~ .,
  data = train_num_ref,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = grid_enet
)


enet_fit_ref
enet_fit_ref$bestTune

enet_prob_ref <- predict(enet_fit_ref, newdata = valid_num_ref, type = "prob")[, "Yes"]


res_enet_ref <- f1_eval(enet_prob_ref, valid_num_ref$Pobre, cutoff = 0.5)

res_enet_ref$f1
res_enet_ref$precision
res_enet_ref$recall
res_enet_ref$confusion

#Validamos el mejor cutoff

cuts <- seq(0.10, 0.90, by = 0.05)
resultados_enet_ref <- data.frame()

for (c in cuts) {
  met <- f1_eval(enet_prob_ref, valid_num_ref$Pobre, cutoff = c)
  
  resultados_enet_ref <- rbind(
    resultados_enet_ref,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_enet_ref
resultados_enet_ref[which.max(resultados_enet_ref$f1), ]


mejor_cutoff_enet_ref <- resultados_enet_ref$cutoff[which.max(resultados_enet_ref$f1)]
mejor_cutoff_enet_ref

#Predecimos  y validamos que no haya nnigun dato faltante para generar nuestra base 

test_prob_enet_ref <- predict(enet_fit_ref, newdata = test_num_ref, type = "prob")[, "Yes"]

test_pred_enet_ref <- ifelse(test_prob_enet_ref >= mejor_cutoff_enet_ref, 1, 0)
test_pred_enet_ref[is.na(test_pred_enet_ref)] <- 0
test_pred_enet_ref <- as.integer(test_pred_enet_ref)

submission_enet_ref <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_enet_ref
)

head(submission_enet_ref)
dim(submission_enet_ref)
sum(is.na(submission_enet_ref$Pobre))
unique(submission_enet_ref$Pobre)
names(submission_enet_ref)

write.csv(submission_enet_ref, "submission_model2-1_v1.csv",
          row.names = FALSE, quote = FALSE)

##Concluimos apesar de que quicimos mejorar con este modelo obtuvimos el mismo resultado 0.63

########################################################
##4 MODELO RANDOM FOREST


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


