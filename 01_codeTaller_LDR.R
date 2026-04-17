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

##############################################################################3
##########Entrenamiento############################################
#####################################

#Modelo -1 
#Acontinuación despues de limpiar nuestros datos vamos a crear un set see en donde se realizar un modelo
#logit con el fin de estimar la probabilidad de que el hogar sea pobre en donde utilizamos variables demograficas
#socieconomicas 

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

#Modelo -2

#De acuerdo a nuestro primer modelo, nuestra idea es realizar un modelo mas robusto, a medida que vamos 
#entrenendo, sin embargo por otro lado nos encontramos en el dile de la incertidumbre, la cual validaremos
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

for (c in cuts) {
  met <- f1_eval(rf_prob, valid_num_rf$Pobre, cutoff = c)
  
  resultados_rf <- rbind(
    resultados_rf,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_rf
resultados_rf[which.max(resultados_rf$f1), ]

mejor_cutoff_rf <- resultados_rf$cutoff[which.max(resultados_rf$f1)]
mejor_cutoff_rf

# Predicción final
test_prob_rf <- predict(rf_fit, newdata = test_num_rf, type = "prob")[, "Yes"]

test_pred_rf <- ifelse(test_prob_rf >= mejor_cutoff_rf, 1, 0)
test_pred_rf[is.na(test_pred_rf)] <- 0
test_pred_rf <- as.integer(test_pred_rf)

# Submission
submission_rf <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_rf
)

head(submission_rf)
dim(submission_rf)
sum(is.na(submission_rf$Pobre))
unique(submission_rf$Pobre)

write.csv(submission_rf, "submission_model4_rf_v1.csv",
          row.names = FALSE, quote = FALSE)