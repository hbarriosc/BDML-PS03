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

# 8. Predicciones
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
test_pred_logit[is.na(test_pred_logit)] <- 0

#Predicción final en test
test_pred_logit <- as.integer(test_pred_logit)

#Submission
submission_logit <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_logit
)

# Chequeos
head(submission_logit)
dim(submission_logit)
sum(is.na(submission_logit$Pobre))
unique(submission_logit$Pobre)
names(submission_logit)

#Guardardamos  CSV
write.csv(submission_logit, "submission_logit_v2.csv", row.names = FALSE, quote = FALSE)