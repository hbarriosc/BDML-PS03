# Prediccion 4 Modelo Elastic Net 

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
