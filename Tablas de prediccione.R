#Registro de modelos

#Primera predicción

registro_modelos <- data.frame(
  modelo = "logit_base",
  variables = "Numericas sin id, Dominio y Depto",
  cutoff = mejor_cutoff_logit,
  f1_validacion = max(resultados_logit$f1),
  kaggle_score = 0.55,
  comentario = "Primer intento, modelo con imputacion 0.55"
)

registro_modelos

View(registro_modelos)

write.csv(registro_modelos, "registro_modelos.csv", row.names = FALSE)

#Segunda predicción 

nueva_fila <- data.frame(
  modelo = "elastic_net",
  variables = "Numericas + imputacion mediana",
  cutoff = mejor_cutoff_enet,
  f1_validacion = max(resultados_enet$f1),
  kaggle_score = 0.63,
  comentario = "Elastic Net con imputacion de 0.63"
)

registro_modelos <- rbind(registro_modelos, nueva_fila)

View(registro_modelos)
write.csv(registro_modelos, "registro_modelos.csv", row.names = FALSE)