logit_prob2 <- predict(logit_fit2, newdata = valid_split2, type = "response")

# 1. Curva ROC usando validación
roc_logit2 <- roc(
  response = valid_split2$Pobre,
  predictor = logit_prob2,
  levels = c("No", "Yes"),
  direction = "<"
)

png(here("03_output","figures","roc_logit_mejorado.png"),
    width = 900, height = 700)

# 2. Graficar ROC
plot(
  roc_logit2,
  col = "#27ae60",
  lwd = 4,
  main = "Curva ROC - Logit mejorado"
)

# 3. Agregar AUC al gráfico
text(
  0.6, 0.2,
  paste("AUC =", round(auc(roc_logit2), 4)),
  cex = 1.2,
  col = "darkgreen"
)

dev.off()
# 4. Ver AUC en consola
auc(roc_logit2)

#2 GRAFICA

#Lo que realizaremos a contunuacion es un grafica que nos muestre algunas de las variables en 
#donde el modelo no solo predice, también muestra que  educación, empleo y protección social son dimensiones
#centrales de la pobreza.

# Importancia de variables del Logit mejorado
coef_logit2 <- summary(logit_fit2)$coefficients

importancia_logit2 <- data.frame(
  variable = rownames(coef_logit2),
  coeficiente = coef_logit2[, "Estimate"],
  p_value = coef_logit2[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

importancia_logit2 <- importancia_logit2[importancia_logit2$variable != "(Intercept)", ]
importancia_logit2$importancia <- abs(importancia_logit2$coeficiente)

# Mirar nombres reales para validar
print(importancia_logit2$variable)

# Elegimos las variables que sí aparecen fuerte en tu gráfica original
vars_clave <- c(
  "prom_P6090",  # Afiliación salud
  "prom_P6240",  # Actividad
  "prom_P6430",  # Tipo empleo
  "prom_P6210"   # Educación
)

idx_vars <- unlist(lapply(vars_clave, function(x) {
  grep(x, importancia_logit2$variable)
}))

importancia_filtrada <- importancia_logit2[idx_vars, ]

# Nos quedamos con el coeficiente más alto de cada grupo
importancia_filtrada$grupo <- NA
importancia_filtrada$grupo[grepl("prom_P6090", importancia_filtrada$variable)] <- "Afiliación a salud"
importancia_filtrada$grupo[grepl("prom_P6240", importancia_filtrada$variable)] <- "Actividad laboral"
importancia_filtrada$grupo[grepl("prom_P6430", importancia_filtrada$variable)] <- "Tipo de empleo"
importancia_filtrada$grupo[grepl("prom_P6210", importancia_filtrada$variable)] <- "Educación"

importancia_final <- aggregate(
  importancia ~ grupo,
  data = importancia_filtrada,
  FUN = max
)

importancia_final <- importancia_final[order(importancia_final$importancia), ]

write.csv(
  importancia_final,
  here("03_output", "tables", "importancia_variables_logit.csv"),
  row.names = FALSE
)

# Gráfica
png(
  here("03_output", "figures", "factores_clave_pobreza.png"),
  width = 900,
  height = 700
)

par(mar = c(5, 10, 3, 2))

barplot(
  importancia_final$importancia,
  names.arg = importancia_final$grupo,
  horiz = TRUE,
  las = 1,
  col = "#2E86C1",
  border = NA,
  main = "Factores clave de la pobreza",
  cex.main = 1,
  cex.names = 0.9,
  xlab = "Importancia (|coeficiente|)"
)

dev.off()

par(mar = c(5, 4, 4, 2))

# Grafica 3

#Para esta grafica La subcobertura representa hogares pobres que quedarían por fuera del programa;
#la fuga representa hogares no pobres que recibirían beneficios.El mejor cutoff busca equilibrar ambos errores mediante F1 
#los falsos negativos = hogares pobres clasificados como no pobres = subcobertura
# falsos positivos = hogares no pobres clasificados como pobres = fuga.


pred_valid_logit2 <- ifelse(logit_prob2 >= mejor_cutoff_logit2, "Yes", "No")
pred_valid_logit2 <- factor(pred_valid_logit2, levels = c("No", "Yes"))

real_valid_logit2 <- factor(valid_split2$Pobre, levels = c("No", "Yes"))

cm_logit2 <- caret::confusionMatrix(
  pred_valid_logit2,
  real_valid_logit2,
  positive = "Yes"
)

cm_logit2$table

mat <- cm_logit2$table

TN <- mat["No", "No"]
FN <- mat["No", "Yes"]
FP <- mat["Yes", "No"]
TP <- mat["Yes", "Yes"]

tasa_subcobertura <- FN / (FN + TP)
tasa_fuga <- FP / (FP + TN)

errores_politica <- c(
  "Subcobertura\nFalsos negativos" = tasa_subcobertura,
  "Fuga\nFalsos positivos" = tasa_fuga
)

png(
  here("03_output", "figures", "errores_politica_logit.png"),
  width = 900,
  height = 700
)

barplot(
  errores_politica,
  ylim = c(0, max(errores_politica) + 0.1),
  main = "Errores de política pública - Logit",
  ylab = "Tasa de error",
  col = "#2E86C1",
  border = NA,
  las = 1
)

text(
  x = c(0.7, 1.9),
  y = errores_politica + 0.03,
  labels = paste0(round(errores_politica * 100, 1), "%")
)

dev.off()

##################

# Redondear todo a 3 decimales
resultados_logit2$precision <- round(resultados_logit2$precision, 3)
resultados_logit2$recall    <- round(resultados_logit2$recall, 3)
resultados_logit2$f1        <- round(resultados_logit2$f1, 3)

# Identificar la fila del mejor cutoff
mejor_fila <- which.max(resultados_logit2$f1)

# Crear tabla con fila resaltada
tabla_cutoff <- resultados_logit2 %>%
  kbl(
    col.names = c("Cutoff", "Precisión", "Recall", "F1"),
    align     = "c",
    caption   = "Búsqueda del cutoff óptimo — maximizando F1"
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width        = FALSE,
    position          = "center"
  ) %>%
  row_spec(
    mejor_fila,
    bold       = TRUE,
    color      = "white",
    background = "#1D9E75"   # verde — la fila ganadora
  )

tabla_cutoff

save_kable(
  tabla_cutoff,
  file = here("03_output", "tables", "tabla_cutoff_logit2.html")
)

#############################

# formato largo para ggplot
resultados_largo <- resultados_logit2 %>%
  pivot_longer(
    cols      = c(precision, recall, f1),
    names_to  = "metrica",
    values_to = "valor"
  )

# Etiqueta
resultados_largo$metrica <- factor(
  resultados_largo$metrica,
  levels = c("f1", "recall", "precision"),
  labels = c("F1", "Recall", "Precisión")
)

# identificacion el cutoff óptimo para marcarlo
cutoff_opt  <- resultados_logit2$cutoff[which.max(resultados_logit2$f1)]
f1_max      <- max(resultados_logit2$f1, na.rm = TRUE)

#  graficar
ggplot(resultados_largo, aes(x = cutoff, y = valor, color = metrica)) +
  
  # Líneas de las tres métricas
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  
  # Línea vertical del cutoff óptimo
  geom_vline(
    xintercept = cutoff_opt,
    linetype   = "dashed",
    color      = "#1D9E75",
    linewidth  = 1
  ) +
  
  # Punto destacado en el máximo F1
  geom_point(
    data = resultados_logit2[which.max(resultados_logit2$f1), ],
    aes(x = cutoff, y = f1),
    color = "#1D9E75",
    size  = 5,
    shape = 21,
    fill  = "#1D9E75"
  ) +
  
  # Etiqueta del cutoff óptimo
  annotate(
    "text",
    x     = cutoff_opt + 0.03,
    y     = f1_max - 0.05,
    label = paste0("Cutoff óptimo\n", cutoff_opt),
    color = "#1D9E75",
    size  = 3.5,
    hjust = 0
  ) +
  
  # Colores manuales
  scale_color_manual(
    values = c(
      "F1"        = "#1D9E75",   # verde
      "Recall"    = "#378ADD",   # azul
      "Precisión" = "#E24B4A"    # rojo
    )
  ) +
  
  # Escalas
  scale_x_continuous(breaks = seq(0.10, 0.90, by = 0.10)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  
  # Etiquetas
  labs(
    title    = "Selección del cutoff óptimo — Logit mejorado",
    subtitle = paste0("Cutoff seleccionado: ", cutoff_opt,
                      "  |  F1 máximo en validación: ", round(f1_max, 3)),
    x        = "Cutoff",
    y        = "Valor de la métrica",
    color    = "Métrica"
  ) +
  
  # Tema limpio
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

#Guardar

ggsave(
  filename = here("03_output", "figures", "cutoff_optimo_logit2.png"),
  width    = 10,
  height   = 6,
  dpi      = 300,
  bg       = "white"
)

##################

# Variables numéricas 
vars_desc <- c(
  "P5000",
  "P5010",
  "Nper",
  "Npersug",
  "prom_P6040",
  "prom_P6800",
  "prom_Oc",
  "n_personas",
  "hacinamiento",
  "cuartos_percapita"
)

vars_desc <- intersect(vars_desc, names(train_split2))

# Tabla descriptiva
tabla_desc <- train_split2 %>%
  select(all_of(vars_desc)) %>%
  summarise(
    across(
      everything(),
      list(
        media   = ~mean(., na.rm = TRUE),
        sd      = ~sd(., na.rm = TRUE),
        min     = ~min(., na.rm = TRUE),
        max     = ~max(., na.rm = TRUE),
        missing = ~sum(is.na(.))
      )
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("variable", ".value"),
    names_sep = "_(?=[^_]+$)"
  ) %>%
  mutate(
    media = round(media, 2),
    sd    = round(sd, 2),
    min   = round(min, 2),
    max   = round(max, 2)
  )

# Nombres 
nombres_variables <- c(
  P5000 = "Número de cuartos",
  P5010 = "Número de dormitorios",
  Nper = "Personas en el hogar",
  Npersug = "Personas subsidiables",
  prom_P6040 = "Edad promedio del hogar",
  prom_P6800 = "Ingreso promedio del hogar",
  prom_Oc = "Ocupación promedio",
  n_personas = "Tamaño del hogar",
  hacinamiento = "Índice de hacinamiento",
  cuartos_percapita = "Cuartos por persona"
)

tabla_desc$variable <- nombres_variables[tabla_desc$variable]

# tabla
tabla_final <- tabla_desc %>%
  kbl(
    caption = "Estadísticas descriptivas de variables principales del mejor modelo",
    col.names = c(
      "Variable",
      "Media",
      "Desv. Est.",
      "Mínimo",
      "Máximo",
      "Missing"
    )
  ) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )

tabla_final

#Guardamos imagen
save_kable(
  tabla_final,
  file = here("03_output/tables", "tabla_descriptiva_logit.png")
)
