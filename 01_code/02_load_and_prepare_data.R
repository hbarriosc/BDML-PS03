#Cargamos los archivos csv que utilizaremos
#Ajustamos los delim, por motivo que se contraban en una sola columna
#Se realiza el mismo proceso con las 4 bases

train_hogares <- read.csv(here("02_data","train_hogares.csv"))
train_personas <- read.csv(here("02_data","train_personas.csv"))
test_hogares <- read.csv(here("02_data","test_hogares.csv"))
test_personas  <- read.csv(here("02_data","test_personas.csv"))

#Para este paso agrupamos la información de cada hogar ya que tiene varias filas

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

#Aqui empezamos a depurar las variables que no contienen tanta información, ya que 
#puede afectar nuestros modelos cuando empecemos a realizar las predicciones 

base_modelo_train_clean <- base_modelo_train[, colMeans(is.na(base_modelo_train)) <= 0.30]
base_modelo_test_clean  <- base_modelo_test[, colMeans(is.na(base_modelo_test)) <= 0.30]

#Validamos que los ajustes estén aplicados correctamente

dim(base_modelo_train_clean)
dim(base_modelo_test_clean)

#Aqui valdiamos que las variables si se encuentren a la hora de predecir y no 
#nos genere errores por la no existencia 

vars_comunes <- intersect(names(base_modelo_train_clean), names(base_modelo_test_clean))
vars_train <- unique(c(vars_comunes, "Pobre"))

train <- base_modelo_train_clean[, vars_train]
test  <- base_modelo_test_clean[, vars_comunes]

#Validamos los ajustes

dim(train)
dim(test)

#Validamos que la variable pobre se encuentre en la base, ya que sera fundamental 
#en nuestras predicciones 

"Pobre" %in% names(train)

train$Pobre <- factor(train$Pobre, levels = c(0, 1), labels = c("No", "Yes"))
table(train$Pobre)
prop.table(table(train$Pobre))

dim(base_modelo_train)
dim(base_modelo_test)
dim(train)
dim(test)

#Aqui podemos encontrar un desbalance en las diferentes clases

table(train$Pobre)
