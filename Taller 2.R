library(tidyverse)
library(dplyr)
library(glmnet)
library(caret)

test_hogares <- read.csv("C:/Users/JuanGuillermoSanchez/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_hogares.csv")
test_personas <- read.csv("C:/Users/JuanGuillermoSanchez/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_personas.csv")
train_hogares <- read.csv("C:/Users/JuanGuillermoSanchez/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_hogares.csv")
train_personas<- read.csv("C:/Users/JuanGuillermoSanchez/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_personas.csv")

length(intersect(names(train_hogares),names(train_personas)))

### Data Wrangling

  ## Merge de conjuntos de TRAIN

    # 1. Agregar personas
personas_hogar_train <- train_personas %>%
  group_by(id) %>%
  summarise(
    across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
    n_personas = n()
  )

    # 2. Merge final (persona-hogar)
base_modelo_train <- train_hogares %>%
  left_join(personas_hogar_train, by = "id")

    # 3. Eliminar variables con mas de 30% de NAs
base_modelo_train_clean <- base_modelo_train[, colMeans(is.na(base_modelo_train)) <= 0.3]


  ## Merge de conjuntos de TEST

    # 1. Agregar personas
personas_hogar_test <- test_personas %>%
  group_by(id) %>%
  summarise(
    across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
    n_personas = n()
  )

    # 2. Merge final
base_modelo_test <- test_hogares %>%
  left_join(personas_hogar_test, by = "id")

    # 3. Eliminar variables con mas de 30% de NAs
base_modelo_test_clean <- base_modelo_test[, colMeans(is.na(base_modelo_test)) <= 0.3]

# Descripción de variables utilizables (* para variables que requieren ajustes)

# id:                     identificador del hogar
# Clase(persona):         1. Cabecera, 2. Resto (centros poblados y área rural dispersa) 
# Dominio(persona):       Cada una de las 24 a.M., otras cabeceras y resto  
# P5000(hogares):         Incluyendo sala-comedor ¿de cuántos cuartos en total dispone este hogar? 
# P5010(hogares):         ¿En cuántos de esos cuartos duermen las personas de este hogar? 
# P5090(hogares):         La vivienda ocupada por este hogar es: 1.Propia, totalmente pagada, 2. Propia, la están pagando 3. En arriendo o subarriendo, 4. En usufructo, 5. Posesión sin titulo, ocupante, 6. Otra
# Nper(hogares):          Número de personas en el hogar
# Npersug(hogares):       Número de personas en la unidad de gasto
# Li(hogares):            Linea de indigencia
# Lp(hogares):            Linea de probreza
# Fex_c(hogares):         Factor de expansión anualizado
# Depto(hogar):           Departamento
# Fex_dpto(hogar):        Factor de expansión departamental
# prom_Orden(personas):   Identificación de la persona en el hogar
# prom_Clase(personas):*  1. Cabecera, 2. Resto (centros poblados y área rural dispersa) 
# prom_P6020(personas):*  Sexo 1 hombre 2 mujer
# prom_P6040(personas):   ¿cuántos años cumplidos tiene?
# prom_P6050(personas):   ¿Cuál es el parentesco  Con el jefe del hogar? 1. Jefe (a) del hogar 2. Pareja, esposo(a), cónyuge, 3. Hijo(a), hijastro(a) 4. Nieto(a) 5. Otro pariente 6. Empleado(a) del servicio 7. Pensionado 8.Trabajador 9. Otro no pariente
# prom_P6090(personas):   ¿Está afiliado, es cotizante o es beneficiario de alguna entidad de seguridad social en salud? 1 sí 2 no 9 no sabe, no informa 
# prom_P6100(personas):   ¿A cuál de los siguientes regímenes de seguridad social en salud está afiliado: a. Contributivo (eps)? b. Especial ? (fuerzas armadas, ecopetrol, universidades públicas) c. Subsidiado? (eps-s) d. No sabe, no informa 
# prom_P6210(personas):   ¿Cuál es el nivel educativo más alto alcanzado y el último grado aprobado en este nivel? 1. Ninguno 2. Preescolar 3. Básica primaria (1o - 5o) 4. Básica secundaria (6o - 9o) 5. Media (10o -13o) 6. Superior o universitaria 7. No sabe, no informa 
# prom_P6210s1(personas): Grado escolar aprobado
# prom_P6240(personas):   ¿en que actividad ocuopó la mayor parte del tiempo la semana pasada? 1.Trabajando 2. Buscando trabajo 3. Estudiando 4. Oficios del hogar 5. Incapacitado permanente para trabajar 6. Otra actividad
# prom_Oficio(personas):  ¿Qué hace en este trabajo?
# prom_P6426(personas):   ¿cuanto tiempo lleva Trabajando en esta empresa, negocio, industria, oficina, firma o finca de manera continua? 
# prom_P6430(personas):   En este trabajo es: 1. Obrero o empleado de empresa particular 2. Obrero o empleado del gobierno 3. Empleado doméstico 4. Trabajador por cuenta propia 5. Patrón o empleador 6. Trabajador familiar sin remuneración 7. Trabajador sin remuneración en empresas o negocios de otros hogares 8. Jornalero o peón 9. Otro,
# prom_P6800(personas):   ¿cuántas horas a la semana trabaja normalmente.... en ese trabajo ? 
# prom_P6870(personas):   ¿cuántas personas en total tiene la empresa, negocio, industria, oficina, firma, finca o sitio donde Trabaja? 1. Trabaja solo 2. 2 a 3 personas 3. 4 a 5 personas 4. 6 a 10 personas 5. 11 a 19 personas 6. 20 a 30 personas 7. 31 a 50 personas 8. 51 a 100 personas 9. 101 o más personas 
# prom_P6920(personas):   ¿Está Cotizando actualmente a un fondo de pensiones? 1 sí 2 no 3 ya es pensionado 
# prom_P7040(personas):   Además de la ocupación principal, ¿tenía la semana pasada otro trabajo o negocio? 1 sí 2 no 
# prom_P7090(personas):   Además de las horas que trabaja actualmente ¿quiere trabajar más horas? 1 sí 2 no 
# prom_P7495(personas):   El mes pasado, ¿recibió pagos por concepto de arriendos y/o pensiones? 1 sí 2 no 
# prom_P7505(personas):   Durante los últimos doce meses, ¿recibió dinero de otros hogares, personas o instituciones no gubernamentales; dinero por intereses, dividendos, utilidades o cesantias? 1 sí 2 no 
# prom_Pet(personas):     Población en edad de trabajar 1: Sí, 2: No
# prom_Oc(personas):*     Ocupado 1: sí
# prom_Fex_c(personas):*  Factor de expansión anualizado
# prom_Depto(personas):*  Departamento
# prom_Fex_dpto:          Factor de expansión departamental
# n_personas(personas):*  numero de personas


# Datasets unicamente con variables en comúnes entre TRAIN y TEST

vars_comunes <- intersect(names(base_modelo_test_clean),
                          names(base_modelo_train_clean))

base_modelo_train_clean_depurado <- base_modelo_train_clean[, vars_comunes]
base_modelo_test_clean_depurado  <- base_modelo_test_clean[, vars_comunes]

