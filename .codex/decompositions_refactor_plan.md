# Plan maestro de refactorización y ampliación de `decompositions`

> Estado del documento: implementación en curso; fase 2 en TT→TR.
>
> Rama de referencia inicial: `tt_rss`.
>
> Última actualización: 2026-08-28.
>
> Nota de versionado: esta guía se mantiene versionada explícitamente aunque
> `.codex/` esté ignorada por el `.gitignore` general.

## 0. Misión y guía para retomar el proyecto

El objetivo de este proyecto es rediseñar por completo la sección
`tensorkrowch.decompositions` para convertirla en una familia coherente,
reutilizable y extensible de algoritmos de descomposición tensorial. La API
habitual debe seguir siendo sencilla —funciones como `tt_svd`, `tt_als` o
`tt_rss` que devuelven listas de cores—, mientras que una API avanzada basada
en objetos (`TTSVD`, `TTALS`, `TTRSS`, etc.) permitirá fijar una función o
tensor una sola vez y ejecutar repetidamente `.fit(...)` con distintos ranks,
samples o estrategias.

El rediseño debe:

- unificar la infraestructura compartida por TT, TR y TTM;
- adoptar en todas partes la semántica actual de
  `tensorkrowch.utils.truncated_svd`;
- separar los drivers algorítmicos de las estrategias intercambiables:
  truncación, entornos ALS, sampling, apertura local de loops, recursión de
  sketches, fitting físico, transformaciones de valores y ejecución;
- reutilizar las partes sólidas de los proyectos externos `tt2tr`,
  `peps-rss`, `vmc-rss-solvers` y `l2g-tn-solvers`, pero reescribirlas con los
  contratos y tests de TensorKrowch en lugar de copiarlas;
- admitir funciones escalares o tensoriales, embeddings y dominios distintos
  por posición, QTT multivariable, fuentes sparse/empíricas y fuentes TT;
- preparar desde el inicio unidades de trabajo serializables y sin estado
  global para una paralelización posterior;
- portar y reorganizar PEPS únicamente al final y únicamente en la rama
  `peps_rss`, distinguiendo claramente código estable, experimental y legacy.

Este documento es la fuente de verdad del proyecto. En un chat nuevo se debe:

1. leer primero esta misión, las decisiones de la sección 2 y el esquema de la
   sección 3;
2. consultar el resumen de progreso de la sección 1;
3. localizar el primer TODO sin marcar cuyas dependencias estén completas;
4. implementar solo esa unidad o el pequeño grupo de unidades acordado;
5. ejecutar sus criterios de aceptación;
6. hacer un commit atómico y marcarlo como pendiente de revisión cuando el
   usuario haya autorizado avance continuo; en otro caso, pedir confirmación
   antes de marcar el TODO y hacer el commit.

No se debe reinterpretar un prototipo externo como comportamiento definitivo.
Ante una diferencia, prevalecen, en este orden: una instrucción explícita del
usuario, este documento, `.codex/tensorkrowch_style_guidelines.md`, los tests de
TensorKrowch y, por último, el código histórico.

---

## 1. Seguimiento del proyecto

### 1.1 Leyenda

- `[ ]`: no iniciado.
- `[x]`: implementado, validado localmente y commiteado. El campo `Estado`
  distingue si está pendiente de revisión del usuario o completamente cerrado.
- Una tarea parcialmente desarrollada permanece `[ ]`; su estado temporal se
  anota debajo como `Estado: en progreso`.
- `EXP`: componente intencionadamente experimental.
- `COMPAT`: fachada o alias temporal de compatibilidad.

Los identificadores (`SVD-01`, `ALS-04`, etc.) son estables y deben aparecer en
el mensaje y, cuando resulte útil, en el commit correspondiente.

### 1.2 Resumen de progreso

| Fase | Objetivo | Estado |
|---|---|---|
| 1 | Infraestructura común y SVD | 10/10 implementadas; 3 pendientes de revisión |
| 2 | ALS, apertura de loops y TT→TR | 20/20 implementadas; 20 pendientes de revisión |
| 3 | Sketching (RS/RSS), transforms y QTT | 0/26 tareas |
| 4 | Ejecución paralela TT/TR | 0/12 tareas |
| 5 | Port y refactorización PEPS en `peps_rss` | 0/20 tareas |

### 1.3 Regla de commits

Cada TODO principal está diseñado como una unidad razonable de commit. Se
pueden agrupar dos TODOs pequeños si forman una única transición atómica, pero
no se mezclarán:

- refactors mecánicos con cambios algorítmicos;
- infraestructura común con el port PEPS;
- código estable con experimentos no señalados;
- cambios de API con eliminaciones de compatibilidad.

Antes de cada commit:

```bash
conda run -n test_tk pytest <tests afectados>
git diff --check
git status --short
```

Además se inspeccionará el diff completo para evitar código muerto, helpers
duplicados y cambios ajenos a la tarea.

---

## 2. Decisiones arquitectónicas y nomenclatura canónica

### 2.1 Dos niveles de API

#### API pública sencilla

Las funciones directas aceptan argumentos ordinarios y devuelven por defecto
`list[torch.Tensor]`, listas bidimensionales en PEPS, o el tipo sencillo que
corresponda. Ejemplo:

```python
cores = tk.decompositions.tt_rss(
    function=function,
    embedding=embedding,
    sketch_samples=samples,
    rank=16,
)
model = tk.models.MPS(tensors=cores)
```

No se obligará al usuario habitual a construir dataclasses de configuración,
políticas ni objetos de resultados.

#### API pública avanzada

Cada familia tendrá una clase que fija el problema y permite repetir `.fit`:

```python
decomposer = tk.decompositions.TTRSS(
    function=function,
    embedding=embedding,
    domain=domain,
)
result_8 = decomposer.fit(sketch_samples=samples, rank=8)
result_16 = decomposer.fit(sketch_samples=samples, rank=16)
```

`.fit(...)` devuelve una clase ligera de resultado, por ejemplo
`TTDecomposition`, con `.cores`, `.rank` y `.metrics`. El objeto algoritmo
puede conservar la función o tensor necesario para repetir fits; el resultado
no conservará callables pesados salvo solicitud explícita.

Las estrategias y dataclasses avanzadas serán opcionales. Los argumentos
comunes de las funciones directas se traducirán internamente a esas piezas.

### 2.2 Terminología

Se usará de forma sistemática:

- **TT**, no MPS, dentro de `decompositions`;
- **TR**, no MPS periódico;
- **TTM** (*Tensor Train Matrix*), no MPO;
- **rank**, no bond dimension/bond dims, también cuando el valor normalizado
  sea una secuencia;
- `rank`: argumento normalmente escalar que impone el mismo upper bound en
  todos los enlaces; donde sea necesario fijar ranks TR concretos, también
  acepta una secuencia de longitud `n_sites`;
- en TR, `rank[k]` siempre designa el enlace derecho del core `k` y
  `rank[-1]` es por tanto el enlace cíclico entre el último y el primer core;
- `tr_rank` solo se usa cuando el enlace cíclico todavía no existe como parte
  de una lista TR, principalmente al convertir TT→TR;
- `embedding`, `domain`, `base`, `level`, etc. en singular aunque acepten un
  valor compartido o una secuencia por variable;
- los argumentos y propiedades públicas que acepten un escalar o una
  secuencia se nombran en singular: `rank`, `input_dim`, `output_dim`,
  `embedding`, `domain`, etc.;
- dentro de `decompositions`, las dimensiones locales de TT/TR se denominan
  `input_dim`; TTM usa `input_dim` y `output_dim`. No se expondrá
  `physical_dim`/`physical_shapes`; “físico” se reserva para coordenadas o
  geometría física real, por ejemplo en mapas QTT;
- los docstrings y explicaciones de `decompositions` usan siempre **input
  dimension** y **rank**. Nombres legacy como `mps.phys_dim` o
  `mps.bond_dim` solo aparecen cuando se cita literalmente un atributo de la
  API de `models`;
- todo método que exponga `rank`, `cutoff`, `atol`, `rtol` y
  `cum_percentage` reutiliza literalmente la sección canónica documentada en
  `truncated_svd`, incluida la fórmula de `cum_percentage`; las clases y
  funciones de una misma familia mantienen además introducciones, niveles de
  verbosity, retornos y ejemplos con estructura paralela;
- toda mención en docstrings a los modelos MPS, MPSData o MPO se escribe como
  enlace Sphinx a `tensorkrowch.models.MPS`, `MPSData` o `MPO`; los snippets de
  código conservan naturalmente sus constructores Python;
- se mantienen plurales únicamente para colecciones inequívocas como
  `cores`, `samples`, `metrics` o registros internos locales;
- `SampledSketch`, no `CoordinateSketch`;
- `GlobalValueTransform` y `LocalValueTransform`;
- `SketchRecursion`, `SketchGaugeRecursion` y `TTCoreGaugeRecursion` para
  destacar la recursión, no “transport”.

Una lista interna de ranks efectivos o caps sigue siendo válida. El uso
público normal continúa siendo un entero compartido; la secuencia es la ruta
explícita para fijar enlaces TR no uniformes sin introducir `tr_rank` aparte.

En métodos con shapes ALS fijas, `rank` es el upper bound con el que se
inicializan/prescriben los virtual spaces; un one-site ALS no reduce esos
ranks por sí solo. Ranks efectivos menores aparecen por factibilidad algebraica
obligatoria o si existe una fase explícita de discovery/deflation/truncación.
Para TT, cada interfaz se inicializa con
`min(rank, prod(dims_left), prod(dims_right))`; este clipping determinista no
se considera rank discovery y se registra. En TR-SVD, una secuencia se
interpreta con la convención de enlaces anterior; con discovery, el rank de la
bipartición se divide en dos factores lo más equilibrados posible, sujeto a
factibilidad algebraica y a los caps disponibles.

### 2.3 Truncación

Todas las descomposiciones usarán exactamente:

```python
rank=None
cutoff=None
atol=None
rtol=None
cum_percentage=None
```

con la semántica actual de `truncated_svd`:

- se conserva el mínimo rank impuesto por todos los criterios;
- `cutoff` conserva `s > cutoff`;
- `atol` y `rtol` se aplican a la suma de cuadrados de la cola;
- `cum_percentage` equivale a `rtol = 1 - cum_percentage`;
- se conserva al menos un valor singular;
- `rank` es un único upper bound global.

El backend exacto se configura globalmente con `tk.set_svd_method("svd" |
"qr_svd")` o temporalmente mediante `with tk.svd_method(...):`. La variable
de entorno `TENSORKROWCH_SVD_METHOD` permite fijar su valor inicial. El default
es `"svd"`, que conserva el comportamiento histórico. `truncated_svd` admite
además `svd_method=None | "svd" | "qr_svd"` como override de bajo nivel;
`None` consulta la configuración activa. Las descomposiciones y modelos no
repiten este argumento en sus APIs.

No se portarán los antiguos `tol`, `eps -> cum_percentage`, ni criterios
basados en `sum(s)` de los proyectos externos.

### 2.4 Retornos y compatibilidad

- Las funciones directas devuelven cores por defecto.
- `return_info=True` seguirá ofreciendo `(cores, info_dict)` donde ya exista.
- La API avanzada devuelve `*Decomposition`.
- Se podrá añadir `return_result=True` a funciones nuevas solo si evita una
  segunda API paralela; no será obligatorio para el uso normal.
- `vec_to_mps` y `mat_to_mpo` quedarán como wrappers `COMPAT` de `tt_svd` y
  `ttm_svd`, con `DeprecationWarning`, `stacklevel=2` y conservación de los
  keywords `vec=` y `mat=`.
- Los helpers actuales `extend_with_output`, `sketching`,
  `create_projector`, `trimming` y `val_error` se caracterizarán antes de
  privatizarlos. Los que aún importe `tt2tr` tendrán wrappers temporales hasta
  migrar el caller.
- Los aliases se mantendrán como mínimo durante un ciclo completo de versión.

### 2.5 Device, dtype y memoria

Se separan dos conceptos:

- `device`: dispositivo de evaluación y álgebra activa;
- `output_device`: dispositivo donde se almacenan los cores finalizados.

La nueva API tendrá `output_device="cpu"` por defecto para liberar memoria de
GPU a medida que los cores dejan de participar en el cálculo. El usuario podrá
usar `output_device=None` para conservar el device activo o pasar un device
explícito. Un core solo se offloadeará cuando no sea necesario para pasos
posteriores y después de absorber toda escala pendiente; no se introducirán
copias CPU/GPU ocultas dentro de un kernel.

Las métricas numéricas se calculan en el device activo; sus agregados escalares
y, cuando hay batches, pequeños vectores per-batch detached se guardan en CPU
en los registros. Los métodos posteriores del resultado
(`.error`, `.fidelity`, etc.) calculan en el device de sus cores; mueven inputs
tensoriales pequeños a ese device, pero nunca devuelven implícitamente todos
los cores a GPU. Para una comparación grande o diferenciable, el usuario debe
hacer antes `.to(device)`. La copia a CPU conserva el grafo de autograd de
PyTorch, pero quien necesite backward y rendimiento en GPU deberá usar
`output_device=None` para evitar el enlace de copia.

Los aliases legacy preservarán inicialmente su comportamiento histórico cuando
sea necesario para no romper autograd o device. Esta diferencia se documentará
durante la transición.

`dtype` se inferirá del tensor o de la función si no se proporciona. Matrices
aleatorias, identidades, acumuladores y regularización usarán el dtype/device
del problema. Se preservarán valores complejos y conjugaciones.

### 2.6 Errores y métricas

Se distinguirán explícitamente:

- error absoluto y relativo de reconstrucción;
- error sobre el conjunto fijo observado en completion;
- error sobre `sketch_samples`;
- residuo de un subproblema local;
- energía descartada por una SVD;
- normalized overlap complejo
  `⟨a,b⟩ / (||a|| ||b||)`;
- fidelity, definida como
  `abs(normalized_overlap) ** 2`;
- diagnósticos de recursión/gauge;
- error global, bound global o mero diagnóstico local.

No se llamará “validation error” al error sobre los mismos samples usados para
construir el método.

Los errores de truncación se obtendrán de los valores singulares ya calculados,
sin SVD adicional. Se separan tres cantidades que no deben compartir nombre:

```text
discarded_squared_norm = sum(s_discarded ** 2)
local_absolute_error   = sqrt(discarded_squared_norm)
local_relative_error   = local_absolute_error / local_input_norm
global_relative_contribution = local_absolute_error / original_tensor_norm
```

En TT-SVD y otros barridos cuya ortogonalidad lo justifique, el error absoluto
acumulado será `sqrt(sum(local_absolute_error**2))` y el relativo se obtendrá
dividiendo ese acumulado por la norma original. En TR-SVD, RSS y cualquier
pipeline donde las truncaciones no sean contribuciones ortogonales en una
misma escala, los descartes se registrarán como diagnósticos locales o como
bound solo después de derivarlo; no se presentará su suma cuadrática como error
de reconstrucción. Un oracle explícito podrá medir el error global en casos
pequeños.

Con `renormalize=True` se guardará la escala original de cada paso y de cada
elemento batch por separado. El error absoluto local se expresará en la escala
del tensor de entrada antes de acumularlo; redistribuir la norma entre cores no
debe alterar las métricas. Normas cero se enmascaran con una política explícita
y valores no finitos producen error; `running_log_scale` y energía descartada
no se agregarán entre batches hasta la fase final.

La reducción final por batches será:

```text
absolute_per_batch[b] = sqrt(discarded_energy[b])
relative_per_batch[b] = absolute_per_batch[b] / input_norm[b]
absolute = sqrt(sum_b discarded_energy[b])
relative = absolute / sqrt(sum_b input_norm[b] ** 2)
```

Los records guardan el agregado Frobenius y, si hay más de un batch, los
vectores per-batch. Un batch de norma cero usa la política de denominador cero
sin contaminar los demás. Con `renormalize=True`, ningún core se offloadea
hasta que su parte de la escala final haya sido absorbida.

No se calcularán validaciones costosas por defecto. Las clases resultado
expondrán `.error(...)` y `.fidelity(...)`. Excepciones deliberadas:

- RSS podrá calcular error absoluto/relativo sobre `sketch_samples` cuando se
  solicite información; podrá desactivarse;
- `tt2tr` calculará por defecto overlap/fidelity entre el TT inicial y el TR
  final sin densificar.

Cada `.fit` expondrá `collect_metrics: bool = False` cuando las métricas no
sean necesarias para controlar el propio algoritmo. La función directa usará
`return_info` como control equivalente, sin duplicar un argumento que produciría
métricas inaccesibles cuando solo se devuelven cores. La política efectiva es:

```text
fit silencioso + collect_metrics=False  -> fast path sin instrumentación
collect_metrics=True                    -> records, errores y timings
return_info=True                        -> instrumentación en función directa
verbose>0 u observer                    -> instrumentación automática
```

El fast path no calculará normas exclusivamente diagnósticas, no pedirá
`_TruncatedSVDInfo`, no creará records/eventos y sustituirá timers por contextos
nulos. Las sincronizaciones necesarias para seleccionar shapes/ranks
adaptativos, comprobar invariantes numéricos, ejecutar kernels o mover un
resultado a `output_device` pertenecen al algoritmo/runtime y no se ocultarán
como “coste de métricas”. En particular, con métricas desactivadas no se
llamará explícitamente a `torch.cuda.synchronize`/`torch.mps.synchronize` para
medir tiempo.

### 2.7 Verbosity y eventos estructurados

`verbose` aceptará `False`, `True` o un entero:

| Nivel | Salida |
|---|---|
| `0` | silencio |
| `1` | títulos de fase/sitio, progreso y resumen final |
| `2` | ranks, tiempos y errores de cada subrutina |
| `3` | shapes, diagnósticos numéricos y cores completos al final |

La salida de consola será espaciosa, jerárquica y legible, con títulos,
indentación y resúmenes; no una línea contraída de logs.

Internamente los drivers emitirán `DecompositionEvent`: registros con fase,
sitio, nivel, métricas y tiempo, sin texto preformateado. `ConsoleObserver`
será quien los convierta en texto. Esto permite:

- guardar historial sin parsear `print`;
- testear eventos;
- agregar eventos de workers paralelos en orden;
- sustituir consola por notebooks o profiling sin cambiar el algoritmo.

Los eventos solo se construirán si existe un consumidor real (`verbose>0` u
observer); un `NullObserver` no justifica crear objetos o calcular valores que
después se descartan.

### 2.8 Estado, repetición y aleatoriedad

- Los argumentos fijos viven en la instancia del algoritmo.
- Todo estado mutable de un fit —caches, muestras, métricas, RNG y cores
  temporales— vive en un contexto creado por `.fit`.
- Dos llamadas consecutivas a `.fit` no reutilizan estado numérico salvo que
  el usuario solicite explícitamente warm start.
- Toda aleatoriedad aceptará `generator: torch.Generator` o `seed`; no se
  dependerá del RNG global.
- En paralelo, las seeds se derivarán determinísticamente de
  `(root_seed, task_id, generation)`.

### 2.9 Dependencia de `tk.models`

El núcleo de `decompositions` trabajará con tensores y contracciones PyTorch
específicas. No construirá grafos TensorKrowch para evaluar algunos métodos.
Si se recibe `tk.models.MPS`, un adaptador extraerá y normalizará sus cores una
vez; desde ese punto se seguirá el mismo camino que para una lista de tensores.
Solo se reutilizará una operación de `models` si se demuestra más sencilla y
eficiente, sin crear dos backends divergentes.

### 2.10 Estabilidad y madurez

Cada API se etiquetará en documentación como:

- **estable**: portada desde comportamiento caracterizado y con tests;
- **experimental**: algoritmo nuevo, BLOSTR, recursión TT-core, leverage TR
  exacto, CTM/PEPS, etc.;
- **compatibilidad**: alias temporal.

Los componentes experimentales deben tener tests de invariantes y errores
claros; “experimental” no significa sin verificar.

---

## 3. Visualización global

### 3.1 Flujo conceptual

```text
                         API FUNCIONAL SIMPLE
     tt_svd · tt_als · tt_rss · tr_rss · qtt_rss · tt2tr · ...
                                  │
                                  ▼
                     OBJETO ALGORITMO REUTILIZABLE
       TTSVD · TTALS · TTRSS · TRRSS · TTRS · TRRS · TT2TR
                                  │
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
          contexto de un fit                 estrategias opcionales
     fuente · layout · device · RNG      sampler · fitter · opener · backend
                  │                                │
                  └───────────────┬────────────────┘
                                  ▼
                         DRIVERS COMPARTIDOS
       SVD sweep · ALS sweep · sketching sites · ring bidirectional sweep
                                  │
          ┌──────────────┬────────┼───────────┬──────────────┐
          ▼              ▼        ▼           ▼              ▼
      truncación      entornos  regiones    Phi lazy    loop/gauges
          │              │        │           │              │
          └──────────────┴────────┴───────────┴──────────────┘
                                  ▼
                     TT/TR/TTM/PEPS raw cores
                                  │
                                  ▼
       TTDecomposition · TRDecomposition · TTMDecomposition · ...
            cores · rank · metrics · error/fidelity · .to(device)
```

### 3.2 Fuentes comunes y relación específica de sketching

```text
TensorSource
    │
    ├── CallableTensorSource
    ├── DenseTensorSource
    ├── SparseTensorSource / EmpiricalDistribution
    └── TTTensorSource
    │
    ├──> ALSProblem ──> TTALS / TRALS
    │
    ▼
RegionSketch ──recursive_projector──> SketchRecursion
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
             PhiOperator (lazy)
                   │
          _EvaluationSession
                   │
       collect/closure/freeze/evaluate
                   │
          GlobalValueTransform (una vez)
                   │
          scatter a vistas PhiView
                   │
          LocalValueTransform (opcional)
                   │
             PhysicalFitter
                   │
      RangeProjector + truncated_svd
                   │
        TT solve o apertura local de TR
```

`TensorSource` es el único proveedor de valores para ALS y sketching;
`ALSProblem` solo añade observaciones, sampling, pesos y loss. `RegionSketch`
describe muestras restringidas a una región; no posee la
función, embeddings, Phi, solver o device. `PhiOperator` compone regiones y
ejes abiertos, planifica las evaluaciones y decide si materializa o evalúa
fibras.

No es una jerarquía profunda de herencia, sino composición de cuatro
responsabilidades:

```text
SiteRegion + _SamplePool ──> RegionSketch
RegionSketch child,parent ──> SketchRecursion
TensorSource + RegionSketch + axes ──> PhiOperator
varios PhiOperator ──> _EvaluationSession ──> PhiView
```

Esta separación mantiene las piezas pequeñas pero evita caminos alternativos:
solo `RegionSketch` conoce la correlación de filas, solo `SketchRecursion`
conoce el gather recursivo, solo `PhiOperator` conoce el tensor local y solo la
sesión evalúa/deduplica globalmente.

### 3.3 Relación común TT→TR / TR-RSS

```text
                     BidirectionalRingDriver
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
     LocalTargetProvider                  LoopOpener
   TT core/supercore o Phi       ALS · BLOSTR init+ALS · callable
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
             left gauge · physical core · right gauge
                               │
                 GaugeRecursion (por dirección)
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     TTCoreGaugeRecursion              SketchGaugeRecursion
      contrae con TT core          contrae con SketchRecursion
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                   siguiente problema local
```

El driver es común; el target y la forma de avanzar la base externa son
distintos. No se duplicará el barrido completo ni se fingirá que ambos
backends realizan la misma contracción.

---

## 4. Estructura de carpetas y archivos objetivo

Leyenda:

- `[P]`: API pública sencilla.
- `[A]`: API pública avanzada.
- `[I]`: implementación interna.
- `[C]`: compatibilidad/deprecación.
- `[EXP]`: experimental.

Los módulos se crearán cuando contengan una subrutina sustancial; no se
fragmentarán helpers de una o dos líneas solo para reproducir este árbol.

```text
tensorkrowch/decompositions/
├── __init__.py                         [P] exports explícitos y __all__
├── results.py                          [A] resultados ligeros TT/TR/TTM
├── metrics.py                          [A/I] registros de error, tiempo y fit
├── observers.py                        [A/I] eventos y verbosity estructurada
├── _runtime.py                         [I] device, dtype, RNG y contexto de fit
├── svd_decompositions.py               [C/TEMP] fachada legacy; eliminar tras Fase 1
├── tt_decompositions.py                [C/TEMP] fachada legacy; eliminar tras Fase 3
│
├── svd/
│   ├── __init__.py                     [P/A] funciones y clases SVD
│   ├── common.py                       [I] specs/adapters de truncación y métricas
│   ├── tt.py                           [P/A] TTSVD y tt_svd
│   ├── ttm.py                          [P/A] TTMSVD y ttm_svd
│   └── tr.py                           [P/A] TRSVD y tr_svd
│
├── sources/
│   ├── __init__.py                     [A] fuentes tensoriales comunes
│   ├── base.py                         [A/I] TensorSource y capacidades
│   ├── dense.py                        [A] DenseTensorSource
│   ├── callable.py                     [A] CallableTensorSource
│   ├── sparse.py                       [A] sparse y distribución empírica
│   └── tt.py                           [A] TTTensorSource
│
├── als/
│   ├── __init__.py                     [P/A] tt_als, tr_als, TTALS, TRALS
│   ├── tt.py                           [P/A] TTALS y wrapper tt_als
│   ├── tr.py                           [P/A] TRALS y wrapper tr_als
│   ├── problem.py                      [I] ALSProblem y ObservedEntries
│   ├── driver.py                       [I] ALSSweepDriver
│   ├── environments.py                 [I] caches TT zip-up y TR segmentado
│   ├── sampling.py                     [A/I] rows exactas/observadas/leverage
│   ├── solvers.py                      [A/I] least squares estable
│   ├── gauges.py                       [A/I] QR/SVD/normalización
│   └── convergence.py                  [I] criterios de parada principales
│
├── ring/
│   ├── __init__.py                     [A] estrategias avanzadas seleccionadas
│   ├── opening.py                      [A/I/EXP] contratos, ALS y adapters
│   ├── blocks.py                       [I] selección de bloque y ranks
│   ├── gauges.py                       [A/I/EXP] mapas y recursión de gauges
│   ├── driver.py                       [I] barrido bidireccional común
│   ├── blostr.py                       [A/EXP] BLOSTR verificado y aislado
│   └── tt2tr.py                        [P/A/EXP] TT2TR y función tt2tr
│
├── sketching/
│   ├── __init__.py                     [P/A] funciones y clases RSS/RS/QTT
│   ├── base.py                         [A/I] RecursiveSketching y contexto
│   ├── specs.py                        [I] embeddings, domains y outputs
│   ├── regions.py                      [I/EXP] SiteRegion/RegionSketch/recursión
│   ├── phi.py                          [I/EXP] PhiOperator y vistas lazy
│   ├── evaluations.py                  [A/I/EXP] views, planes y sesiones
│   ├── sketches.py                     [A/I/EXP] operadores y sistemas RS
│   ├── transforms.py                   [A/EXP] Global/LocalValueTransform
│   ├── fitting.py                      [A/EXP] fitting físico fijo/entrenable/QTT
│   ├── projections.py                  [A/I] range finder y randomized SVD
│   ├── quantization.py                 [A/I] layout, mapas y source adapter QTT
│   ├── tt.py                           [P/A] TT-RSS/RS/QTT y QTT-Tucker
│   └── tr.py                           [P/A/EXP] TR-RSS/RS/QTT y QTR-Tucker
│
├── execution/                          [Fase 4]
│   ├── __init__.py                     [A] backends seleccionados
│   ├── backends.py                     [A/I] serial, procesos y distribuido
│   ├── tasks.py                        [I] task graph y dependencias
│   └── shared_store.py                 [I] evaluaciones y tensores compartidos
│
└── peps/                               [rama peps_rss; Fase 5]
    ├── __init__.py                     [P/A]
    ├── specs.py                        [A/I] PEPSRanks y PEPSOutputLayout
    ├── geometry.py                     [I] grid, fronteras y shapes
    ├── svd.py                          [P/A] PEPSSVD / peps_svd
    ├── als.py                          [P/A] PEPSALS / peps_als
    ├── environments.py                 [I/EXP] exact/sampled PEPS envs
    ├── gauges.py                       [A/EXP] PEPSGaugeConditioner/PEPSOrbit
    ├── vo.py                           [P/A/EXP] PEPSVO
    ├── natural_gradient.py             [P/A/EXP] PEPSNaturalGradient
    ├── initializers.py                 [A/EXP] jerárquico/column compression
    ├── ctm.py                          [I/EXP] fronteras y traversal CTM
    ├── rss.py                          [P/A/EXP] PEPSRSS y wrappers
    └── parallel.py                     [I/EXP] schedule checkerboard PEPS

tests/decompositions/
├── test_results.py
├── test_metrics.py
├── svd/
│   ├── test_common.py
│   ├── test_tt.py
│   ├── test_ttm.py
│   └── test_tr.py
├── test_svd_decompositions.py       [TEMP] caracterización legacy
├── sources/
│   ├── test_common.py
│   ├── test_sparse.py
│   └── test_tt.py
├── als/
│   ├── test_tt_als.py
│   ├── test_tr_als.py
│   ├── test_environments.py
│   ├── test_sampling.py
│   └── test_solvers.py
├── ring/
│   ├── test_opening.py
│   ├── test_gauge_recursion.py
│   ├── test_blostr.py
│   └── test_tt2tr.py
├── sketching/
│   ├── test_regions.py
│   ├── test_phi.py
│   ├── test_fitting.py
│   ├── test_tt_rss.py
│   ├── test_tr_rss.py
│   ├── test_tt_rs.py
│   ├── test_tr_rs.py
│   ├── test_quantization.py
│   └── test_qtt_tucker.py
└── peps/                                [rama peps_rss]
    ├── test_svd.py
    ├── test_als.py
    ├── test_gauges.py
    ├── test_vo.py
    ├── test_natural_gradient.py
    ├── test_initializers.py
    ├── test_ctm.py
    └── test_rss.py
```

Archivos relacionados que también cambiarán:

```text
tensorkrowch/config.py                  configuración runtime de SVD
tensorkrowch/utils.py                   _compact_svd/method/return_info
docs/decompositions.rst                 API y referencias
tests/test_config.py                    default/contexto/variable de entorno
tests/test_utils.py                     semántica de truncated_svd
tests/test_operations.py                regresión de split/svd/svdr
```

---

## 5. Catálogo de API, clases y métodos principales

### 5.1 Funciones públicas sencillas

| Nombre | Fase | Estado | Qué hace |
|---|---:|---|---|
| `tt_svd(tensor, ...)` | 1 | estable | Divide un tensor denso en TT mediante SVDs sucesivas. |
| `ttm_svd(tensor, ...)` | 1 | estable | Reordena/agrupa pares input-output y delega la descomposición en TT-SVD. |
| `tr_svd(tensor, ...)` | 1 | estable tras port | Realiza una bipartición interior, abre el rank como dos factores cíclicos y aplica TT-SVD a las subcadenas. |
| `vec_to_mps(vec, ...)` | 1 | `COMPAT` | Wrapper deprecated de `tt_svd` con firma histórica. |
| `mat_to_mpo(mat, ...)` | 1 | `COMPAT` | Wrapper deprecated de `ttm_svd` con firma histórica. |
| `tt_als(source, ...)` | 2 | estable nuevo | Ajusta un TT por ALS exacto, sampled o completion. |
| `tr_als(source, ...)` | 2 | estable tras port | Ajusta un TR por ALS usando entornos cíclicos cacheados. |
| `tr_blostr(tensor, ...)` | 2 | `EXP` | Descomposición/apertura TR mediante BLOSTR verificado. |
| `tt2tr(cores, ...)` | 2 | `EXP` hasta validar recursión | Convierte un TT explícito en un TR y calcula fidelity por defecto. |
| `tt_rss(function, ...)` | 3 | estable tras refactor | TT Recursive Sketching from Samples. |
| `tr_rss(function, ...)` | 3 | `EXP` | TR-RSS con apertura local y recursión bidireccional. |
| `tt_rs(source, ...)` | 3 | `EXP` | TT Recursive Sketching sobre fuente sparse, empírica o TT. |
| `tr_rs(source, ...)` | 3 | `EXP` | Extensión cíclica de RS; no hereda automáticamente garantías TT. |
| `qtt_rss(function, ...)` | 3 | `EXP` | RSS sobre variables cuantizadas con layout y mapa de coordenadas explícitos. |
| `qtr_rss(function, ...)` | 3 | `EXP` | Variante TR de QTT-RSS. |
| `qtt_tucker_rss(function, ...)` | 3 | `EXP` | Construye factores QTT locales conectables a un TT superior. |
| `qtr_tucker_rss(function, ...)` | 3 | `EXP` | Variante con factores QTT locales y tensor superior TR. |
| `peps_svd`, `peps_als`, `peps_vo` | 5 | rama `peps_rss` | Backends PEPS separados por familia. |
| `peps_natural_gradient` | 5 | `EXP`, rama `peps_rss` | Ajuste PEPS por natural gradient/Gauss–Newton. |
| `peps_rss`, `peps_rss_als`, `peps_rss_vo` | 5 | `EXP`, rama `peps_rss` | Variantes PEPS-RSS sobre la infraestructura común. |

Todas estas funciones mostrarán el flujo de alto nivel, validarán la API
pública y delegarán en la clase correspondiente. No serán aliases opacos salvo
los nombres `COMPAT`.

### 5.2 Resultados ligeros (`results.py`)

#### `TensorDecomposition` `[A, dataclass base]`

Almacena `cores`, `rank`, `input_dim`, `output_dim`, `metrics` y metadatos
pequeños. No representa un
grafo TensorKrowch.

- `.to(device=None, dtype=None, copy=False)`: mueve/castea cores y devuelve el
  resultado transformado con semántica explícita.
- `.cpu()`: shorthand de `.to("cpu")`.
- `.norm()`: norma estable sin densificar cuando la topología lo permita.
- `.error(function, samples, **kwargs)`: error absoluto/relativo en samples
  proporcionados por el usuario.
- `.normalized_overlap(other)`: overlap normalizado estable.
- `.fidelity(other)`: `abs(normalized_overlap) ** 2`.
- `.as_info()`: vista `dict` compatible con `return_info`.

#### `TTDecomposition(TensorDecomposition)` `[A]`

- `.evaluate(samples, embedding=None)`: contracción batched directa de cores.
- `.contract_dense()`: oracle explícito solo para tensores pequeños.
- Valida shapes OBC y deriva `rank` e `input_dim` efectivos.

#### `TRDecomposition(TensorDecomposition)` `[A]`

- `.evaluate(samples, embedding=None)`: contracción cíclica batched.
- `.contract_dense()`: oracle pequeño cerrando la traza.
- Admite overlap/fidelity con TT sin convertir una topología en la otra.

#### `TTMDecomposition(TensorDecomposition)` `[A]`

- `.apply(inputs, embedding=None)`: aplica el operador TT a inputs.
- `.contract_dense()`: reconstruye la matriz/tensor para tests pequeños.
- Conserva el layout de cores compatible con `tk.models.MPO`, aunque la API se
  denomine TTM.

#### `QTTTuckerDecomposition(TensorDecomposition)` `[A, EXP]`

Almacena el TT superior y, por variable, un factor QTT cuyo core terminal deja
abierto el índice Tucker `gamma_k`. No lo aplana implícitamente ni contrae ese
índice fuera del TT superior.

- `.evaluate(points)`: aplica los mapas de coordenadas y contrae los dos
  niveles.
- `.flatten()`: conversión explícita a TT plano cuando sea posible.

#### `QTRTuckerDecomposition(TensorDecomposition)` `[A, EXP]`

Comparte los factores QTT locales y los índices Tucker `gamma_k`, pero el
tensor superior tiene topología TR. Evalúa cerrando el anillo superior y no
presupone que sus garantías sean idénticas a las del formato QTT-Tucker TT.

#### `PEPSDecomposition(TensorDecomposition)` `[A, Fase 5]`

Almacena grid de cores y geometría; implementa evaluación/error solo mediante
contracciones PEPS explícitamente seleccionadas.

### 5.3 Métricas, runtime y observers

#### Dataclasses de `metrics.py`

| Clase | Visibilidad | Contenido/objetivo |
|---|---|---|
| `ErrorRecord` | `[A]` | Agregados y valores per-batch opcionales, `kind`, tamaño y denominador. |
| `TruncationRecord` | `[A]` | Corte/sitio, ranks, energía/escala agregada y per-batch opcional. |
| `TimingRecord` | `[A]` | nombre de fase, elapsed, sitio/worker e hijos opcionales. |
| `LocalSolveRecord` | `[A]` | Sitio/sweep, residual, regularización, exactitud del sampling y estado. |
| `SweepRecord` | `[A]` | error objetivo, cambio relativo, tiempo y generación de samples. |
| `GaugeRecord` | `[A]` | shape, rank numérico, condición, error de cancelación y si es proyectivo. |
| `EvaluationStats` | `[A]` | puntos pedidos/únicos, batches, cache hits y llamadas a la fuente. |
| `FidelityRecord` | `[A]` | Overlap normalizado complejo, `abs(overlap)**2` y error compatible. |
| `DecompositionMetrics` | `[A]` | colección tipada de los registros anteriores y warnings. |

Las singular values completas solo se guardarán si un nivel de diagnóstico lo
solicita; por defecto se guardan escalares y ranks.

#### `DecompositionEvent` `[A, dataclass]`

Campos principales: `name`, `phase`, `level`, `site`, `sweep`, `elapsed`,
`values` y `worker`. Es información estructurada, no texto de consola.

#### `DecompositionObserver` `[A, Protocol]`

- `.emit(event)`: recibe un evento.
- `.close(metrics)`: recibe el resumen final.

Implementaciones:

- `NullObserver` `[I]`: coste mínimo;
- `ConsoleObserver` `[A]`: formato jerárquico según verbosity;
- `HistoryObserver` `[A]`: conserva eventos para tests/notebooks;
- `_CompositeObserver` `[I]`: reenvía a varios observers.

#### `_RuntimePolicy` `[I, dataclass]`

Normaliza `device`, `output_device`, `dtype`, batch size, generator, offloading
y sincronización de timers CUDA. No se exige al usuario instanciarla.

#### `_FitContext` `[I, dataclass]`

Contiene runtime, observer, métricas, caches y estado temporal de una única
llamada a `.fit`. Nunca se comparte implícitamente entre fits.

### 5.4 SVD (`decompositions/svd/`)

#### `_TruncationSpec` `[I, frozen dataclass]`

Agrupa `rank`, `cutoff`, `atol`, `rtol` y `cum_percentage`, valida una vez y
pasa exactamente esos valores a `truncated_svd`.

#### `_compact_svd` y selección de backend (`utils.py`) `[I]`

Ofrece dos rutas exactas con la misma salida economy-size:

```text
svd:      A = U S Vᴴ
qr_svd:   A = Q R; R = Uᵣ S Vᴴ; U = Q Uᵣ              si m >= n
qr_svd:   Aᴴ = Q R; Rᴴ = U S Vᵣᴴ; Vᴴ = Vᵣᴴ Qᴴ          si m < n
```

El kernel vive junto a `truncated_svd`, por debajo de `decompositions`, para
que `operations.split` y cualquier otro caller lo reutilicen sin dependencia
circular. La ruta QR siempre usa QR reducida. El backend activo procede de
`tensorkrowch.config`: default global `"svd"`, contexto temporal o variable de
entorno inicial. No existe selección automática dependiente del hardware.

`utils.py` definirá además `_TruncatedSVDInfo` `[I, NamedTuple]` con full rank,
selected rank, energía total/descartada por batch y backend efectivo
`"svd"|"qr_svd"`. Al vivir junto al kernel evita que `utils.py` importe
`decompositions.metrics` y cree una dependencia circular. `TruncationRecord`
convierte ese registro numérico en la métrica de alto nivel.

#### `TTSVD` `[A]`

- `__init__(tensor, n_batches=0, *, output_device="cpu")`: fija tensor,
  batches y política de salida.
- `.fit(rank=None, cutoff=None, atol=None, rtol=None,
  cum_percentage=None, renormalize=False, verbose=0)`: ejecuta el sweep y
  devuelve `TTDecomposition`; usa el backend SVD activo.
- `._split_site(...)` `[I helper]`: un corte SVD sustancial, con error y escala.
- `._finalize_norm(...)` `[I helper]`: redistribuye log-norma sin alterar las
  métricas.

#### `TTMSVD` `[A]`

- `__init__(tensor, input_dim=None, output_dim=None, ...)`: acepta tensor con
  ejes ya intercalados o matriz más dimensions explícitas.
- `.fit(...)`: reordena a
  `(in_0, out_0, in_1, out_1, ...)`, fusiona cada pareja, llama al motor
  TT-SVD y reabre cada eje input/output en el layout TTM.
- `._interleave_axes(...)` `[I helper]`: transformación de layout validada.
- `._unfuse_input_output_axes(...)` `[I helper]`: forma cores TTM.

No habrá un segundo algoritmo de truncación para TTM.

#### `TRSVD` `[A]`

- `__init__(tensor, center=None, ...)`: fija el tensor y un corte preferido
  interior válido `1 <= center < n_sites`.
- `.fit(rank=None, center=None, ...)`: permite repetir el mismo problema con
  discovery, un cap entero compartido o una secuencia de `n_sites` ranks TR.
  La SVD inicial usa como cap el producto de los dos enlaces cortados por la
  bipartición y después aplica TT-SVD a ambas subcadenas.
- `._split_cycle_rank(...)` `[I helper]`: busca factores individuales
  equilibrados bajo cap; si el producto excede el rank SVD, añade únicamente
  dimensiones estructurales cero y las registra.

Con una secuencia, `rank[-1]` es el enlace cíclico y el otro factor se obtiene
del enlace que también corta `center`. Con un entero, ambos reciben el mismo
cap. Con discovery se factoriza el rank seleccionado en la pareja admisible
más equilibrada, como en los algoritmos de referencia. Para un rank primo o
no factorizable, cualquier padding estructural cero debe ser explícito y
registrado; nunca se inventa señal ni se introduce un argumento `tr_rank`.

Debe corregir los bugs de índices del prototipo `tt2tr/src/blostr.py:tr_svd`,
funcionar con `input_dim` heterogéneo y no convertir silenciosamente
el dtype a `cdouble`.

### 5.5 ALS

#### Fuentes comunes (`decompositions/sources/`)

`ConfigurationBatch` `[A/EXP, dataclass]` representa configuraciones packed o
heterogéneas sin mezclar índices discretos y coordenadas físicas.

`TensorSource` `[A, Protocol]` es el contrato compartido por ALS y sketching:

- `.evaluate(configurations: ConfigurationBatch)`;
- metadata de shape, dtype, device y output;
- batching determinista.

Implementaciones: `CallableTensorSource`, `DenseTensorSource`,
`SparseTensorSource`, `EmpiricalDistribution` y `TTTensorSource`. Capacidades
ortogonales como acceso a fibers o soporte sparse se expresan mediante
protocolos opcionales. Las contracciones específicas de un tipo de sketch se
definen en `sketching`, no en el contrato base.

Una función se normaliza exactamente de la misma forma al pasarla a ALS o a
sketching; ninguno mantiene una segunda jerarquía de proveedores de valores.

#### Problema ALS (`als/problem.py`)

`ALSProblem` `[I, dataclass]` compone:

- `source: TensorSource`;
- `observations: Optional[ObservedEntries]`;
- selector/sampler de filas;
- pesos y definición del objetivo;
- acceso al objetivo global/fijo cuando exista.

`ObservedEntries` `[A, frozen dataclass]` guarda índices globales fijos,
valores, shape y pesos opcionales. No es una fuente sparse: fuera de `Ω` el
valor es desconocido y no participa en el objetivo, mientras que fuera del
soporte de `SparseTensorSource` el valor declarado es cero.

No existirán `ALSTarget`, `DenseALSTarget`, `CallableALSTarget` ni
`ObservedALSTarget`; `ALSProblem` añade semántica de optimización sin duplicar
la evaluación de `TensorSource`.

#### Sampling (`als/sampling.py`)

`SampleBatch` `[I, dataclass]` contiene ids, probabilidades de extracción,
pesos, generación y `proposal_core_versions`. Su
`.is_exact_for(current_core_versions)` calcula la exactitud para el diseño del
solve sin mutar probabilidades históricas.

`RowSampler` `[A, Protocol]`:

- `.draw(state, site, n_samples, generator)`;
- `.update_after_core(state, site)`;
- propiedad `.proposal_exact`.

Implementaciones:

- `ExactRows`: todas las filas;
- `ObservedRows`: las observaciones fijas, sin refresh;
- `UniformRows`: sampling uniforme;
- `TTLeverageRows`: leverage exacto/estructurado en mixed-canonical TT;
- `TRProductLeverageRows` `[EXP]`: aproximación producto por unfoldings;
- `TRExactLeverageRows` `[EXP posterior]`: Gram cíclico y sampling condicional.

En TR no se asumirán entornos isométricos como en TT. La ruta exacta define
los scores del diseño local mediante
`diag(A @ pinv(AᴴA) @ Aᴴ)`: construye `AᴴA` contrayendo el double-layer
cíclico sin el core activo y samplea configuraciones condicionalmente mediante
prefixes del mismo entorno. Debe recomputarse al cambiar el diseño y puede ser
demasiado cara; por eso primero se ofrece la aproximación producto, siempre
marcada `exact=False`.

Todo sampling no uniforme multiplica diseño y target por
`1 / sqrt(n_samples * probability)`. Un error medido sobre batches aleatorios o
leverage no se usará como convergencia global.

`SampleRefreshPolicy` `[I, dataclass]` fija `reuse_sweeps`; al refrescar invalida
slices, targets y entornos dependientes de samples.

La reutilización se aplica a sampling uniforme y a leverage **congelado**:
ids y probabilidades de extracción permanecen inmutables durante toda la
generación. El leverage TT exacto redibuja site a site al cambiar el diseño y
no admite reutilizar esos samples. Se expondrá esta diferencia mediante una
estrategia avanzada (`mode="exact"|"frozen"`), no mediante probabilidades que
cambian debajo de unos ids fijos.

#### Entornos (`als/environments.py`)

`EnvironmentCache` `[I, Protocol]`:

- `.prepare_sweep(order, samples=None)`;
- `.local_environment(site)`;
- `.commit(update_set)`;
- `.invalidate(reason)`.

`CoreUpdateSet` `[I, frozen dataclass]` contiene atómicamente todos los cores
alterados por un solve, absorción QR/SVD o normalización, junto con sus nuevas
versiones. La validez de una entrada de cache depende como mínimo de site,
dirección, versiones de todos los cores contraídos, generación de samples,
device y dtype; no basta con versionar el core objetivo.

Implementaciones:

- `TTEnvironmentCache`: suffixes antiguos precalculados y prefix actualizado,
  siguiendo el patrón zip-up de `MPS.sample`;
- `TRSegmentEnvironmentCache`: divide el anillo en segmentos, mantiene
  resúmenes de segmentos externos y prefix/suffix dentro del segmento;
- `DirectTREnvironment`: oracle que reconstruye `_build_tr_env` completo para
  tests;
- `PEPSEnvironmentCache` en Fase 5.

Nunca se “descontraerá” un entorno mediante pseudoinversa. La antigua
`_shift_tr_env` no se migra al backend estable.

#### Least squares (`als/solvers.py`)

`LeastSquaresSolver` `[A]`:

- `__init__(l2_reg=0, rcond=None, column_scaling="auto",
  system_scaling=True)`;
- `.solve(environment, target) -> (solution, LocalSolveRecord)`.

Propiedades:

- Tikhonov mediante sistema aumentado, no ecuaciones normales;
- finite checks antes/después;
- escalado global y de columnas con desescalado exacto;
- regularización absoluta o relativa a la escala declarada;
- drivers `torch.linalg.lstsq` y fallback controlado donde proceda;
- soporte real/complejo.

El orden de scaling preserva exactamente el problema original
`min ||Ax-b||² + λ||x||²`:

```text
A_aug = [A; sqrt(λ) I]       b_aug = [b; 0]
z = D x
solve c * A_aug * D^{-1} z = c * b_aug
x = D^{-1} z
```

`D` se calcula con la política de columnas declarada, pero se aplica a **todas**
las filas aumentadas; `c` escala matriz y right-hand side completos. La
regularización relativa determina `λ` en la escala original antes de estas
transformaciones. No se escalan solo las filas de datos ni se deja
`sqrt(λ) I` sin transformar.

`UpdatePolicy` `[I]` aplica damping y aceptación; damping `1` es default.

#### Gauges (`als/gauges.py`)

`GaugePolicy` `[A, Protocol]`:

- `.factor(core, direction)`;
- `.absorb(factor, neighbor)`;
- `.invalidated_regions(...)`.

Implementaciones `NoGauge`, `QRGauge` y `SVDGauge`. Un gauge nunca se absorbe
en un core fijo. Antes de factorizar se comprueba que exista receptor legal:
si el vecino de la dirección del sweep es fijo, el default aplica `NoGauge` en
ese update; una policy avanzada puede pedir error o usar la dirección opuesta
solo si el driver la valida. Nunca se factoriza y descarta el factor. La
normalización escalar puede moverse al siguiente core entrenable sin cambiar
la red, y todos los cores tocados entran en el mismo `CoreUpdateSet`.

#### Convergencia (`als/convergence.py`)

`ConvergencePolicy` `[I, dataclass]` usa:

- `error_atol`: error absoluto objetivo;
- `error_rtol`: error relativo objetivo;
- `change_rtol`: cambio relativo entre objetivos de sweep;
- `patience`;
- `max_sweeps`.

Registra únicamente las métricas principales: error absoluto/relativo,
variación relativa, sweeps, tiempo y razón de parada. Condiciones y residuos
locales quedan disponibles para `verbose>=2`, pero no forman la API de
convergencia normal.

Con sampling renovable:

- completion mide siempre el conjunto completo fijo `Ω`;
- si existe otro objetivo exacto/fijo accesible, se evalúa por sweep cuando
  gobierne convergencia, paciencia o best state;
- si no existe, el batch cambiante no se compara entre generaciones;
- al refrescar, se reinicia cualquier historial local al batch; un historial
  basado en objetivo global/fijo sí puede continuar;
- sin objetivo comparable se rechazan `error_atol`, `error_rtol`,
  `change_rtol`, `patience` y best-state basado en error. El criterio
  obligatorio es `max_sweeps` y cualquier parada adicional debe venir de un
  callback explícito.

#### `ALSSweepDriver` `[I]`

- `.fit(problem, initial_cores, policies, context)`;
- `._run_sweep(direction)` prepara cache, resuelve sitios y commitea updates;
- `._measure_objective()` produce el `SweepRecord`;
- `._refresh_samples_if_needed()` aplica invalidación atómica.

Orquesta; no conoce fórmulas concretas TT/TR/PEPS.

#### `TTALS` y `TRALS` `[A]`

- `__init__(source, shape=None, *, output_device="cpu")`: normaliza una
  función/tensor/source y fija el objeto aproximado.
- `.completion(observations, shape, ...)` `[classmethod]`: construye el
  problema de matrix/tensor completion sin usar “from”.
- `.fit(initial_cores=None, rank=None, sampling="exact", n_samples=None,
  sample_reuse_sweeps=1, ...)`: traduce opciones simples a `ALSProblem` y
  estrategias. En TR, un entero se comparte y una secuencia de longitud
  `n_sites` sigue la convención `rank[k] = enlace derecho del core k`.

`TTALS` usa mixed-canonical/zip-up; `TRALS` usa segmentos cíclicos. Sus
wrappers sampled/QR históricos, si se conservan, serán presets finos y no
clases adicionales.

Si no se proporcionan `initial_cores`, `rank` es obligatorio salvo default
legacy documentado. Si se proporcionan, sus ranks se infieren y cualquier core
que exceda el cap solicitado produce error; redondearlo requiere una operación
explícita previa. Un cap mayor que la factibilidad algebraica se clipea de forma
determinista en la inicialización y se registra.

### 5.6 Apertura de loops, gauges y TT→TR (`ring/`)

#### `LoopOpening` `[A, dataclass]`

Contiene `left_gauge`, `cores`, `right_gauge`, ranks efectivos,
`LocalSolveRecord` y diagnósticos.

#### `LoopOpener` `[A, Protocol]`

- `.open(target, rank, fixed_left=None, fixed_right=None,
  context=None) -> LoopOpening`.
- propiedad `.capabilities -> LoopOpenerCapabilities`.

`LoopOpenerCapabilities` `[I, frozen dataclass]` declara
`supports_fixed_left`, `supports_fixed_right`, `supports_two_fixed_gauges` y
`supports_blocks`; el driver valida capacidades antes de ejecutar.

Implementaciones:

- `ALSLoopOpener`: encapsula una instancia/configuración de `TRALS`;
- `BLOSTRLoopOpener` `[EXP]`: apertura espectral **no restringida**, válida
  para el bloque central pero no para imponer gauges;
- `CompositeLoopOpener` `[EXP]`: BLOSTR como inicializador y ALS como refino;
- `FixedGaugeCoreOpener`: con ambos gauges fijos resuelve directamente el core
  físico por least squares, sin ejecutar TR-ALS;
- `CallableLoopOpener`: adaptador para un callable parcialmente evaluado.

Las funciones `tt2tr` y `tr_rss` aceptarán un literal sencillo
`loop_opener="als"|"blostr+als"` o un objeto/callable avanzado. BLOSTR puro se
expone solo como estrategia central avanzada o mediante `tr_blostr`; no se
selecciona para sites con gauges fijos. Las funciones no expondrán en su firma
todos los hiperparámetros de TR-ALS.

#### `CentralBlockSelector` `[I]`

- `.select(provider, rank_spec, center=None) -> BlockSelection`;
- crece desde el centro en ambas direcciones;
- comprueba injectividad/refinabilidad y límites;
- maneja bloques de un sitio, multi-sitio y cercanos a frontera.

`BlockSelection` `[I, dataclass]` guarda sites, dimensiones, ranks externos y
razón de selección.

#### `RingRankEstimator` `[I]`

- `.estimate(left_dim, right_dim, auxiliary_rank, rank_caps)`;
- reproduce de forma controlada la estimación
  `sqrt(D_left * D_right / auxiliary_rank)`;
- respeta caps y registra infeasibilidad, sin zero-padding silencioso.

#### `GaugeMap` `[I, dataclass]`

Representa una transformación orientada de base virtual. Métodos:

- `.inverse_or_pinv(policy)`;
- `.diagnostics() -> GaugeRecord`;
- `.require_cancellable(tolerance, allow_projective=False)`.

Los gauges cuadrados bien condicionados usan solve/inversa; los rectangulares
usan least squares/pseudoinversa y declaran si la transformación es proyectiva.

#### `GaugeRecursion` `[A, Protocol]`

- `.advance_left(opening, local_target, recursion_context)`;
- `.advance_right(...)`.

Implementaciones:

- `SketchGaugeRecursion`: contrae gauge, core físico, embedding y
  `SketchRecursion`;
- `TTCoreGaugeRecursion` `[EXP]`: contrae gauge y core TR abierto con el propio
  core TT para cambiar a la base virtual del siguiente sitio.

`TTCoreGaugeRecursion` se implementará tal como fue especificada, pero
permanecerá experimental hasta contrastar cada transición con una contracción
densa/local directa, ambos sentidos, gauges cuadrados/rectangulares y
boundaries.

#### `BidirectionalRingDriver` `[I]`

- `.fit(provider, opener, recursion, block_selector, context)`;
- abre bloque central;
- avanza hacia izquierda y derecha;
- fija cero, uno o dos gauges según el sitio;
- resuelve boundaries;
- valida ranks y compatibilidad de gauges;
- ensambla cores en orden original.

#### `TT2TR` `[A, EXP]`

- `__init__(tt, *, output_device="cpu")`: acepta cores o adaptador de modelo.
- `.fit(rank, tr_rank=None, center=None, loop_opener="als",
  gauge_recursion="pseudoinverse", allow_projective_gauges=False, ...)`.
- Usa por defecto la recursión de pseudoinversa caracterizada y permite
  seleccionar `TTCoreGaugeRecursion` mediante `gauge_recursion="tt_core"`
  mientras la nueva estrategia permanezca experimental.
- Calcula normalized overlap, fidelity y error relativo por defecto.

La primera referencia de comportamiento será `tt2tr_fixed_rank` de
`tt2tr/src/blostr.py`; las versiones `tt2tr` y `tt2tr_local` del prototipo no
se portarán como engines independientes.

`TT2TR` conserva `tr_rank` porque su entrada TT solo tiene `n_sites - 1`
enlaces: el argumento añade el nuevo enlace cíclico antes de construir el
`rank_spec` TR normalizado. Es una excepción deliberada, no una convención de
los métodos que ya reciben un TR.

### 5.7 Infraestructura de sketching (RS/RSS)

RSS se trata como el caso particular de recursive sketching que usa samples;
la carpeta se denomina `sketching` y no `rss`.

#### Specs (`sketching/specs.py`)

Dataclasses internas, no requeridas al usuario:

- `_EmbeddingSpec`: normaliza un callable compartido o una lista por site,
  infiere cada dimensión física y valida sus outputs;
- `_DomainSpec`: normaliza un dominio compartido o uno por variable;
- `_OutputSpec`: output shape, sites, flatten/unflatten de labels y embeddings
  `basis`;
- `_SketchingFitSpec`: rank/truncación, projection, batch y diagnósticos.

`_OutputSpec` acepta:

- función escalar con shape `(batch,)` o `(batch, 1)`, sin output site;
- función tensorial `(batch, o_1, ..., o_m)`;
- `out_position=None`, un entero o una secuencia;
- labels enteros sobre la dimensión aplanada `prod(o_i)`;
- sampling de labels proporcional a `abs(output)**2`;
- descomposición del label plano en un índice por output axis.

Por defecto se reparten los inputs en `m+1` grupos cuyos tamaños difieren como
máximo en uno y se inserta un output entre grupos. Todos los output sites usan
embedding `basis` durante sketching y recursión.

Las fuentes proceden de `decompositions/sources/`. Sketching puede definir
protocolos/adaptadores opcionales como `SketchContractableSource`, pero no
duplica `TensorSource` ni sus implementaciones.

#### Regiones (`sketching/regions.py`)

`SiteRegion` `[I/EXP, frozen dataclass]`:

- colección ordenada de sites, válida en 1D, grids y mallas N-D;
- `.contains`, `.difference`, `.union` y validación de orden/topología.

`RegionSketch` `[I/EXP]`:

- guarda `pool_id`, `region`, `representative_row_ids`, representantes únicos
  e `inverse_ids` respecto a un pool de samples;
- `.restrict(region)`: proyecta las muestras a una subregión;
- `.combine(*others)`: unión **correlacionada** de regiones pertenecientes a
  las mismas filas y al mismo `pool_id`;
- `.compare(other)`: parte común y parte nueva;
- `.recursive_projector(target) -> SketchRecursion`.

En `small.recursive_projector(large)`, `small.region` debe estar contenida en
`large.region`; el resultado selecciona para cada fila de `large` su
representante en `small` y conserva los valores de `large - small`.

`SketchRecursion` `[I/EXP, frozen dataclass]`:

- representa la relación child/parent mediante índices gather y valores de la
  región nueva;
- `.apply(tensor, axis)`: recursión sin matriz densa;
- `.compose(next_recursion)`: compone pasos compatibles;
- `.inverse()` solo cuando la relación sea realmente biyectiva.

El gather se obtiene mediante los IDs de filas del pool, por ejemplo
`small.inverse_ids[large.representative_row_ids]`; nunca se indexa
`inverse_ids` con los valores multidimensionales de un representante. La
generalización no debe crear matrices projector densas.

`_SamplePool` `[I]` conserva muestras originales y una cache de restricciones.
No se expone como requisito público.

#### Phi lazy (`sketching/phi.py`, `sketching/evaluations.py`)

`PhiOperator` `[I/EXP]` une una fuente, region sketches laterales, axes físicos
abiertos y output layout.

- `.evaluate(index_selection)`: evalúa solo configuraciones pedidas;
- `.fiber(axis, fixed_indices)`: acceso funcional a una fibra continua/discreta;
- `.materialize(batch_size=None)`: construye el Phi completo;
- `.select(index_selection)`: devuelve una vista lazy;
- `.configuration_batch(index_selection)`: produce configuraciones finales sin
  evaluar la función, para el caso sampled PEPS;
- `.fit(axis, fitter)`: delega dependencia física a un `PhysicalFitter`.

La unión correlacionada de regions produce `RegionSketch`; un producto
cartesiano o selección batched produce un `ConfigurationBatch` y un
`_EvaluationPlan`, nunca se disfraza de unión correlacionada. Si un algoritmo
necesita después tratar ese batch como nuevas muestras correlacionadas, debe
crear explícitamente un `_SamplePool` nuevo con identidad propia.

`PhiView` `[A/EXP, Protocol]` es la interfaz mínima que consumen fitters y
transforms locales: `.evaluate`, `.fiber` y `.materialize`. La implementan
`PhiOperator`, sus vistas lazy y `_MaterializedPhi`, sin obligar a convertir
una representación en otra.

`EvaluationView` `[A/EXP, frozen dataclass]` es una vista read-only para
callbacks avanzados: configuraciones, valores opcionales, incidencias y fase.
No permite mutar el draft, plan ni mapas internos.

`_EvaluationPlanBuilder` `[I]` es el único estado mutable durante
`collect/expand`; puede producir snapshots `EvaluationView` sin exponer su
storage. `.freeze()` genera `_EvaluationPlan` `[I, frozen dataclass]`, que
contiene puntos globales únicos y mapas gather/scatter definitivos desde cada
Phi. `_EvaluationRegistry` deduplica entre sites y registra `EvaluationStats`.
`_MaterializedPhi` conserva tensor, layout e incidencias.

`_EvaluationSession` `[I]` da un ciclo de vida determinista:

```text
collect requests in _EvaluationPlanBuilder
    -> expand closure required by global transform
    -> freeze as _EvaluationPlan
    -> evaluate source once
    -> apply global transform once
    -> scatter PhiView views
    -> apply local transforms
```

No se evalúa mientras el builder esté en `collect/expand`, ni se añaden puntos
a un plan congelado. En la primera versión no existe extensión incremental:
fitters y transforms deben declarar todas sus queries antes del freeze; una
query tardía falla o crea una sesión independiente que vuelve a planificar el
Phi afectado. Esto elimina resultados dependientes del orden en que los sites
pidan una fibra.

#### Sketch operators (`sketching/sketches.py`)

`SketchOperator` `[A/EXP, Protocol]` describe cómo una fuente se proyecta sobre
regiones. Los tres operadores concretos se exportan al completar `RSS-15`; las
ecuaciones/sistemas que los adaptan al driver permanecen internos. La función
simple acepta literales y la clase avanzada puede aceptar el objeto estrategia.

- `SampledSketch`: selección/evaluación RSS mediante muestras;
- `MarginalSketch`: contrae la región eliminada con unos u otro factor;
- `MarginalSketch.markov(order=1)`: preset que deja vecindarios según
  Markovianity, sin crear una clase `MarkovianSketch`;
- `TTStackSketch(tt_rank, n_stacks, orthogonal=False)` `[EXP]`: proyección
  lineal aleatoria en formato TT; `n_stacks=1` es una **Gaussian TT random
  projection** (caso `P=1` de TTStack), no por ello la recursión `s_k/T_k` de
  TT-RS; `tt_rank=1` es el límite separable/Khatri–Rao.

`SampledSketch`, `MarginalSketch` y `TTStackSketch` son operaciones
matemáticamente distintas; no comparten garantías ni se implementarán como
modos booleanos de una misma fórmula.

`CoreDeterminingSystem` `[I/EXP, dataclass]` guarda los left/right sketches,
sus recursiones y las ecuaciones locales resultantes. Un
`SketchSystemBuilder` `[I/EXP, Protocol]` construye ese sistema para un
`SketchOperator` concreto y valida shapes/ranks. Así, `TTRS` no presupone que
toda proyección lineal global induce automáticamente ecuaciones
core-determining compatibles.

#### Transformaciones (`sketching/transforms.py`)

`GlobalValueTransform` `[A, Protocol]`:

- `.required_points(view, context) -> ConfigurationBatch | None`: declara
  closure adicional antes de congelar el plan;
- `.apply(view: EvaluationView, context) -> values`: actúa una vez
  sobre la tabla de evaluaciones globales únicas antes de ensamblar los Phi;
- puede preservar restricciones globales como norma o consistencia entre
  apariciones.

`LocalValueTransform` `[A, Protocol]`:

- `.required_queries(phi_view, context)`: declara en `collect` cualquier punto
  adicional;
- `.apply(phi_view: PhiView, context) -> PhiView`;
- actúa por región/Phi, por ejemplo sobre marginales locales;
- puede devolver una vista materializada o un nuevo operador lazy con la misma
  interfaz.

Los defaults identidad tienen coste despreciable. No se añadirán nombres
particulares de aplicación al núcleo. Tests sintéticos demostrarán
transformación global coherente, transformación local y composición de ambas.

Un fitter o transform local adaptativo debe declarar de antemano los puntos que
consultará. La extensión incremental se pospone hasta que exista un caso que
justifique un protocolo transaccional completo; nunca se reaplica una
transformación global sobre subconjuntos parciales.

La optimización `SketchContractableSource.contract_sketch(...)` solo puede
saltarse la tabla punto a punto con el transform global identidad. Con
cualquier transform global no trivial, el driver cae explícitamente a
`_EvaluationSession`; si la fuente no puede ofrecer esa evaluación, rechaza la
combinación. Ningún backend estructurado puede omitir silenciosamente el
transform.

#### Fitting físico (`sketching/fitting.py`)

`PhysicalFitter` `[A, Protocol]`:

- `.required_queries(phi_view, axis, domain, context)`: declara consultas para
  la fase `collect`, si no están ya en el plan;
- `.fit(phi_view: PhiView, axis, domain, context) -> FittedPhysicalAxis`.

Implementaciones:

- `FixedEmbeddingFitter`: least squares/de-embedding con embedding dado;
- `BasisFitter`: selección exacta para output sites y bases discretas;
- `TrainableEmbeddingFitter` `[EXP]`: entrena un modelo que mapea `x_k` al
  espacio físico;
- `QTTPhysicalFitter` `[EXP]`: llama a RSS local sobre la fibra cuantizada.

`FittedPhysicalAxis` `[I, dataclass]` guarda tensor/core, dimensión física,
residuo y metadatos. El fitter consume una fibra funcional y por tanto no exige
materializar Phi antes de entrenar o cuantizar.

`SiteRegion`, `RegionSketch`, `SketchRecursion`, `PhiOperator`,
`_EvaluationPlan` y `FittedPhysicalAxis` permanecen internos/experimentales
hasta validar parity de TT-RSS y el adapter PEPS. Después se decidirá si alguno
merece export público avanzado; la API funcional no depende de ello.

#### Range projection (`sketching/projections.py`)

`RangeProjector` `[A, Protocol]`:

- `.project(matrix, target_rank, generator) -> ProjectedRange`.

`IdentityRangeProjector` no cambia la matriz.

`RandomizedRangeProjector` implementa un range finder consistente:

```text
Y = A @ Omega
Q = qr(Y)
B = Qᴴ @ A
small SVD(B)
U = Q @ U_small
```

La proyección se aplica sobre el lado que se va a comprimir. Argumentos
públicos sencillos:

- `random_projection: bool = True`;
- `projection_dim: Optional[int] = None`;
- `projection_oversampling` y `n_power_iter` solo en API avanzada.

Si `projection_dim` no se fija, usa `rank`; si `rank is None`, conserva una
proyección cuadrada/no reductora. La antigua multiplicación aislada por
`randu` queda como ruta `COMPAT` hasta comparar resultados.

#### `RecursiveSketching` `[A, abstracta]`

Fija fuente, embedding, domain, outputs y runtime. Métodos:

- `.fit(...)`: abstracto;
- `._build_regions(context)` `[I]`;
- `._build_phi(site, regions, context)` `[I]`;
- `._fit_physical_axis(phi, site, context)` `[I]`;
- `._trim(phi, site, context)` `[I]`;
- `._solve_local(...)` `[I]`;
- `._assemble_result(...)` `[I]`.

La herencia comparte el workflow; regions, source, fitters y ring opening se
inyectan por composición. No se implementarán grandes cascadas de
`if topology == ...`.

#### `TTRSS`, `TRRSS`, `TTRS`, `TRRS` `[A]`

- `TTRSS`/`TRRSS.__init__(function/source, embedding, domain=None,
  out_position=None, ...)` fija el objeto tensorizado;
- `TTRSS`/`TRRSS.fit(sketch_samples, rank, ..., warm_start=None)` usa samples
  correlacionadas RSS;
- `TTRS`/`TRRS.__init__(source=None, dataset=None, sketch_operator=..., ...)`
  fija una fuente sparse/empírica/TT y el tipo de proyección;
- `TTRS`/`TRRS.fit(rank, ..., warm_start=None)` proyecta la fuente completa
  mediante su `SketchSystemBuilder`, sin exigir `sketch_samples` RSS.

`TTRSS` resuelve ecuaciones TT entre bases sketch vecinas. `TRRSS` usa
`BidirectionalRingDriver` y `SketchGaugeRecursion`. `TTRS`/`TRRS` cambian la
fuente y el operador de sketch, no copian el pipeline.

### 5.8 QTT (`sketching/quantization.py`, `sketching/tt.py`, `sketching/tr.py`)

#### `QuantizedLayout` `[A, frozen dataclass]`

Campos: número de variables, `base`, `level`, `ordering` y orden de digits.

- `.sites()`: schedule final;
- `.decode_digits(digits)`: índice entero por variable;
- `.encode_indices(indices)`: digits;
- `.reorder_configurations(digits, target_ordering)`: permuta columnas de
  configuraciones antes de evaluar, no cores TT.

`base` y `level` aceptan escalar o lista por variable.

Ordenaciones:

- `"grouped"`:
  `(x_1,...,x_l,y_1,...,y_l,z_1,...,z_l)`;
- `"interleaved"`:
  `(x_1,y_1,z_1,...,x_l,y_l,z_l)`;
- schedule personalizado avanzado.

Con levels distintos, `"interleaved"` omite una variable al agotar sus digits.
Se documentará si el primer digit es coarse o fine y se permitirá seleccionar
ese orden.

Cambiar grouped↔interleaved **no** es una simple permutación de una lista de
cores: intercambiar sites no vecinos altera la función representada salvo que
se apliquen swaps tensoriales. Si se necesita convertir una descomposición ya
ajustada, `reorder_decomposition(...)` será una utilidad `EXP` separada basada
en adjacent swaps + SVD/truncación, con posible crecimiento de ranks y error
registrado. El camino normal ajusta cada layout directamente.

#### `CoordinateMap` `[A, Protocol]`

- `.forward(unit_coordinates, domain) -> physical_coordinates`;
- `.inverse(...)` es opcional y solo necesario al indexar muestras físicas.

Implementaciones:

- `UniformCoordinateMap(grid="endpoints")`: grid uniforme + mapa afín al
  dominio;
- `WarpedCoordinateMap(callable, inverse=None)`: contracción/expansión
  separable o acoplada;
- `ExplicitGridMap(points)`: grids arbitrarias;
- `_CompositeCoordinateMap` `[I]`: lista por variable.

`domain` acepta un intervalo compartido o lista de dominios, uno por variable.
Es obligatorio para el mapa uniforme; un mapa custom puede contener toda la
geometría física. La cuantización siempre separa:

```text
digits -> índice -> coordenada computacional u -> CoordinateMap -> x físico
```

Para `N = base**level`, el default uniforme usa endpoints
`u_j = j / (N - 1)`; `grid="cell_centers"` usa
`u_j = (j + 1/2) / N`. El inverse cuantiza al punto más cercano, resuelve
empates hacia el índice menor y rechaza puntos fuera del dominio salvo
`out_of_domain="clip"` explícito.

#### `QuantizedSourceAdapter` `[I/EXP]`

Presenta cualquier fuente en el layout de digits:

- callable física: decode + coordinate map + llamada;
- sparse/empírica en coordenadas físicas: inverse, cuantización y agregación de
  colisiones;
- sparse sobre índices de grid: encode directo;
- TT sobre una dimensión física de tamaño `base**level` por variable: decode a
  índices y evaluación estructurada;
- TT ya cuantizado: bypass solo con metadata de layout compatible.

En `qtt_rss`/`qtr_rss`, `sketch_samples` vive por defecto en espacio
**físico**, con shape `(n_samples, n_variables)`. La API avanzada permite
`sample_space="digits"` para mapas sin inverse; mezclar ambos espacios en una
misma llamada es error.

#### Constructores y funciones

- `TTRSS.quantized(...)` y `TRRSS.quantized(...)`: crean el adapter cuantizado;
- `qtt_rss(...)` y `qtr_rss(...)`: wrappers explícitos;
- no se usará `from_qtt`;
- no se pide `embedding`: los digits usan `basis`.

#### `QTTTuckerRSS` y `QTRTuckerRSS` `[A, EXP]`

Para cada variable `x_k` crea un factor
`F_k(i_{k,1}, ..., i_{k,L_k}, gamma_k)` mediante una fibra y un RSS local
cuantizado. Todos los output axes locales se sitúan en el mismo extremo y se
fusionan temporalmente. Un split rank-revealing entre los digits y los axes
del entorno,

```text
Phi_k[digits, alpha_left * alpha_right]
    ≈ U_k[digits, gamma_k]
      @ R_k[gamma_k, alpha_left * alpha_right],
```

crea el índice comprimido `gamma_k`: `U_k` se representa como factor QTT y
`R_k` se reshapea a
`C_k(alpha_left, gamma_k, alpha_right)`. El TT superior contiene esos cores
`C_k(alpha_{k-1}, gamma_k, alpha_k)` y conecta cada `gamma_k` con su factor.

`QTTTuckerRSS` ensambla esos cores superiores como TT; `QTRTuckerRSS` usa el
mismo fitting local y los ensambla como TR, interpretando el último rank como
enlace cíclico. Los algoritmos viven respectivamente en `sketching/tt.py` y
`sketching/tr.py`, no en un módulo QTT separado.

Este layout boundary es específico de QTT-Tucker y no usa la política
equispaciada de outputs de RSS ordinario. La implementación y sus bounds de
rounding se contrastarán también con la corrección publicada para el algoritmo
QTT-Tucker original, no solo con el artículo inicial. Construir estos factores
mediante RSS es una adaptación propia: la errata de rounding no demuestra
recovery ni error del sketch RSS.

### 5.9 Ejecución paralela (`execution/`)

#### `ExecutionBackend` `[A, Protocol]`

- `.submit(task)`;
- `.gather(handles)`;
- `.map(tasks)`;
- `.close()`.

Implementaciones:

- `SerialBackend`: referencia determinista;
- `ProcessBackend`: procesos locales CPU/GPU explícitamente asignados;
- `DistributedBackend` `[EXP]`: backend posterior, sin fijar aún framework.

#### `_DecompositionTask` `[I, dataclass]`

ID estable, dependencias, payload tensorial serializable, device solicitado,
seed y metadata de eventos.

#### `_TaskGraph` `[I]`

- `.add(task, depends_on=...)`;
- `.ready(completed)`;
- `.validate_acyclic()`;
- `.execute(backend)`.

#### `_SharedEvaluationStore` `[I]`

Tabla shardeable de puntos/valores únicos, incidence maps y cache stats. Evita
reevaluar la función en cada worker y permite devolver valores transformados a
todos los Phi.

### 5.10 PEPS (solo Fase 5 y rama `peps_rss`)

#### `PEPSRanks` `[A, dataclass]`

Normaliza el upper bound `rank` y registra ranks efectivos horizontales y
verticales. Sustituye `PepsBondDims`.

#### `PEPSGeometry` `[I]`

Grid, fronteras, orden de ejes, vecindad, checkerboard y layouts de cores.

#### `PEPSOutputLayout` `[I/EXP, frozen dataclass]`

Asigna cada axis de output a una coordenada explícita del grid y los inputs a
las coordenadas restantes según un traversal declarado. Valida capacidad del
grid, colisiones, orden de axes y flatten/unflatten de labels. Si no se pasan
coordenadas, distribuye los outputs de forma aproximadamente equiespaciada a lo
largo del traversal seleccionado. Todos esos sites usan `basis` en la
recursión y se eliminan de la configuración antes de llamar a la fuente.

#### `PEPSSVD`, `PEPSALS`, `PEPSVO`, `PEPSNaturalGradient` `[A]`

Cada clase fija source/geometría y expone `.fit(...)`. Los métodos de
optimización construyen un `ALSProblem` u objetivo equivalente. Comparten
resultados, métricas, runtime y least squares, pero mantienen drivers separados:

- `PEPSSVD`: sequential frontier e hierarchical;
- `PEPSALS`: exact, sampled y observed;
- `PEPSVO`: optimización conjunta por autograd;
- `PEPSNaturalGradient` `[EXP]`: Gauss–Newton/natural gradient.

#### `PEPSGaugeConditioner` `[A, EXP]`

Agrupa QR/SVD, normalización, Minimal Canonical Form y órbitas. La MCF no entra
en `GaugePolicy` común porque es una optimización PEPS costosa.

#### Inicializadores `[A, EXP]`

- `HierarchicalPEPSInitializer`;
- `ColumnCompressionInitializer`;
- `PEPSRSSInitializer`.

Son estrategias explícitas; no modos ocultos dentro de `peps_als`.

#### `PEPSCTMDriver` y `PEPSRSS` `[I/A, EXP]`

`PEPSCTMDriver` gestiona boundaries, incoming environments, traversal y Phi
explícito/lazy. `PEPSRSS` compone dicho driver con `PhiOperator`,
`PhysicalFitter` y un solver local ALS/VO/natural-gradient seleccionado. La
firma pública no reproducirá los ~100 argumentos de `_peps_rss_ctm`; las
opciones avanzadas viven en objetos de estrategia.

---

## 6. Contratos transversales de implementación

Toda tarea debe respetar los siguientes requisitos cuando apliquen:

### 6.1 Validación y shapes

- `TypeError` para tipos inválidos; `ValueError` para valores, shapes o layouts.
- Mensajes en el estilo `` `argument` should ... ``.
- No usar `assert` en validación pública ni `except:` general.
- Documentar ejes input/output, ranks, batches, outputs y orden de cores.
- Probar primer, último y sitio interior; TT OBC y TR cíclico.
- Aceptar `input_dim`/`output_dim` heterogéneos donde el algoritmo lo permita.
- Un valor compartido de `embedding`/`domain`/`base`/`level` se broadcast;
  una secuencia debe tener la longitud correcta.

### 6.2 Numérica

- Preservar `float32/64`, `complex64/128` y conjugación.
- Evitar ecuaciones normales para least squares salvo justificación medida.
- No usar pseudoinversa para invalidar/descontraer caches.
- Guardar scaling explícito y finite checks en solves delicados.
- Comparar kernels nuevos con un oracle denso/directo en tamaños pequeños.
- Los algoritmos no deben convertir silenciosamente a `cdouble`.

### 6.3 Autograd

- TT-SVD/TTM-SVD conservarán el comportamiento diferenciable actual cuando
  PyTorch lo permita.
- ALS, RSS, TT→TR estructural y sampling se ejecutarán por defecto sin grad.
- Un `TrainableEmbeddingFitter` abrirá un contexto de grad local y explícito,
  sin quitar `@no_grad` de todo el pipeline de forma accidental.

### 6.4 Rendimiento y memoria

- No densificar TT/TR para métricas ordinarias.
- No repetir SVD para calcular error.
- Deduplicar evaluaciones antes de llamar a la función.
- Procesar productos grandes por batch/fibra.
- Mover a `output_device` únicamente objetos finalizados.
- Instrumentar llamadas, puntos únicos, tiempos y, opcionalmente, memoria pico.
- Añadir benchmarks antes/después de reemplazar una ruta madura.
- Mantener una ruta silenciosa sin records, eventos, normas diagnósticas ni
  timers sincronizados cuando `collect_metrics=False`/`return_info=False`.
- Distinguir sincronización explícita de profiling de sincronizaciones
  implícitas necesarias por `.item()`, transferencia CPU, shapes dinámicas o
  kernels de PyTorch; eliminar las primeras del fast path.

### 6.5 Tests y documentación

- Tests públicos antes que tests de helpers.
- Seeds/generator deterministas en toda ruta aleatoria.
- Tests CUDA condicionales, sin hacer de CUDA un requisito.
- Docstrings NumPy/Sphinx detallados en clases/métodos públicos: shapes,
  interacción y semántica exacta de criterios de truncación, renormalización,
  device/output device, retorno y niveles concretos de verbosity.
- Cada clase algorítmica y función directa principal incluirá una sección
  `Examples` autocontenida y ejecutable: uso básico, shapes/ranks resultantes y,
  cuando aplique, repetición de `.fit` o `return_info`/métricas.
- Las clases resultado explican que son contenedores lightweight sin grafo e
  incluyen ejemplos de composición con `tk.models` a partir de `.cores`.
- `docs/decompositions.rst` y exports se actualizan en el mismo commit que una
  API pública.
- Código compatible con Python `>=3.8`, según `pyproject.toml`.

---

## 7. Plan de implementación por fases

### Fase 1 — Infraestructura común y descomposiciones SVD

#### Objetivo de la fase

Introducir el mínimo núcleo común que necesitarán las demás familias y
comenzar por optimizar y caracterizar el kernel exacto SVD/QR+SVD usado en toda
la librería. Después se refactorizan TT-SVD y TTM-SVD y se porta TR-SVD sobre
el mismo motor. Al terminar, la API nueva y los aliases históricos deben ser
equivalentes, medibles y bien testeados.

#### Dependencias

Ninguna otra tarea SVD comienza antes de `SVD-KERNEL-00`; ninguna fase
posterior puede estabilizarse antes de completar al menos `SVD-KERNEL-00` y
`SVD-01` a `SVD-06`.

#### TODOs

- [x] **SVD-KERNEL-00 — Implementar SVD exacta directa/QR+SVD y configuración**

  Estado: implementado, validado y confirmado por el usuario. Commit
  `bfbebb0`.

  Decisión cerrada (2026-08-25):

  - solo existen `"svd"` y `"qr_svd"`; no se implementa modo `"auto"`;
  - el default global es `"svd"`, conservando el comportamiento histórico;
  - `tk.set_svd_method` cambia el default global y `tk.svd_method` proporciona
    un contexto temporal, anidable y seguro ante excepciones;
  - `TENSORKROWCH_SVD_METHOD` fija únicamente el valor inicial;
  - las APIs de alto nivel usan la configuración activa sin repetir el
    argumento; `truncated_svd` conserva un override opcional de bajo nivel.

  Implementación y evidencia:

  - configuración en `tensorkrowch/config.py` y API en `tensorkrowch/__init__.py`;
  - kernel exacto en `tensorkrowch/utils.py`;
  - tests de configuración en `tests/test_config.py`;
  - tests numéricos y de autograd en `tests/test_utils.py`;
  - regresión explícita de `split`, `svd`, `svdr`, `vec_to_mps`, `mat_to_mpo`
    TT-RSS y `canonicalize` en los tests correspondientes.

  Validación local registrada:

  - suite dirigida final de callers parametrizados con `"svd"` y
    `"qr_svd"`: `734 passed`;
  - suite oficial completa bajo `tests/`: `14184 passed, 188 skipped` y un
    fallo estocástico preexistente en `canonicalize_univocal`, ruta ajena a
    SVD; el caso pasó al repetirlo de forma aislada;
  - getter/setter, contextos anidados, restauración ante excepciones y variable
    de entorno comprobados;
  - propagación comprobada en `split`, `svd`, `svdr`, `vec_to_mps`,
    `mat_to_mpo`, TT-RSS, `MPS.entropy`, `MPS.canonicalize`,
    `MPO.canonicalize`, `Tree.canonicalize` y la contracción aproximada de
    PEPS.

  Es la primera tarea del proyecto. Primero se conserva una baseline de
  `truncated_svd`; después se implementa
  `tensorkrowch.utils._compact_svd` con:

  - `svd_method="svd"`: `torch.linalg.svd(..., full_matrices=False)`;
  - `svd_method="qr_svd"`: QR reducida + SVD de la matriz cuadrada pequeña;
  - ruta tall `A = QR`, seguida de SVD de `R` y `U = Q U_R`;
  - ruta wide mediante QR de `Aᴴ`, seguida de la reconstrucción correcta de
    `Vᴴ`;
  - misma semántica batched, real/compleja, device, dtype y autograd;
  - consulta de la configuración activa sin alterar `(u, s, vh)` por defecto.

  Tests de corrección:

  - matrices tall, wide y cuadradas, pequeñas y grandes;
  - batches homogéneos, real/complejo, CPU y CUDA cuando esté disponible;
  - reconstrucción, singular values y subespacios frente a SVD directa;
  - rank deficiente, tensor cero, singular values repetidos y no contiguos;
  - gradcheck/grad parity donde la SVD sea diferenciable;
  - todos los criterios `rank`, `cutoff`, `atol`, `rtol` y
    `cum_percentage`;
  - paridad de callers actuales: `operations.split`, `svd`, `svdr`,
    `vec_to_mps`, `mat_to_mpo` y TT-RSS legacy.

  Configuración y tests:

  - setter global, getter y contexto temporal con restauración garantizada;
  - contextos anidados y aislamiento mediante `ContextVar`;
  - variable de entorno validada al importar;
  - override explícito de `truncated_svd` prevalece sobre el contexto;
  - callers reutilizan el kernel común sin argumentos ni estado duplicado.

  Criterio de finalización: configuración documentada y suite de callers
  completa sin cambios semánticos con el default `"svd"`.

- [x] **SVD-00 — Congelar comportamiento actual y añadir oracles**

  Estado: implementado, validado y confirmado por el usuario. Commit
  `97796e9`.

  Evidencia local:

  - oracles densos TT/TTM implementados únicamente con PyTorch;
  - reconstrucción exacta cubierta para un site, batches, real/complejo y
    `renormalize=True|False` con ambos backends SVD;
  - device cubierto en CPU y preparado condicionalmente para CUDA y MPS;
  - gradcheck cubierto para TT y TTM;
  - errores públicos actuales y layout TTM intercalado documentados;
  - suite de descomposiciones: `124 passed, 4 skipped` (CUDA y MPS no
    disponibles en la máquina local).

  Alcance:

  - conservar los 66 casos actuales de
    `tests/decompositions/test_svd_decompositions.py`;
  - añadir caracterización de un sitio, batches, complejo, device y autograd;
  - verificar explícitamente layouts TT y TTM contra contracción densa;
  - añadir tests de errores/warnings de argumentos actuales;
  - documentar que `mat_to_mpo` recibe ejes intercalados;
  - crear tests que no dependan de `tk.models` para el oracle principal.

  Criterio de finalización: existe una baseline reproducible que falla si se
  altera shape, dtype, rank, reconstrucción o semántica de renormalización.

- [x] **SVD-01 — Crear resultados, métricas y runtime mínimos**

  Estado: implementado, validado y confirmado por el usuario. Commit
  `5e7b713`.

  Evidencia local:

  - resultados públicos ligeros y validación de cores en
    `decompositions/results.py`, sin dependencia de `tk.models`;
  - records estructurados y almacenamiento CPU detached de diagnósticos en
    `decompositions/metrics.py`;
  - `_RuntimePolicy` con device activo, output device, dtype y timer
    sincronizado;
  - contracciones escaladas verificadas frente a tensores densos para TT–TT,
    TR–TR, TT–TR y TTM, incluyendo fase compleja y batches;
  - `evaluate`, `apply`, `error`, `.to`, `.cpu` y `.as_info` cubiertos;
  - suite `tests/decompositions`: `157 passed, 4 skipped`;
  - suite oficial completa: `14632 passed, 192 skipped` y un único fallo
    estocástico preexistente en `canonicalize_univocal`; el caso pasó al
    repetirlo de forma aislada.

  Implementar:

  - `TensorDecomposition`, `TTDecomposition`, `TTMDecomposition` y
    `TRDecomposition`;
  - `ErrorRecord`, `TruncationRecord`, `TimingRecord`,
    `FidelityRecord` y `DecompositionMetrics`;
  - `_RuntimePolicy` con `device`, `output_device`, `dtype` y timer;
  - `.to`, `.cpu`, `.as_info`, validación de cores y derivación de `rank`,
    `input_dim` y `output_dim`;
  - contracciones TT/TR estables para `.norm`, `.normalized_overlap`,
    `.fidelity` y `.error`.

  Propiedades:

  - sin dependencia de `tk.models`;
  - overlap complejo con conjugación correcta;
  - normalized overlap conserva la fase y fidelity usa su módulo al cuadrado;
  - `normalized_overlap` y `fidelity` no se confunden;
  - norma cero produce un error claro para fidelity;
  - TT-vs-TR permitido cuando shapes físicas coinciden;
  - el resultado no conserva la función origen por defecto.

  Tests: identidad, escala, fase compleja, estados ortogonales, TT-vs-TR,
  `.to`, dtype y comparación densa.

- [x] **SVD-02 — Instrumentar `truncated_svd` y su backend seleccionado**

  Estado: implementado, validado y confirmado por el usuario. Commit
  `6434bb5`.

  Evidencia local:

  - `return_info=False` conserva exactamente la tupla histórica de tres
    tensores;
  - `_TruncatedSVDInfo` registra backend efectivo, ranks y energías agregadas
    y per-batch sin conservar el espectro completo;
  - `TruncationRecord.from_svd_info` aplica escalas logarítmicas, reducción
    Frobenius y política explícita para denominadores cero;
  - un spy confirma una única llamada al kernel SVD;
  - kernel/métricas: `82 passed, 4 skipped`;
  - operaciones Split/SVD/SVDR y configuración: `104 passed, 116 deselected`;
  - suite oficial completa: `14642 passed, 192 skipped` y un único fallo
    estocástico preexistente en `MPS.entropy`; el caso pasó al repetirlo de
    forma aislada.

  Añadir `return_info=False` como keyword opcional, manteniendo siempre por
  defecto exactamente la salida `(u, s, vh)` y el backend activo decidido en
  `SVD-KERNEL-00`. Con `return_info=True` devolver
  `(u, s, vh, _TruncatedSVDInfo)` y convertirlo después en
  `TruncationRecord`. La información se obtiene de las singular values
  completas ya calculadas.

  Propiedades:

  - mismos argumentos y rank seleccionado que la implementación actual;
  - backend directo/QR centralizado y registrado;
  - no se ejecuta una segunda SVD;
  - batches mantienen un rank común;
  - energía descartada por batch, reducción Frobenius y relativos per-batch
    correctamente agregados;
  - no guardar todas las singular values salvo petición;
  - ninguna regresión en `tests/test_utils.py`, `split`, `svd` y `svdr`.

  Tests obligatorios:

  ```bash
  conda run -n test_tk pytest tests/test_utils.py
  conda run -n test_tk pytest tests/test_operations.py -k "SVD or SVDR or Split"
  ```

- [x] **SVD-03 — Implementar `TTSVD` y `tt_svd`**

  Estado: implementado, validado y confirmado por el usuario. Commit
  `b028ae3`.

  Evidencia local:

  - `TTSVD` fija tensor, batches y `output_device`; cada `.fit` crea estado,
    cores y métricas independientes;
  - `tt_svd` devuelve cores y admite `return_info`; `vec_to_mps` delega en la
    clase preservando el device histórico;
  - `.fit(collect_metrics=False)` y
    `tt_svd(return_info=False, verbose=0)` usan un fast path sin norma global
    diagnóstica, `_TruncatedSVDInfo`, records, timers sincronizados ni eventos;
    `collect_metrics=True`, `return_info=True`, verbosity positiva u observer
    activan automáticamente la instrumentación completa;
  - `_TruncationSpec`, `_split_site` y `_finalize_norm` centralizan criterios,
    cortes, errores, renormalización y offload;
  - `truncated_svd` conserva la implementación general y sencilla cerrada en
    SVD-02; no incorpora normalización ni aritmética logarítmica específica de
    TT-SVD;
  - con `renormalize=True`, TT-SVD normaliza el residual completo justo antes
    de cada corte y acumula en logaritmos la escala extraída; en tensores
    batched se usa una escala común para conservar una única selección de rank;
  - `cutoff` y `atol` se expresan en la escala local normalizada antes de llamar
    a `truncated_svd`, mientras que `rank`, `rtol` y `cum_percentage` no cambian;
  - las contribuciones descartadas se acumulan como errores relativos
    cuadrados acotados y las cantidades absolutas solo se materializan al
    construir las métricas; la escala global solo se exponencia al final para
    redistribuirla entre los cores;
  - errores acumulados contrastados con reconstrucción densa, incluyendo
    batches con norma cero, escalas `1e-200`/`1e200`, tensores
    reales/complejos y ambos backends;
  - en un tensor moderado, normas, errores locales y errores acumulados en
    log-escala coinciden con un cálculo TT-SVD directo fuera de log-escala;
  - observers estructurados y verbosity 0–3 implementados en
    `decompositions/observers.py`;
  - suite dirigida transversal: `496 passed, 10 skipped`;
  - tres ejecuciones oficiales completas alcanzan
    `14687 passed, 194 skipped` con un único fallo estocástico distinto y no
    relacionado en cada ejecución (`mat_to_mpo`, contracción TR y
    `MPS.condition`); los tres casos pasan al repetirlos aisladamente. La
    baseline anterior a este fast path sí obtuvo
    `14680 passed, 194 skipped` sin fallos.

  Refactorizar `vec_to_mps` sobre `TTSVD.fit`.

  Funcionalidad:

  - tensor de uno o varios sites;
  - `n_batches` conservado en cada core;
  - mismo `rank` escalar y criterios de truncación en cada corte;
  - backend SVD activo delegado al kernel común, sin configuración local;
  - `renormalize=False/True`;
  - offload de cores finalizados a `output_device`; con `renormalize=True` se
    retrasa hasta absorber la escala final correspondiente;
  - errores locales y acumulados absolutos/relativos;
  - total y tiempo por corte;
  - `collect_metrics=False` público y por defecto en `.fit`, con métricas
    vacías y sin trabajo diagnóstico; `verbose>0` u observer prevalecen porque
    consumen esa información;
  - soporte real/complejo y autograd;
  - verbosity limpia mediante observer.

  Invariantes de error con renormalización:

  - guardar la norma original global y por elemento batch antes del sweep;
  - propagar una `running_log_scale` común antes de registrar cada descarte;
  - acumular cuadrados relativos y convertir a escala absoluta solo al final;
  - conservar la semántica original de los criterios absolutos reescalando
    `cutoff` y `atol` para cada residual normalizado;
  - enmascarar batches de norma cero y rechazar escalas no finitas;
  - repartir log-norma al final sin modificar el tensor representado.

  Criterio de finalización: `tt_svd(...).cores` vía clase y la lista devuelta por
  función reconstruyen lo mismo que `vec_to_mps` en toda la baseline.

- [x] **SVD-03D — Documentar resultados lightweight y creación de modelos**

  Estado: implementado y validado como commit independiente posterior a
  SVD-03. Commit `1817733`; ejemplos principales en `0f07342`.

  Subtarea documental:

  - ampliar los docstrings de `TTDecomposition` y `TRDecomposition` para
    explicar que almacenan cores y métricas, pero no construyen un grafo;
  - incluir ejemplos directos con
    `tk.models.MPS(tensors=result.cores)`, dejando que las shapes identifiquen
    respectivamente las fronteras OBC o PBC;
  - indicar la alternativa `MPSData` cuando los cores conservan dimensiones
    batch y documentar las limitaciones correspondientes;
  - añadir la misma explicación a `TTMDecomposition`/`MPO` y
    `PEPSDecomposition`/`PEPS` cuando se implementen esas rutas;
  - mantener las clases resultado independientes de `models`: la relación se
    explica y prueba, pero no se añade un método que cree grafos internamente.

  Criterio de finalización: la documentación permite pasar del resultado
  lightweight al modelo TensorKrowch apropiado con una única construcción y
  sin ambigüedad sobre topology, batches o propiedad de los tensores.

  Evidencia local: `84 passed, 2 skipped` en resultados, métricas y TT-SVD;
  tests explícitos de TT→MPS/OBC, TR→MPS/PBC y resultados batched→MPSData.
  Los docstrings de `TTSVD.fit` y `tt_svd` explican el contrato completo
  `(*batch_shape, d_1, ..., d_n)`, las shapes first/interior/last y el caso de
  un site; sus ejemplos se ejecutan además con `doctest`. Contrato de shapes y
  nomenclatura consolidados en `e764fac`.

- [x] **SVD-04 — Implementar `TTMSVD` y `ttm_svd` como layout + TT-SVD**

  Estado: implementado, validado y cerrado en `b88756d`.

  Funcionalidad:

  - argumento `layout="interleaved"|"grouped"`, con `"interleaved"` como
    default compatible;
  - `layout="interleaved"`: tensor
    `(in_1, out_1, ..., in_n, out_n)`, como en la ruta histórica;
  - `layout="grouped"`: tensor
    `(in_1, ..., in_n, out_1, ..., out_n)`, intercalado internamente antes del
    sweep;
  - ruta matriz `(prod(input_dim), prod(output_dim))` con dims explícitas;
  - dimensiones input/output distintas por site;
  - fusión local `(in_k, out_k)` antes de TT-SVD;
  - reapertura al layout compatible con `tk.models.MPO`;
  - errores y ranks heredados sin segunda SVD;
  - validación de números de sites y productos de dimensiones.

  No se copiará el sweep de `mat_to_mpo`.

  Implementación local:

  - `svd/ttm.py` normaliza una sola vez la entrada tensorizada o matricial,
    fusiona cada pareja `(input_k, output_k)` y conserva un motor `TTSVD` para
    repetir fits con distintos criterios;
  - `_interleave_axes` es la única frontera entre layouts y
    `_unfuse_input_output_axes` reabre los cores sin una segunda
    factorización;
  - la ruta matriz requiere `input_dim` y `output_dim`, valida sus productos y
    admite dimensiones heterogéneas; las entradas tensorizadas pueden inferir
    ambas secuencias o validarlas explícitamente;
  - `TTMSVD.fit` conserva el fast path de `TTSVD` y hereda truncación,
    renormalización logarítmica, errores, timings, backend y política de
    dispositivo; sus eventos se presentan limpiamente como `TTM-SVD` y
    muestran los cores ya reabiertos;
  - `ttm_svd(..., return_info=False)` devuelve la lista de cores y
    `return_info=True` activa las métricas y devuelve también el diccionario
    estructurado;
  - `TTMDecomposition` documenta y prueba la conversión directa a
    `tk.models.MPO(tensors=result.cores)`; los batches TTM siguen fuera de
    alcance;
  - `TTMSVD` y `ttm_svd` ya se exportan desde `decompositions.svd` y
    `decompositions`; el wrapper legacy `mat_to_mpo` se migrará en SVD-06.
  - los docstrings de TT/TTM siguen la misma estructura y todos los métodos de
    `decompositions` que exponen truncación reutilizan literalmente el bloque
    canónico de `truncated_svd`; un test documental evita divergencias futuras.

  Tests: equivalencia entre `"interleaved"` y `"grouped"`, layouts permutados,
  una posición, complejo, truncación combinada y equivalencia exacta con
  TT-SVD sobre ejes fusionados.

  Evidencia local:

  - `41 passed, 2 skipped` en la suite canónica nueva de TTM-SVD;
  - `93 passed, 4 skipped` en TT-SVD, TTM-SVD y el contrato documental común;
  - `1739 passed, 8 skipped` en SVD, resultados, métricas, legacy SVD y MPO;
  - ejemplos de `TTMSVD.fit` y `ttm_svd` ejecutados con `doctest`;
  - `ruff` y `git diff --check` sin incidencias;
  - al combinar inicialmente suites apareció una comparación aleatoria
    float32 preexistente en contracción TR batched; pasó aislada y la
    ejecución dirigida final fue limpia.

- [x] **SVD-05 — Portar y corregir `TRSVD` / `tr_svd`**

  Estado: implementado, validado y commiteado en `1196324`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Fuente conceptual: `tt2tr/src/blostr.py:tr_svd`, reescrita.

  Funcionalidad:

  - `center` es un corte interior, nunca un índice fuera de rango;
  - `input_dim` heterogéneo;
  - SVD de bipartición inicial con criterios modernos y cap de producto;
  - `rank` entero compartido, secuencia TR completa o discovery;
  - `rank[-1]` interpretado como enlace cíclico, sin `tr_rank` separado;
  - elegir la pareja admisible de menor producto que cubra el rank inicial,
    con desempate equilibrado;
  - padding cero estructural si el producto no coincide exactamente, siempre
    registrado;
  - TT-SVD de subcadenas mediante `TTSVD`, no `vec_to_mps`;
  - ensamblado cíclico y ranks coherentes;
  - descartes locales etiquetados como diagnósticos; no sumar como bound global
    hasta derivar la ortogonalidad/escala del pipeline TR;
  - error absoluto/relativo global mediante oracle denso en tests pequeños;
  - ningún cast fijo.

  Casos de regresión:

  - bug histórico `dims[center]` frente a `dims[0]`;
  - center más a la izquierda/derecha permitido;
  - rank inicial no divisible por los caps de los dos enlaces cortados;
  - rank inicial primo y caps asimétricos;
  - complejo y tensor cero.

  Implementación local:

  - `svd/tr.py` fija el tensor y un corte interior preferido en `TRSVD`, pero
    permite repetir `fit` variando `center`, truncación y `rank`; `tr_svd`
    mantiene la interfaz funcional simple y activa diagnósticos solo con
    `return_info=True` o verbosity/observer;
  - el corte inicial se resuelve mediante `TTSVD` sobre la matriz bipartita;
    las subcadenas izquierda y derecha vuelven a usar el mismo sweep TT-SVD,
    con una política privada de caps por corte cuando `rank` es una secuencia;
  - un `rank` entero limita todos los enlaces; una secuencia contiene un upper
    bound por enlace derecho y `rank[-1]` es el enlace cíclico. La pareja que
    abre el rank inicial minimiza primero su capacidad, después su desequilibrio
    y nunca supera los caps;
  - discovery conserva una factorización exacta del rank inicial siempre que
    no hay caps (un rank primo produce factores `1` y `rank`); el padding cero
    estructural solo aparece cuando los caps requieren una capacidad superior;
  - los records de las tres etapas se etiquetan como `left_subchain`,
    `initial_bipartition` y `right_subchain`; se eliminan contribuciones
    globales heredadas y `metrics.errors` queda vacío para no presentar un
    bound TR no justificado;
  - `TruncationRecord.phase` es opcional y reutilizable por futuros drivers;
    timings y eventos conservan jerarquía por fase, mientras el fast path evita
    records, observers y timers sincronizados;
  - no hay cast de dtype, se preservan autograd, real/complejo y la política
    `output_device` común.

  Evidencia local:

  - `116 passed, 2 skipped` en TR-SVD, métricas y contrato documental en la
    primera validación dirigida; tras ajustar la factorización mínima,
    `105 passed, 2 skipped` en TR-SVD y docstrings;
  - `639 passed, 14 skipped` en `tests/decompositions`, `test_utils.py` y
    `test_operations.py`;
  - `11527 passed, 28 skipped` en las suites completas de MPS y MPO;
  - ejemplos de `TRSVD.fit` y `tr_svd` ejecutados con `doctest`; `ruff` y
    `git diff --check` sin incidencias.

- [x] **SVD-06 — API, aliases y deprecaciones**

  Estado: implementado, validado y commiteado en `e49410d`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Implementar:

  - exports de clases/resultados y `tt_svd`, `ttm_svd`, `tr_svd`;
  - wrappers reales `vec_to_mps` y `mat_to_mpo`;
  - warning, stacklevel y keywords históricos;
  - `return_info` coherente;
  - docs y ejemplos TT/TTM;
  - migración de callers internos de TensorKrowch a nombres canónicos.

  No se eliminarán aliases en esta fase.

  Implementación:

  - `vec_to_mps` vive junto a `tt_svd` en `svd/tt.py` y `mat_to_mpo` junto a
    `ttm_svd` en `svd/ttm.py`; ambos preservan los keywords históricos,
    delegan una sola vez en la API canónica y mantienen los cores en el device
    de entrada como hacía la implementación anterior;
  - los wrappers aceptan `verbose` y `return_info`, emiten exactamente un
    `FutureWarning` con `stacklevel=2` y documentan la migración;
  - `svd_decompositions.py` queda temporalmente como fachada de imports, sin
    ningún sweep numérico; se elimina en SVD-07;
  - `decompositions.__init__` exporta directamente las clases, funciones
    canónicas, resultados y wrappers desde sus módulos definitivos;
  - los notebooks fuente usan ya `tt_svd`/`ttm_svd` y
    `docs/decompositions.rst` documenta TT-SVD, TTM-SVD, TR-SVD, resultados
    lightweight y aliases de compatibilidad;
  - `pyproject.toml` incluye explícitamente el nuevo subpaquete
    `tensorkrowch.decompositions.svd`; el wheel construido contiene todos sus
    módulos.

  Evidencia local:

  - `265 passed, 8 skipped` en API, TT/TTM-SVD, aliases legacy y contrato de
    docstrings;
  - `366 passed, 10 skipped` en toda la suite `tests/decompositions`;
  - wheel `tensorkrowch-1.1.6-py3-none-any.whl` construido sin dependencias y
    verificado para `svd/{__init__,common,tt,ttm,tr}.py`;
  - build Sphinx mínimo con autodoc completado sin warnings; ejemplos
    canónicos de TT/TTM ejecutados con `doctest`;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **SVD-07 — Gate final de fase**

  Estado: implementado, validado y commiteado en `a37bf93`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Ejecutar:

  ```bash
  conda run -n test_tk pytest tests/decompositions
  conda run -n test_tk pytest tests/test_utils.py tests/test_operations.py
  conda run -n test_tk pytest tests/models/test_mps.py tests/models/test_mpo.py
  ```

  Revisar documentación, tiempos y memoria frente a baseline. Marcar la fase
  solo tras aprobación del usuario y commit(s) limpios. Una vez migrados todos
  los callers y comprobada la API pública, eliminar
  `svd_decompositions.py`; `vec_to_mps` y `mat_to_mpo` siguen exportándose
  desde `tensorkrowch.decompositions` como wrappers de compatibilidad.

  Resultado:

  - `svd_decompositions.py` eliminado; los wrappers compatibles permanecen en
    sus módulos canónicos y en `tensorkrowch.decompositions`;
  - `366 passed, 10 skipped` en `tests/decompositions`;
  - `283 passed, 4 skipped` en `tests/test_utils.py` y
    `tests/test_operations.py`;
  - el gate completo MPS/MPO recorrió `11527` casos en cada una de dos
    ejecuciones: en ambas apareció un único fallo estocástico preexistente en
    `MPS.entropy` con tensores complejos (`svd` primero y `qr_svd` después), y
    ambos casos pasaron al repetirlos de forma aislada; una comprobación
    adicional sin entropy encontró un único fallo estocástico de tolerancia en
    `MPS.condition`, que también pasó de forma aislada;
  - la ejecución completa MPS/MPO inmediatamente anterior, durante SVD-05,
    terminó con `11527 passed, 28 skipped`;
  - wheel reconstruido desde un directorio `build/` limpio: contiene
    `svd/{__init__,common,tt,ttm,tr}.py` y no contiene
    `svd_decompositions.py`;
  - la retirada de la fachada no añade cálculos, copias ni sincronizaciones al
    camino numérico; los wrappers son delegaciones directas y el motor canónico
    conserva sus fast paths;
  - test dirigido de API, `ruff` y `git diff --check` sin incidencias.

#### Entregable de la fase

Carpeta `svd/` canónica, resultados/metrics reutilizables, aliases compatibles
y un motor exacto directo/QR+SVD listo para ALS, sketching, TT→TR y PEPS.

---

### Fase 2 — ALS, apertura de loops y conversión TT→TR

#### Objetivo de la fase

Crear un driver ALS reutilizable sin forzar las contracciones de TT, TR y PEPS
en una única implementación. Primero se construyen sources, `ALSProblem`, solver, sampling y
caches; después se implementa TT-ALS, seguido de TR-ALS. Sobre esas piezas se
separan los mecanismos comunes de apertura de loops y se implementa TT→TR.

#### Decisiones específicas

- ALS y sketching reciben la misma familia `TensorSource`; `ALSProblem` añade
  el dominio/pesos del objetivo sin convertirse en otro proveedor de valores.
- ALS exacto, sampled aleatorio y completion observado son modos distintos.
- En completion, `ObservedEntries` permanece fijo durante todo el fit; su error
  absoluto/relativo es el objetivo.
- En sampling aleatorio/leverage, el residuo del batch local no es criterio
  global de convergencia.
- `sample_reuse_sweeps` conserva ids y probabilidades de extracción durante
  varios sweeps para uniform/frozen leverage; leverage TT exacto redibuja al
  cambiar el diseño y exige `sample_reuse_sweeps=1`.
- TR no quitará cores de un entorno mediante pseudoinversa.
- Column scaling, system scaling, Tikhonov, damping y gauges son estrategias
  del solver/update, no drivers ALS duplicados.

#### TODOs

- [x] **ALS-00 — Construir suite de caracterización y oracles**

  Estado: implementado, validado y commiteado en `db1bee4`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Antes del port:

  - convertir `_build_tr_env` del proyecto `tt2tr` en oracle de tests;
  - crear tensores TT/TR sintéticos pequeños con ranks heterogéneos;
  - reconstrucción densa para TT/TR ALS;
  - fixed cores conservados bit a bit;
  - QR con vecino fijo y sin vecino fijo;
  - target cero, complejo y sistemas rank-deficient;
  - sampled determinista con generator;
  - completion con observaciones fijas;
  - registrar resultados actuales de `tr_als` sin adoptar sus fallos.

  Los notebooks externos no cuentan como tests.

  Implementación:

  - oracles densos independientes para contracción TT/TR, mapas locales y
    solves exactos o restringidos a filas;
  - `_build_tr_env` convertido en `build_tr_environment` de tests y contrastado
    contra productos directos de slices para todos los sites y ranks
    heterogéneos;
  - sweep TR deliberadamente denso que caracteriza la estrategia estable del
    prototipo, reconstruyendo cada entorno en vez de adoptar el shift mediante
    pseudoinversa;
  - caracterización de fixed cores bit a bit, absorción QR permitida/omitida,
    target cero, dtype complejo, sistemas rank-deficient, sampling con
    generator y observaciones permanentes de completion.

  Evidencia local:

  - `14 passed` en la suite ALS-00;
  - la suite completa de decompositions recorrió `380` tests y solo reprodujo
    el conocido fallo aleatorio float32 de contracción TR batched; pasó al
    repetirlo aisladamente;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-01 — Implementar fuentes comunes y `ALSProblem`**

  Estado: implementado, validado y commiteado en `c0ba038`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `decompositions/sources/` con `TensorSource`, `ConfigurationBatch`,
  `CallableTensorSource`, `DenseTensorSource`, `SparseTensorSource`,
  `EmpiricalDistribution` y `TTTensorSource`. Crear además `ALSProblem` y
  `ObservedEntries` en `als/problem.py`.

  Propiedades:

  - una función se pasa y normaliza igual en `TTALS`/`TRALS` y en sketching;
  - evaluación/fibers sin materializar el tensor cuando la fuente lo permita;
  - `ALSProblem` compone source, selector/sampler, observations, pesos y loss;
  - observations con ids globales, values y pesos;
  - deduplicación/validación de observaciones repetidas;
  - errores:

    ```text
    absolute = ||W * P_Ω(approx - target)||
    relative = absolute / ||W * P_Ω(target)||
    ```

  - denominador cero tratado explícitamente;
  - ninguna semántica de refresh en `ObservedEntries`;
  - completion distingue desconocido fuera de `Ω` de cero fuera de un
    soporte sparse;
  - no crear jerarquía paralela `ALSTarget`;
  - shapes y dtype consistentes.

  Implementación:

  - `ConfigurationBatch` separa explícitamente índices discretos de
    coordenadas y admite storage packed o heterogéneo por site;
  - `TensorSource`/`FiberTensorSource` definen el contrato común, y
    `as_tensor_source` normaliza tensor, callable, resultado TT o source ya
    construido por la misma ruta para ALS y sketching;
  - `DenseTensorSource`, `CallableTensorSource`, `SparseTensorSource`,
    `EmpiricalDistribution` y `TTTensorSource` implementan evaluación
    determinista; dense, callable, sparse y TT ofrecen fibers sin construir un
    tensor denso nuevo;
  - callable admite batching contiguo, inputs heterogéneos e inferencia
    controlada de dtype/output; TT evalúa y forma fibers mediante contracciones
    PyTorch específicas, sin construir un modelo;
  - sparse coalesce entradas repetidas y declara cero fuera del soporte;
    `ObservedEntries` deduplica solo observaciones idénticas y mantiene
    desconocido todo lo exterior a su conjunto fijo;
  - `ALSProblem` admite source conocido o completion solo observada, compone
    selector, pesos y loss L2, y aplica la política explícita de denominador
    cero a errores absolutos/relativos;
  - las piezas avanzadas se exportan desde `tensorkrowch.decompositions` y los
    nuevos subpaquetes se incluyen explícitamente en el wheel.

  Evidencia local:

  - `42 passed` en toda la base ALS-00/ALS-01;
  - `408 passed, 10 skipped` en toda la suite de decompositions;
  - wheel verificado con los módulos completos de `sources/` y `als/problem`;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-02 — Portar el least-squares solver estable**

  Estado: implementado, validado y commiteado en `6d24470`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Semillas:

  - `peps-rss/src/peps_als.py:_augment_lstsq_system`;
  - `_lstsq_column_scales`;
  - `_solve_regularized_lstsq`.

  Implementar `LeastSquaresSolver` con:

  - Tikhonov aumentado;
  - `column_scaling=False|True|"auto"`;
  - `system_scaling`;
  - regularización absoluta/relativa documentada;
  - finite checks y errores informativos;
  - fallback sin ocultar el driver usado;
  - `LocalSolveRecord`;
  - soporte de múltiples right-hand sides y complejo.

  Orden obligatorio: formar `[A; sqrt(λ)I]`, aplicar el column scaling como
  cambio de variable a todas sus filas y después escalar globalmente matriz y
  right-hand side completos. `λ` relativo se fija antes del scaling.

  Tests contra solución densa, magnitudes desequilibradas, regularización,
  rank-deficiency, NaN/Inf, reescalado exacto y equivalencia de la solución
  regularizada con/sin ambos scalings.

  Implementación:

  - `LeastSquaresSolver` resuelve targets vectoriales o múltiples right-hand
    sides reales/complejos y valida dtype, device, shapes y finitud antes del
    kernel;
  - Tikhonov se forma mediante `[A; sqrt(lambda) I]`; el modo relativo usa la
    norma RMS de columnas del entorno original y fija `lambda` antes de todo
    scaling;
  - column scaling explícito o `"auto"` actúa como cambio de variable sobre
    todas las filas aumentadas, y system scaling divide después matriz y
    target completos por una misma escala;
  - las normas de columna extraen primero máximos para reducir overflow y
    conservan un fallback finito;
  - fallback CPU recorre drivers de `torch.linalg.lstsq` de forma visible y
    `LocalSolveRecord` conserva el driver efectivo, residuo, regularización y
    scalings;
  - `return_record=False` mantiene la solución pero omite residuos, records y
    conversiones escalares, para el futuro fast path de los drivers ALS.

  Evidencia local:

  - `71 passed` en ALS más contratos de métricas;
  - `426 passed, 10 skipped` en toda la suite de decompositions;
  - equivalencia con solves densos, Tikhonov cerrado, scalings combinados,
    magnitudes `1e150`, sistemas rank-deficient, fallback y complejo;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-03 — Implementar sampling y refresh**

  Estado: implementado, validado y commiteado en `5c1313c`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `SampleBatch`, `RowSampler`, `ExactRows`, `ObservedRows`,
  `UniformRows` y `SampleRefreshPolicy`.

  Propiedades:

  - probabilidades y pesos siempre explícitos;
  - reponderación `1/sqrt(J p_i)`;
  - samples y draw-probabilities conservados exactamente
    `sample_reuse_sweeps` en modos congelados;
  - `proposal_core_versions` guardados y exactitud dinámica derivada/registrada
    en cada solve;
  - generación incremental y eventos de refresh;
  - invalidación centralizada;
  - no resamplear completion;
  - generator determinista CPU/CUDA.

  Tests estadísticos pequeños y equivalencia exacta cuando se seleccionan todas
  las filas.

  Implementación:

  - `SampleBatch` conserva ids, draw probabilities, pesos
    `1/sqrt(J p_i)`, generación, site, versiones del proposal y exactitud sin
    permitir que se reescriban al reutilizar el batch;
  - `.is_exact_for(...)` distingue proposals independientes del diseño,
    dependientes de versiones y aproximados;
  - `ExactRows` y `ObservedRows` enumeran determinísticamente con peso efectivo
    uno; completion usa `ObservedRows.refreshable=False`;
  - `UniformRows` muestrea con reemplazo y generator del mismo device, con
    probabilidades uniformes explícitas y soporte CPU/CUDA;
  - `_RowSamplingState` versiona commits de cores sin estado global y
    `SampleRefreshPolicy` asigna generaciones por sweep, devuelve un indicador
    de refresh e invoca una única callback central de invalidación;
  - los batches uniformes congelados preservan el mismo objeto, ids y
    probabilidades durante `reuse_sweeps`; las observaciones permanecen fijas
    durante todo el fit.

  Evidencia local:

  - `70 passed, 1 skipped` en toda la suite ALS; el skip es CUDA no disponible;
  - tests estadísticos de Gram/right-hand side, equivalencia exacta de todas
    las filas, generators, versiones y refresh;
  - `436 passed, 11 skipped` en toda la suite de decompositions;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-04 — Implementar cache TT zip-up**

  Estado: implementado, validado y commiteado en `be1acc4`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Fuente de patrón:
  `MPS._build_sample_right_envs` y el sweep de `MPS.sample`.

  `TTEnvironmentCache` debe:

  - precalcular suffixes de los cores aún no actualizados;
  - crecer un prefix con cores ya actualizados;
  - invertir el sentido al cambiar el sweep;
  - trabajar con slices sampled sin construir el entorno completo;
  - invalidarse al cambiar sample generation;
  - admitir canonical shortcuts solo cuando se demuestre la isometría;
  - renormalizar intermediates sin cambiar el solve.

  Cada entorno se compara con contracción directa para todos los sites y ambos
  sentidos. Añadir commits atómicos `CoreUpdateSet` y oracles tras: update
  local, QR/SVD absorbido en vecino, normalización que salta un core fijo,
  gauge que cruza un segmento y cambio de dirección. Las keys incluyen
  versiones de todos los cores dependientes, sample generation, device y
  dtype.

  Implementación:

  - `TTEnvironmentCache` precomputa suffixes y crece prefixes commiteados en
    forward, con la construcción espejo en reverse;
  - `TTLocalEnvironment` materializa el diseño exacto solo bajo demanda o
    forma directamente las filas correlacionadas de un `SampleBatch`, sin
    construir primero el entorno completo;
  - `CoreUpdateSet` valida y aplica atómicamente todos los cores/versiones
    tocados por solve, QR/SVD o normalización; una incompatibilidad de ranks
    falla antes de mutar el cache;
  - updates current+neighbor conservan el coste zip-up; updates más lejanos
    reconstruyen únicamente los prefixes/suffixes cuyas dependencias han
    cambiado;
  - las keys incluyen site, dirección, versiones de todos los cores del
    entorno, sample generation, device y dtype;
  - invalidación explícita libera referencias a entornos y obliga a preparar
    de nuevo; no se implementan shortcuts canónicos sin una isometría
    demostrada;
  - renormalización opcional usa una escala global por entorno, acumulada en
    log; `scale_target` y `scale_l2_reg` preservan exactamente el problema
    local, incluido Tikhonov absoluto.

  Evidencia local:

  - `16 passed` en caches: todos los sites, ambas direcciones, exact/sampled,
    real/complejo y ranks heterogéneos;
  - oracles tras update local, QR current+neighbor, normalización que salta
    sites, cambio de dirección, refresh y fallo atómico;
  - test de coste estructural: cada core se contrae una vez por lado en un
    sweep estándar;
  - `86 passed, 1 skipped` en ALS y `452 passed, 11 skipped` en toda la suite
    de decompositions;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-05 — Implementar `ALSSweepDriver` y convergencia**

  Estado: implementado, validado y commiteado en `3f4c914`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `ConvergencePolicy`, `UpdatePolicy`, observers de sweep y driver.

  Métricas principales:

  - error objetivo absoluto;
  - error objetivo relativo;
  - cambio relativo entre finales de sweep;
  - `n_sweeps`, tiempo y razón de parada.

  Razones normalizadas:

  - `"error_atol"`;
  - `"error_rtol"`;
  - `"relative_stability"`;
  - `"max_sweeps"`;
  - `"all_cores_fixed"`;
  - `"nonfinite_local_system"`;
  - `"nonfinite_solution"`.

  No se usa el máximo error observado durante updates como error final del
  sweep. Se conserva best state solo si la política lo solicita y existe un
  objetivo fijo/global comparable. Completion mide siempre `Ω`; si no existe
  tal objetivo se rechazan tolerancias/patience/change y se usa
  `max_sweeps` o callback. Un refresh reinicia solo el historial ligado al
  batch, no el de un objetivo fijo.

  Implementación:

  - `ALSSweepDriver` alterna sweeps forward/reverse y delega en un backend
    topology-specific la preparación de entornos, el solve local y el commit
    atómico;
  - `ConvergencePolicy` evalúa `error_atol`, `error_rtol` y estabilidad solo
    en objetivos globales/fijos comparables; los batches renovables se limitan
    a `max_sweeps` o callback;
  - `UpdatePolicy` centraliza damping y la aceptación local no creciente sin
    mezclar esas decisiones con el driver;
  - `SweepRecord` y `LocalSolveRecord` separan claramente el objetivo al final
    del sweep de los residuos de cada subproblema;
  - los eventos estructurados incluyen inicio/final de sweep, refresh de
    samples, site completado y resumen, con formato jerárquico en consola;
  - estados no finitos del sistema o de la solución producen razones de parada
    normalizadas y no commits parciales;
  - `keep_best` solo opera con objetivo fijo y restaura un snapshot completo;
  - el fast path sin métricas, observer ni criterio dependiente del error omite
    objetivos globales, records, timers y conversiones a CPU/Python.

  Evidencia local:

  - `25 passed` en driver y contratos de métricas;
  - `100 passed, 1 skipped` en toda la suite ALS;
  - `466 passed, 11 skipped` en toda la suite de decompositions;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-06 — Implementar TT-ALS exacto**

  Estado: implementado, validado y commiteado en `dc755cb`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `TTALS` y `tt_als` con:

  - inicialización random, cores dados o resultado SVD;
  - ranks limitados por el `rank` escalar;
  - clipping TT algebraicamente factible
    `min(rank, prod(dims_left), prod(dims_right))`;
  - `rank` obligatorio sin init; initial cores sobre el cap producen error;
  - sweeps left-to-right/right-to-left;
  - least squares común;
  - QR/SVD gauge seleccionable;
  - fallback `NoGauge` antes de factorizar si el receptor es fijo;
  - fixed cores;
  - normalización estable;
  - error exacto por sweep cuando se soliciten métricas, un observer o un
    criterio de convergencia basado en el objetivo; el fast path silencioso
    conserva la decisión transversal de no calcular diagnósticos;
  - salida `TTDecomposition` y wrapper de cores.

  Verificar monotonicidad esperada del objetivo exacto sin damping ni
  truncación; documentar excepciones numéricas.

  Implementación:

  - `TTALS` fija y normaliza tensor, callable, `TensorSource` o resultado TT;
    enumera la malla discreta una sola vez y cachea el target escalar para
    fits repetidos;
  - `tt_als` conserva la interfaz funcional sencilla de cores y construye
    internamente solver, convergencia y update policy a partir de argumentos
    opcionales;
  - inicialización `"random"` o `"svd"`, cores dados en shapes standard o
    lightweight y clipping determinista
    `min(rank, prod(dims_left), prod(dims_right))`;
  - `rank` es obligatorio sin cores iniciales; un init que exceda el cap
    solicitado o la factibilidad algebraica falla sin truncación implícita;
  - `_TTALSBackend` compone `TTEnvironmentCache`, `LeastSquaresSolver` y
    `ALSSweepDriver`; el target completo y su orden global se reutilizan en
    todos los subproblemas;
  - `GaugePolicy`, `NoGauge`, `QRGauge` y `SVDGauge` viven en `als/gauges.py`;
    QR/SVD preservan exactamente el tensor vecino en ambas direcciones y
    soportan complejo;
  - si el receptor inmediato no existe o es fijo, se elige `NoGauge` antes de
    factorizar; los cores fijos no reciben gauges y permanecen bit a bit;
  - los updates current+neighbor se commitean mediante un solo
    `CoreUpdateSet`; damping y aceptación se aplican antes del gauge;
  - la normalización logarítmica del cache preserva target y Tikhonov absoluto
    mediante un factor tensorial de regularización, sin `.item()` en el camino
    numérico normal;
  - source densa/callable, un site, target complejo, ranks heterogéneos,
    target cero, fixed cores, gauges y `output_device` siguen el runtime
    declarado;
  - `TTDecomposition` contiene cores, métricas y metadata de convergencia;
    `return_info=False`, verbosity cero y convergencia por `max_sweeps` omiten
    records, timers y evaluaciones globales al final del sweep.

  Evidencia local:

  - `27 passed` en TT-ALS y gauges antes de ampliar regresiones;
  - `129 passed, 1 skipped` en toda la suite ALS;
  - `495 passed, 11 skipped` en toda la suite de decompositions;
  - examples de `TTALS.fit` y `tt_als` ejecutados con `doctest`;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-07 — Añadir TT completion y sampled ALS**

  Estado: implementado, validado y commiteado en `36f51ad`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Implementar:

  - `TTALS.completion(...)`;
  - sampling uniforme sobre source densa/callable;
  - observations permanentes;
  - fibers/valores de la source cacheados;
  - `sample_reuse_sweeps`;
  - error observado como objetivo en completion;
  - ausencia de “sample validation error” en batches renovables.

  Tests de matrix completion y tensor completion con un conjunto `Ω` pequeño,
  incluyendo pesos y observaciones no ordenadas.

  Implementación:

  - `TTALS.completion(...)` acepta `ObservedEntries` o índices, valores,
    `input_dim` y pesos; fija `Ω` permanentemente y mantiene desconocido todo
    lo exterior;
  - `TTALS.fit` resuelve `sampling=None` como `"exact"` para source conocida y
    `"observed"` para completion; la interfaz funcional admite además
    `sampling="uniform"`;
  - completion usa `ObservedRows`, nunca refresca ids y mide por sweep el error
    absoluto/relativo ponderado exactamente sobre `Ω`;
  - uniform sampling usa `UniformRows`, `SampleRefreshPolicy` y un solo
    `SampleBatch` global por generación para todos los sites del sweep;
  - ids, probabilidades, pesos `1/sqrt(J p_i)` y valores de la source se
    conservan exactamente durante `sample_reuse_sweeps`; solo una nueva
    generación invalida el cache y reevalúa la source;
  - el cache TT construye directamente filas correlacionadas para los ids
    globales sampleados; no materializa primero el entorno exacto;
  - sampling uniforme no enumera la source completa y no registra ni compara
    un error global entre batches renovables; se rechazan tolerancias que
    requieran tal objetivo;
  - initial random, gauges, fixed cores, regularización y normalización
    reutilizan exactamente el backend de ALS-06; SVD init se rechaza cuando el
    tensor completo es desconocido o deliberadamente no enumerado;
  - source values se cachean por generación; no se evalúan fibers porque el
    diseño sampled usa directamente configuraciones globales completas;
  - el resultado registra modo, `n_samples`, reutilización y generaciones sin
    introducir una métrica de validation opcional.

  Evidencia local:

  - matrix completion rank-1 con una entrada desconocida y descenso monótono
    del objetivo observado;
  - tensor completion con observaciones no ordenadas, pesos y error final
    contrastado directamente;
  - tests de refresh/reuse, evaluaciones cacheadas, determinismo y wrapper
    uniform sampled;
  - `136 passed, 1 skipped` en toda la suite ALS;
  - `502 passed, 11 skipped` en toda la suite de decompositions;
  - examples de `TTALS.completion`, `TTALS.fit` y `tt_als` ejecutados con
    `doctest`; `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-08 — Implementar TT leverage sampling**

  Estado: implementado, validado y commiteado en `85aca88`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `TTLeverageRows`.

  Funcionalidad:

  - mantener mixed-canonical respecto al core objetivo;
  - row norms de entornos isométricos como leverage scores;
  - sampling recursivo izquierda/derecha;
  - si se usa Vidal `Γ–Λ`, incorporar `Λ` explícitamente;
  - modo exacto: actualizar distribución y redibujar tras cada core;
  - modo frozen: conservar ids y probabilidades originales durante la
    generación, etiquetando `exact=False` tras cambiar el diseño;
  - reponderar ambos lados del solve;
  - registrar si las probabilidades son exactas.

  Tests:

  - probabilities frente a leverage denso;
  - suma uno; soporte completo solo al mezclar una componente uniforme;
  - Gram y right-hand side insesgados en esperanza; el objetivo completo solo
    con soporte total, mientras leverage puro puede omitir una constante
    independiente de la solución;
  - nunca afirmar que la solución del least squares sea insesgada;
  - prohibición de cambiar probabilidades conservando ids;
  - ranks uno, heterogéneos y complejo.

  Implementación:

  - `TTLeverageRows` calcula probabilidades de leverage desde row norms de un
    TT mixed-canonical, sin materializar el diseño local completo;
  - la distribución factoriza en recursión izquierda, input actual uniforme y
    recursión derecha; mensajes densidad normalizan cada probabilidad
    condicional y admiten ranks heterogéneos y complejo;
  - `uniform_mix` mezcla la proposal con uniform global; un valor positivo da
    soporte completo, mientras leverage puro puede omitir filas de diseño
    nulas cuya contribución al objetivo es constante respecto al core;
  - los pesos siguen siendo `1/sqrt(J p_i)` y hacen insesgados Gram y
    right-hand side bajo soporte suficiente; no se afirma que la solución del
    least squares sea un estimador insesgado;
  - `_TTLeverageALSBackend` prepara el TT mixed-canonical una vez y mantiene la
    canonicalidad mediante gauges QR o SVD en ambas direcciones;
  - el modo `"exact"` redibuja por site con las versiones vigentes y exige
    `sample_reuse_sweeps=1`; el modo `"frozen"` conserva por site ids,
    probabilidades y valores durante la generación;
  - cada `LocalSolveRecord` registra `sampling_exact` y `sample_generation`;
    al cambiar los cores, un batch frozen conserva sus probabilidades
    originales pero pasa explícitamente a `sampling_exact=False`;
  - como los ids cambian por site, esta ruta construye directamente solo las
    filas sampled de los entornos izquierdo/derecho; exact, uniform y
    completion conservan el cache zip-up común;
  - leverage no enumera el target ni calcula error global entre batches;
    evalúa solo `n_samples` configuraciones por site y cachea los valores en
    modo frozen;
  - fixed cores se rechazan en esta primera versión porque impiden preparar la
    canonicalidad requerida sin modificarlos; no se usa forma Vidal
    `Gamma-Lambda`, por lo que no hay singular values separados que incorporar.

  Evidencia local:

  - probabilidades recursivas comparadas con la diagonal leverage de un QR
    denso para real y complejo, con suma uno;
  - soporte puro/mixto y Gram/right-hand side contrastados estadísticamente;
  - integración QR/SVD, exact/frozen, generaciones, exactness, evaluaciones
    exclusivas por site y validaciones de gauges/reuse/fixed cores;
  - `148 passed, 1 skipped` en toda la suite ALS;
  - `514 passed, 11 skipped` en toda la suite de decompositions;
  - examples de `TTALS.fit` y `tt_als` ejecutados con `doctest`; `ruff` sobre
    todos los archivos modificados y `git diff --check` sin incidencias.

- [x] **ALS-09 — Implementar cache TR por segmentos**

  Estado: implementado, validado y commiteado en `1168b5e`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `TRSegmentEnvironmentCache`.

  Algoritmo:

  ```text
  ring = segment_0 | ... | segment_B-1
  active segment:
      external segment summaries
      + old suffix inside segment
      + growing updated prefix
  end segment:
      replace its summary atomically
  ```

  Propiedades:

  - número de segmentos configurable; tres es preset, no hardcode;
  - solo contracciones hacia delante, nunca pseudoinversa;
  - coste lineal por sweep salvo costes de ranks;
  - sampled slices batched;
  - gauges que crucen frontera producen un `CoreUpdateSet` atómico e invalidan
    todas las dependencias receptoras;
  - referencia `DirectTREnvironment`;
  - diseño de segmentos ya utilizable como partición futura de workers.

  Implementación:

  - `TRLocalEnvironment` materializa las filas locales en el mismo orden que
    `core.reshape(-1)` y conserva la escala acumulada en log;
  - `DirectTREnvironment` contrae directamente el anillo para actuar como
    oracle de exactitud, no como ruta productiva;
  - `TRSegmentEnvironmentCache` divide la cadena en intervalos contiguos y
    balanceados, con tres segmentos como preset limitado por el número de
    sitios;
  - cada entorno se forma como `suffix @ external @ prefix`: suffix/prefix se
    actualizan en zip-up dentro del segmento y `external` reutiliza resúmenes
    de los demás segmentos;
  - todos los productos siguen la orientación del anillo y usan solamente
    multiplicaciones batched, sin inversas ni pseudoinversas;
  - exact enumera configuraciones en orden tensorial y sampled conserva orden,
    duplicados, generación y slices batched;
  - las actualizaciones current-plus-receiver se validan completas antes de
    mutar el cache, incluidas absorciones de gauge que cruzan segmentos;
  - `segments` expone intervalos half-open estables para una futura partición
    de workers, sin introducir todavía ejecución paralela.

  Evidencia local:

  - diseños exactos comparados con el oracle denso para 1, 2, 3 y 5 segmentos,
    ambos sentidos, ranks heterogéneos, real y complejo;
  - rows sampled, duplicados, generaciones, renormalización logarítmica,
    absorción QR en frontera y atomicidad de errores cubiertos por tests;
  - `174 passed, 1 skipped` en toda la suite ALS;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **ALS-10 — Portar `TRALS` sobre el driver común**

  Estado: implementado, validado y commiteado en `d42cc42`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Fuente funcional: `_tr_als_impl` estable de `tt2tr/src/blostr.py`, sin copiar
  su reconstrucción completa por site.

  Funcionalidad:

  - ranks cíclicos normalizados desde un entero compartido o una secuencia;
  - `rank[k]` es el enlace derecho del core `k` y `rank[-1]` el cierre;
  - no existe argumento `tr_rank` en `TRALS`;
  - inicialización/fixed cores;
  - cache segmentado;
  - exact/sampled/observed;
  - QR solo cuando el factor puede absorberse legalmente;
  - normalización que salta cores fijos sin modificarlos;
  - error final de sweep;
  - historial y razones normalizadas;
  - wrappers históricos sampled/QR como presets si se necesitan.

  Tests comparan cache y resultado con el oracle directo en cada update,
  absorción de gauge, frontera de segmento y cambio de sentido.

  Implementación:

  - `TRALS` reutiliza `ALSSweepDriver`, `LeastSquaresSolver`, políticas de
    convergencia/update, observers y contratos de sampling de TT;
  - un `rank` entero se replica en todos los enlaces y una secuencia contiene
    exactamente el rank derecho de cada core; `rank[-1]` cierra el ciclo y no
    existe argumento `tr_rank`;
  - inicialización random y TR-SVD exacta, `TRDecomposition`/secuencias como
    estado inicial y caps de rank se validan sin truncado silencioso;
  - `_TRALSBackend` implementa exact, uniform y observed/completion sobre
    `TRSegmentEnvironmentCache`, con generaciones y valores sampled cacheados;
  - QR/SVD solo se aplican si el receptor cíclico inmediato es entrenable y la
    forma conserva los ranks prescritos; en otro caso se usa `NoGauge` antes
    de factorizar;
  - la normalización escalar extrae la raíz de la norma del core resuelto y la
    absorbe en el siguiente sitio entrenable del sentido del sweep, saltando
    cores fijos sin modificarlos;
  - objective exacto y de completion se mide una vez por sweep; uniform no
    presenta como global el error de una generación renovable;
  - fast path sin métricas/observer evita records, objective y timings, y
    `output_device` conserva la política común de devolver cores finales en
    CPU por defecto;
  - `tr_als` ofrece la interfaz funcional simple y devuelve lista de cores o
    `(cores, info)`.

  Evidencia local:

  - un sweep exacto real/complejo comparado core a core con el oracle denso;
  - gauges none/QR/SVD, ranks heterogéneos y cíclicos, SVD init, fixed/all-fixed,
    fast path, uniform con reuse, completion y wrapper cubiertos por tests;
  - `188 passed, 1 skipped` en toda la suite ALS;
  - `554 passed, 11 skipped` en toda la suite de decompositions;
  - docstring example ejecutado con namespace estándar, `ruff` y
    `git diff --check` sin incidencias.

- [x] **ALS-11 — Añadir leverage sampling TR por etapas**

  Estado: etapa A corregida para seguir literalmente Malik--Becker, validada y
  commiteada en `32e70f5`. Etapa B exacta implementada, validada y commiteada
  junto con esta anotación; queda pendiente de revisión detallada del usuario
  antes de considerar ALS-11 completamente cerrada.

  Etapa A, necesaria:

  - `TRProductLeverageRows` basado en leverage de unfoldings individuales;
  - actualización tras modificar core;
  - etiqueta `exact=False`;
  - reponderación correcta;
  - documentar que es aproximación/cota.

  Etapa B, posterior y `EXP`:

  - `TRExactLeverageRows`;
  - Gram implícito `AᴴA` del diseño cíclico double-layer;
  - pseudoinversa pequeña;
  - leverage `diag(A @ pinv(AᴴA) @ Aᴴ)` sin formar `A`;
  - sampling físico condicional mediante prefixes/contracciones;
  - redraw tras cada cambio del diseño;
  - benchmarks que justifiquen coste.

  Las garantías de TT no se atribuirán a TR.

  Implementación de etapa A:

  - `TRProductLeverageRows` sigue el Algorithm 2 de Malik--Becker (ICML 2021):
    calcula row leverage del unfolding mode-input
    `input x (left rank * right rank)` de cada core;
  - el producto de las marginales no activas forma la propuesta acotada del
    paper, sin materializar el diseño cíclico completo;
  - `n_samples` cuenta configuraciones del entorno y cada una se expande a la
    fibra input completa del sitio activo; el backend usa `source.fiber(...)`
    cuando esa capacidad está disponible;
  - `uniform_mix` añade soporte global y las probabilidades mixtas exactas del
    draw generan pesos `1/sqrt(J p_i)` inmutables;
  - el sampler conserva versiones de proposal pero fija siempre
    `proposal_exact=False`; ni un batch recién dibujado se presenta como
    leverage exacto del diseño TR;
  - `TRALS(...).fit(sampling="leverage")` redibuja por sitio después de cada
    cambio de cores, evalúa solo esas configuraciones y registra generación y
    `sampling_exact=False` en cada solve;
  - esta primera versión exige `sample_reuse_sweeps=1`, permite fixed cores y
    reutiliza gauges/normalización cíclica del backend TR;
  - la reponderación apunta al Gram y right-hand side del objetivo completo,
    pero no convierte la solución least-squares no lineal en un estimador
    insesgado ni aporta las garantías mixed-canonical de TT.

  Implementación de etapa B (`EXP`):

  - [x] `TRExactLeverageRows` especializa Sections 4.1--4.2 y Appendix B.2 de
    Malik--Bharadwaj--Murray (2022) al entorno local TR;
  - construye `AᴴA` mediante una cadena double-layer que elimina primero los
    inputs y obtiene una pseudoinversa Hermitian de shape
    `(left rank * right rank) x (left rank * right rank)`, sin formar el
    diseño alto;
  - muestrea la configuración conjunta en orden cíclico mediante
    probabilidades condicionadas y suffix metrics, sin enumerar el vector
    leverage completo;
  - conserva la fibra input activa completa, repondera con la probabilidad
    conjunta exacta y redibuja tras cada cambio del diseño;
  - `TRALS.fit(..., sampling="leverage", leverage_method="exact")` y
    `tr_als` exponen la ruta; `leverage_method="product"` permanece como
    default barato y retrocompatible;
  - `uniform_mix` es una extensión robusta: se conoce exactamente la
    probabilidad mixta, pero solo mix cero es la distribución leverage pura
    analizada en el paper.

  Evidencia local:

  - probabilidades enumeradas suman uno y coinciden con el unfolding
    mode-input del paper, con soporte puro/mixto, fibras completas, versiones,
    pesos y etiqueta aproximada cubiertos;
  - Gram y right-hand side reponderados se contrastan estadísticamente con el
    sistema denso;
  - integración verifica redraw por sitio/generación, etiqueta aproximada en
    product, etiqueta exacta en exact, metadata y validación del refresh;
  - en etapa B, probabilidades reales y complejas coinciden con el leverage de
    un QR/SVD denso usado solo como oracle; las frecuencias del sampling
    condicional, fibras, versiones, pesos, wrapper y metadata están cubiertos;
  - microbenchmark orientativo CPU (`N=6`, `input=4`, `rank=2`, `J=128`):
    product `0.16 ms`, exact `1.21 ms`; confirma mantener product como default,
    sin convertir estos tiempos dependientes de hardware en tests frágiles;
  - `202 passed, 1 skipped` en toda la suite ALS;
  - `568 passed, 11 skipped` en toda la suite de decompositions;
  - docstring example, `ruff` y `git diff --check` sin incidencias.

- [x] **RING-01 — Extraer contratos de apertura local**

  Estado: implementado, validado y commiteado en `98bfbd2`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Implementar `LoopOpening`, `LoopOpenerCapabilities`, `LoopOpener`,
  `ALSLoopOpener`, `FixedGaugeCoreOpener`, `CallableLoopOpener` y
  `CompositeLoopOpener`.

  Propiedades:

  - source/local target, lista normalizada de ranks, gauges fijos y contexto;
  - orientación izquierda/derecha explícita;
  - una API local sirve a TT→TR y TR-RSS;
  - configuración TR-ALS encapsulada, no expandida en la firma caller;
  - un gauge fijo mediante ALS restringido;
  - dos gauges fijos mediante solve directo del core físico;
  - BLOSTR puro solo donde no se imponen gauges;
  - diagnósticos homogéneos.

  Implementación:

  - `LoopOpening` conserva gauge izquierdo, cores físicos, gauge derecho,
    ranks efectivos, orientación, `LocalSolveRecord` y diagnósticos; valida la
    cadena cíclica y puede contraer el target local sin construir un modelo;
  - `LoopOpenerCapabilities` declara y valida soporte de gauge izquierdo,
    derecho, ambos gauges y bloques antes de ejecutar una estrategia;
  - `LoopOpener` fija un contrato común con target/source, `rank`, gauges,
    orientación explícita y contexto extensible;
  - `ALSLoopOpener` encapsula opciones avanzadas de `TRALS`, admite cero o un
    gauge fijo, targets densos/callables/`TensorSource` y bloques físicos;
  - la orientación `left` invierte variables, ranks y cores mediante el espejo
    TR exacto y restaura la apertura al orden original al terminar;
  - `FixedGaugeCoreOpener` materializa un único sistema least-squares cuando
    ambos gauges están fijados y solo queda el core físico;
  - `CallableLoopOpener` adapta estrategias parciales con capabilities
    declaradas y `CompositeLoopOpener` usa una apertura irrestricta como
    inicialización de un refino que impone las restricciones;
  - las capabilities permiten que BLOSTR futuro declare limpiamente que no
    admite gauges, sin introducirlo todavía en este paso.

  Evidencia local:

  - `16 passed` en contratos/openers con real y complejo, ambas orientaciones,
    gauges fijos, target callable, adapter y composición;
  - `218 passed, 1 skipped` en ring+ALS;
  - `584 passed, 11 skipped` en toda la suite de decompositions;
  - `ruff` y `git diff --check` sin incidencias.

- [x] **RING-02 — Centralizar selección de bloques y ranks**

  Estado: implementado, validado y commiteado en `c00d157`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Portar y generalizar:

  - `_estimate_block_ranks`;
  - `_split_block_ttsvd`;
  - crecimiento central/boundary de `tr_rss`;

  Implementar `CentralBlockSelector`, `BlockSelection`,
  `RingRankEstimator` y split mediante `TTSVD`.

  Propiedades:

  - crecimiento bilateral;
  - ranks caps insuficientes producen diagnóstico;
  - bloque central y bloques de frontera;
  - padding solo si el usuario lo solicita y siempre registrado;
  - exactitud del split contra supercore;
  - `rank` público escalar; listas efectivas internas.

  Implementación:

  - `CentralBlockSelector` acepta un provider con `input_dim` o la secuencia
    directamente, normaliza el `rank` TR compartido/por enlace y conserva el
    historial de crecimiento bilateral;
  - `BlockSelection` distingue capacidad de inputs, capacidad requerida,
    factibilidad, razón de parada y contacto con frontera izquierda/derecha;
  - agotar los sites disponibles produce un resultado no factible y
    diagnosticable, no una excepción que pierda la selección alcanzada;
  - `RingRankEstimator` porta la heurística
    `sqrt(D_left * D_right / auxiliary_rank)`, adapta los tres ranks a sus caps
    y enumera por separado toda capacidad izquierda, derecha o auxiliar no
    satisfecha;
  - `split_block_ttsvd` fusiona únicamente los ranks externos con el primer y
    último input, delega todos los cortes en `TTSVD` y restaura supercores con
    ranks externos arbitrarios;
  - `BlockSplit` separa ranks efectivos, ranks almacenados y padding por corte;
    el zero-padding solo ocurre con `pad_rank=True` y queda en metadata;
  - el split conserva el fast path sin métricas, admite los criterios de
    truncación comunes, ambos backends SVD, real/complejo y `output_device`;
  - el subpaquete `tensorkrowch.decompositions.ring` queda incluido en el
    empaquetado instalado, además de funcionar desde el checkout.

  Evidencia local:

  - `18 passed` en selección central/boundary, caps, split exacto/truncado,
    padding opt-in, real/complejo y `svd`/`qr_svd`;
  - `236 passed, 1 skipped` en ring+ALS;
  - `602 passed, 11 skipped` en toda la suite de decompositions;
  - configuración de paquetes, `ruff` y `git diff --check` sin incidencias.

- [x] **RING-03 — Implementar mapas y diagnósticos de gauge**

  Estado: implementado, validado y commiteado en `34ba6c4`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `GaugeMap` y un único kernel para:

  - matricización/orientación;
  - solve, inverse o pseudoinverse;
  - rank numérico y condición;
  - error de cancelación;
  - política `allow_projective_gauges`;
  - mirrors left/right;
  - eventos y `GaugeRecord`.

  Sustituye los bloques duplicados de diagnóstico de
  `tt2tr_fixed_rank`.

  Implementación:

  - `GaugeMap` normaliza gauges izquierdos y derechos a una matriz común
    `external_dim x (cyclic_rank * local_rank)` y su mirror conserva
    exactamente esa transformación;
  - `inverse_or_pinv` admite `auto`, `solve`, `inverse` y `pinv`; `auto` usa
    solve para matrices cuadradas y pseudoinversa para rectangulares o como
    fallback singular;
  - el dual usa la transposición direccional `(G⁺).T` para verificar
    `G.T @ F = I`: los enlaces tensoriales son contracciones bilineales y no
    deben conjugar un gauge complejo;
  - `rank_rtol` controla de forma coherente el cutoff de pseudoinversa y el
    diagnóstico de rank numérico;
  - `GaugeRecord` almacena orientación, shape, rank numérico/cancelable,
    condición, error relativo de cancelación, carácter proyectivo, método y
    tolerancias; `DecompositionMetrics` solo añade `gauges` a `as_info` cuando
    existen records, preservando las salidas legacy vacías;
  - `require_cancellable` implementa la política opt-in de gauges proyectivos
    y produce un error con site, shape, rank y error de cancelación;
  - `as_event` genera eventos estructurados que el futuro driver puede enviar
    al observer sin duplicar formateo ni diagnósticos;
  - construir/matricizar/invertir gauges conserva los tensores en el device;
    las conversiones a escalares CPU ocurren solo al solicitar diagnósticos.

  Evidencia local:

  - `28 passed` en gauges y métricas, cubriendo mirrors, los cuatro métodos,
    real/complejo bilineal (distinguido explícitamente de `G.mH @ F`), fallback
    singular, proyectores, tolerancias y eventos;
  - `264 passed, 1 skipped` en ring+métricas+ALS;
  - `619 passed, 11 skipped` en toda la suite de decompositions;
  - `ruff` dirigido a los archivos nuevos/modificados y `git diff --check` sin
    incidencias; el barrido Ruff histórico completo sigue señalando cuatro
    incidencias preexistentes en `tt_decompositions.py`.

- [x] **RING-04 — Implementar `BidirectionalRingDriver`**

  Estado: implementado, validado y commiteado en `88f3c7f`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Primero con un provider sintético:

  - central block;
  - dos sweeps;
  - boundaries;
  - fixed-left, fixed-right y ambos;
  - ensamblado/order;
  - errores de incompatibilidad.

  No integrar aún TT→TR ni RSS hasta pasar tests del driver aislado.

  Implementación:

  - `RingTargetProvider` desacopla el driver de la semántica futura TT/RSS y
    proporciona target, ranks y contexto solo para el bloque/sites solicitados;
  - `GaugeRecursion` define `advance_left`/`advance_right` y
    `GaugeRecursionStep` transporta el gauge fijo, `GaugeRecord` y diagnósticos
    sin imponer una implementación TT o sketching;
  - `BidirectionalRingDriver` abre el bloque central, alterna sites por derecha
    e izquierda y reserva el último site de la región cíclica restante para un
    solve con ambos gauges;
  - el esquema cruza de forma explícita las fronteras `0`/`n_sites - 1`, por lo
    que centers interiores y de frontera usan el mismo algoritmo;
  - un `boundary_opener` separado permite usar `FixedGaugeCoreOpener` cuando la
    estrategia central/one-gauge no soporta dos gauges;
  - cada apertura comprueba orientación, número de cores y conservación bit a
    bit de gauges fijados antes de incorporarse al ensamblado;
  - `BidirectionalRingResult` valida cobertura exacta de sites, orden original,
    compatibilidad cíclica de ranks mediante `TRDecomposition`, métricas y
    trazabilidad completa de aperturas/recursiones;
  - un bloque central que agota todos los sites se rechaza porque no queda un
    site para reconciliar sus dos gauges salientes;
  - no se ha conectado aún el driver a TT→TR ni sketching, tal como exige el
    orden de esta subtarea.

  Evidencia local:

  - `26 passed` en driver+gauges sintéticos: bloque central simple/multisite,
    dos barridos, centers de frontera, fixed-left/right/both, opener de frontera,
    orden de ensamblado y errores de capacidad/gauge/rank;
  - `274 passed, 1 skipped` en ring+ALS+métricas;
  - `629 passed, 11 skipped` en toda la suite de decompositions;
  - Ruff dirigido y `git diff --check` sin incidencias.

- [x] **TT2TR-01 — Refactorizar `tt2tr_fixed_rank`**

  Estado: implementado, validado y commiteado en `eaeac8a`; pendiente de
  revisión detallada del usuario antes de considerarlo completamente cerrado.

  Crear `TT2TR` y `tt2tr` usando:

  - `ALSLoopOpener`;
  - `BidirectionalRingDriver`;
  - primera versión de recursión equivalente al comportamiento caracterizado;
  - `GaugeMap`;
  - results/metrics;
  - fidelity sin densificar.

  El wrapper no recibe todos los argumentos ALS: acepta `loop_opener`.

  Tests:

  - ranks uno y mayores;
  - `rank` y `tr_rank` de TT→TR convertidos una sola vez al `rank_spec` TR;
  - ranks menores solo en modo adaptativo explícito;
  - TT pequeño exacto;
  - complejo/fase;
  - projective gauges on/off;
  - boundaries y centers;
  - normalized overlap/fidelity frente a denso.

  Implementación:

  - `TT2TR` fija una `TTDecomposition`, una secuencia de cores TT o un
    adaptador `MPS` OBC, y permite repetir `.fit(...)` con distintos `rank`,
    `tr_rank`, centers y estrategias de apertura;
  - `tt2tr` conserva la API sencilla: devuelve cores por defecto y
    `(cores, info)` con `return_info=True`; las opciones ALS avanzadas quedan
    encapsuladas en `loop_opener`, sin ensanchar la firma pública;
  - `rank` prescribe todos los links no cíclicos y `tr_rank` únicamente el
    link cíclico; el `rank_spec` interno se construye una vez y nunca se reduce
    silenciosamente en esta ruta fixed-rank;
  - `_TTCoreProvider` reutiliza `TTTensorSource`, forma supercores locales solo
    cuando son necesarios y declara boundary mode abierto;
  - el modo abierto de `BidirectionalRingDriver` abre un center interno,
    realiza dos sweeps independientes y absorbe los gauges finales en los
    cores TT de rango unidad de ambos extremos; el modo cíclico previo se
    conserva sin cambios para TR-RSS;
  - `PseudoinverseGaugeRecursion` dualiza el gauge saliente con `GaugeMap`, lo
    refleja a la orientación entrante del siguiente site y aplica de forma
    explícita la política de gauges proyectivos;
  - fidelity, normalized overlap y error de reconstrucción absoluto/relativo
    se calculan siempre con contracciones TT/TR escaladas y sin densificar;
    para redes prácticamente idénticas, el residuo obtenido a partir del
    overlap tiene el límite numérico inherente `O(sqrt(eps))`, mientras los
    tests de exactitud comparan además las contracciones densas pequeñas;
  - todas las métricas locales, gauges, tiempos y eventos estructurados se
    conservan en `TRDecomposition`; los cores finales respetan
    `output_device` y la ejecución numérica intermedia permanece en el device
    de entrada;
  - la ruta no trivial sigue siendo una aproximación no convexa cuando el
    opener es ALS: se preservan exactamente los ranks solicitados y las
    métricas cuantifican la calidad obtenida, sin afirmar conversión exacta.

  Evidencia local:

  - `41 passed` en TT→TR, driver y gauges: ranks uno/mayores, tres centers,
    real/complejo, rank cíclico distinto, projective on/off, boundaries,
    adaptador MPS, wrapper, observer y recursión direccional;
  - fidelity, normalized overlap y errores de una aproximación no trivial se
    contrastan con un oracle denso construido solo en tests;
  - `277 passed, 1 skipped` en ring+ALS;
  - `644 passed, 11 skipped` en toda la suite de decompositions;
  - Ruff dirigido y `git diff --check` sin incidencias.

- [x] **TT2TR-02 — Implementar `TTCoreGaugeRecursion`**

  Estado: implementado, validado y guardado en el commit `a0f0f2b`; pendiente
  de revisión detallada del usuario antes de considerarlo completamente
  cerrado.

  Implementación experimental especificada:

  1. abrir TT core como gauge izquierdo, core TR y gauge derecho;
  2. contraer gauge propagado y core TR con el TT core original;
  3. convertir la base virtual TT externa en la del siguiente site;
  4. conservar internamente los dos ranks TR;
  5. repetir simétricamente en ambas direcciones.

  Verificación obligatoria:

  - transición local contra contracción directa;
  - gauges cuadrados: inverse y solve;
  - rectangulares: pinv/projector;
  - left/right mirror;
  - cadena completa;
  - fidelity final;
  - estabilidad por condición.

  Aunque estos tests queden completos localmente, API y docs mantendrán
  `ExperimentalWarning` hasta la revisión detallada del usuario.

  Implementación:

  - `TTCoreGaugeRecursion` construye la base de prefix/suffix contrayendo el
    gauge entrante con el core TR ya retenido; después expresa esa base en el
    siguiente enlace TT mediante un solve local contra el unfolding del core
    TT original;
  - hacia la derecha resuelve
    `TT_(left,input;right) @ E = (left_gauge · TR_core)` y devuelve `E` como
    nuevo gauge izquierdo; hacia la izquierda aplica exactamente el unfolding
    y orientación espejo;
  - `auto`, `solve`, `inverse` y `pinv` comparten la semántica de `GaugeMap`:
    solve/inverse exigen unfolding cuadrado, auto usa solve cuadrado y pinv
    rectangular, y `rank_rtol` controla rango numérico y pseudoinversa;
  - cada transición registra rank, condición y residuo de proyección; un
    transporte rank-deficient o fuera de la imagen se rechaza salvo opt-in
    explícito mediante `allow_projective_gauges=True`;
  - los boundaries no contraen directamente el entorno recursivo con el core
    TT extremo: primero construyen su dual mediante `GaugeMap`, equivalente al
    least-squares final de TR-RSS, y después absorben ese mapa saliente en el
    core OBC de rango unidad;
  - el driver solo invoca este hook de boundary cuando la recursión lo ofrece;
    por tanto la recursión de pseudoinversa y el comportamiento TT2TR-01 no
    cambian;
  - `TT2TR.fit` y `tt2tr` aceptan
    `gauge_recursion="pseudoinverse"|"tt_core"|GaugeRecursion`; el default
    permanece en la ruta caracterizada y seleccionar `"tt_core"` emite
    `ExperimentalWarning`;
  - la primera versión experimental requiere aperturas de un único site. Los
    bloques centrales multi-site deberán dividirse antes de la recursión, como
    ya prevé `split_block_ttsvd` para rank discovery/TR-RSS.

  Evidencia local:

  - `19 passed` en la suite específica: transiciones directas left/right,
    real/complejo, solve/inverse/auto, rectangulares pinv, rechazo/opt-in de
    proyectores, cutoff, condición, boundary dual, cadena completa y fidelity;
  - la cadena rank-one reproduce el TT y una cadena rank mayor conserva ranks
    y hace coincidir normalized overlap/error con el oracle denso;
  - `296 passed, 1 skipped` en ring+ALS;
  - una primera ejecución completa tuvo un único fallo estocástico de
    tolerancia en el test preexistente que compara `TTDecomposition` con dos
    órdenes de contracción float32; el caso pasó aislado inmediatamente y la
    repetición completa terminó con `663 passed, 11 skipped`;
  - doctests de TT2TR, Ruff dirigido y `git diff --check` sin incidencias.

- [x] **BLOSTR-01 — Aislar y verificar BLOSTR**

  Estado: implementado, validado y guardado en el commit `a7eadc4`; pendiente
  de revisión detallada del usuario antes de considerarlo completamente
  cerrado.

  Portar únicamente tras tests:

  - slicing espectral;
  - agrupamiento/reordenación con generator;
  - apertura del primer loop;
  - recuperación completa opcional;
  - `BLOSTRLoopOpener` con capabilities que rechacen gauges fijos;
  - preset `"blostr+als"` que usa BLOSTR como init y ALS para restricciones.

  No portar:

  - `mps_pbc_all_together.py`;
  - `Orbit` como parte de decompositions;
  - prototipos `aux_files` como autoridad.

  Tests con TR sintéticos, autovalores degenerados, seeds, fallo limpio,
  comparación densa y shapes. BLOSTR permanece `EXP`.

  Implementación:

  - `ring/blostr.py` aísla el slicing, selección de eigenspaces,
    balanced clustering reproducible, alineamiento por bloques, primer core y
    recuperación completa de la cola mediante `truncated_svd`;
  - `tr_blostr(tensor, rank, ...)` es la API pública simple y devuelve cores
    TR o `(cores, info)`; `BLOSTRLoopOpener` adapta el mismo motor al contrato
    avanzado de apertura local;
  - la implementación actual conserva de forma explícita la hipótesis del
    paper de rank TR uniforme. Acepta un escalar o una secuencia uniforme y
    rechaza ranks heterogéneos en vez de devolver una factorización espectral
    no caracterizada;
  - las dimensiones de input primera y última deben admitir `rank ** 2`; las
    slices aleatorias y el clustering comparten `torch.Generator`, incluyendo
    generators asociados a aceleradores;
  - una entrada real puede requerir gauges complejos. La descomposición BLOSTR
    lo documenta y mantiene esos cores; `ALSLoopOpener` promueve de forma
    coherente target, inicialización y gauges fijos, y el driver ensambla todos
    los cores con el dtype común promovido;
  - `CompositeLoopOpener(..., fallback_on_error=True)` conserva el error del
    inicializador en diagnósticos y ejecuta el refino sin init cuando las
    hipótesis espectrales fallan;
  - `TT2TR.fit(loop_opener="blostr+als")` intenta BLOSTR sin gauges y usa ALS
    para imponer las restricciones; el preset ALS estable sigue siendo el
    default;
  - tanto la clase como la función emiten `ExperimentalWarning`; no se han
    portado `Orbit`, monolitos PBC ni prototipos auxiliares.

  Evidencia local:

  - tests real/complejo contra oracle denso, tres y cuatro sites, shapes,
    recuperación con varios cortes, slices explícitas, seeds reproducibles,
    source callable, orientaciones left/right y rechazo previo de gauges;
  - espectro degenerado, dimensiones insuficientes y ranks no uniformes fallan
    limpiamente; el composite caracteriza tanto fallback como propagación del
    error;
  - el preset TT→TR se verifica con una cadena exacta y cubre la promoción de
    dtype entre BLOSTR, ALS, recursión y boundaries;
  - `311 passed, 1 skipped` en ring+ALS y `678 passed, 11 skipped` en toda la
    suite de decompositions;
  - doctests de BLOSTR/TT2TR, Ruff dirigido y `git diff --check` sin
    incidencias. El Ruff global de `decompositions` sigue mostrando cuatro
    incidencias legacy ya existentes en `tt_decompositions.py`, fuera de esta
    tarea.

- [x] **ALS-12 — Gate final de fase**

  Estado: gate ejecutado correctamente y guardado en el commit `2ebdd04`;
  pendiente de revisión detallada del usuario antes de considerar la fase
  completamente cerrada.

  Ejecutar toda la suite SVD+ALS+ring, benchmarks de entornos y tests de
  fidelity. Confirmar:

  - ausencia de regresión SVD;
  - TT/TR caches equivalentes al oracle;
  - completion usa siempre los mismos índices;
  - batches sampled no controlan convergencia;
  - fixed cores no cambian;
  - TT→TR calcula fidelity por defecto.

  Resultado:

  - `467 passed, 7 skipped` en la ejecución conjunta de `svd/`, `als/` y
    `ring/`; la suite completa de decompositions de BLOSTR-01 permanece en
    `678 passed, 11 skipped`;
  - `283 passed, 4 skipped` en `tests/test_utils.py` y
    `tests/test_operations.py`, cubriendo también callers subyacentes de SVD;
  - los tests de entornos contrastan todos los diseños TT y TR, forward y
    reverse, real y complejo, exactos y sampled contra el oracle denso; los
    commits atómicos y cambios de segmento/dirección también están cubiertos;
  - completion TT/TR conserva `ObservedRows` y los mismos ids globales durante
    todos los sweeps; el error medido es exactamente el objetivo observado;
  - uniform/leverage sampled reutilizan un batch por generación y rechazan
    criterios de convergencia de error global que no tengan un objetivo fijo;
  - los tests TT/TR comprueban igualdad bitwise de cores fijos y el caso con
    todos los cores fijos; TT→TR contrasta por defecto normalized overlap,
    fidelity y error absoluto/relativo con un oracle denso;
  - microbenchmark orientativo CPU, no usado como umbral de tests (`N=10`,
    `input=2`, `rank=3`, mediana de 15 repeticiones): TT cacheado `0.499 ms`
    frente a oracle directo `5.370 ms` (`10.77x`), TR segmentado `2.493 ms`
    frente a entorno directo `4.366 ms` (`1.75x`);
  - doctests de BLOSTR/TT2TR, Ruff dirigido y `git diff --check` pasaron. Los
    warnings observados en operations son avisos preexistentes de PyTorch
    sobre indexing con secuencias.

#### Entregable de la fase

TT-ALS y TR-ALS reutilizables, sampling correcto, entornos eficientes,
infraestructura de apertura de loops y una conversión TT→TR estructurada que
TR-RSS podrá reutilizar.

---

### Fase 3 — Sketching (RS/RSS), transforms y QTT

#### Objetivo de la fase

Consumir las fuentes comunes creadas para ALS y construir regiones,
recursiones, Phi lazy y fitting físico. Usar esa infraestructura para
refactorizar TT-RSS como caso sampled de recursive sketching, sin cambiar
innecesariamente su matemática. A continuación implementar TR-RSS, fuentes
sparse/empíricas/TT, nuevos operadores RS, QTT/QTR y Tucker cuantizado.

#### Orden conceptual interno

```text
common TensorSource + sketching specs
    -> regions/recursion
    -> Phi/evaluation plan
    -> transforms
    -> physical fitting
    -> projection/truncation
    -> TTRSS estable
    -> TRRSS/RS/QTT experimentales
```

Aunque se implemente en ese orden, todos los contratos se diseñan desde el
principio para 1D, N-D, lazy fibers, sparse y ejecución paralela.

#### TODOs

- [ ] **RSS-00 — Caracterizar TT-RSS legacy**

  Añadir tests antes del refactor para:

  - función escalar `(batch, 1)`;
  - función vectorial;
  - labels explícitos y muestreados;
  - output primero/medio/último;
  - dominio compartido, por site e inferido;
  - samples 2D y `(batch, n_features, in_dim)`;
  - criterios modernos de truncación;
  - dtype real/complejo y device;
  - `return_info`;
  - cores cargables en `models.MPS`/`MPSLayer`;
  - generator introducido de forma determinista;
  - producto cartesiano/projector frente a oracle.

  Registrar limitaciones actuales como tests `xfail` o especificaciones, no
  perpetuarlas: output único, embedding homogéneo, escalar obligado a
  `(batch,1)`, verbosity booleana y `randu`.

- [ ] **RSS-01 — Implementar specs de embedding, domain y outputs**

  Crear `_EmbeddingSpec`, `_DomainSpec`, `_OutputSpec` y
  `_SketchingFitSpec`.

  Requisitos de embedding/domain:

  - un callable/tensor compartido o lista por input variable;
  - cada embedding puede tener dimensión física distinta;
  - cada domain puede tener distinto número/shape de valores;
  - validación por variable con mensajes que indiquen site;
  - output sites no consumen `embedding`, siempre `basis`;
  - evitar evaluar repetidamente embeddings para inferir shapes.

  Requisitos de output:

  - escalar acepta `(batch,)` y `(batch,1)` sin output site;
  - tensor output acepta cualquier shape `(batch,o_1,...,o_m)`;
  - `out_position` entero/lista o default equiespaciado;
  - outputs separados y orden conservado;
  - label plano row-major sobre `prod(output_shape)`;
  - flatten/unflatten reversible;
  - labels ausentes: samplear `abs(values)**2`, normalizado por fila;
  - filas de norma cero producen política/error explícito;
  - insertar un índice por output axis en sketch samples;
  - recursion de todos los output axes con `basis`.

  Tests exhaustivos de posiciones, shapes heterogéneas y complejos.

- [ ] **RSS-02 — Adaptar las fuentes comunes a sketching**

  Reutilizar las clases de `decompositions/sources/` creadas en `ALS-01` y
  añadir solo capacidades/adaptadores específicos de sketching, como
  `SketchContractableSource`.

  Propiedades comunes:

  - evaluación batched;
  - shape/output metadata;
  - device/dtype explícitos;
  - stats de llamadas y puntos;
  - capacidades opcionales sin obligar a todo source a implementar todo.
  - exactamente la misma normalización de callable que en ALS.

  Sparse/empírica:

  - índices enteros `(n_nonzero, n_sites)`;
  - values escalares/tensoriales;
  - agrupar duplicados determinísticamente;
  - pesos empíricos opcionales y normalización;
  - lookup sparse/join sin producto denso;
  - dataset directo aceptado por el wrapper RS y convertido internamente.

  TT:

  - cores raw como camino canónico;
  - adapter de modelo solo extrae cores;
  - evaluación por prefix/suffix;
  - contracción con sketches TT/TTStack cuando sea más eficiente;
  - no crear TensorKrowch graph.

- [ ] **RSS-03 — Implementar geometría de regiones**

  Crear `SiteRegion`, `_SamplePool`, `RegionSketch` y `SketchRecursion`.

  Requisitos:

  - sites 1D enteros o coordenadas N-D hashables;
  - orden determinista separado de pertenencia;
  - unique/inverse ids vectorizados;
  - `restrict`, unión correlacionada y comparación;
  - detección de contenido child⊂parent;
  - `recursive_projector` como gather, no matriz;
  - composición de recursiones;
  - inputs con `in_dim`;
  - left/right en 1D emergen de la misma operación;
  - cuatro direcciones PEPS serán adapters, no branches aquí.

  Benchmarks contra `create_projector` Python-loop y tests de igualdad exacta.

- [ ] **RSS-04 — Implementar Phi lazy y evaluación deduplicada**

  Crear `PhiView`, `EvaluationView`, `PhiOperator`,
  `_EvaluationPlanBuilder`, `_EvaluationPlan`, `_EvaluationSession`,
  `_EvaluationRegistry`, `_MaterializedPhi` e incidence maps.

  Funcionalidad:

  - construir layout de regions y axes físicos;
  - acceso completo, por ids y por fibra;
  - ensamblar configuraciones globales;
  - eliminar output sites de la llamada y hacer gather tensorial correcto;
  - deduplicar puntos entre todos los sites antes de evaluar;
  - scatter/gather de valores a cada Phi;
  - materialización batched;
  - ruta sparse basada en soporte;
  - ruta TT basada en contracciones parciales;
  - `ConfigurationBatch` para combinar selector+regions sin fingir una unión
    correlacionada;
  - ciclo `builder.collect -> expand -> freeze(plan) -> evaluate -> scatter`;
  - stats y diseño shardeable.

  Pruebas:

  - explícito vs lazy vs fibers;
  - múltiples output axes;
  - puntos repetidos entre Phi;
  - funciones tensoriales;
  - sparse y TT contra callable denso;
  - selección batched PEPS sintética.

- [ ] **RSS-05 — Implementar transforms de valores**

  Crear protocolos `GlobalValueTransform` y `LocalValueTransform`, adaptadores
  de callable e identidades.

  Pipeline exacto:

  ```text
  collect requests
      -> GlobalValueTransform.required_points
      -> freeze EvaluationPlan
      -> evaluate unique points
      -> GlobalValueTransform.apply once
      -> scatter into PhiView views
      -> LocalValueTransform.apply
      -> PhysicalFitter
  ```

  Requisitos:

  - acceso global a puntos, valores e incidencias;
  - una misma configuración transformada produce el mismo valor en todos los
    Phi;
  - transformación global puede declarar puntos/closure antes del freeze;
  - transformación local puede trabajar sobre materialización o fibra y
    declara antes cualquier query;
  - transformación local siempre devuelve `PhiView`;
  - la primera versión rechaza extensión posterior al freeze;
  - fuente con `contract_sketch` y transform global no identidad fuerza ruta
    puntual o error explícito;
  - composición y contexto inmutables donde sea posible;
  - identity no añade materialización ni copia;
  - no introducir nombres específicos de VMC o estimación de densidad.

  Tests de norma global, closure sintético, transformación local genérica,
  rechazo de extensión tardía y orden independiente. Añadir una
  integración vertical sin nombre de aplicación:
  `EmpiricalDistribution -> MarginalSketch.markov -> callback local de
  suavizado por kernel -> fitting local -> core-determining equations`.

- [ ] **RSS-06 — Implementar fitting físico**

  Crear `PhysicalFitter`, `FittedPhysicalAxis`, `FixedEmbeddingFitter` y
  `BasisFitter`.

  `FixedEmbeddingFitter`:

  - least squares sobre el eje físico indicado;
  - embedding distinto por site;
  - regularización/escalado mediante `LeastSquaresSolver`;
  - puede consumir Phi completo o fibers;
  - declara previamente las queries adicionales;
  - residuo y condición etiquetados como fitting, no truncación.

  `BasisFitter`:

  - output sites y variables discretas;
  - selección/identidad exacta cuando proceda;
  - múltiples output axes.

  Tests de de-embedding exacto, sobredeterminado, complejo, domains distintos y
  no materialización.

- [ ] **RSS-07 — Sustituir `randu` por range projection explícita**

  Crear `RangeProjector`, `IdentityRangeProjector` y
  `RandomizedRangeProjector`.

  Entregas:

  1. caracterizar la multiplicación `randu` actual;
  2. implementar range finder `AΩ -> QR -> QᴴA -> small SVD`;
  3. asegurar que el eje proyectado es el eje que se comprime;
  4. `projection_dim=rank` por defecto;
  5. `rank=None` produce proyección cuadrada/no reductora;
  6. permitir `random_projection=False`;
  7. generator, complejo, oversampling y power iterations;
  8. registrar dimensión, tiempo y error de proyección.

  Comparar precisión/coste con SVD directa y ruta legacy. No llamar
  “randomized SVD” a una mera multiplicación por matriz aleatoria.

- [ ] **RSS-08 — Implementar `RecursiveSketching` y observabilidad**

  Crear clase base, `_SketchingFitContext` y eventos:

  - `source.prepare`;
  - `regions.build`;
  - `phi.plan`;
  - `source.evaluate`;
  - `values.global_transform`;
  - `values.local_transform`;
  - `physical.fit`;
  - `range.project`;
  - `svd.trim`;
  - `recursion.apply`;
  - `core.solve`;
  - `result.validate`.

  Verbosity:

  - nivel 1: títulos por site/fase y resumen;
  - nivel 2: tiempos, ranks, errores;
  - nivel 3: shapes/diagnósticos y cores al final;
  - salida ordenada incluso si en el futuro llegan eventos paralelos.

  La base no conoce TT/TR/PEPS mediante flags; expone hooks sustanciales y usa
  composición.

- [ ] **RSS-09 — Refactorizar TT-RSS sobre la infraestructura**

  Portar el algoritmo existente conservando primero su matemática:

  - prefix/current/suffix como `RegionSketch`;
  - `sketching` como `PhiOperator.materialize`;
  - `create_projector` como `SketchRecursion`;
  - de-embedding con `FixedEmbeddingFitter`;
  - trimming con `truncated_svd`;
  - ecuación `pinv/least-squares(A_{k-1}) @ B_k`;
  - cores OBC y orden existente;
  - `output_device="cpu"`.

  Estructura de tareas por sites:

  1. planificar/evaluar valores únicos globales;
  2. construir/fit/trim cada `B_k`;
  3. construir cada `A_k` mediante recursión;
  4. resolver cada core usando `A_{k-1}` y `B_k`;
  5. ensamblar.

  Esta separación debe revelar dependencias locales y preparar paralelización,
  pero la primera ejecución será serial.

  Compatibilidad:

  - `tt_rss` mantiene lista de cores;
  - `return_info` mantiene keys legacy además de las nuevas durante transición;
  - wrappers temporales para helpers importados por `tt2tr`;
  - mismo resultado numérico bajo modo `legacy_projection`.

- [ ] **RSS-10 — Activar generalizaciones TT-RSS**

  Una vez equivalencia básica esté validada, habilitar:

  - `embedding`/`domain`/`input_dim` por site;
  - outputs tensoriales múltiples y separados;
  - scalar output sin eje artificial;
  - output sampling por `abs(f)**2`;
  - labels planos y unflatten;
  - range projection opcional;
  - `verbose=0..3`;
  - generator;
  - `device`/`output_device`;
  - `TTRSS` reusable con fits independientes;
  - warm start explícito, nunca accidental.

  Métricas:

  - tiempo total y por fase/site;
  - ranks y shapes;
  - error de truncación local/acumulado correctamente etiquetado;
  - error absoluto/relativo sobre `sketch_samples` si se solicita;
  - puntos únicos, llamadas/batches y cache hits;
  - warnings de condicionamiento/fitting.

- [ ] **RSS-11 — Portar TR-RSS usando ring común**

  Fuentes:

  - `tt2tr/src/tr_rss.py`;
  - piezas maduras de rank discovery/boundary blocks;
  - `BidirectionalRingDriver`;
  - `SketchGaugeRecursion`.

  Implementar primero ruta de rank cap fijo:

  - `rank` entero compartido o secuencia de `n_sites` enlaces derechos;
  - `rank[-1]` como enlace cíclico, sin argumento `tr_rank`;
  - central site/block;
  - doble trimming;
  - `LoopOpener` configurable;
  - recursión izquierda/derecha;
  - solve de boundaries;
  - scalar y outputs múltiples;
  - domains/embeddings heterogéneos;
  - sample error y métricas locales;
  - no firma expandida con opciones ALS.

  Después ruta adaptativa:

  - `CentralBlockSelector`;
  - `RingRankEstimator`;
  - bloques no injectivos de frontera;
  - split con TT-SVD;
  - ranks efectivos registrados;
  - factorización de ranks descubiertos tan equilibrada como permitan caps y
    factibilidad;
  - padding solo opt-in.

  El antiguo `_create_right_projector` desaparece; left/right son la misma
  abstracción orientada.

- [ ] **RSS-12 — Verificar reutilización TT→TR / TR-RSS**

  Crear tests de contrato comunes:

  - mismo `LoopOpener` sobre TT core y Phi de shape equivalente;
  - mismo `GaugeMap` y policy de cancelación;
  - misma selección/split de bloque;
  - mismos diagnostics de apertura;
  - diferentes `GaugeRecursion` contrastadas con oracles.

  No aceptar una refactorización que mantenga dos copias de:

  - init BLOSTR;
  - ALS local;
  - gauge inversion/pinv;
  - rank/block search;
  - bidirectional sweep;
  - split TT-SVD.

- [ ] **RSS-13 — Implementar schedule TR par/impar serial**

  Antes de paralelizar, implementar y validar el algoritmo propuesto de forma
  serial:

  1. seleccionar sites pares o bloques alternos;
  2. abrir cada site par con solve local completo y gauges libres;
  3. estabilizar ambos lados mediante SVDs/gauge canonicalization compatible
     con una forma tipo Vidal;
  4. propagar recursiones a ambos vecinos;
  5. resolver sites impares con ambos gauges fijos, idealmente mediante
     least squares/pseudoinversa sin ALS;
  6. reconciliar la base cíclica y medir error.

  Variantes:

  - anillo par/impar;
  - número impar de sites;
  - bloques mayores para menos particiones;
  - ranks heterogéneos efectivos;
  - fallback al center-out si gauges no son compatibles.

  El schedule sigue `EXP` hasta superar al center-out en estabilidad o coste.

- [ ] **RSS-14 — Implementar sources sparse/empíricas en RS**

  Añadir wrappers para:

  ```python
  tt_rs(source=sparse_source, ...)
  tt_rs(dataset=samples, weights=None, ...)
  ```

  Requisitos:

  - exactamente uno entre `source` y `dataset`;
  - dataset se convierte a `EmpiricalDistribution`;
  - evaluar/proyectar solo soporte no nulo;
  - weights y duplicados correctos;
  - outputs/distribution normalization explícitos;
  - coste proporcional al soporte y sketch, no al grid completo;
  - distribución empírica no se confunde con samples RSS de una función densa.

- [ ] **RSS-15 — Implementar operadores y sistemas Sampled/Marginal/TTStack**

  Crear también `CoreDeterminingSystem` y `SketchSystemBuilder`; cada operador
  debe aportar un builder validado, no entrar por un branch ambiguo del driver.

  `SampledSketch`:

  - evaluación por producto de left/current/right regions;
  - ruta sparse que detecta entradas no nulas sin materializar ceros;
  - semántica RSS existente.

  `MarginalSketch`:

  - contraer sites eliminados con unos/factor;
  - `.markov(order)` deja vecinos locales;
  - marginales locales de extremos/interior;
  - fuente empírica: histogramas/marginales desde dataset;
  - no afirmar que es gaussiano.

  `TTStackSketch`:

  - `tt_rank` y `n_stacks`, separados del `rank` objetivo de la
    descomposición;
  - `tt_rank=1` reduce a filas separables/Khatri–Rao;
  - `n_stacks=1` es el caso Gaussian TT projection del operador, sin garantía
    TT-RS automática;
  - dimensión sketch de salida `n_stacks * tt_rank`;
  - aplicación sobre soporte sparse;
  - contracción eficiente con `TTTensorSource`;
  - variante orthogonal claramente separada de las garantías gaussianas.

  Antes de implementar su `SketchSystemBuilder`, completar un gate
  matemático independiente: construir los right sketches `T_k`, la recursión
  de left blocks `s_k` y las core-determining equations; verificar dimensiones,
  compatibilidad recursiva y condiciones de row/column space y rank. Ser una
  proyección lineal global no basta por sí solo para ser un sketch válido del
  sistema TT-RS.

- [ ] **RSS-16 — Implementar `TTRS` y `TRRS`**

  Reutilizar:

  - sources;
  - regions/Phi;
  - fitters;
  - trimming;
  - solve TT o ring.

  `TTRS` implementa primero las core-determining equations TT para:

  - `MarginalSketch.markov`;
  - `TTStackSketch`, solo tras superar el gate de `RSS-15`;
  - `SampledSketch`.

  `TRRS` adapta el driver cíclico y permanece experimental. Documentar que las
  garantías del paper TT abierto no se transfieren automáticamente.

  Tests con fuente sparse exacta, distribución empírica y TT source.

- [ ] **RSS-17 — Optimizar `TTTensorSource`**

  Portar como kernels, no como nombres MPS:

  - left evaluation;
  - right evaluation;
  - construcción de Phi local/bloque;
  - contracción con TTStack.

  Comparar:

  - evaluación puntual genérica;
  - contracción estructurada;
  - densificación oracle.

  Elegir siempre el backend por capacidad de source, no por `isinstance` en el
  driver principal.

- [ ] **RSS-18 — Añadir fitting entrenable y Phi funcional**

  Implementar `TrainableEmbeddingFitter`:

  - recibe `PhiOperator.fiber(axis, fixed_indices)`;
  - declara su conjunto de entrenamiento/query antes de congelar la sesión;
  - una consulta posterior exige crear/replanificar una sesión independiente,
    no extender un plan congelado;
  - entrena callable `x_k -> embedding_dim`;
  - controla optimizer, stopping y seed dentro del fitter;
  - devuelve residual y modelo/estado solo si se solicita;
  - no obliga a materializar Phi;
  - grad local aislado.

  Verificar que `SampledPhysicalFitter` no es un nombre necesario:
  `SampledSketch` describe el sketch; el fitter describe la representación del
  eje físico.

- [ ] **QTT-01 — Implementar layouts cuantizados multivariables**

  Crear `QuantizedLayout` con:

  - `base`/`level` broadcast o por variable;
  - grouped/interleaved;
  - levels desiguales;
  - coarse-to-fine/fine-to-coarse;
  - permutation custom;
  - encode/decode reversible;
  - shapes y overflow validados.

  Tests concretos para `f(x,y,z)` en las dos ordenaciones estándar.

- [ ] **QTT-02 — Implementar mapas de coordenadas**

  Crear `CoordinateMap`, `UniformCoordinateMap`, `WarpedCoordinateMap` y
  `ExplicitGridMap`.

  Funcionalidad:

  - dominio 1D por variable;
  - lista de dominios para N variables;
  - mapa separable por variable;
  - mapa conjunto `u -> (x,y,z)` para coordenadas curvilíneas;
  - leyes geométricas/contracción-expansión expresables como callable;
  - inverse opcional;
  - grid uniforme por endpoints por defecto y cell centers opcional;
  - nearest-grid con desempate inferior y out-of-domain error/clip explícito;
  - domain requerido solo cuando el mapa no contiene la geometría física.

  Inspiración de tests/documentación:

  - grid computacional uniforme y mapa físico/curvilíneo;
  - celdas variables y remapeo de dominio;
  - ningún mapeo de un paper queda hardcodeado en el driver.

- [ ] **QTT-03 — Implementar `.quantized`, `qtt_rss` y `qtr_rss`**

  Crear `QuantizedSourceAdapter`.

  La adaptación:

  1. recibe `sketch_samples` físicos por defecto, o digits si
     `sample_space="digits"`;
  2. aplica inverse + cuantización cuando las muestras son físicas;
  3. `QuantizedLayout.decode_digits`;
  4. normaliza índices a coordenadas computacionales;
  5. aplica `CoordinateMap`;
  6. llama a la fuente física original;
  7. usa `basis` como embedding de cada digit;
  8. delega en `TTRSS` o `TRRSS`.

  Requisitos:

  - no argumento `embedding`;
  - constructor `TTRSS.quantized`, no `from_qtt`;
  - functions N-D;
  - callable, sparse/empírica y TT mediante el adapter;
  - outputs tensoriales;
  - domain/map metadata;
  - rounding, endpoints, colisiones sparse y out-of-domain verificados;
  - las versiones grouped/interleaved aproximan la misma función física tras
    decodificar cada layout;
  - no se afirma equivalencia por permutar una lista de cores;
  - error comparado en coordenadas físicas.

- [ ] **QTT-04 — Implementar `QTTPhysicalFitter`**

  Sobre una fibra local continua:

  - discretizar solo `x_k`;
  - llamar recursivamente a TT-RSS;
  - admitir output tensorial del entorno izquierdo/derecho;
  - conservar el eje conector;
  - controlar recursión para evitar observers/RNG/global caches mezclados;
  - permitir distinto base/level/map por variable.

  Tests frente a materializar la fibra en grid y aplicar TT-SVD/RSS ordinario.

- [ ] **QTT-05 — Implementar QTT-Tucker y QTR-Tucker nativos**

  Crear `QTTTuckerRSS`, `QTRTuckerRSS`, sus resultados ligeros y los wrappers
  `qtt_tucker_rss`/`qtr_tucker_rss`, ubicados en `sketching/tt.py` y
  `sketching/tr.py`.

  Algoritmo:

  - construir el Phi/fibra local con axes
    `(digits, alpha_{k-1}, alpha_k)`;
  - colocar todos los output axes locales consecutivos en un extremo;
  - usar `basis` en su recursión;
  - hacer un split rank-revealing
    `Phi_k[digits, alpha_left*alpha_right] ≈ U_k[:,gamma_k] @ R_k`;
  - representar `U_k` como
    `F_k(i_{k,1}, ..., i_{k,L_k}, gamma_k)`;
  - reshape `R_k` como core superior
    `C_k(alpha_{k-1}, gamma_k, alpha_k)`;
  - ensamblar los `C_k` como TT o TR según la variante; en TR el último rank
    es el enlace cíclico;
  - registrar truncación/residuo del split y conectar cada factor al
    `gamma_k`, sin eliminarlo;
  - mantener representación jerárquica, no fingir TT plano.

  Tests de dos niveles TT/TR, ranks uno, múltiples variables, evaluación y
  `.flatten()` en casos pequeños cuando proceda; contrastar el bound con la
  errata/corrección del algoritmo QTT-Tucker; etiquetar la construcción RSS
  como adaptación sin garantía de recovery heredada.

- [ ] **RSS-19 — Limpieza de compatibilidad y docs**

  Cuando TT-RSS y TR-RSS nuevos estén validados:

  - migrar imports de `tt2tr` y otros callers;
  - convertir helpers legacy en wrappers privados/deprecated;
  - eliminar `tt_decompositions.py` solo después de validar TT-RSS nuevo y
    migrar callers; `tt_rss` sigue exportándose desde la API pública;
  - actualizar `docs/decompositions.rst`;
  - ejemplos de función escalar, tensorial, sources sparse/TT y QTT;
  - tabla de madurez estable/experimental.

- [ ] **RSS-20 — Gate final de fase**

  Ejecutar suites:

  - SVD, ALS y ring completas;
  - sketching regions/Phi/source adapters/fitting;
  - TT-RSS equivalencia y generalizaciones;
  - TR-RSS center-out y par/impar;
  - sparse/RS;
  - QTT/QTR y QTT-Tucker/QTR-Tucker.

  Medir:

  - número de evaluaciones únicas;
  - memoria máxima Phi explícito vs lazy;
  - precisión y tiempo projection on/off;
  - coste callable vs sparse vs TT source;
  - overhead de metrics/verbosity 0.

#### Entregable de la fase

Un motor de sketching común que conserva TT-RSS como caso sampled, comparte
apertura cíclica con TT→TR, admite sources y outputs generales, y proporciona
RS, QTT/QTR y Tucker cuantizado sin contaminar la API sencilla.

---

### Fase 4 — Paralelización y ejecución distribuible de TT/TR

#### Objetivo de la fase

Convertir las unidades independientes ya existentes en task graphs ejecutables
por backends serial, multiproceso y, posteriormente, distribuido. La primera
versión de cada algoritmo será un schedule serial expresado como tareas; solo
después se activará concurrencia. No se cambia la matemática durante la
paralelización.

#### Principios

- El backend serial es el oracle.
- Una task es pura respecto a su payload; no accede a globals del driver.
- No se serializan objetos `tk.models` ni grafos.
- Fuentes callable no serializables producen un error temprano o se ejecutan
  en el coordinador.
- Las evaluaciones únicas se calculan una vez y se comparten/shardean.
- Los workers no imprimen; emiten eventos estructurados.
- Reducciones de errores, ranks y métricas tienen orden determinista.
- La asignación de GPU es explícita; no se comparte memoria CUDA entre procesos
  por defecto.
- Los bloques permiten usar menos workers que sites.

#### TODOs

- [ ] **PAR-00 — Perfilar y fijar baselines seriales**

  Para SVD, TT/TR-ALS, TT→TR, TT/TR-RSS, sparse RS y QTT:

  - tiempo por fase;
  - memoria pico;
  - evaluaciones/unique points;
  - tamaños de Phi;
  - costes de entornos;
  - porcentaje serial inevitable.

  Definir tamaños pequeños de corrección y medianos de rendimiento. No
  paralelizar una subrutina cuyo coste sea marginal.

- [ ] **PAR-01 — Implementar task graph y backend serial**

  Crear `_DecompositionTask`, `_TaskGraph`, `ExecutionBackend` y
  `SerialBackend`.

  Requisitos:

  - IDs/dependencias estables;
  - detección de ciclos;
  - propagación de excepción con contexto de task/site;
  - seed derivada;
  - device/output device;
  - eventos y tiempos;
  - cancelación limpia;
  - resultado bitwise o numéricamente equivalente al driver serial directo.

- [ ] **PAR-02 — Implementar store de evaluaciones compartidas**

  Crear `_SharedEvaluationStore` sobre una `_EvaluationSession` ya congelada.

  Funcionalidad:

  - particionar puntos únicos;
  - evaluar por shards;
  - recomponer en orden canónico;
  - ejecutar la expansión/closure antes de repartir;
  - aplicar `GlobalValueTransform` exactamente una vez tras recomponer;
  - exponer gathers de cada Phi;
  - cache stats agregadas;
  - soporte CPU shared memory y serialización normal;
  - política explícita para valores GPU.

  Tests con puntos repetidos, transform global y fallos parciales.

- [ ] **PAR-03 — Implementar `ProcessBackend`**

  Requisitos:

  - número de workers configurable;
  - contexto de procesos seguro para PyTorch;
  - inicialización por worker;
  - devices asignados;
  - no oversubscription accidental de threads BLAS;
  - shutdown/cancelación;
  - orden determinista de outputs/events;
  - fallback serial documentado para callables no serializables.

- [ ] **PAR-04 — Paralelizar evaluación y fitting común**

  Aplicable a RSS/RS/QTT:

  - shards de source evaluation;
  - embeddings/domains por site;
  - local transforms;
  - physical fitters independientes;
  - range projection/trimming por site.

  Comparar resultados con serial usando las mismas matrices aleatorias/seeds.

- [ ] **PAR-05 — Paralelizar TT-RSS por oleadas**

  Task graph:

  ```text
  global evaluation plan
          │
          ├── B_0 ... B_n        (Phi + fit + projection + trim)
          │
          ├── A_0 ... A_n        (recursión; depende del B local)
          │
          └── G_k solves         (depende de A_{k-1}, B_k)
                   │
                assemble
  ```

  Requisitos:

  - un worker por site o bloque de sites;
  - dependencia vecina mínima;
  - outputs separados;
  - métricas locales agregadas;
  - offload temprano;
  - igualdad serial;
  - speedup medido en funciones suficientemente costosas.

- [ ] **PAR-06 — Paralelizar TR-RSS par/impar o por bloques**

  Solo después de validar `RSS-13`.

  Oleadas:

  1. abrir sites/bloques pares independientemente con gauges libres;
  2. propagar gauges/recursions a vecinos;
  3. comunicar un entorno por frontera de proceso;
  4. resolver impares con ambos gauges fijos;
  5. reconciliar cierre y ensamblar.

  Verificar:

  - forma tipo Vidal/SVD bilateral;
  - anillos pares/impares;
  - menos workers que sites;
  - condición/cancelación de gauges;
  - fallback center-out;
  - mismo error que schedule serial.

- [ ] **PAR-07 — Paralelizar TT→TR**

  Reutilizar el mismo schedule por bloques:

  - providers TT serializables;
  - `TTCoreGaugeRecursion`;
  - local openings independientes cuando gauges libres;
  - solves con gauges fijados;
  - fidelity final centralizada.

  No compartir mutablemente cores entre workers.

- [ ] **PAR-08 — Paralelizar construcción de entornos ALS**

  Mantener updates ALS secuenciales cuando exista dependencia de Gauss–Seidel,
  pero paralelizar:

  - contracción inicial de segmentos;
  - suffix/prefix scans mediante reducción en árbol donde compense;
  - sistemas batched independientes;
  - objetivos/metrics.

  Añadir un modo Jacobi/checkerboard solo como estrategia separada y tras
  validar convergencia serial. No presentar Jacobi como el mismo ALS.

- [ ] **PAR-09 — Paralelización razonable de SVD/QTT**

  TT-SVD tiene dependencia entre cortes y no se paraleliza artificialmente.
  Se permiten:

  - batches independientes;
  - SVD backend ya paralela;
  - factores QTT por variable;
  - factors QTT-Tucker y fitting local.

  Documentar la parte secuencial en lugar de prometer scaling lineal.

- [ ] **PAR-10 — Backend distribuido experimental**

  Tras estabilizar procesos locales:

  - definir `DistributedBackend` sin acoplar algoritmos a un framework;
  - sharding de evaluation store;
  - transferencia explícita de tensores;
  - retries únicamente para tasks idempotentes;
  - aggregation de events/metrics;
  - timeout/cancelación;
  - tests de dos workers en local.

  La elección concreta de framework se decide mediante benchmark y
  disponibilidad del entorno; no se fija anticipadamente en las APIs.

- [ ] **PAR-11 — Gate final de fase**

  Para cada algoritmo:

  - serial task graph == driver original;
  - process backend == serial dentro de tolerancia;
  - misma seed y resultados reproducibles;
  - errors/warnings llegan con site/task;
  - no hay evaluaciones duplicadas;
  - no hay regresión significativa con `workers=1`;
  - speedups publicados solo con workload representativo.

#### Entregable de la fase

Backends y task graphs genéricos, TT-RSS y TR-RSS/TT→TR paralelos por sites o
bloques, y ALS con contracciones paralelas sin alterar su semántica de sweep.
La infraestructura queda lista para ser reutilizada por PEPS.

---

### Fase 5 — PEPS en la rama `peps_rss`

#### Objetivo de la fase

Portar las implementaciones modernas del proyecto externo `peps-rss`,
limpiarlas y conectarlas a la infraestructura común. Todo el trabajo PEPS se
realiza en la rama TensorKrowch `peps_rss`. Primero se valida el algoritmo
serial; solo al final se activa su paralelización.

#### Clasificación de fuentes

##### Referencias modernas principales

- `peps-rss/src/peps_core.py` para contratos de ranks/layout/storage;
- `peps-rss/src/peps_svd.py`;
- `peps-rss/src/peps_als.py`;
- `peps-rss/src/peps_rss_ctm.py`;
- `peps-rss/src/peps_vo.py`;
- `peps-rss/src/peps_natural_gradient.py`;
- `peps-rss/src/peps_column_compression.py`;
- `peps-rss/src/peps_rss_hierarchical.py`;
- `peps-rss/src/orbits.py`;
- tests `peps-rss/tests/test_peps_rss_als.py` y
  `test_peps_rss_ctm.py`.

##### Legacy, no base de implementación

- `tensorkrowch/decompositions/peps_decompositions.py` de la rama actual
  `peps_rss`;
- `peps-rss/src/old/*`;
- `peps-rss/src/peps_rss_als.py` como fachada dinámica;
- copias PEPS de `blostr.py`, `utils.py` y helpers TT/TR.

El código legacy solo sirve para caracterización/migración de API. No se
mezclan sus helpers con el port moderno.

#### Riesgo algorítmico central

En PEPS-RSS, reconstruir exactamente un Phi local fusionado no garantiza que
el environment futuro pueda refinarse en cores PEPS con el rank solicitado.
Cada solve debe distinguir:

- error de reconstrucción local;
- rank/refinabilidad de virtual spaces abiertos;
- error de truncación de boundary;
- compatibilidad de incoming gauges;
- error global PEPS cuando pueda calcularse.

La versión CTM y los schedules checkerboard permanecen experimentales hasta
controlar esta condición.

#### TODOs

- [ ] **PEPS-00 — Preparar rama e importar infraestructura común**

  Pasos:

  - cambiar a `peps_rss` sin perder cambios;
  - integrar commits finalizados de Fases 1–4;
  - comprobar qué archivo legacy existe en la rama;
  - guardar baseline de sus exports/warnings;
  - no sobrescribir trabajo experimental no relacionado;
  - crear estructura `decompositions/peps/`.

  Criterio: tests TT/TR siguen pasando en `peps_rss` antes de portar PEPS.

- [ ] **PEPS-01 — Portar tests de caracterización modernos**

  Adaptar primero los tests externos para:

  - grids 2×2, 3×3 y casos 4×4 razonables;
  - boundaries soportadas;
  - PEPS-SVD exact/adaptive;
  - ranks explícitos/effectivos;
  - ALS exact/fixed/sampled;
  - cache prefix/suffix;
  - gauges/MCF;
  - SVD projectors;
  - Phi explícito vs implicit;
  - CTM movement;
  - column compression.

  Marcar experimentos inestables de forma explícita; no relajar tolerancias
  globalmente para hacerlos pasar.

- [ ] **PEPS-02 — Implementar specs, geometría y resultado**

  Crear:

  - `PEPSRanks`;
  - `PEPSGeometry`;
  - `PEPSOutputLayout`;
  - `PEPSDecomposition`;
  - layouts de cores y fronteras;
  - contracción/evaluación oracle pequeña.

  Propiedades:

  - terminología rank;
  - `rank` escalar como upper bound normal;
  - ranks horizontales/verticales efectivos internos;
  - shapes OBC y otras boundaries documentadas;
  - grids con `input_dim` heterogéneo;
  - axes de output en coordenadas explícitas o default equiespaciado por
    traversal;
  - validación `n_grid_sites = n_input_sites + n_output_axes`, colisiones,
    orden y flatten/unflatten de labels;
  - output_device/dtype;
  - compatibilidad con constructor `tk.models.PEPS` cuando proceda.

- [ ] **PEPS-03 — Portar PEPS-SVD secuencial**

  Fuente: `peps_svd_sequential`.

  Implementar `PEPSSVD.fit(method="sequential")`:

  - traversal frontier row-major;
  - `truncated_svd` moderno;
  - rank cap global;
  - error acumulado con `sum(discarded_energy)` únicamente en los pasos cuya
    ortogonalidad respecto a la escala original se demuestre; el resto queda
    como diagnóstico local;
  - storage/offload;
  - metrics/events;
  - real/complejo;
  - no helper PEPS de truncación antiguo.

  Comparar reconstrucción y error reportado con tensor denso pequeño.

- [ ] **PEPS-04 — Portar PEPS-SVD jerárquico e inicializadores SVD**

  Fuente: `peps_svd_hierarchical` y
  `_tt_svd_with_storage`.

  Reutilizar `TTSVD` para:

  - TT de columnas;
  - TT verticales;
  - splits de supercores;
  - errores/truncation records.

  Añadir `HierarchicalPEPSInitializer`; no duplicar TT-SVD dentro de PEPS.

- [ ] **PEPS-05 — Adaptar sources, `ALSProblem` y entornos PEPS-ALS**

  Portar conceptos:

  - source dense/callable/indexed normalizada por `TensorSource`;
  - sistemas exactos;
  - fibers de la source;
  - componentes completamente fijos;
  - suffixes/prefix incremental;
  - cache de valores locales por site.

  Implementar adapters a:

  - `TensorSource` + `ALSProblem`;
  - `SampleBatch`/`SampleRefreshPolicy`;
  - `LeastSquaresSolver`;
  - `EnvironmentCache`.

  Mantener la advertencia: un orden lineal PEPS puede hacer crecer frontier;
  el cache es backend PEPS, no promesa de coste TT.

- [ ] **PEPS-06 — Portar PEPS-ALS exacto y sampled**

  Crear `PEPSALS` y `peps_als`.

  Funcionalidad:

  - init random/SVD/cores;
  - fixed cores;
  - exact/callable/sampled/observed;
  - `sample_reuse_sweeps`;
  - fibers/valores de la source cacheados;
  - Tikhonov, column/system scaling, finite checks;
  - damping;
  - normalización;
  - error objetivo por sweep;
  - primary convergence metrics comunes;
  - profile avanzado opt-in.

  Sampling leverage PEPS no se declarará exacto: sin una forma canónica exacta
  de PEPS, una futura estrategia basada en boundaries/CTM será aproximada,
  `exact=False`, y solo se añadirá tras validar primero esos entornos.

  Si un movimiento de gauge invalida suffixes, reconstruir/fallback directo
  como hace el prototipo; nunca usar cache obsoleta.

- [ ] **PEPS-07 — Consolidar gauges y Minimal Canonical Form**

  Crear `PEPSGaugeConditioner` y consolidar duplicados de
  `peps_als.py`/`orbits.py`.

  Separar:

  - QR/SVD local exacta;
  - normalización escalar;
  - SVD projectors;
  - `PEPSOrbit`;
  - Minimal Canonical Form.

  Invariantes:

  - cores fijos no reciben gauges;
  - una factorización exacta se absorbe en un vecino válido;
  - invalidación de cache declarada;
  - MCF opt-in por su coste;
  - complejo y conjugación.

- [ ] **PEPS-08 — Portar column compression e initializers**

  Crear `ColumnCompressionInitializer` y `PEPSRSSInitializer`.

  Portar de forma aislada:

  - adaptive SVD por columnas;
  - projectors;
  - annealing de ranks verticales;
  - reparación ALS por columnas;
  - inicializador jerárquico mediante subtensor cartesiano.

  Unificar sus registros con `TruncationRecord` y métricas comunes. No
  presentar el hierarchical restricted tensor como el algoritmo PEPS-RSS
  definitivo.

- [ ] **PEPS-09 — Portar PEPS-VO**

  Crear `PEPSVO`/`peps_vo` como driver separado:

  - optimización conjunta autograd;
  - source dense y, si se valida, callable/indexed;
  - optimizer configurable;
  - sampled objective correctamente etiquetado;
  - MCF opcional mediante conditioner;
  - convergence/metrics comunes.

  No mezclar optimizer VO en branches de `PEPSALS`.

- [ ] **PEPS-10 — Portar natural gradient**

  Crear `PEPSNaturalGradient`/`peps_natural_gradient` `EXP`:

  - source dense/callable;
  - Gauss–Newton/natural-gradient local;
  - damping/backtracking;
  - sample reuse;
  - Jacobiano explícito como primera ruta;
  - límites de tamaño documentados;
  - finite checks y razones de parada.

  No afirmar escalabilidad hasta implementar operadores Jacobian-vector.

- [ ] **PEPS-11 — Adaptar regiones/Phi comunes a grid PEPS**

  Sustituir el `RegionSketch` local de `peps_rss_ctm.py` por:

  - `SiteRegion` con coordenadas `(row,col)`;
  - `RegionSketch`;
  - `SketchRecursion`;
  - `PhiOperator`;
  - `_EvaluationPlan`.

  Implementar:

  - cuatro direcciones;
  - regions de boundary/frontier;
  - Phi denso;
  - Phi lazy/indexado;
  - fibers para ALS sampled;
  - combinación de region sketches con selector batched;
  - evaluación únicamente de configuraciones finales;
  - `PEPSOutputLayout`: output sites usan `basis`, conservan el orden tensorial
    y nunca se pasan como argumentos a la fuente.

  Tests explícito==lazy==implementación moderna externa en casos pequeños.

- [ ] **PEPS-12 — Portar estado y traversal CTM**

  Reorganizar dataclasses modernas:

  - `BoundaryTriple`;
  - `IncomingEnvironments`;
  - `LocalCTMState`;
  - `EnvironmentCache`;
  - `PathRecord`.

  Integrarlas como tipos internos de `PEPSCTMDriver`, reutilizando métricas
  comunes. Un `PathRecord` que suma errores relativos se etiqueta diagnóstico
  de path, no bound global.

  Portar:

  - movement de boundaries;
  - orientation;
  - incoming environments;
  - local SVD initialization;
  - projectors y lifting;
  - normalización de Phi con regularización consistente.

- [ ] **PEPS-13 — Implementar solver local PEPS-RSS por estrategias**

  Sustituir el solver monolítico por:

  - `PEPSLocalSolver` protocol;
  - `PEPSALSLocalSolver`;
  - `PEPSVOLocalSolver`;
  - `PEPSNaturalGradientLocalSolver`.

  Cada uno recibe Phi/PhiOperator, ranks e incoming gauges; devuelve cores,
  environments y métricas. Las opciones se encapsulan en el solver, no en
  `PEPSRSS.fit`.

- [ ] **PEPS-14 — Implementar `PEPSRSS` y wrappers**

  Componer:

  - source/specs comunes;
  - `PEPSOutputLayout`;
  - regions/Phi;
  - CTM driver;
  - physical fitter;
  - local solver;
  - truncation/metrics/observer.

  API:

  - `PEPSRSS(...).fit(...)`;
  - `peps_rss` default ALS estable;
  - `peps_rss_als`;
  - `peps_rss_vo`;
  - `peps_rss_natural_gradient`;
  - alias temporal `peps_rss_ng` si era público.

  Evitar una firma de decenas de argumentos. La función simple expone opciones
  frecuentes; usuarios avanzados pasan un local solver configurado.

- [ ] **PEPS-15 — Implementar refinability diagnostics**

  Para cada environment fusionado:

  - abrir su estructura física oculta;
  - calcular rank requerido frente al `rank` permitido;
  - guardar singular spectrum resumido/energía descartada;
  - distinguir local reconstruction y PEPS refinability;
  - fallar, truncar o elegir otro opener según policy explícita;
  - propagar el diagnóstico al siguiente site.

  Casos 2×2 del documento `peps_rss_analysis.md` deben reproducir el fallo
  histórico y detectar su causa.

- [ ] **PEPS-16 — Implementar checkerboard serial**

  Antes de procesos:

  1. resolver sites blancos con gauges libres;
  2. canonicalizar/estabilizar outgoing gauges;
  3. propagar a vecinos;
  4. resolver negros con incoming gauges fijos;
  5. comprobar compatibilidad/refinabilidad;
  6. repetir por bloques si procede.

  Comparar con traversal CTM serial. Mantener fallback y etiqueta `EXP`.

- [ ] **PEPS-17 — Paralelizar PEPS**

  Reutilizar `ExecutionBackend`:

  - una task por site blanco/bloque;
  - barrera y comunicación de gauges;
  - tasks negras;
  - evaluation store global;
  - menos workers que sites;
  - events agregados;
  - serial checkerboard como oracle.

  No activar por default hasta demostrar estabilidad y speedup.

- [ ] **PEPS-18 — Limpiar legacy y documentación**

  - reemplazar fachada dinámica por imports explícitos;
  - no copiar `src/old`;
  - mantener wrapper deprecated de `peps_rss` histórico si es necesario;
  - eliminar copias TT/TR/BLOSTR/utils;
  - portar docs útiles de CTM, implicit ALS y failure modes;
  - documentar madurez de cada backend;
  - ejemplos pequeños reproducibles.

- [ ] **PEPS-19 — Gate final del proyecto**

  Ejecutar:

  - suite completa TensorKrowch;
  - suite PEPS portada;
  - equivalencia explicit/implicit;
  - fixed cores/gauges;
  - refinability failures;
  - serial/parallel;
  - real/complejo/device;
  - benchmarks 2×2–4×4 apropiados.

  Revisar que ninguna API TT/TR importe módulos PEPS y que los protocolos
  comunes no contengan branches específicos de grid.

#### Entregable de la fase

PEPS-SVD/ALS/VO/natural-gradient y PEPS-RSS separados, con infraestructura
común y moderna, legacy aislado, diagnósticos de refinability y un schedule
checkerboard validado antes de paralelizar.

---

## 8. Justificación del diseño híbrido

El diseño adopta ideas de otras librerías sin copiar su superficie:

- [TensorLy](https://tensorly.org/dev/modules/api.html) ofrece funciones de
  descomposición y también clases configurables. Las funciones son cómodas
  para una ejecución; las clases encajan con repetición y pipelines. De ahí la
  pareja `tt_rss(...)` / `TTRSS(...).fit(...)`.
- [TensorCrossInterpolation.jl](https://tensor4all.org/TensorCrossInterpolation.jl/dev/documentation/)
  usa estructuras mutables porque un interpolante adaptativo acumula pivots,
  ranks y matrices y puede seguir refinándose. Aquí el estado mutable se limita
  al objeto problema y al `_FitContext`; el resultado no se convierte en un
  gran solver mutable.
- [SeeMPS](https://pypi.org/project/seemps/) integra algoritmos alrededor de
  objetos MPS/QTT ricos. TensorKrowch ya tiene `models`; duplicar ese nivel en
  `decompositions` introduciría grafos y caminos de evaluación innecesarios.
  Por ello se devuelven raw cores o resultados ligeros.

Objetivos del híbrido:

| Necesidad | Función | Clase algoritmo | Clase resultado |
|---|---:|---:|---:|
| llamada corta | excelente | innecesaria | invisible |
| repetir fits sobre misma función | recrea estado | excelente | un resultado por fit |
| estrategias avanzadas | firma crecería | composición limpia | solo registra |
| compatibilidad con `tk.models` | lista directa | `.fit().cores` | `.cores` |
| inspección de métricas | `return_info` | `.fit().metrics` | excelente |
| paralelización | delega | posee plan/contexto | sin estado de ejecución |

---

## 9. Inventario de origen y mapa de migración

### 9.1 TensorKrowch actual

Snapshot inspeccionado al preparar este documento:

```text
branch: tt_rss
commit: 1e19ff2

tensorkrowch/decompositions/
├── __init__.py
├── svd_decompositions.py
└── tt_decompositions.py
```

Exports actuales: `vec_to_mps`, `mat_to_mpo`, `tt_rss`.

Cobertura:

- `tests/decompositions/test_svd_decompositions.py`;
- `tests/test_utils.py` para `truncated_svd`;
- tests indirectos de operations/models;
- no existe suite TT-RSS.

Mapa:

| Pieza actual | Destino |
|---|---|
| `vec_to_mps` | `svd/tt.py:TTSVD` + wrapper deprecated |
| `mat_to_mpo` | `svd/ttm.py:TTMSVD` + wrapper deprecated |
| `tt_rss` | `sketching/tt.py:TTRSS` + wrapper funcional |
| `extend_with_output` | `_OutputSpec` + evaluation plan |
| `sketching` | `PhiOperator` |
| `trimming` | `truncated_svd` instrumentado |
| `create_projector` | `RegionSketch.recursive_projector` |
| `val_error` | `TTDecomposition.error` / sample error RSS |

`svd_decompositions.py` se elimina al cerrar la Fase 1;
`tt_decompositions.py` se elimina al cerrar la Fase 3. Las funciones públicas
compatibles sobreviven en `decompositions.__init__` y delegan en los módulos
nuevos.

### 9.2 Proyecto `tt2tr`

Referencia inspeccionada: `/Users/jose/VSCodeProjects/tt2tr`, rama `main`,
commit `a1a4b82`.

| Archivo/pieza | Uso futuro |
|---|---|
| `src/blostr.py:tr_svd` | algoritmo a reescribir sobre TT-SVD |
| `_build_tr_env` | oracle de tests TR-ALS |
| `_tr_als_impl` | referencia funcional, no cache |
| `tt2tr_fixed_rank` | referencia preferente TT→TR |
| `tr_blostr_first`/`tr_blostr_svd` | `BLOSTRLoopOpener`, experimental |
| `src/tr_rss.py:_extend_A_left/right` | `SketchGaugeRecursion` |
| rank/block discovery de `tr_rss` | `CentralBlockSelector`/estimator |
| `src/utils.py:fidelity` | overlap estable, contrastado con versión L2G |
| `mps_pbc_all_together.py` | no migrar |
| `src/orbits.py` | fuera del refactor TT/TR inicial |

No hay tests automatizados en este proyecto; toda pieza necesita tests nuevos.

Los helpers externos llamados `fidelity` se usarán como referencia de
contracción/escalado, no como oracle semántico: algunos devuelven el módulo del
overlap normalizado. La API nueva define fidelity únicamente como
`abs(normalized_overlap) ** 2`.

### 9.3 Proyecto `peps-rss`

Referencia: `/Users/jose/VSCodeProjects/peps-rss`, rama `main`, commit
`78e89b2`. La rama TensorKrowch `peps_rss` estaba en `f05f406`.

| Pieza moderna | Destino en rama `peps_rss` |
|---|---|
| `peps_core.py` | contratos de cores/ranks/layout; no sus helpers RSS antiguos |
| `peps_svd.py` | `peps/svd.py` |
| `peps_als.py` | `peps/als.py`, common solver/caches |
| `peps_vo.py` | `peps/vo.py` |
| `peps_natural_gradient.py` | `peps/natural_gradient.py` |
| `peps_column_compression.py` | `peps/initializers.py` |
| `peps_rss_hierarchical.py` | initializer, no engine final |
| `peps_rss_ctm.py` | `peps/ctm.py` + adapters Region/Phi |
| `orbits.py:PepsOrbit` | `peps/gauges.py` |
| tests externos | caracterización antes del port |

`src/old/*`, la fachada dinámica `peps_rss_als.py` y las copias TT/TR no se
migran.

### 9.4 Prototipos VMC y L2G

Referencias:

- `/Users/jose/VSCodeProjects/vmc-rss-solvers`, commit `1290b2b`;
- `/Users/jose/VSCodeProjects/l2g-tn-solvers`, commit `77f434e`.

Ideas a portar con tests nuevos:

- `tt_rss_vmc.py:{AmplitudeTable,PhiSketch,build_phi_sketch,
  construct_tt_cores_from_phis}` como referencia de puntos únicos,
  incidencias y ensamblado;
- transforms global/local;
- `mps_mpo_linear_rss.py:{_build_phi_from_mps,
  _build_block_phi_from_mps}` como referencia de Phi desde TT;
- `mps_mpo_linear_rss.py:{mps_inner_product_scaled,relative_mps_error}` para
  contrastar overlap/norma/fidelity con scaling;
- evaluación de TT por left/right environments.

No se planifica ningún solver lineal procedente de `l2g-tn-solvers`; el
proyecto solo se usa como referencia de kernels de evaluación y métricas que
también resulten útiles para las descomposiciones.

No se portan:

- nombres de aplicación al núcleo genérico;
- callbacks order-dependent;
- inicializaciones aleatorias de valores sparse ausentes;
- generadores de experimentos;
- utilidades SVD/norma duplicadas.

---

## 10. Matriz mínima de cobertura

| Propiedad | SVD | ALS | Sketching | TT→TR | QTT/QTR | PEPS |
|---|---:|---:|---:|---:|---:|---:|
| `input_dim`/`output_dim` heterogéneos | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| real y complejo | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CPU y CUDA condicional | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ranks uno/adaptativos | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| error absoluto/relativo | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| generator determinista | n/a | ✓ | ✓ | ✓ si random | ✓ | ✓ |
| output tensorial múltiple | n/a | según source/problema | ✓ | n/a | ✓ | ✓ RSS |
| source callable | n/a | ✓ | ✓ | n/a | ✓ | ✓ |
| source sparse/empírica | n/a | completion | ✓ | n/a | ✓ | ✓ RSS |
| source TT | input cores | init/source | ✓ | ✓ | ✓ | initializer |
| oracle denso pequeño | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| serial == paralelo | dependencias | Fase 4 | Fase 4 | Fase 4 | Fase 4 | Fase 5 |

Tests comunes adicionales:

- un site, dos sites y boundaries;
- source/función cero;
- NaN/Inf;
- rank cap mayor que rank algebraico;
- criterios de truncación combinados;
- denominadores de error cero;
- inputs no contiguos;
- outputs en orden no trivial;
- offload CPU y conservación de dtype;
- `verbose=0` sin stdout;
- events jerárquicos en `verbose=1..3`;
- funciones directas y clases producen resultados equivalentes;
- aliases emiten un único warning con stack correcto.

---

## 11. Medidas de rendimiento que deben acompañar al refactor

Las métricas automáticas no deben dominar el algoritmo. Se clasifican:

### Sin coste asintótico adicional

- singular energy descartada;
- ranks y shapes;
- tiempo alrededor de subrutinas ya ejecutadas;
- número de puntos/evaluaciones;
- residuos devueltos por least squares cuando puedan obtenerse sin nueva
  contracción;
- cache hits.

### Coste adicional pequeño y opt-in/default contextual

- error sobre `sketch_samples`;
- condición mediante singular values ya disponibles;
- overlap/fidelity TT→TR;
- error de objetivo por sweep si exige contracción completa.

### Solo diagnóstico explícito

- singular spectra completos;
- condición SVD adicional;
- dense reconstruction;
- validation samples independientes;
- memoria pico precisa con sincronización CUDA;
- error global PEPS.

Benchmarks registrarán tanto el fast path
`verbose=0, collect_metrics=False/return_info=False` como el modo diagnóstico.

---

## 12. Gates y decisiones aún no cerradas

Estas decisiones se posponen deliberadamente hasta tener una implementación o
benchmark concreto:

1. **`return_result=True` en funciones directas.** La clase avanzada ya cubre
   el caso; decidir en `SVD-06` si aporta suficiente valor.
2. **Duración exacta de aliases deprecated.** Mínimo un ciclo de versión;
   fijar versión de eliminación al preparar release.
3. **BLOSTR como método completo público.** Resuelto en `BLOSTR-01`:
   `tr_blostr` y `BLOSTRLoopOpener` se exponen como APIs experimentales sobre
   un único motor caracterizado para ranks uniformes.
4. **Framework distribuido.** Se elige en `PAR-10`.
5. **Schedule par/impar como default TR.** Solo si `RSS-13/PAR-06` superan al
   center-out.
6. **Checkerboard PEPS como default.** Solo tras refinability y benchmarks.
7. **Autograd a través de RSS entero.** Fuera del alcance inicial; solo fitter
   entrenable local.
8. **Persistencia de caches entre fits.** No se implementa hasta demostrar una
    cache inmutable segura y útil.

Decisiones ya cerradas y que no deben reabrirse sin evidencia:

- `rank` escalar como upper bound público normal y secuencia opcional para
  fijar enlaces TR; el último elemento es el enlace cíclico;
- vocabulario TT/TR/TTM/ranks;
- argumentos modernos de `truncated_svd`;
- funciones sencillas + clases avanzadas;
- resultados ligeros/raw cores, no nuevos modelos ricos;
- multiple output sites con `basis`;
- `GlobalValueTransform`/`LocalValueTransform`;
- leverage TR producto como default y leverage TR exacto como método
  experimental opt-in mediante `leverage_method="exact"`;
- `.quantized`, `qtt_rss` y `qtr_rss`, no `from_qtt`;
- `MarginalSketch` con preset `.markov(...)`;
- serial antes de paralelo;
- PEPS solo en rama `peps_rss`.

---

## 13. Fuera de alcance o antipatrones explícitos

- Sustituir `tk.models.MPS/MPO/PEPS` por las clases resultado.
- Usar términos MPS/MPO/bond dims en la nueva API de decompositions.
- Exigir listas de ranks por site en la API normal.
- Copiar monolitos externos y “limpiarlos después”.
- Una clase `RegionSketch` que también sea source, Phi, solver y cache.
- Una callback ambigua que mezcle transformaciones globales y locales.
- Un booleano único que haga equivalentes Sampled, Marginal y TTStack.
- Calcular Phi completo antes de seleccionar fibers cuando la ruta lazy esté
  disponible.
- Usar batch error aleatorio/leverage como convergencia global.
- Descontraer entornos TR/PEPS mediante pseudoinversa.
- Paralelizar antes de verificar el schedule serial equivalente.
- Ejecutar código PEPS experimental en la rama TT/TR principal.

---

## 14. Referencias algorítmicas

### Política de referencias en la implementación

Todo método cuyo algoritmo proceda de un artículo debe identificarlo en el
docstring de su clase principal y de su función pública. La referencia debe
incluir autores, título, año y un enlace estable, e indicar con precisión:

- cuando exista una versión pública, usar un enlace abierto y la sintaxis reST
  habitual de la librería: `` `paper <https://...>`_ ``;

- qué algoritmo, ecuación o sección del artículo se implementa;
- qué extensiones o adaptaciones son propias de TensorKrowch;
- qué garantías del artículo siguen siendo aplicables y cuáles no.

Esta regla se aplicará de forma general a SVD/ALS, BLOSTR, TT→TR, RS/RSS,
QTT, solvers y futuras descomposiciones PEPS. Los helpers privados solo
repetirán la referencia cuando implementen por sí mismos un subalgoritmo
publicado que no quede inequívocamente documentado por su caller principal.

- TT recursive sketching y marginales Markov:
  [arXiv:2202.11788](https://arxiv.org/abs/2202.11788).
- TTStack:
  [arXiv:2603.11009](https://arxiv.org/abs/2603.11009).
- Transformaciones locales sobre distribuciones empíricas:
  [arXiv:2212.00759](https://arxiv.org/abs/2212.00759).
- QTT-Tucker:
  [SIAM DOI 10.1137/120882597](https://doi.org/10.1137/120882597).
- Corrección/errata del algoritmo y bounds QTT-Tucker:
  [SIAM DOI 10.1137/15M104089X](https://doi.org/10.1137/15M104089X).
- Grid computacional y coordenadas curvilíneas:
  [arXiv:2507.05222](https://arxiv.org/abs/2507.05222).
- Grids variables/remapeo de dominio:
  [arXiv:2509.10142](https://arxiv.org/abs/2509.10142).
- Leverage sampling TT:
  [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/hash/86c1fd74fa25bd6be0072937803e0bd1-Paper-Conference.pdf).
- Leverage sampling aproximado para TR:
  [arXiv:2010.08581](https://arxiv.org/abs/2010.08581).
- Sampling leverage para tensor networks:
  [arXiv:2210.03828](https://arxiv.org/abs/2210.03828).

Antes de implementar un método procedente de un artículo se debe volver a
consultar la versión concreta y registrar qué parte es algoritmo publicado y
qué parte es adaptación propia.

---

## 15. Definición global de proyecto terminado

El proyecto se considerará completo cuando:

- todas las tareas no opcionales estén `[x]`;
- las funciones directas y clases avanzadas estén documentadas;
- aliases tengan calendario de deprecación;
- TT/TTM/TR SVD, ALS y RSS pasen sus oracles;
- TT→TR calcule fidelity estable por defecto;
- completion y leverage tengan semántica correcta;
- Region/Phi/source/transform/QTT estén cubiertos de forma independiente;
- las rutas serial/paralela sean equivalentes;
- PEPS moderno esté portado únicamente en `peps_rss`;
- código legacy y duplicados estén aislados o eliminados;
- la suite completa, `git diff --check`, docs y benchmarks estén limpios;
- el usuario haya confirmado el último gate y los commits correspondientes.

Cada vez que el usuario confirme una subtarea, se actualizarán:

1. su checkbox;
2. el resumen de progreso de la sección 1;
3. cualquier decisión de la sección 12 que haya quedado resuelta;
4. el commit o referencia de implementación, anotado bajo el TODO si resulta
   útil.
