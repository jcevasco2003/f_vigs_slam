/**
 * @file KEYFRAME_SELECTOR_INTEGRATION.md
 * @brief Guía de integración de KeyframeSelector en GSSlam
 */

# KeyframeSelector - Guía de Integración

## Descripción General

`KeyframeSelector` es una clase genérica que permite elegir dinámicamente cómo seleccionar keyframes para optimización en tu SLAM. Soporta 5 métodos diferentes y es fácil agregar más.

## Métodos Disponibles

### 1. **beta_binomial** (default - VIGS-Fusion)
```cpp
// Parámetros: alpha (float), beta (float)
auto indices = selector.sample(4, keyframes.size(), 
                               "beta_binomial", {0.7f, 2.0f});
```

**Características:**
- α=0.7 → bias del 70% hacia keyframes recientes
- β=2.0 → permite exploración de keyframes históricos
- Ideal para balance entre optimización local y consistencia global

**Matemática:**
$$p \sim \text{Beta}(\alpha, \beta) \quad \Rightarrow \quad k \sim \text{Binomial}(n-1, p)$$

### 2. **uniform**
```cpp
// Sin parámetros
auto indices = selector.sample(4, keyframes.size(), "uniform", {});
```

**Características:**
- Muestreo completamente aleatorio
- Útil para debugging o baseline

### 3. **exponential**
```cpp
// Parámetros: lambda (float, decay rate)
auto indices = selector.sample(4, keyframes.size(), 
                               "exponential", {2.0f});
```

**Características:**
- λ > 1: fuerte bias hacia recientes
- λ = 0.5: débil bias (casi uniforme)
- Determinístico pero sesgado

**Matemática:**
$$P(k) \propto e^{\lambda \cdot k/n}$$

### 4. **sliding_window**
```cpp
// Parámetros: window_size (float)
auto indices = selector.sample(4, keyframes.size(), 
                               "sliding_window", {8.0f});
```

**Características:**
- Considera solo los últimos N keyframes
- Perfecto para tracking local
- Evita cambios drásticos en el mapa

### 5. **covisibility**
```cpp
// Parámetros: threshold (float)
// Primero proporcionar datos:
selector.set_covisibility_scores(covis_ratio_vector);

auto indices = selector.sample(4, keyframes.size(), 
                               "covisibility", {0.3f});
```

**Características:**
- Selecciona keyframes con alta covisibilidad con frame actual
- Threshold adaptable
- Ideal para optimización visual consistente

---

## Integración en GSSlam

### Paso 1: Agregar miembro en GSSlam.cuh

```cpp
// En f_vigs_slam/include/f_vigs_slam/GSSlam.cuh

class GSSlam {
private:
    KeyframeSelector kf_selector_;
    std::string kf_selection_method_;  // "beta_binomial", "exponential", etc
    std::vector<float> kf_selection_args_;
    
    // ...
};
```

### Paso 2: Inicializar en constructor

```cpp
// En src/GSSlam.cu constructor

GSSlam::GSSlam() : 
    // ... otros miembros ...
    kf_selector_(0),  // seed = 0 (random)
    kf_selection_method_("beta_binomial"),
    kf_selection_args_({0.7f, 2.0f})  // default: β-Binomial VIGS-Fusion
{
    // ...
}
```

### Paso 3: Crear setter para cambiar método

```cpp
// En GSSlam.hpp

void set_keyframe_selection_method(const std::string &method, 
                                    const std::vector<float> &args = {})
{
    kf_selection_method_ = method;
    kf_selection_args_ = args;
}
```

### Paso 4: Usar en compute() o loop de optimización

**Opción A: Dentro de `compute()`**

```cpp
// En compute() después de PASO 8 (covisibilidad)

float covis_ratio = computeCovisibilityRatio();

if (covis_ratio < covisibility_threshold_) {
    // Nuevo keyframe
    prune();
    addKeyframe();
    densify();
    
    // AQUÍ: Seleccionar keyframes para optimización
    std::vector<int> selected_kfs = kf_selector_.sample(
        4,  // seleccionar 4 keyframes
        keyframes_.size(),
        kf_selection_method_,
        kf_selection_args_
    );
    
    // Optimizar solo los seleccionados
    for (int kf_idx : selected_kfs) {
        optimizeGaussians(keyframes_[kf_idx], eta);
    }
}
```

**Opción B: En un loop de optimización dedicado**

```cpp
// En un método optimizationLoop() asincrónico

void GSSlam::optimizationLoop()
{
    while (running_) {
        if (keyframes_.size() < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Seleccionar keyframes
        auto selected = kf_selector_.sample(
            4, 
            keyframes_.size(),
            kf_selection_method_,
            kf_selection_args_
        );
        
        // Optimizar
        for (int idx : selected) {
            optimizeGaussians(keyframes_[idx], eta);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

---

## Ejemplo Completo: Cambiar método en runtime

```cpp
// Crear selector
KeyframeSelector selector(42);

// Comenzar con Beta-Binomial
auto kfs = selector.sample(4, 20, "beta_binomial", {0.7f, 2.0f});
std::cout << "Beta-Binomial: ";
for (int idx : kfs) std::cout << idx << " ";
std::cout << "\n";

// Cambiar a exponencial después de X frames
kfs = selector.sample(4, 20, "exponential", {1.5f});
std::cout << "Exponential: ";
for (int idx : kfs) std::cout << idx << " ";
std::cout << "\n";

// Cambiar a covisibilidad si disponible
std::vector<float> covis = {0.1, 0.3, 0.5, 0.7, 0.8, 0.9, ...};
selector.set_covisibility_scores(covis);
kfs = selector.sample(4, 20, "covisibility", {0.3f});
std::cout << "Covisibility: ";
for (int idx : kfs) std::cout << idx << " ";
std::cout << "\n";
```

---

## Agregar un Nuevo Método

Es muy simple. Solo necesitas:

1. **Agregar declaración privada en `.hpp`:**
   ```cpp
   std::vector<int> sample_my_custom_method(int n, int total_kfs, 
                                            const std::vector<float> &args);
   ```

2. **Implementar en `.cpp`:**
   ```cpp
   std::vector<int> KeyframeSelector::sample_my_custom_method(int n, int total_kfs, 
                                                              const std::vector<float> &args)
   {
       // Tu lógica aquí
       // Retornar vector de índices [0, 1, 3, 7, ...]
   }
   ```

3. **Agregar dispatch en `sample()`:**
   ```cpp
   else if (method == "my_custom_method") {
       return sample_my_custom_method(n, total_kfs, args);
   }
   ```

4. **¡Listo!**
   ```cpp
   auto indices = selector.sample(4, 20, "my_custom_method", {arg1, arg2});
   ```

---

## Notas de Implementación

### Rendimiento
- **Beta-Binomial**: O(n log n) - más costoso pero mejor exploración
- **Uniforme**: O(n) - muy rápido
- **Exponencial**: O(n log n) - intermedio
- **Sliding Window**: O(n) - muy rápido
- **Covisibilidad**: O(n log n) - depende de datos externos

### Reproducibilidad
```cpp
// Para resultados determinísticos (testing):
selector.set_seed(42);  // seed fijo

// Para aleatorio (producción):
selector.set_seed(0);   // seed aleatorio del dispositivo
```

### Datos Externos (Covisibilidad)
```cpp
// Opción 1: Vector de scores (respecto último KF)
std::vector<float> scores(total_kfs);
// calcular scores...
selector.set_covisibility_scores(scores);

// Opción 2: Matriz completa (para análisis futuro)
std::vector<std::vector<float>> matrix(total_kfs, 
                                       std::vector<float>(total_kfs));
// calcular matriz...
selector.set_covisibility_data(matrix);
```

---

## Parámetros Recomendados por Escenario

| Escenario | Método | Args | Razón |
|-----------|--------|------|-------|
| Mapping rápido | sliding_window | {5.0} | Reciente + compacto |
| Robustez global | beta_binomial | {0.7, 2.0} | VIGS-Fusion proven |
| Relocalization | covisibility | {0.4} | Coherencia visual |
| Loop closure | exponential | {1.0} | Historical recovery |
| Testing | uniform | {} | Baseline neutral |

