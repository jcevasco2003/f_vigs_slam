/**
 * @file KeyframeSelector_example.cpp
 * @brief Ejemplos de uso de KeyframeSelector con diferentes métodos
 */

#include <f_vigs_slam/KeyframeSelector.hpp>
#include <iostream>
#include <iomanip>

using namespace f_vigs_slam;

void print_selection(const std::string &method, const std::vector<int> &indices) {
    std::cout << "  [" << method << "] Seleccionados: ";
    for (int idx : indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Crear selector con seed fijo para reproducibilidad
    KeyframeSelector selector(42);

    int total_keyframes = 20;
    int num_to_select = 4;

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "KeyframeSelector - Ejemplos de Uso\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    std::cout << "Total de keyframes disponibles: " << total_keyframes << "\n";
    std::cout << "Keyframes a seleccionar: " << num_to_select << "\n\n";

    // ═══════════════════════════════════════════════════════════════
    // 1. MÉTODO: Beta-Binomial (default VIGS-Fusion)
    // ═══════════════════════════════════════════════════════════════
    std::cout << "1. BETA-BINOMIAL (α=0.7, β=2.0)\n";
    std::cout << "   Descripción: Exploración controlada con bias hacia recientes\n";
    
    auto result = selector.sample(num_to_select, total_keyframes, 
                                   "beta_binomial", {0.7f, 2.0f});
    print_selection("beta_binomial", result);
    std::cout << "\n";

    // ═══════════════════════════════════════════════════════════════
    // 2. MÉTODO: Uniforme
    // ═══════════════════════════════════════════════════════════════
    std::cout << "2. UNIFORME\n";
    std::cout << "   Descripción: Muestreo aleatorio sin preferencia\n";
    
    result = selector.sample(num_to_select, total_keyframes, "uniform", {});
    print_selection("uniform", result);
    std::cout << "\n";

    // ═══════════════════════════════════════════════════════════════
    // 3. MÉTODO: Exponencial
    // ═══════════════════════════════════════════════════════════════
    std::cout << "3. EXPONENCIAL (λ=2.0)\n";
    std::cout << "   Descripción: Bias fuerte hacia keyframes más recientes\n";
    
    result = selector.sample(num_to_select, total_keyframes, 
                             "exponential", {2.0f});
    print_selection("exponential", result);
    std::cout << "\n";

    // ═══════════════════════════════════════════════════════════════
    // 4. MÉTODO: Sliding Window
    // ═══════════════════════════════════════════════════════════════
    std::cout << "4. SLIDING WINDOW (tamaño=8)\n";
    std::cout << "   Descripción: Ventana deslizante sobre keyframes recientes\n";
    
    result = selector.sample(num_to_select, total_keyframes, 
                             "sliding_window", {8.0f});
    print_selection("sliding_window", result);
    std::cout << "\n";

    // ═══════════════════════════════════════════════════════════════
    // 5. MÉTODO: Covisibilidad (con scores)
    // ═══════════════════════════════════════════════════════════════
    std::cout << "5. COVISIBILIDAD (threshold=0.3)\n";
    std::cout << "   Descripción: Selección basada en covisibilidad con último KF\n";
    
    // Crear scores simulados: decrecer hacia el pasado
    std::vector<float> covis_scores(total_keyframes);
    for (int i = 0; i < total_keyframes; i++) {
        // Score simulado: gaussiana con máximo en el último keyframe
        float dist = total_keyframes - 1 - i;
        covis_scores[i] = std::exp(-dist * dist / 50.0f);
    }
    
    selector.set_covisibility_scores(covis_scores);
    result = selector.sample(num_to_select, total_keyframes, 
                             "covisibility", {0.3f});
    print_selection("covisibility", result);
    std::cout << "\n";

    // ═══════════════════════════════════════════════════════════════
    // Ejemplo de uso genérico (en código real)
    // ═══════════════════════════════════════════════════════════════
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "EJEMPLO: Cómo integrar en tu código SLAM\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    std::cout << "// En GSSlam.cpp:\n\n";
    
    std::cout << "KeyframeSelector kf_selector(0);\n\n";
    
    std::cout << "// Opción 1: Beta-Binomial (VIGS-Fusion style)\n";
    std::cout << "auto selected = kf_selector.sample(4, keyframes_.size(),\n";
    std::cout << "                                    \"beta_binomial\", {0.7f, 2.0f});\n\n";
    
    std::cout << "// Opción 2: Cambiar dinámicamente a exponencial\n";
    std::cout << "selected = kf_selector.sample(4, keyframes_.size(),\n";
    std::cout << "                               \"exponential\", {1.5f});\n\n";
    
    std::cout << "// Opción 3: Usar covisibilidad\n";
    std::cout << "kf_selector.set_covisibility_scores(covisibility_ratios);\n";
    std::cout << "selected = kf_selector.sample(4, keyframes_.size(),\n";
    std::cout << "                               \"covisibility\", {0.4f});\n\n";
    
    std::cout << "// Optimizar los keyframes seleccionados\n";
    std::cout << "for (int idx : selected) {\n";
    std::cout << "    optimizeGaussians(keyframes_[idx], eta);\n";
    std::cout << "}\n\n";

    return 0;
}
