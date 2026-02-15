#pragma once

#include <vector>
#include <string>
#include <functional>
#include <random>
#include <stdexcept>
#include <cmath>

namespace f_vigs_slam
{
    /**
     * @class KeyframeSelector
     * @brief Selector genérico de keyframes para optimización con múltiples métodos
     * 
     * Permite cambiar dinámicamente entre diferentes estrategias de selección:
     * - "beta_binomial": Muestreo Beta-Binomial (exploración controlada)
     * - "uniform": Muestreo uniforme
     * - "exponential": Muestreo exponencial (bias hacia recientes)
     * - "sliding_window": Ventana deslizante
     * - "covisibility": Selección por covisibilidad (requiere argumentos externos)
     */
    class KeyframeSelector
    {
    public:
        KeyframeSelector();
        ~KeyframeSelector();
        
        /**
         * @brief Muestrean índices de keyframes
         * 
         * @param n             Número de keyframes a seleccionar
         * @param total_kfs     Total de keyframes disponibles (índices: 0 a total_kfs-1)
         * @param method        Método de selección (default: "beta_binomial")
         * @param args          Argumentos específicos del método (ver detalles abajo)
         * @return              Vector de índices seleccionados (sorted)
         * 
         * MÉTODOS DISPONIBLES:
         * 
         * - "gumbel"
         *   Args: alpha (float), beta (float)
         *   Ej: sample(4, 10, "gumbel", {0.7f, 2.0f})
         * 
         * - "beta_binomial"
         *   Args: alpha (float), beta (float)
         *   Ej: sample(4, 10, "beta_binomial", {0.7f, 2.0f})
         *   Nota: muestreo con reemplazo (independiente) como en VIGS-Fusion
         * 
         * - "uniform"
         *   Args: ninguno (ignorados si existen)
         *   Ej: sample(4, 10, "uniform", {})
         * 
         * - "exponential"
         *   Args: lambda (float, decay rate)
         *   Ej: sample(4, 10, "exponential", {1.0f})
         * 
         * - "sliding_window"
         *   Args: window_size (int)
         *   Ej: sample(4, 10, "sliding_window", {5})
         * 
         */
        std::vector<int> sample(int n, int total_kfs, 
                                 const std::string &method = "beta_binomial",
                                 const std::vector<float> &args = {});


        // ============================================================
        // Configuración
        // ============================================================
        
        void set_seed(unsigned long seed);
        void set_recent_bias_weight(float weight);  // Cómo de sesgado hacia recientes (default: 0.5)

    private:
        // ============================================================
        // Miembros
        // ============================================================
        
        std::mt19937 rng_;
        float recent_bias_weight_ = 0.5f;

        // ============================================================
        // Métodos de muestreo privados
        // ============================================================
        
        std::vector<int> sample_gumbel(int n, int total_kfs, const std::vector<float> &args);
        std::vector<int> sample_beta_binomial(int n, int total_kfs, const std::vector<float> &args);
        std::vector<int> sample_uniform(int n, int total_kfs, const std::vector<float> &args);
        std::vector<int> sample_exponential(int n, int total_kfs, const std::vector<float> &args);
        std::vector<int> sample_sliding_window(int n, int total_kfs, const std::vector<float> &args);
        // ============================================================
        // Utilidades
        // ============================================================
        
        /**
         * @brief Muestrea de distribución Beta(alpha, beta)
         * Usa método de composición con Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
         */
        float sample_beta(float alpha, float beta);
        
        /**
         * @brief Muestrea de distribución Gamma(shape, scale=1)
         * Usa transformación de exponencial
         */
        float sample_gamma(float shape);

        /**
         * @brief Muestrea n índices sin reemplazo del rango [0, total)
         */
        std::vector<int> sample_without_replacement(int n, int total);
    };

} // namespace f_vigs_slam