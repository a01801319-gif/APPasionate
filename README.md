# APPasionate
# 📈 App de Elasticidad de Demanda & Pricing Dinámico

Esta aplicación interactiva de ciencia de datos está diseñada para limpiar bases de datos transaccionales, estimar modelos de elasticidad precio de la demanda a nivel SKU/Trimestre mediante modelos estadísticos avanzados (`statsmodels`) y simular escenarios óptimos de precios y promociones.

## 🚀 Características Clave
- **Carga y Diagnóstico Automatizado:** Cuenta con un pipeline de limpieza de datos integrado que evalúa la calidad de la información a través de un semáforo de control analítico.
- **Modelación de Elasticidad (YoY):** Cálculo e interpretación automatizada de coeficientes $\beta$ y elasticidades asociadas a variables geográficas y temporales.
- **Simulador de Pricing Dinámico:** Proyección de impactos financieros (ingresos, volúmenes y márgenes) bajo múltiples escenarios comerciales (p. ej., +15%, 3x2, 2do al 50%).

## 🛠️ Estructura del Repositorio
- `app.py`: Archivo principal de la aplicación desarrollado en Streamlit.
- `requirements.txt`: Dependencias del entorno de Python necesarias para la ejecución.
- `nse_predeterminado.csv`: Base de datos de Niveles Socioeconómicos por municipio/estado utilizada por defecto.

## 📦 Instalación Local
Si deseas clonar este repositorio y ejecutarlo localmente, sigue estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
Instalar dependencias:
Bash
pip install -r requirements.txt
Ejecutar la aplicación:
Bash
streamlit run app.py
