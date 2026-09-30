"""Neurona artificial para decidir si se debe regar una planta.

Entradas: humedad del suelo (%) y temperatura ambiental (°C).
Salida: 1 = regar, 0 = no regar.
"""

import numpy as np

# ---------------------------------------------------------------
# 1. Datos de entrenamiento
# ---------------------------------------------------------------
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# ---------------------------------------------------------------
# 2. Normalización
# ---------------------------------------------------------------
# La humedad se encuentra aproximadamente entre 0 y 100.
# La temperatura se encuentra aproximadamente entre 0 y 50.
#
# Dividimos cada columna por su valor máximo esperado
# para dejar los datos en una escala cercana a 0 y 1.
escala = np.array([100, 50])

X_normalizado = X / escala


# ---------------------------------------------------------------
# 3. Función de activación
# ---------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------
# 4. Entrenamiento
# ---------------------------------------------------------------
def entrenar(tasa_aprendizaje, epocas, semilla=42, mostrar_cada=None):
    """Entrena la neurona y devuelve pesos, sesgo y error final (MSE)."""
    rng = np.random.default_rng(semilla)
    pesos = rng.uniform(-1, 1, size=(2, 1))  # [peso_humedad, peso_temperatura]
    sesgo = rng.uniform(-1, 1)
    n = len(X_normalizado)

    for epoca in range(1, epocas + 1):
        # Propagación hacia adelante
        z = X_normalizado @ pesos + sesgo
        probabilidades = sigmoide(z)

        # Error
        error = probabilidades - y
        mse = np.mean(error ** 2)

        # Gradientes (derivada del MSE con respecto a z, usando la sigmoide)
        gradiente_z = error * probabilidades * (1 - probabilidades)
        gradiente_pesos = X_normalizado.T @ gradiente_z / n
        gradiente_sesgo = np.mean(gradiente_z)

        # Actualización
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar_cada and (epoca % mostrar_cada == 0 or epoca == 1):
            print(f"Época {epoca:>6} | Error (MSE): {mse:.6f}")

    probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
    error_final = np.mean((probabilidades - y) ** 2)
    return pesos, sesgo, error_final


def predecir(datos, pesos, sesgo, umbral=0.5):
    """Normaliza con la MISMA escala y devuelve (probabilidad, decisión)."""
    datos_normalizados = np.array(datos, dtype=float) / escala
    probabilidad = sigmoide(datos_normalizados @ pesos + sesgo)
    respuesta = (probabilidad >= umbral).astype(int)
    return probabilidad, respuesta


def texto_decision(valor):
    return "Regar" if valor == 1 else "No regar"


# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Datos originales (humedad %, temperatura °C):")
    print(X)
    print("\nDatos normalizados:")
    print(X_normalizado)

    # ---------- Configuración inicial ----------
    tasa_aprendizaje = 0.5
    epocas = 10000

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO BASE (tasa={tasa_aprendizaje}, épocas={epocas})")
    print("=" * 60)
    pesos, sesgo, error_final = entrenar(tasa_aprendizaje, epocas, mostrar_cada=1000)

    print(f"\nPeso humedad:     {pesos[0, 0]:.4f}")
    print(f"Peso temperatura: {pesos[1, 0]:.4f}")
    print(f"Sesgo:            {sesgo:.4f}")
    print(f"Error final:      {error_final:.6f}")

    prob, resp = predecir(X, pesos, sesgo)
    print("\nResultados sobre los datos de entrenamiento:")
    print(f"{'Caso':>4} {'Hum %':>6} {'Temp °C':>8} {'Prob':>8} {'Pred':>5} {'Esperado':>9}")
    for i in range(len(X)):
        print(f"{i + 1:>4} {X[i, 0]:>6.0f} {X[i, 1]:>8.0f} {prob[i, 0]:>8.4f} "
              f"{resp[i, 0]:>5} {int(y[i, 0]):>9}")
    print(f"Aciertos: {int((resp == y).sum())}/10")

    # ---------- Condiciones nuevas ----------
    print("\n" + "=" * 60)
    print("PRUEBAS CON CONDICIONES NUEVAS")
    print("=" * 60)
    nuevos = np.array([[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float)
    prob_n, resp_n = predecir(nuevos, pesos, sesgo)
    print(f"{'Hum %':>6} {'Temp °C':>8} {'Prob':>8}  Decisión")
    for i in range(len(nuevos)):
        print(f"{nuevos[i, 0]:>6.0f} {nuevos[i, 1]:>8.0f} {prob_n[i, 0]:>8.4f}  "
              f"{resp_n[i, 0]} ({texto_decision(resp_n[i, 0])})")

    # ---------- Experimentos ----------
    print("\n" + "=" * 60)
    print("EXPERIMENTOS CON LOS PARÁMETROS")
    print("=" * 60)
    experimentos = [
        ("Prueba base", 10000, 0.5),
        ("Pocas épocas", 100, 0.5),
        ("Cantidad intermedia", 1000, 0.5),
        ("Más épocas", 20000, 0.5),
        ("Tasa pequeña", 10000, 0.01),
        ("Tasa moderada", 10000, 0.1),
        ("Tasa alta", 10000, 1.0),
        ("Tasa muy alta", 10000, 2.0),
    ]
    print(f"{'Experimento':<20} {'Épocas':>6} {'Tasa':>5} {'Error final':>12} "
          f"{'Aciertos':>9} {'P(45%,34°C)':>12}")
    for nombre, ep, tasa in experimentos:
        p, b, err = entrenar(tasa, ep)
        _, r = predecir(X, p, b)
        p45, _ = predecir([[45, 34]], p, b)
        print(f"{nombre:<20} {ep:>6} {tasa:>5} {err:>12.6f} "
              f"{int((r == y).sum()):>6}/10 {p45[0, 0]:>12.4f}")

    # ---------- Umbral ----------
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE UMBRALES (misma neurona, sin reentrenar)")
    print("=" * 60)
    todos = np.vstack([X, nuevos])
    etiquetas = [f"Caso {i + 1}" for i in range(10)] + [f"Nuevo {i + 1}" for i in range(5)]
    prob_t, _ = predecir(todos, pesos, sesgo)
    print(f"{'Dato':<9} {'Hum':>4} {'Temp':>5} {'Prob':>8} {'u=0.4':>6} {'u=0.5':>6} {'u=0.6':>6}")
    for i in range(len(todos)):
        decisiones = [int(prob_t[i, 0] >= u) for u in (0.4, 0.5, 0.6)]
        marca = "  <- cambia" if len(set(decisiones)) > 1 else ""
        print(f"{etiquetas[i]:<9} {todos[i, 0]:>4.0f} {todos[i, 1]:>5.0f} {prob_t[i, 0]:>8.4f} "
              f"{decisiones[0]:>6} {decisiones[1]:>6} {decisiones[2]:>6}{marca}")