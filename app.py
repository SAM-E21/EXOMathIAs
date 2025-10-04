import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

from flask import Flask, request, jsonify, render_template_string, redirect, url_for

app = Flask(__name__)

# ===================== Carga de modelos =====================
try:
    classifier = joblib.load('exoplanet_classifier.pkl')
    preprocessor = joblib.load('exoplanet_preprocessor.pkl')
    importances_list = joblib.load('feature_importances.pkl')
    model_metrics = joblib.load('model_metrics.pkl')
except FileNotFoundError as e:
    # Si faltan archivos .pkl, la app sigue levantando pero muestra error
    print(f"ERROR: {str(e)}")
    classifier = preprocessor = importances_list = model_metrics = None

CLASSES = {0: 'FALSE POSITIVE (FP)', 1: 'EXOPLANETA CONFIRMADO', 2: 'CANDIDATO PLANETARIO'}
FEATURES = ['period', 'duration', 'impact', 'depth', 'teq', 'prad', 'slogg', 'steff', 'srad']
N_FEATURES = len(FEATURES)
GLOBAL_DF = pd.DataFrame(np.random.rand(500, N_FEATURES), columns=FEATURES)
GLOBAL_DF['prad'] = GLOBAL_DF['prad'] * 10 + 1
GLOBAL_DF['period'] = GLOBAL_DF['period'] * 300 + 1

# ===================== CSS =====================
CSS_STYLE = """
<style>
... tu CSS actual aquí ...
</style>
"""

# ===================== Funciones auxiliares =====================
def clean_input_value(value_str):
    value_str = value_str.split('±')[0].strip()
    value_str = value_str.split('+')[0].strip()
    return float(value_str)

def create_plot(input_data_list, prediction_int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    idx_period = FEATURES.index('period')
    idx_prad = FEATURES.index('prad')
    idx_depth = FEATURES.index('depth')
    idx_steff = FEATURES.index('steff')
    color = {0: 'red', 1: 'green', 2: 'orange'}.get(prediction_int, 'blue')

    ax1.scatter(GLOBAL_DF['period'], GLOBAL_DF['prad'], c='lightgray', alpha=0.5, label='Contexto Histórico')
    ax1.scatter(input_data_list[idx_period], input_data_list[idx_prad],
               c=color, s=200, marker='*', label='Objeto Clasificado', zorder=5)
    ax1.set_xscale('log'); ax1.set_title('1. Periodo Orbital vs. Radio Planetario')
    ax1.set_xlabel('Periodo Orbital (días)'); ax1.set_ylabel('Radio Planetario (Radios Terrestres)')
    ax1.legend(); ax1.grid(True, which="both", ls="--", linewidth=0.5); ax1.set_facecolor('#161b22')
    ax1.tick_params(colors='#c9d1d9'); ax1.spines['left'].set_color('#c9d1d9'); ax1.spines['bottom'].set_color('#c9d1d9')

    ax2.scatter(GLOBAL_DF['steff'], GLOBAL_DF['depth'], c='lightgray', alpha=0.5)
    ax2.scatter(input_data_list[idx_steff], input_data_list[idx_depth],
               c=color, s=200, marker='*', label='Objeto Clasificado', zorder=5)
    ax2.set_yscale('log'); ax2.set_title('2. Profundidad de Tránsito vs. Temp. Estelar')
    ax2.set_xlabel('Temperatura Estelar (Kelvin)'); ax2.set_ylabel('Profundidad de Tránsito (ppm)')
    ax2.legend(); ax2.grid(True, which="both", ls="--", linewidth=0.5); ax2.set_facecolor('#161b22')
    ax2.tick_params(colors='#c9d1d9'); ax2.spines['left'].set_color('#c9d1d9'); ax2.spines['bottom'].set_color('#c9d1d9')

    plt.tight_layout(); plt.gcf().set_facecolor('#0d1117')
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ===================== Rutas =====================
@app.route('/')
def home():
    if model_metrics:
        metrics_html = f"""
        <div class="card">
            <h3>Estado y Precisión del Modelo</h3>
            <table>
                <tr><th>Métrica</th><th>Valor</th><th>Interpretación</th></tr>
                <tr><td>Precisión General (Accuracy)</td><td align="right"><b>{model_metrics['accuracy']:.4f}</b></td><td>Porcentaje de predicciones correctas.</td></tr>
                <tr><td>Recall de Confirmados</td><td align="right"><b>{model_metrics['recall_confirmed']:.4f}</b></td><td>Capacidad para no perder exoplanetas reales.</td></tr>
                <tr><td>Precisión de Falsos Positivos</td><td align="right"><b>{model_metrics['precision_fp']:.4f}</b></td><td>Fiabilidad al descartar candidatos como FP.</td></tr>
            </table>
        </div>
        """
    else:
        metrics_html = "<div class='card'><h3>Modelo no disponible</h3><p>Los archivos .pkl no se cargaron.</p></div>"

    form_html = f"""
    {CSS_STYLE}
    <div class="main-container">
        <h1>EXOMathIAs</h1>
        {metrics_html}
        <div class="card">
            <h3>Introducir Nuevos Datos de Tránsito</h3>
            <form action='{url_for('predict')}' method='post'>
            {"".join([f"<label for='{f}'>{f.capitalize()}:</label><input type='text' id='{f}' name='{f}' value='0.0'>" for f in FEATURES])}
            <input type='submit' value='Clasificar Tránsito'>
            </form>
        </div>
    </div>
    """
    return render_template_string(form_html)

@app.route('/predict', methods=['POST'])
def predict():
    if not classifier or not preprocessor:
        return render_template_string(f"{CSS_STYLE}<div class='main-container'><h1>Error</h1><p>Modelo no disponible.</p></div>")

    data = request.form.to_dict()
    try:
        input_data_clean = [clean_input_value(data[f]) for f in FEATURES]
        input_df = pd.DataFrame([input_data_clean], columns=FEATURES)
        input_scaled = preprocessor.transform(input_df)
        prediction_int = classifier.predict(input_scaled)[0]
        result = CLASSES.get(prediction_int, "Clasificación Desconocida")
        plot_base64 = create_plot(input_data_clean, prediction_int)

        top_importances_html = "".join([
            f"<li><b>{i+1}. {f.capitalize()}</b> (Score: {score:.0f})</li>"
            for i, (f, score) in enumerate(importances_list[:3])
        ])
        status_class = {0:'status-false-positive',1:'status-confirmed',2:'status-candidate'}.get(prediction_int,'status-unknown')

        html_response = f"""
        {CSS_STYLE}
        <div class="main-container">
            <a href='{url_for('home')}'>&lt; Volver al Panel de Control</a>
            <div class="status-banner {status_class}">RESULTADO: {result}</div>
            <div class="result-container">
                <div class="data-panel">
                    <div class="card"><h3>Parámetros Ingresados</h3><ul>{"".join([f"<li><b>{f.capitalize()}:</b> {input_df[f].iloc[0]:.4f}</li>" for f in FEATURES])}</ul></div>
                    <div class="card"><h3>Justificación</h3><ol>{top_importances_html}</ol></div>
                </div>
                <div class="graph-panel">
                    <div class="card"><h3>Previsualización</h3>
                    <img src="data:image/png;base64,{plot_base64}" style="max-width:100%;"/></div>
                </div>
            </div>
        </div>
        """
        return render_template_string(html_response)
    except Exception as e:
        return render_template_string(f"{CSS_STYLE}<div class='main-container'><h1>Error</h1><p>{str(e)}</p></div>")

# ===================== App listo para producción =====================
# NO usar app.run() en Railway, Gunicorn lo levanta automáticamente
