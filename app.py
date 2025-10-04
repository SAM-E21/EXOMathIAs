import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, request, jsonify, render_template_string, redirect, url_for

app = Flask(__name__)


try:
    classifier = joblib.load('exoplanet_classifier.pkl')
    preprocessor = joblib.load('exoplanet_preprocessor.pkl')
    importances_list = joblib.load('feature_importances.pkl')
    model_metrics = joblib.load('model_metrics.pkl')

    CLASSES = {0: 'FALSE POSITIVE (FP)', 1: 'EXOPLANETA CONFIRMADO', 2: 'CANDIDATO PLANETARIO'}

    FEATURES = ['period', 'duration', 'impact', 'depth', 'teq', 'prad', 'slogg', 'steff', 'srad']

    N_FEATURES = len(FEATURES)
    GLOBAL_DF = pd.DataFrame(np.random.rand(500, N_FEATURES), columns=FEATURES)
    GLOBAL_DF['prad'] = GLOBAL_DF['prad'] * 10 + 1
    GLOBAL_DF['period'] = GLOBAL_DF['period'] * 300 + 1

except FileNotFoundError:
    print("¡ERROR! Asegúrate de que todos los archivos .pkl estén presentes.")
    exit()



CSS_STYLE = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        margin: 0;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-container {
        padding: 40px;
        max-width: 1200px;
        margin: auto;
    }
    h1, h2, h3 { color: #58a6ff; font-weight: 300; border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-top: 20px;}
    h1 { color: #58a6ff; }
    h3 { border: none; padding-bottom: 0; margin-top: 0; }

    a { color: #58a6ff; text-decoration: none; }
    a:hover { text-decoration: underline; }

    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Form Styles */
    label { display: block; margin-top: 10px; font-weight: bold; }
    input[type="text"] {
        width: 100%;
        padding: 8px;
        margin-top: 5px;
        box-sizing: border-box;
        background-color: #0d1117;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 3px;
    }
    input[type="submit"] {
        width: 100%;
        padding: 12px;
        margin-top: 20px;
        background-color: #2ea043; /* Green for Action */
        color: white;
        border: none;
        cursor: pointer;
        border-radius: 6px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    input[type="submit"]:hover {
        background-color: #3fbc55;
    }

    /* Table Styles (Metrics) */
    table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    th, td { border: 1px solid #30363d; padding: 12px; text-align: left; }
    th { background-color: #161b22; color: #58a6ff; }
    tr:nth-child(even) { background-color: #161b22; }

    /* Prediction Page Layout */
    .result-container { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
    .data-panel, .graph-panel { flex: 1; min-width: 300px; }
    .justification-list { list-style: none; padding: 0; }
    .justification-list li { margin-bottom: 5px; font-size: 1.1em; }

    /* Status Banner */
    .status-banner {
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
    }
    .status-confirmed { background-color: #2ea043; color: white; }
    .status-false-positive { background-color: #f85149; color: white; }
    .status-candidate { background-color: #f2a900; color: #161b22; }
    .status-unknown { background-color: #444; color: white; }
</style>
"""



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

    ax1.set_xscale('log')
    ax1.set_title('1. Periodo Orbital vs. Radio Planetario')
    ax1.set_xlabel('Periodo Orbital (días)')
    ax1.set_ylabel('Radio Planetario (Radios Terrestres)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    ax1.set_facecolor('#161b22')
    ax1.tick_params(colors='#c9d1d9')
    ax1.spines['left'].set_color('#c9d1d9')
    ax1.spines['bottom'].set_color('#c9d1d9')

    ax2.scatter(GLOBAL_DF['steff'], GLOBAL_DF['depth'], c='lightgray', alpha=0.5)
    ax2.scatter(input_data_list[idx_steff], input_data_list[idx_depth],
               c=color, s=200, marker='*', label='Objeto Clasificado', zorder=5)

    ax2.set_yscale('log')
    ax2.set_title('2. Profundidad de Tránsito vs. Temp. Estelar')
    ax2.set_xlabel('Temperatura Estelar (Kelvin)')
    ax2.set_ylabel('Profundidad de Tránsito (ppm)')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", linewidth=0.5)
    ax2.set_facecolor('#161b22')
    ax2.tick_params(colors='#c9d1d9')
    ax2.spines['left'].set_color('#c9d1d9')
    ax2.spines['bottom'].set_color('#c9d1d9')

    plt.tight_layout()
    plt.gcf().set_facecolor('#0d1117')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')



@app.route('/')
def home():

    metrics_html = f"""
    <div class="card">
        <h3>Estado y Precisión del Modelo (Kepler/TESS/K2)</h3>
        <table>
            <tr><th>Métrica</th><th>Valor</th><th>Interpretación</th></tr>
            <tr>
                <td>Precisión General (Accuracy)</td>
                <td align="right"><b>{model_metrics['accuracy']:.4f}</b></td>
                <td>Porcentaje de predicciones correctas.</td>
            </tr>
            <tr>
                <td>Recall de Confirmados</td>
                <td align="right"><b>{model_metrics['recall_confirmed']:.4f}</b></td>
                <td>Capacidad para no perder exoplanetas reales.</td>
            </tr>
            <tr>
                <td>Precisión de Falsos Positivos</td>
                <td align="right"><b>{model_metrics['precision_fp']:.4f}</b></td>
                <td>Fiabilidad al descartar candidatos como FP.</td>
            </tr>
        </table>
    </div>
    """

    form_html = f"""
        {CSS_STYLE}
        <div class="main-container">
            <h1>EXOMathIAs</h1>
            <a href="exomathias.web.app" target="_blank" class="button-link">Ir a la pagina principal</a>
            <br>
            {metrics_html}
            <div class="card">
                <h3>Introducir Nuevos Datos de Tránsito</h3>
                <p>Ingresa los 9 parámetros astrofísicos para obtener una clasificación:</p>
                <form action='{url_for('predict')}' method='post'>
                {"".join([f"<label for='{f}'>{f.replace('_', ' ').capitalize()} (Media):</label><input type='text' id='{f}' name='{f}' value='0.0'>" for f in FEATURES])}
                <input type='submit' value='Clasificar Tránsito'>
                </form>
            </div>
        </div>
    """
    return render_template_string(form_html)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

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

        status_class = {
            0: 'status-false-positive',
            1: 'status-confirmed',
            2: 'status-candidate'
        }.get(prediction_int, 'status-unknown')

        html_response = f"""
        {CSS_STYLE}
        <div class="main-container">
            <a href='{url_for('home')}'>&lt; Volver al Panel de Control</a>

            <div class="status-banner {status_class}">
                RESULTADO DE LA CLASIFICACIÓN: {result}
            </div>

            <div class="result-container">

                <div class="data-panel">
                    <div class="card">
                        <h3>Parámetros Ingresados</h3>
                        <ul class="justification-list">
                            {"".join([f"<li><b>{f.capitalize()}:</b> {input_df[f].iloc[0]:.4f}</li>" for f in FEATURES])}
                        </ul>
                    </div>

                    <div class="card">
                        <h3>Justificación del Modelo (Global)</h3>
                        <p>El modelo basó su decisión principalmente en estas 3 características:</p>
                        <ol class="justification-list" style='margin-left: 20px;'>
                            {top_importances_html}
                        </ol>
                    </div>
                </div>

                <div class="graph-panel">
                    <div class="card">
                        <h3>Previsualización Astrofísica</h3>
                        <img src="data:image/png;base64,{plot_base64}" alt="Gráfico de Previsualización" style="max-width: 100%; height: auto; border-radius: 4px;">
                    </div>
                </div>
            </div>
        </div>
        """
        return render_template_string(html_response)

    except Exception as e:
        return render_template_string(f"{CSS_STYLE}<div class='main-container'><h1>Error de Procesamiento</h1><p>Asegúrate de que todos los campos contienen valores numéricos limpios. Error: {str(e)}</p><a href='{url_for('home')}'>&lt; Volver</a></div>")

if __name__ == '__main__':
    app.run(port=5000)
    
