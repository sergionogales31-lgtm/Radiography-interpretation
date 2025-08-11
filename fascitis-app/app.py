import os
from math import exp
from flask import Flask, render_template, request, flash, redirect, url_for
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # openai es opcional
    OpenAI = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def clamp(value, min_value=0.0, max_value=1.0):
    """Restringe value al rango [min_value, max_value]."""
    return max(min_value, min(value, max_value))

def sigmoid(x):
    """Función sigmoide clásica."""
    return 1 / (1 + exp(-x))

def normalize_demo(value, max_value):
    """Normaliza valores para el modelo DEMO."""
    if max_value <= 0:
        return 0
    return clamp(value / max_value)

def calculate_demo(pcr, wbc, esr):
    """Calcula riesgo DEMO combinando PCR, leucocitos y VSG."""
    npcr = normalize_demo(pcr, 300)
    nwbc = normalize_demo(wbc, 40)
    nesr = normalize_demo(esr, 150)
    weights = [0.4, 0.35, 0.25]
    z = npcr * weights[0] + nwbc * weights[1] + nesr * weights[2]
    # Ajuste simple con sigmoide
    return sigmoid(3 * z - 1.5)

def calculate_lrinec(crp, wbc, hb, na, creat, glucose):
    """Calcula puntaje LRINEC y lo mapea a probabilidad."""
    score = 0
    # CRP
    if crp >= 150:
        score += 4
    # Leucocitos
    if 15 <= wbc <= 25:
        score += 1
    elif wbc > 25:
        score += 2
    # Hemoglobina
    if 11 <= hb <= 13.5:
        score += 1
    elif hb < 11:
        score += 2
    # Sodio
    if na < 135:
        score += 2
    # Creatinina
    if creat > 1.6:
        score += 2
    # Glucosa
    if glucose > 180:
        score += 1
    # Probabilidad suave usando sigmoide centrado en 6.5
    prob = sigmoid((score - 6.5) / 1.5)
    if score >= 8:
        level = "Alto"
    elif score >= 6:
        level = "Intermedio"
    else:
        level = "Bajo"
    return score, level, prob

def keywords_factor(text):
    """Ajuste por palabras clave cuando no hay IA."""
    keywords = [
        "dolor desproporcionado",
        "crepitación",
        "bullas",
        "necrosis",
        "progresión rápida",
        "sepsis",
        "hipotensión",
    ]
    t = text.lower()
    hits = sum(1 for kw in keywords if kw in t)
    return clamp(0.05 * hits, 0, 0.3)

def ai_text_factor(text):
    """Obtiene factor textual [0,1] desde la API de OpenAI."""
    if not (OPENAI_API_KEY and OpenAI):
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Devuelve SOLO un número entre 0 y 1 (máx 3 decimales) que represente "
        "la probabilidad de fascitis necrotizante según el texto dado:\n" + text
    )
    try:
        resp = client.responses.create(
            model="gpt-3.5-turbo-instruct",
            input=prompt,
            max_tokens=5,
            temperature=0,
        )
        value = resp.output_text.strip().replace(",", ".")
        return clamp(float(value))
    except Exception:
        return None

def combine_risk(base_prob, text_factor, ai_active):
    if ai_active and text_factor is not None:
        return clamp(0.7 * base_prob + 0.3 * text_factor)
    elif text_factor:
        return clamp(base_prob + text_factor)
    return base_prob

def risk_label(prob):
    if prob >= 0.6:
        return "Alto", "danger", "#dc3545"
    if prob >= 0.3:
        return "Intermedio", "warning", "#ffc107"
    return "Bajo", "success", "#198754"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    model = "demo"
    lrinec_score = None
    lrinec_level = None
    form_values = {}
    if request.method == "POST":
        form_values = request.form.to_dict()
        model = request.form.get("model", "demo")
        text = request.form.get("notes", "")
        try:
            if model == "demo":
                pcr = float(request.form.get("pcr", ""))
                wbc = float(request.form.get("wbc", ""))
                esr = float(request.form.get("esr", ""))
                if min(pcr, wbc, esr) < 0:
                    raise ValueError
                base_prob = calculate_demo(pcr, wbc, esr)
                inputs = {
                    "PCR (mg/L)": pcr,
                    "Leucocitos (×10⁹/L)": wbc,
                    "VSG (mm/h)": esr,
                }
            else:
                crp = float(request.form.get("crp", ""))
                wbc = float(request.form.get("wbc", ""))
                hb = float(request.form.get("hb", ""))
                na = float(request.form.get("na", ""))
                creat = float(request.form.get("creat", ""))
                glucose = float(request.form.get("glucose", ""))
                if min(crp, wbc, hb, na, creat, glucose) < 0:
                    raise ValueError
                lrinec_score, lrinec_level, base_prob = calculate_lrinec(
                    crp, wbc, hb, na, creat, glucose
                )
                inputs = {
                    "CRP (mg/L)": crp,
                    "Leucocitos (×10⁹/L)": wbc,
                    "Hemoglobina (g/dL)": hb,
                    "Sodio (mmol/L)": na,
                    "Creatinina (mg/dL)": creat,
                    "Glucosa (mg/dL)": glucose,
                }
            ai_active = bool(OPENAI_API_KEY and OpenAI)
            tfactor = ai_text_factor(text) if ai_active else keywords_factor(text)
            final_prob = combine_risk(base_prob, tfactor, ai_active)
            level, color_class, color_hex = risk_label(final_prob)
            risk_percent = round(final_prob * 100, 1)
            result = {
                "prob": final_prob,
                "percent": risk_percent,
                "level": level,
                "color_class": color_class,
                "color_hex": color_hex,
                "inputs": inputs,
            }
        except ValueError:
            flash("Valores inválidos o negativos.")
            return redirect(url_for("index"))
        except KeyError:
            flash("Faltan campos obligatorios.")
            return redirect(url_for("index"))
    return render_template(
        "index.html",
        result=result,
        model=model,
        lrinec_score=lrinec_score,
        lrinec_level=lrinec_level,
        form_values=form_values,
        ai_active=bool(OPENAI_API_KEY and OpenAI),
    )

if __name__ == "__main__":
    app.run(debug=True)
