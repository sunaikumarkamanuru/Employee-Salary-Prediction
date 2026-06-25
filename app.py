from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'employee-salary-prediction-secret-key-2026'

model = joblib.load("model.pkl")


def load_choices():
    df = pd.read_csv("job_salary_prediction_dataset.csv")
    return {
        "job_title": sorted(df["job_title"].dropna().unique().tolist()),
        "education_level": sorted(df["education_level"].dropna().unique().tolist()),
        "industry": sorted(df["industry"].dropna().unique().tolist()),
        "company_size": sorted(df["company_size"].dropna().unique().tolist()),
        "location": sorted(df["location"].dropna().unique().tolist()),
        "remote_work": sorted(df["remote_work"].dropna().unique().tolist())
    }


@app.route('/')
def home():
    choices = load_choices()
    return render_template('index.html', choices=choices)


@app.route('/predict', methods=['POST'])
def predict():
    choices = load_choices()
    input_data = pd.DataFrame({
        "job_title": [request.form["job_title"]],
        "experience_years": [int(request.form["experience_years"])],
        "education_level": [request.form["education_level"]],
        "skills_count": [int(request.form["skills_count"])],
        "industry": [request.form["industry"]],
        "company_size": [request.form["company_size"]],
        "location": [request.form["location"]],
        "remote_work": [request.form["remote_work"]],
        "certifications": [int(request.form["certifications"])]
    })
    prediction = model.predict(input_data)[0]
    return render_template(
        'index.html',
        choices=choices,
        prediction=f"{prediction:,.2f}"
    )


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/salary-guide')
def salary_guide():
    return render_template('salary-guide.html')


@app.route('/career-growth')
def career_growth():
    return render_template('career-growth.html')


@app.route('/skills-vs-salary')
def skills_salary():
    return render_template('skills-vs-salary.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/ads.txt')
def ads():
    return send_file('ads.txt', mimetype='text/plain')


if __name__ == "__main__":
    app.run(debug=True)