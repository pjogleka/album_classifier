print("Starting app...")

from flask import Flask, request, redirect, url_for, render_template_string, session
from io import BytesIO
import requests
import joblib

print("Imported dependencies")

app = Flask(__name__)
app.secret_key = "tsunami"

print("Created app")

# Load model

MODEL_URL = "https://myawsbucketjoblib.s3.us-east-2.amazonaws.com/best_model.joblib"
model = None

def load_model():
    global model
    if model is None:
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        print(f"Status code: {response.status_code}")
        response.raise_for_status()
        print("Loading model...")
        try:
            model = joblib.load(BytesIO(response.content))
        except Exception as e:
            print(e)
            print(response.content[:500])
            raise
    return model

@app.route("/", methods=["GET", "POST"])
def home():
    
    clf = load_model()
    
    if request.method == "POST":
        lyric = request.form.get("lyric", "").strip()
        album = clf.predict([lyric])[0] if lyric else None
        session["album"] = album
        session["lyric"] = lyric
        return redirect(url_for("home"))
    
    album = session.pop("album", None)
    lyric = session.pop("lyric", "")
    
    # HTML template
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>TS Lyric Classifier</title>
            <meta http-equiv="Cache-Control" content="no-store" />
            <style>
                body {
                    font-family: 'AppleGothic', sans-serif;
                    font-size: 16px;
                    background: #f5f5f5;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding-top: 50px;
                }
                h1 {
                    color: #444;
                }
                form {
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    width: 550px;
                }
                textarea {
                    width: 95%;
                    padding: 10px;
                    border-radius: 6px;
                    border: 1px solid #ccc;
                    font-size: 14px;
                    resize: vertical;
                }
                input[type="submit"] {
                    font-family: 'AppleGothic', sans-serif;
                    margin-top: 0px;
                    padding: 10px 20px;
                    background: pink;
                    color: black;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background: #fa8ea1;
                }
                .prediction {
                    margin-top: 20px;
                    font-size: 21px;
                    color: #333;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <h1>Taylor Swift Lyric Classifier</h1>
            <form method="POST">
                <textarea name="lyric" rows="4" cols="50" placeholder="Type your lyric here...">{{lyric}}</textarea><br><br>
                <input type="submit" value="Predict album">
            </form>
            {% if album %}
                <h2 class="prediction">Predicted album: {{album}}</h2>
            {% endif %}

        </body>
        </html>
    ''', album=album, lyric=lyric)

print("Defined route")

if __name__ == "__main__":
    print("Running app")
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
