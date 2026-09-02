from pathlib import Path

from flask import Flask, send_from_directory

from api import analytics_bp, api_bp

# U15: the form and its server-rendered table are gone. client/dist is a
# build artifact (npm run build inside client/), not committed, so a fresh
# checkout needs that build before this route has anything to serve.
CLIENT_DIST = Path(__file__).parent / "client" / "dist"

app = Flask(__name__, static_folder=str(CLIENT_DIST), static_url_path="")
app.register_blueprint(api_bp)
app.register_blueprint(analytics_bp)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True)
