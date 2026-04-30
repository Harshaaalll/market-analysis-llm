# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import app as analysis_app   # uses your uploaded app.py (must be in same folder)

server = Flask(__name__)
CORS(server)  # allow calls from frontend dev server

@server.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True, silent=True) or request.form
        stock_name = None
        if isinstance(data, dict):
            stock_name = data.get("stock_name") or data.get("name") or data.get("stock")
        if not stock_name:
            return jsonify({"error": "Missing parameter 'stock_name'"}), 400

        # call analyze_stock from your app.py
        result_text, avg_score = analysis_app.analyze_stock(stock_name)
        return jsonify({
            "ok": True,
            "result": result_text,
            "scores": avg_score
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    server.run(host="127.0.0.1", port=5001, debug=True)
