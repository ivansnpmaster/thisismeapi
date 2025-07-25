from flask import Flask
from flask_cors import CORS
from predict_route import predict_blueprint

app = Flask(__name__)
CORS(app)
app.register_blueprint(predict_blueprint)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)