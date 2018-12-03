from flask import Flask
from flask import request, current_app, abort
from functools import wraps
from flask import jsonify
from engines import ContentEngine

app = Flask(__name__)
app.config.from_object('settings')


def token_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-TOKEN', None) != current_app.config['API_TOKEN']:
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


@app.route('/predict', methods=['POST'])
@token_auth
def predict():
    content_engine = ContentEngine()
    req = request.get_json()
    item = req['item']
    num_predictions = req['num'] or 10
    if not item:
        return jsonify([])
    result = content_engine.predict(str(item), num_predictions)
    result = [{"id": item[0].decode("utf-8"), "probability": item[1]} for item in result]
    return jsonify(result)


@app.route('/train')
@token_auth
def train():
    content_engine = ContentEngine()
    data_url = request.args.get('data-url', None)
    content_engine.train(data_url)
    return jsonify({"message": "Success!", "success": 1})


if __name__ == '__main__':
    app.debug = True
    app.run()
