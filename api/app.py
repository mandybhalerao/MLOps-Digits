from flask export Flask

app = Flask(__name__)

@app.route('/hello'):
    return "Hello World"

@app.route('/model',method=['POST']):
def pred():
    js = request.get_json()
    x = js['x']
    y = js['y']
    return x+y