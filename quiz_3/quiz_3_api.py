from flask export Flask

app = Flask(__name__)

// Quiz 3- Comparison of two images.

    @app.route('/model', methods=['POST'])
def pred_model():
    js = request.get_json()
    image1 = js['image1']
    image2 = js['image2']
    #Let this be the Best Trained Model
    model = load('./models/svm_gamma:0.001_C:0.1.joblib')
    prediction_image_1 = model.predict(image1)
    prediction_image_2 = model.predict(image2)
    if(prediction_image_1 == prediction_image_2):
        return "True"
    else:
        return "False"