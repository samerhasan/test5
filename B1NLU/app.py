# the main logic to make NLP api works as web service based on Flask

# import required libraries
from flask import Flask, jsonify
from api.core import inference

# flask app
app = Flask(__name__)



# add the main route
@app.route('/B1NLP/api/v1.0/command/<string:input>', methods=['GET'])
# given an input, infer the nlp results and return it as JSON object
def getJSON(input):
    # get the infered data
    infered = inference.infer(input)

    # return the data in a JSON format
    return jsonify({'Query': input,'Result': infered})



if __name__ == '__main__':
    app.run(debug=True)


