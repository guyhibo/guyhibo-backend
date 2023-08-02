from flask import Response, request, Flask, jsonify, make_response
from flask_cors import CORS
from translate import translate_video
import json
import ssl


app = Flask(__name__)
app.debug = True

@app.route('/api/', methods=['GET'])
def main_page():
    return "ONLINE"

@app.route('/api/translate/', methods=['POST'])
def translate():
    params = json.loads(request.get_data())
    # print(params)
    if len(params) == 0:
        return 'No params'
    
    filename = params["name"]
    sentence = translate_video(filename)
    print(jsonify({"translated_word":sentence}))
    return jsonify({"translated_word":sentence})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
