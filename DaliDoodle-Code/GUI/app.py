from flask import Flask, render_template, request, flash
from flask_socketio import SocketIO, emit
from Conv_Engine import Conv_Engine
from Free_Story import Free_Story
from NLP_Engine import NLP_Engine

from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('user_input')
def handle_user_input(input_data):
    # Your desktop app logic goes here, for example:
    print(input_data)
    output = f'You entered: {input_data}'

    # Emit the output to the client
    socketio.emit('output', output)

# @app.route("/greet", methods=['POST', 'GET'])
# def convo():
#     initial_input = request.form['user_input']
#     nlp_engine = NLP_Engine(socketio=socketio)
#     story_teller = Free_Story(nlp_engine)
#     conv_eng = Conv_Engine(nlp_engine, story_teller)
#     conv_eng.paint()
#     conv_eng.ranking()
#     return render_template("convo.html")


if __name__=="__main__":


    socketio.run(app, allow_unsafe_werkzeug=True ,debug=True)



# def print_and_emit(msg, socket):
#     print(msg)
#     socket.emit('output', msg)

