from flask import Flask, render_template, request
from NLP_Engine import NLP_Engine

app = Flask(__name__, template_folder='template')

## TODO:: this:
# import threading
#
# # Create a condition object
# cond = threading.Condition()
#
# # Define a function that waits for the condition to be notified
# def wait_for_notification():
#     with cond:
#         print('Waiting for notification...')
#         cond.wait()
#         print('Received notification!')
#
# # Define a function that notifies the condition
# def notify_condition():
#     with cond:
#         print('Notifying condition...')
#         cond.notify()
#
# # Create a new thread that waits for notification
# thread = threading.Thread(target=wait_for_notification)
# thread.start()
#
# # Wait for a few seconds
# print('Waiting...')
# threading.Timer(2.0, notify_condition).start()
#
# # Wait for the thread to finish
# thread.join()
#
# print('Done')

def get_output(user_input):
    nlp.speak_input(user_input)


@app.route('/', methods=['POST', 'GET'])
def submit():
    if request.method == 'GET':
        print("Got to get")
        my_data = "This is some data that I want to insert into my page."
        return render_template("form.html", my_data=my_data)

    if request.method == 'POST':
        print("submit")
        user_input = request.form['user_input']
        print(user_input)
        # get from the server what print
        my_data = get_output(user_input)
        return render_template('form.html', user_input=user_input, my_data=my_data)


nlp = NLP_Engine()
print(0)
app.run(host='localhost', port=5000)
print(1)
