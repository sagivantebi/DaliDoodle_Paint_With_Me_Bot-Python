import csv
import json
import os
import random
from urllib.parse import urlencode
import pymongo
from Conv_Engine import Conv_Engine
from Hard_coded import WELLCOME_CONVO
from NLP_Engine import NLP_Engine
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from Web_Free_Story import Web_Free_Story
from rank_class import Rank, RankStat

app = Flask(__name__, template_folder='template')

nlp_engine = NLP_Engine(True)
story_teller = Web_Free_Story(nlp_engine)
conv_eng = Conv_Engine(nlp_engine, story_teller)


@app.errorhandler(502)
def handle_bad_gateway_error(e):
    print(e)
    return render_template('error_landing_page502.html'), 502


@app.errorhandler(404)
def handle_bad_gateway_error2(e):
    print(e)
    return render_template('error_landing_page404.html'), 404


@app.errorhandler(Exception)
def handle_exception(e):
    print(e)
    return render_template('error_landing_page_else.html'), 500


@app.route('/', methods=['POST', 'GET'])
def home():
    """
    This is the home page of the website, it will show the user the tutorial
    :return:
    """
    if request.method == 'GET':
        return render_template("tutorial.html")
    if request.method == 'POST':
        return redirect(url_for('submit'))


@app.route('/tutorial', methods=['POST', 'GET'])
def tutorial():
    """
    tutorial page of the website
    :return:
    """
    if request.method == 'GET':
        return render_template("tutorial.html")
    if request.method == 'POST':
        return redirect(url_for('submit'))


@app.route('/images_generated', methods=['POST', 'GET'])
def images_generated():
    """
        Images Generated Route

        This route handles requests for viewing generated images. It supports both GET and POST methods.

        GET Method:
        - Displays a list of generated images available in the designated 'static/Imgs' folder.
        - Renders the 'images_generated.html' template to display the images.

        POST Method:
        - Redirects the user to the 'submit' route.

        Returns:
            If GET: HTML rendering of 'images_generated.html' template with image list.
            If POST: Redirect to 'submit' route.
        """
    if request.method == 'GET':
        current_dir = os.getcwd()
        folder_path = os.path.join(current_dir, "static")
        folder_path = os.path.join(folder_path, "Imgs")
        images = os.listdir(folder_path)
        print(images)
        return render_template("images_generated.html", images=images, folder_path=folder_path)
    if request.method == 'POST':
        return redirect(url_for('submit'))


@app.route('/draw', methods=['POST', 'GET'])
def submit():
    """
    this is the main conversation page where most of the time is spent.
    will initialize a conversation engine and a storyteller for each user
    then will start the conversation and show the user the appropriate prompts ate each stage
    :return:
    """
    # if the user doesnt exist in the database, create a new user
    if 'user_id' not in request.form or request.form['user_id'] == '':
        # TODO: handle the case that the random return a used id
        user_id = str(random.randint(0, 100000))
        conv_eng.add_user_story_teller(user_id)
        print("add new user id is: " + user_id)
    # if the user exist in the database, get the user id
    else:
        user_id = request.form['user_id']
        print("user id is: " + user_id)
    st = conv_eng.get_user_story_teller(user_id)
    # if the user is new, start the conversation
    if request.method == 'GET':
        print("Got to get")
        # my_data = "Let\'s start our Painting together\n What would you like to draw?\n"
        my_data = WELLCOME_CONVO
        return render_template("first_form.html", my_data=my_data, user_id=user_id)
    if request.method == 'POST':
        user_input = request.form['user_input']
        next_question = st.create_story(user_input)
        if type(next_question) is dict:
            question = next_question["question"]
            url_picture = next_question["url"][0]
            description = next_question["description"]
            print(next_question)
            return render_template('form.html', user_input=user_input, my_data=question, my_picture=url_picture,
                                   description=description, user_id=user_id)
        elif next_question == "redirect to ranking":
            st = conv_eng.get_user_story_teller(user_id)
            prompt = st.get_prompt()
            url = st.get_url()
            first_prompt = st.get_first_prompt()
            user_enhance_prompt = st.get_user_enhance_prompt()
            gpt_prompt = st.get_gpt_prompt()
            conv_eng.del_user_story_teller(user_id)
            print("Rank - Get")
            return render_template("rank2.html", prompt=prompt, url=url, user_id=user_id, first_prompt=first_prompt,
                                   user_enhance_prompt=user_enhance_prompt, gpt_prompt=gpt_prompt)
        return render_template('form.html', user_input=user_input, my_data=next_question, user_id=user_id)


# @app.route('/draw/<user_input>',methods=["POST"])
# def submit1(user_input):
#     user_id = request.form['user_id']
#     st = conv_eng.get_user_story_teller(user_id)
#     next_question = st.create_story(user_input)
#     if type(next_question) is dict:
#         question = next_question["question"]
#         url_picture = next_question["url"][0]
#         description = next_question["description"]
#         print(next_question)
#         return render_template('form.html', user_input=user_input, my_data=question, my_picture=url_picture,
#                                description=description, user_id=user_id)
#     if next_question == "redirect to ranking":
#         conv_eng.del_user_story_teller(user_id)
#         return redirect(url_for('ranking'))
#     return render_template('form.html', user_input=user_input, my_data=next_question, user_id=user_id)


@app.route('/rank', methods=['POST', 'GET'])
def ranking():
    """
    This is the ranking page of the website, it will show the user the ranking page to get user feedback
    :return:
    """
    if request.method == 'GET':
        print("Rank - GET")
        return render_template('thank.html')
    if request.method == 'POST':
        print("Rank - POST")
        prompt = request.form['prompt']
        url = request.form['url']
        first_prompt = request.form['first_prompt']
        user_enhance_prompt = request.form['user_enhance_prompt']
        gpt_prompt = request.form['gpt_prompt']
        first_time = request.form['choice']
        if(first_time == ""):
            first_time = "yes"

        r = Rank(request, prompt, url, first_prompt, user_enhance_prompt, gpt_prompt,first_time)
        r.add_new_rank()
        return render_template('thank.html')


# def delete_all():
#     connection_string = "mongodb+srv://omribh:313255242@dalidoodle.lyjrxxl.mongodb.net/?retryWrites=true&w=majority"
#     client = pymongo.MongoClient(connection_string)
#     db = client["ranking_table"]
#     collections = db["rank"]
#     result = collections.delete_many({})
#
#
# @app.route('/delete_all_db_please', methods=['POST', 'GET'])
# def delete_all_db_please():
#     if request.method == 'GET' or request.method == 'POST':
#         delete_all()
#         return render_template("thank.html")


@app.route('/stat', methods=['GET'])
def stat():
    """
    stat page to show all the ranking stats
    :return:
    """
    if request.method == 'GET':
        print("Stat - Get")
        rs = RankStat()
        rs.set_stats()
        dict_avg = rs.get_stats_avg()
        return render_template("stats.html", Ease_of_Use=dict_avg["ease_of_use"],
                               Recommendation=dict_avg["recommendation"], Satisfaction=dict_avg["satisfaction"],
                               Overall_Experience=dict_avg["overall_experience"], issues=rs.issues,
                               improvements=rs.improvements,
                               question_feedback=rs.question_feedback,
                               next_time_question=rs.next_time_question,
                               prompt=rs.prompt,
                               url=rs.url,
                               first_prompt=rs.first_prompt,
                               user_enhance_prompt=rs.user_enhance_prompt,
                               gpt_prompt=rs.gpt_prompt,
                               first_time=rs.first_time
                               )


@app.route('/measures', methods=['post'])
def measures():
    """
        Measures Route

        This route handles incoming POST requests containing user measures data. It is used to record and store user measures
        in a CSV file.

        POST Method:
        - Decodes the incoming data as JSON and converts it into a dictionary.
        - Appends the user measures to a CSV file named after the user's ID.
        - If the CSV file doesn't exist, it creates the file and writes the header.
        - Returns a 204 No Content response upon successful data processing.

        Returns:
            If successful POST: A 204 No Content response.
            If unsuccessful POST: A 500 Internal Server Error response.
        """
    print("measures")
    if request.method == 'POST':
        user_measures = request.data.decode('utf-8')
        user_measures = json.loads(user_measures)
        print(user_measures)
        # Specify the output file path
        output_file = 'users_history/' + user_measures.get('userID') + ".csv"
        file_exists = os.path.isfile(output_file)
        # Write JSON data to CSV file
        with open(output_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=user_measures.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(user_measures)
        return Response(status=204)
    return Response(status = 500)

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    """
    Reset Counter Route

    This route handles incoming POST requests for resetting a user's counter or session. It is used to reset the
    session-related data associated with a specific user.

    POST Method:
    - Checks if the 'user_id' key is present in the submitted form data.
    - If 'user_id' is found, the corresponding user's story-teller data is deleted.
    - Redirects the user to the 'submit' route after performing the reset.

    Returns:
        Redirect to the 'submit' route.
    """
    if 'user_id' in request.form:
        user_id = request.form['user_id']
        conv_eng.del_user_story_teller(user_id)
    print("reset Session")
    # new_story_taller = Web_Free_Story(nlp_engine)
    # conv_eng.set_new_story_teller(new_story_taller)

    return redirect(url_for('submit'))


@app.route('/return_back', methods=['POST'])
def return_back():
    """
    Return Back Route

    This route handles incoming POST requests for returning to the previous step during a conversation with a user.
    It manages the process of undoing the last question and presenting the previous state of the conversation.

    POST Method:
    - Retrieves the user ID from the submitted form data.
    - Retrieves the user's story-teller instance from the conversation engine.
    - Checks the answer counter to determine if it's appropriate to initiate a reset of the conversation state.
    - Reverts to the previous user input by invoking the `last_answer()` method on the story-teller instance.
    - If the user input is None, redirects the user to the 'submit' route.
    - Decreases the answer counter by 2.
    - Generates the next question based on the previous user input using the `create_story()` method.
    - Renders the appropriate template based on the generated question or redirection to ranking.

    Returns:
        If redirection to ranking: HTML rendering of 'rank2.html' template with ranking data.
        If next question: HTML rendering of 'form.html' template with the next question and user input data.
        If user input is None: Redirect to the 'submit' route.
    """
    user_id = request.form['user_id']
    st = conv_eng.get_user_story_teller(user_id)
    answer_counter = st.get_answer_counter()
    if answer_counter == 4 or answer_counter == 3:
        st.init_when_to_stop()
        st.clear_all_descriptions()
        user_input = st.last_answer()
    else:
        # need to be twice because we want to undo the last question (that's the whole point of return XD)
        user_input = st.last_answer()
        user_input = st.last_answer()

    if user_input is None:
        return redirect(url_for('submit'))

    # Decrease the answer counter by 2
    st.decrease_answer_counter()

    # render the last user_input again
    next_question = st.create_story(user_input)
    if type(next_question) is dict:
        question = next_question["question"]
        url_picture = next_question["url"][0]
        description = next_question["description"]
        print(next_question)
        return render_template('form.html', user_input=user_input, my_data=question, my_picture=url_picture,
                               description=description, user_id=user_id)
    if next_question == "redirect to ranking":
        st = conv_eng.get_user_story_teller(user_id)
        prompt = st.get_prompt()
        url = st.get_url()
        conv_eng.del_user_story_teller(user_id)
        print("Rank - Get")
        return render_template("rank2.html", prompt=prompt, url=url)
    return render_template('form.html', user_input=user_input, my_data=next_question, user_id=user_id)


# def test_func():
#     nlp_engine = NLP_Engine(args.dev)
#     test_in = input("Enter a sentence: ")
#     while test_in != "":
#         corection = nlp_engine.enhance_sentence(test_in)
#         print(corection)
#         test_in = input("Enter a sentence: ")


#print_picture - print the picture and the description of the picture and skip all the questions
@app.route('/print_picture', methods=['POST', 'GET'])
def print_picture():
    """
       Print Picture Route

       This route handles printing the generated picture and story for the user. It supports both POST and GET methods.

       POST Method:
       - Retrieves the user data from the submitted form data.
       - Retrieves the user's story-teller instance from the conversation engine.
       - Constructs a redirect URL for redirection after printing.
       - If the prompt is empty, renders a message on the form page.
       - Otherwise, generates the final question using the 'create_story' method.
       - Renders the form page with the final question and generated picture.

       GET Method:
       - Redirects the user to the 'submit' route.

       Returns:
           If POST and prompt is not empty: HTML rendering of 'form.html' template with final question and generated picture.
           If POST and prompt is empty: HTML rendering of 'form.html' template with an error message.
           If GET: Redirect to the 'submit' route.
       """
    if request.method == 'POST':
        print("print_picture")
        data = request.form.get('data')
        print(data)
        if 'user_id' not in request.form or request.form['user_id'] == '':
            return redirect(url_for('submit'))
        user_id = request.form['user_id']
        print("user id is: " + user_id)
        st = conv_eng.get_user_story_teller(user_id)
        # Construct the redirect URL with query parameters
        params = {'user_id': user_id, 'user_input': ""}
        redirect_url = url_for('submit') + '?' + urlencode(params)
        if st.get_prompt() == "":
            # return redirect(redirect_url)
            return render_template("form.html", user_input="",my_data="Please describe the painting that you would like to paint together.\n I can't paint an empty description.\n",user_id=user_id)
        else:
            st.set_user_counter(6)
            user_input = "y"
            next_question = st.create_story(user_input)
            if type(next_question) is dict:
                question = next_question["question"]
                url_picture = next_question["url"][0]
                description = next_question["description"]
                print(next_question)
                return render_template('form.html', user_input=user_input, my_data=question, my_picture=url_picture,
                                       description=description, user_id=user_id)
    else:
        return redirect(url_for('submit'))

if __name__ == '__main__':
    app.run(host='localhost', port=5010)
