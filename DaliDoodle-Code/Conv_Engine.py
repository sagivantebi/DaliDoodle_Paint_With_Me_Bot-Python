
from Web_Free_Story import Web_Free_Story

from PIL import Image


class Conv_Engine:
    """
    A class representing a Conversation Engine that manages Natural Language Processing (NLP) and story telling components.

    Attributes:
        nlp_engine (NLP_Engine): An instance of the NLP engine used for processing natural language.
        story_teller (Story_Teller): An instance of the story teller used for generating stories.
        user_story_teller (dict): A dictionary to store user-specific story teller instances.
    """

    def __init__(self, nlp_engine, story_teller):
        """
        Initialize a new instance of the Conv_Engine class.

        :param nlp_engine: An instance of the NLP engine used for processing natural language.
        :type nlp_engine: NLP_Engine

        :param story_teller: An instance of the story teller used for generating stories.
        :type story_teller: Story_Teller
        """
        self.nlp_engine = nlp_engine
        self.story_teller = story_teller
        self.user_story_teller = {}

    def get_nlp_engine(self):
        """
        Get the NLP engine instance.

        :return: The NLP engine instance.
        :rtype: NLP_Engine
        """
        return self.nlp_engine

    def add_user_story_teller(self, user_id):
        """
        Add a new user-specific story teller instance.

        :param user_id: The unique identifier of the user.
        :type user_id: str
        """
        st = Web_Free_Story(self.get_nlp_engine())
        self.user_story_teller[user_id] = st

    def get_user_story_teller(self, user_id):
        """
        Get the story teller instance associated with a user.

        :param user_id: The unique identifier of the user.
        :type user_id: str

        :return: The story teller instance associated with the user, or None if not found.
        :rtype: Story_Teller
        """
        return self.user_story_teller.get(user_id)

    def del_user_story_teller(self, user_id):
        """
        Delete the user-specific story teller instance.

        :param user_id: The unique identifier of the user.
        :type user_id: str
        """
        if self.check_if_user_exist(user_id):
            del self.user_story_teller[user_id]

    def check_if_user_exist(self, user_id):
        """
        Check if a user-specific story teller instance exists.

        :param user_id: The unique identifier of the user.
        :type user_id: str

        :return: True if a user-specific story teller instance exists, False otherwise.
        :rtype: bool
        """
        return user_id in self.user_story_teller

    def show_img(self, file_name):
        """
        Display an image using the default image viewer.

        :param file_name: The name of the image file to be displayed.
        :type file_name: str
        """
        with Image.open(file_name) as image:
            image.show()

    def set_new_story_teller(self, story_teller):
        """
        Set a new story teller instance.

        :param story_teller: The new story teller instance.
        :type story_teller: Story_Teller
        """
        self.story_teller = story_teller

    def get_story_teller(self):
        """
        Get the current story teller instance.

        :return: The current story teller instance.
        :rtype: Story_Teller
        """
        return self.story_teller
