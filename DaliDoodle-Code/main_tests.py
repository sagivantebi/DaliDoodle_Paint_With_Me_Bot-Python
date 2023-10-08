import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from NLP_Engine import NLP_Engine


"""File we used to test some functions """


def testQA():

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")


    text = "i would like to replace the white cat in the sentence a man riding a white cat"
    question = "what i want to replace in the sentence?"

    # Tokenize the input text and question
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")

    # Use the model to predict the answer
    start_scores, end_scores = model(**inputs).start_logits, model(**inputs).end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    answer_tokens = inputs["input_ids"][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)

    # Print the answer
    print(answer)


def test_add_description(nlp):
    description = "the horse is bigs"
    word = "horse"
    new_sentence = nlp.add_descriptions_QA(description, word)
    print("new sentence: " + new_sentence)
    correct_sentence = nlp.correct_grammar(new_sentence)
    print(correct_sentence)

def test_QA(nlp):
    text = "i would like to replace the white cat in the sentence a man riding a white cat"
    question = "what i want to replace in the sentence?"
    answer = nlp.QA_extract(question, text)
    return answer


def test_mask(nlp):
    sentence = "a woman walking down the street"
    new_sentence = nlp.extend_sentence_with_MASK(sentence)
    return new_sentence


def test_extract_user_description(nlp_eng):
    answer = "I think a good description for the painting in a big cat in a big and clean house."
    new_answer = nlp_eng.QA_extract_user_descriptions(answer)
    print(new_answer)



if __name__ == '__main__':
    print("start")
    nlp_engine = NLP_Engine(True)
    test_extract_user_description(nlp_engine)
    # from diffusers import StableDiffusionPipeline
    # import torch
    #
    # model_id = "runwayml/stable-diffusion-v1-5"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    #
    # prompt = "a photo of an astronaut riding a horse on mars"
    # image = pipe(prompt).images[0]
    #
    # image.save("astronaut_rides_horse.png")

