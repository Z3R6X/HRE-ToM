QUESTION_TEMPLATE = "What emotion might {} feel?"


def get_question(row):
    return QUESTION_TEMPLATE.format(row["question_name"])


def get_question_template():
    return QUESTION_TEMPLATE


def get_data_file():
    return "Evaluation/data/inferring_emotion_test.json" 


def get_cot_file(): # DUMMY!
    return "Evaluation/data/inferring_emotion_cot.json" 