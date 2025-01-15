QUESTION_TEMPLATE = "Where does {} think the {} is?"


def get_question(row):
    return QUESTION_TEMPLATE.format(row["question_name"], row["object"])


def get_question_template():
    return QUESTION_TEMPLATE


def get_data_file():
    return "Evaluation/data/knowledge_perception.json" 


def get_cot_file(): # DUMMY!
    return "Evaluation/data/knowledge_perception_cot.json" 