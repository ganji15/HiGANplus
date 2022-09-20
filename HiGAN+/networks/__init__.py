from networks.model import RecognizeModel, WriterIdentifyModel, GlobalLocalAdversarialModel

all_models = {
    'gl_adversarial_model': GlobalLocalAdversarialModel,
    'recognize_model': RecognizeModel,
    'identifier_model': WriterIdentifyModel
}


def get_model(name):
    return all_models[name]