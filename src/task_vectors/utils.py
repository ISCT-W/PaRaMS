import logging
import os


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def create_log_dir(path, filename='record.txt'):
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    hdlf = logging.FileHandler(path + '/' + filename)
    hdlf.setLevel(logging.DEBUG)
    hdls = logging.StreamHandler()
    hdls.setLevel(logging.DEBUG)
    logger.addHandler(hdlf)
    logger.addHandler(hdls)
    return logger
