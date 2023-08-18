from .modeling_berts import BertSeriesConfig,RobertaSeriesConfig,BertSeriesModelWithTransformation,RobertaSeriesModelWithTransformation
STUDENT_CONFIG_DICT={
    'hfl/chinese-roberta-wwm-ext':BertSeriesConfig,
    'hfl/chinese-roberta-wwm-ext-large':BertSeriesConfig,
    'xlm-roberta-large':RobertaSeriesConfig,
    'xlm-roberta-base':RobertaSeriesConfig,
    'bert-base-uncased':BertSeriesConfig,
    'bert':BertSeriesConfig,
    'xlm-roberta':RobertaSeriesConfig,
    'clip_text_model':BertSeriesConfig,
}
    
STUDENT_MODEL_DICT={
    'hfl/chinese-roberta-wwm-ext':BertSeriesModelWithTransformation,
    'hfl/chinese-roberta-wwm-ext-large':BertSeriesModelWithTransformation,
    'xlm-roberta-large':RobertaSeriesModelWithTransformation,
    'xlm-roberta-base':RobertaSeriesModelWithTransformation,
    'bert':BertSeriesModelWithTransformation,
    'xlm-roberta':RobertaSeriesModelWithTransformation,
    'clip_text_model':BertSeriesModelWithTransformation,
}
