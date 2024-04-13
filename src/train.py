from transformers import AutoTokenizer, AutoModelForTokenClassification


# Apresentar modelo com n-gram e depois apresentar o finetuning do modelo da hugging face
# Ajustar função de custo para penalizar erros na classe menos representada
# Under - oversmapling
# Realizar teste posterior com o dataset do cardniffnlp
# Como no @pysientimiento, realizar ajuste retirando o emoticon utilizado para query do texto

id2label = {0: "Neutro", 1: "Positivo", 2: "Negativo"}
label2id = {"Neutro": 0, "Positivo": 1, "Negativo": 2}

model_checkpoint = "neuralmind/bert-base-portuguese-cased"
