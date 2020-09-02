from fairseq.models import register_model_architecture

from encoder_decoder import Vae_encoder_decoder_Model

@register_model_architecture('bert_vae', 'bert_transformer_vae')
def bert_transformer_vae(args):
    pass
