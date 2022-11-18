import logging

from fairseq import checkpoint_utils, utils, tasks
from fairseq.models.speech_to_text.xstnet import XSTNet, base_architecture
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from .MoyuNetEncoder import MoyuNetEncoder
from .MoyuNetDecoder import MoyuNetDecoder

logger = logging.getLogger(__name__)

@register_model('moyunet')
class MoyuNet(XSTNet):
    @staticmethod
    def add_args(parser):
        XSTNet.add_args(parser)
        parser.add_argument("--using-attn", action="store_true",
                            help="using attention as feature extract")

    @classmethod
    def build_encoder(cls, args, dict, embed_tokens):
        encoder = MoyuNetEncoder(args, dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder    
    
    @classmethod
    def build_decoder(cls, args, dict, embed_tokens):
        return MoyuNetDecoder(args, dict, embed_tokens)

@register_model_architecture('moyunet', 'moyunet')
def base_architecture_Moyunet(args):
    args.using_attn = getattr(args, "using_attn", False)
    base_architecture(args)
