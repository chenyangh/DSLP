# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from json import decoder
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.fairseq_decoder import FairseqDecoder
import torch
import logging
from .nat_sd_shared import NATransformerDecoder
import random
from lunanlp import torch_seed

logger = logging.getLogger(__name__)


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("nat_sd_glat")
class NATransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.inference_decoder_layer = getattr(args, 'inference_decoder_layer', -1)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--restore-decoder-from",
            default="off",
            action="store",
        )

        # parser.add_argument(
        #     '--hard-argmax',
        #     action='store_true',
        #     default=False
        # )
        # parser.add_argument(
        #     '--yhat-temp',
        #     type=float,
        #     default=0.1
        # )

        parser.add_argument(
            '--share-ffn',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--share-attn',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--inference-decoder-layer',
            type=int,
            default=-1
        )

        parser.add_argument(
            '--sample-option',
            type=str,
            default='hard'
        )

        parser.add_argument(
            '--softmax-temp',
            type=float,
            default=1
        )
        parser.add_argument(
            '--temp-anneal',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--num-topk',
            default=1,
            type=int
        )
        parser.add_argument(
            '--force-detach',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--concat-yhat',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--concat-dropout',
            type=float,
            default=0
        )
        parser.add_argument(
            '--layer-drop-ratio',
            type=float,
            default=0.0
        )
        parser.add_argument(
            '--all-layer-drop',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--yhat-posemb',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dropout-anneal',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--dropout-anneal-end-ratio',
            type=float,
            default=0
        )
        parser.add_argument(
            '--full-layer-loss',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--length-ls',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--length-dropout',
            type=float,
            default=0
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, train_ratio=None, **kwargs
    ):
        if train_ratio is not None:
            self.encoder.train_ratio = train_ratio
            self.decoder.train_ratio = train_ratio

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )
        tgt_mask = tgt_tokens.ne(self.pad)
        rand_seed = random.randint(0, 19260817)
        # decoding
        glat_info = None
        anneal_info = None
        if glat and tgt_tokens is not None:
            if "context_p" in glat:
                with torch.no_grad():
                    with torch_seed(rand_seed):
                        word_ins_out_list = self.decoder(
                            normalize=False,
                            prev_output_tokens=prev_output_tokens,
                            encoder_out=encoder_out,
                            train_ratio=train_ratio
                        )
                    pred_tokens = word_ins_out_list[-1].argmax(-1)
                    nonpad_positions = tgt_mask
                    same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)
                    seq_lens = (nonpad_positions).sum(1)
                    keep_prob = ((seq_lens - same_num) / seq_lens * glat['context_p']).unsqueeze(-1)
                    # keep: True, drop: False
                    keep_word_mask = (torch.rand(prev_output_tokens.shape,
                                                 device=word_ins_out_list[-1].device) < keep_prob).bool()
                    glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask,
                                                                             0) + tgt_tokens.masked_fill(
                        ~keep_word_mask, 0)
                    glat_tgt_tokens = tgt_tokens.masked_fill(keep_word_mask, self.pad)

                    prev_output_tokens, tgt_tokens = glat_prev_output_tokens, glat_tgt_tokens

                    glat_info = {
                        "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                        "glat_context_p": glat['context_p'],
                        "glat_keep": keep_prob.mean().item()
                    }

                    all_layer_acc_list = []
                    for per_layer_output_logits in word_ins_out_list:
                        per_layer_acc = torch.div(
                            torch.sum((per_layer_output_logits.argmax(-1) == tgt_tokens) & tgt_mask, dim=-1,
                                      dtype=per_layer_output_logits.dtype),
                            torch.sum(tgt_mask, dim=-1))
                        all_layer_acc_list.append(per_layer_acc)
                    anneal_info = {
                        "glat_anneal": [torch.mean(x).item() for x in all_layer_acc_list]
                    }

        with torch_seed(rand_seed):
            word_ins_out_list = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )

        if self.args.length_ls:
            ret_val = {
                "length": {
                    "out": length_out,
                    "tgt": length_tgt,
                    "factor": self.decoder.length_loss_factor,
                    "ls": self.args.label_smoothing,
                    "nll_loss": True
                },
            }
        else:
            ret_val = {
                "length": {
                    "out": length_out,
                    "tgt": length_tgt,
                    "factor": self.decoder.length_loss_factor,
                },
            }

        for _idx, word_ins_out in enumerate(word_ins_out_list):
            ret_val[f"word_ins_{_idx}"] = {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 1.0 if self.args.full_layer_loss else 1.0 / self.decoder.num_layers,
            }

        if glat_info is not None:
            ret_val.update(glat_info)
        if anneal_info is not None:
            ret_val.update(anneal_info)
        return ret_val

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_logits_list = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )

        inference_decoder_layer = self.inference_decoder_layer

        output_logits = output_logits_list[inference_decoder_layer]  # take the last layer by default

        _scores, _tokens = output_logits.max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


@register_model_architecture(
    "nat_sd_glat", "nat_sd_glat_base"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nat_sd_glat", "nat_sd_glat"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nat_sd_glat", "nat_sd_glat_12d"
)
def base_architecture_12d(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nat_sd_glat", "nat_sd_glat_24d"
)
def base_architecture_24d(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nat_sd_glat", "nat_sd_glat_12e"
)
def big_architecture_12e(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    base_architecture(args)
