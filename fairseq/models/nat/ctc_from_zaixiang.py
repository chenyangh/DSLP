# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.models.nat import NATransformerModelBase
from fairseq.models import register_model, register_model_architecture
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from typing import Union
import torch.nn.functional as F
import torch
from torch import Tensor


def sequence_ctc_loss_with_logits(
        logits: torch.FloatTensor,
        logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
        targets: torch.LongTensor,
        target_mask: Union[torch.FloatTensor, torch.BoolTensor],
        blank_index: torch.LongTensor,
        label_smoothing=0,
        reduction='mean',  # or batch_sum
) -> torch.FloatTensor:
    # lengths : (batch_size, )
    # calculated by counting number of mask
    logit_lengths = (logit_mask.bool()).long().sum(1)
    target_lengths = (target_mask.bool()).long().sum(1)

    # (batch_size, T, n_class)
    log_probs = logits.log_softmax(-1)
    # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
    log_probs_T = log_probs.transpose(0, 1)

    #     assert (target_lengths == 0).any()
    targets = targets.long()
    targets = targets[target_mask.bool()]

    loss = F.ctc_loss(
        log_probs_T.float(),  # compatible with fp16
        targets,
        logit_lengths,
        target_lengths,
        blank=blank_index,
        reduction="none" if reduction == 'batch_sum' else 'mean',
        zero_infinity=True,
    )

    if reduction == 'batch_sum':
        return loss

    # n_invalid_samples = (logit_lengths < target_lengths).long().sum()
    # if n_invalid_samples > 0:
    #     logger.warning(
    #         f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
    #     )
    #     raise Exception

    if label_smoothing > 0:
        # n_vocob = log_probs.size(-1)
        # kl_loss = F.kl_div(log_probs, torch.full_like(log_probs, 1/n_vocob), reduction='none', log_target=False).sum(-1)
        # # kl_loss = log_probs.neg().sum(-1) / n_vocob
        # kl_loss = ((kl_loss * logit_mask.float()).sum(-1) / logit_lengths)[logit_lengths >= target_lengths].mean()

        smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
        loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss

    return loss


@register_model("ctc_from_zaixiang")
class CTCNATModel(NATransformerModelBase):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.blank_index = getattr(decoder.dictionary, "blank_index", None)
        from ctcdecode import CTCBeamDecoder
        self.ctc_decoder = CTCBeamDecoder(
            decoder.dictionary.symbols,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=1,
            num_processes=20,
            blank_id=decoder.dictionary.blank_index,
            log_probs_input=False
        )

    @property
    def allow_ensemble(self):
        return False

    @staticmethod
    def add_args(parser):
        NATransformerModelBase.add_args(parser)
        parser.add_argument(
            "--ctc-loss",
            action="store_true",
            default=False,
            help="use custom param initialization for BERT",
        )    
        parser.add_argument(
            "--upsampling",
            type=int, metavar='N',
            default=1,
            help="upsampling ratio",
        )
        parser.add_argument(
            "--src-upsample-scale",
            type=int,
            default=2
        )
        parser.add_argument(
            "--upsampling-source",
            action="store_true",
            default=False,
            help="upsampling ratio",
        )
        parser.add_argument(
            '--use-ctc-decoder',
            action='store_true',
            default=True
        )

    def _full_mask(self, target_tokens):
        pad = self.pad
        bos = self.bos
        eos = self.eos
        unk = self.unk

        target_mask = target_tokens.eq(bos) | target_tokens.eq(
            eos) | target_tokens.eq(pad)
        return target_tokens.masked_fill(~target_mask, unk)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, train_ratio=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        if self.args.upsampling_source:
            prev_output_tokens = self._full_mask(src_tokens)
        prev_output_tokens = self._maybe_upsample(prev_output_tokens)

        # decoding
        logits = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out)
        logit_mask, target_mask = prev_output_tokens.ne(self.pad), tgt_tokens.ne(self.pad)

        # loss
        ctc_loss = sequence_ctc_loss_with_logits(
            logits=logits,
            logit_mask=logit_mask,
            targets=tgt_tokens,
            target_mask=target_mask,
            blank_index=self.blank_index,
            label_smoothing=self.args.label_smoothing,
        )

        net_output = {
            "word_pred_ctc": {
                "loss": ctc_loss,
                "nll_loss": False,
            },
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }

        return net_output

    def _maybe_upsample(self, tokens):
        if self.args.upsampling <= 1:
            return tokens

        def _us(x, s):
            B = x.size(0)
            _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
            return _x
        return _us(tokens, self.args.upsampling)

    def initialize_output_tokens(self, encoder_out, src_tokens, **kwargs):
        if self.args.upsampling_source:
            initial_output_tokens = self._full_mask(src_tokens)
        else:
            # length prediction
            length_tgt = self.decoder.forward_length_prediction(
                self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
                encoder_out=encoder_out
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

        # upsampling decoder input here
        initial_output_tokens = self._maybe_upsample(initial_output_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out['encoder_out'][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None
        )

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # set CTC decoder beam size
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history

        # execute the decoder
        output_logits = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
        # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
        # _scores == beam_results[:,0,:]
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
        beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1), output_length)
        top_beam_tokens = beam_results[:, 0, :]
        top_beam_len = out_lens[:, 0]
        mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len).\
            repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
        top_beam_tokens[~mask] = self.decoder.dictionary.pad()
        # output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=top_beam_tokens.to(output_logits.device),
            output_scores=torch.full(top_beam_tokens.size(), 1.0),
            attn=None,
            history=history,
        )

@register_model_architecture("ctc_from_zaixiang", "ctc_from_zaixiang")
def ctc_nat_base_architecture(args):
    args.ctc_loss = getattr(args, "ctc_loss", True)
    args.upsampling = getattr(args, "upsampling", 2)
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)  # NOTE
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
