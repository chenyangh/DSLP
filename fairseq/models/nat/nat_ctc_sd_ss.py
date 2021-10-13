# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATSharedDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Union
import logging
import random, math
from .nat_sd_ss import NATransformerDecoder
from fairseq.torch_imputer import best_alignment, imputer_loss
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


def _gumbel_softmax(logits, temperature=1.0, withnoise=True, hard=True, eps=1e-10):
    def sample_gumbel(shape, eps=1e-10):
        U = torch.rand(shape).cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    if withnoise:
        gumbels = sample_gumbel(logits.size())
        y_soft = ((logits + gumbels) * 1.0 / temperature).softmax(2)
    else:
        y_soft = (logits * 1.0 / temperature).softmax(2)

    index = y_soft.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(2, index, 1.0)
    ret = (y_hard - y_soft).detach() + y_soft
    if hard:
        return ret
    else:
        return y_soft


@register_model("nat_ctc_sd_ss")
class NATransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        from ctcdecode import CTCBeamDecoder
        self.ctc_decoder = CTCBeamDecoder(
            decoder.dictionary.symbols,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=args.ctc_beam_size,
            num_processes=20,
            blank_id=decoder.dictionary.blank_index,
            log_probs_input=False
        )
        self.inference_decoder_layer = getattr(args, 'inference_decoder_layer', -1)
        self.plain_ctc = getattr(args, 'plain_ctc', False)

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
            "--src-upsample-scale",
            type=int,
            default=1
        )
        parser.add_argument(
            '--use-ctc-decoder',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--ctc-beam-size',
            default=1,
            type=int
        )
        parser.add_argument(
            '--ctc-beam-size-train',
            default=1,
            type=int
        )
        parser.add_argument(
            '--num-cross-layer-sample',
            default=0,
            type=int
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
            '--inference-decoder-layer',
            type=int,
            default=-1
        )
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
            '--copy-src-token',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--force-detach',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--softcopy',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--softcopy-temp',
            default=5,
            type=float
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
            '--force-ls',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--repeat-layer',
            type=int,
            default=0
        )
        parser.add_argument(
            '--masked-loss',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--ss-ratio',
            type=float,
            default=0.3
        )
        parser.add_argument(
            '--fixed-ss-ratio',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--no-empty',
            action='store_true',
            default=False
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        decoder.repeat_layer = getattr(args, 'repeat_layer', 0)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0,
                                      reduce=True,
                                      force_emit=None
                                      ) -> torch.FloatTensor:
        # # lengths : (batch_size, )
        # if self.args.force_ls:  # NOTE temp fix, to really try ls without mess up previous exps
        #     label_smoothing = self.args.label_smoothing
        # else:
        #     label_smoothing = label_smoothing
        # calculated by counting number of mask
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        targets = targets[target_mask.bool()]
        

        if self.args.masked_loss and force_emit is not None:
            loss = imputer_loss(
                    log_probs_T.float(),
                    targets,
                    force_emit,
                    logit_lengths,
                    target_lengths,
                    blank=blank_index,
                    reduction="mean",
                    zero_infinity=True,
                )
        else:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="mean",
                zero_infinity=True,
            )

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, reduce=True, train_ratio=None, **kwargs
    ):
        if train_ratio is not None:
            self.encoder.train_ratio = train_ratio
            self.decoder.train_ratio = train_ratio

        prev_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        target_mask = tgt_tokens.ne(self.pad)
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
        rand_seed = random.randint(0, 19260817)
        if train_ratio is not None:
            # NOTE: Find best CTC alignments as pseudo GT
            with torch.no_grad():
                # decoding
                with torch_seed(rand_seed):
                    output_logits_list = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                        train_ratio=train_ratio
                    )
                target_mask = tgt_tokens.ne(self.pad)
                target_length = target_mask.sum(dim=-1) 

                output_masks = prev_output_tokens.ne(self.pad)
                output_length = output_masks.sum(dim=-1)

                logits_T = output_logits_list[-1].transpose(0, 1).float()

                best_aligns = best_alignment(logits_T, tgt_tokens, output_length, target_length, self.pad, zero_infinity=True)
                best_aligns_pad = torch.tensor([a + [0] * (logits_T.size(0) - len(a)) for a in best_aligns], device=logits_T.device, dtype=tgt_tokens.dtype)
                oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1]-1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                if not self.args.no_empty:
                    oracle = oracle.masked_fill(best_aligns_pad % 2 == 0, self.tgt_dict.mask_index)
            

            if self.args.fixed_ss_ratio:
                ss_ratio = self.args.ss_ratio
            else:
                ss_ratio = self.args.ss_ratio * (1 - train_ratio)
                
            ss_mask = (torch.rand(oracle.size(), device=oracle.device) < ss_ratio).bool()
            force_emit = best_aligns_pad.masked_fill(~ss_mask, -1)
        else:
            ss_mask = None
            force_emit = None
            oracle = None


        with torch_seed(rand_seed):
            output_logits_list = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            tgt_tokens=oracle,
            ss_mask=ss_mask
        )

        if self.args.num_cross_layer_sample != 0:
            output_logits_list = torch.stack(output_logits_list, dim=0)

            N_SAMPLE = self.args.num_cross_layer_sample

            num_decoder_layer = output_logits_list.size(0)
            num_tokens = prev_output_tokens.size(1)
            num_vocab = output_logits_list.size(-1)
            batch_size = prev_output_tokens.size(0)

            all_sample_ctc_loss = 0

            for _ in range(N_SAMPLE):
                cross_layer_sampled_ids_ts = torch.randint(num_decoder_layer, (batch_size * num_tokens,), device=output_logits_list.device)
                output_logits_list = output_logits_list.view(num_decoder_layer, -1, num_vocab)
                gather_idx = cross_layer_sampled_ids_ts.unsqueeze(1).expand(-1, num_vocab).unsqueeze(0)
                gather_logits = output_logits_list.gather(0, gather_idx).view(batch_size, num_tokens, num_vocab)

                # if self.args.use_ctc:
                ctc_loss = self.sequence_ctc_loss_with_logits(
                    logits=gather_logits,
                    logit_mask=prev_output_tokens_mask,
                    targets=tgt_tokens,
                    target_mask=target_mask,
                    blank_index=self.tgt_dict.blank_index,
                    label_smoothing=self.args.label_smoothing,
                    reduce=reduce,
                    force_emit=force_emit
                )
                all_sample_ctc_loss += ctc_loss

            ret_val = {
                "ctc_loss": {"loss": all_sample_ctc_loss / len(output_logits_list)},
            }

        else:
            # if self.args.use_ctc:
            all_layer_ctc_loss = 0
            normalized_factor = 0
            for layer_idx, word_ins_out in enumerate(output_logits_list):
                ctc_loss = self.sequence_ctc_loss_with_logits(
                    logits=word_ins_out,
                    logit_mask=prev_output_tokens_mask,
                    targets=tgt_tokens,
                    target_mask=target_mask,
                    blank_index=self.tgt_dict.blank_index,
                    label_smoothing=self.args.label_smoothing, #NOTE: enable and double check with it later
                    reduce=reduce,
                    force_emit=force_emit
                )
                factor = 1  # math.sqrt(layer_idx + 1)
                all_layer_ctc_loss += ctc_loss * factor
                normalized_factor += factor
            ret_val = {
                "ctc_loss": {"loss": all_layer_ctc_loss / normalized_factor},
            }


        softcopy_learnable = getattr(self.decoder, "softcopy_learnable", False)
        if softcopy_learnable:
            ret_val.update(
                {
                    "stat:softcopy_temp": self.decoder.para_softcopy_temp.item()
                }
            )

        return ret_val

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # set CTC decoder beam size
        self.ctc_decoder._beam_width = self.args.ctc_beam_size
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history

        # execute the decoder
        output_logits_list = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )

        inference_decoder_layer = self.inference_decoder_layer
        output_logits = output_logits_list[inference_decoder_layer]

        if self.plain_ctc:
            output_scores = decoder_out.output_scores
            _scores, _tokens = output_logits.max(-1)
            output_masks = output_tokens.ne(self.pad)
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
        else:
            # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
            # _scores == beam_results[:,0,:]
            output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                     output_length)
            top_beam_tokens = beam_results[:, 0, :]
            top_beam_len = out_lens[:, 0]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
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

    def get_search_results(self, decoder_out, encoder_out, beam_size=None, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history
        # Set ctc beam size
        if beam_size is not None:
            self.ctc_decoder._beam_width = beam_size
        else:
            beam_size = self.ctc_decoder._beam_width

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
        beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                 output_length)

        beam_results = beam_results[:, :, :out_lens.max()]
        beam_size = beam_scores.size(1)
        for beam_idx in range(beam_size):
            top_beam_tokens = beam_results[:, beam_idx, :]
            top_beam_len = out_lens[:, beam_idx]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
        return beam_results, beam_scores

    def initialize_output_tokens_by_src_tokens(self, src_tokens):
        if not self.args.copy_src_token:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            if self.args.src_upsample_scale > 2:
                length_tgt = length_tgt * self.args.src_upsample_scale
            else:
                length_tgt = length_tgt * self.args.src_upsample_scale  # + 10
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
            return initial_output_tokens
        else:
            if self.args.src_upsample_scale <= 1:
                return src_tokens

            def _us(x, s):
                B = x.size(0)
                _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
                return _x

            return _us(src_tokens, self.args.src_upsample_scale)

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        initial_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)

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
    "nat_ctc_sd_ss", "nat_ctc_sd_ss"
)
def base_architecture(args):
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

