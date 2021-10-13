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
import numpy as np

class ModifiedLayerDropModuleList(torch.nn.ModuleList):
    # Note, this will also return index
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield i, m


def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    _device = logits.device
    _dtype = logits.dtype
    gumebel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=_device, dtype=_dtype),
                                                     torch.tensor(1., device=_device, dtype=_dtype))
    y_soft = torch.softmax(
        (logits + gumebel_dist.sample(logits.size())) / tau, dim=-1)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


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


@register_model("nat_sd_shared")
class NATransformerModel(FairseqNATModel):
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
            '--yhat-temp',
            type=float,
            default=0.1
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

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, rain_ratio=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out_list = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

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
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 1 / self.decoder.num_layers,
            }

        return ret_val

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )[-1].max(-1)

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

    def get_search_results(self, decoder_out, encoder_out, beam_size=None, decoding_format=None, **kwargs):

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
        output_logits = torch.log_softmax(output_logits, dim=-1)
        # NAT beam search
        # Ver. 1, dummy order
        # input: B * T * V
        B, T, V = output_logits.size()
        K = beam_size


        prev_top_k_scores, prev_top_k_index = torch.topk(output_logits[:, 0, :], k=beam_size, dim=1)
        top_k_scores = torch.zeros((B, K)).to(prev_top_k_scores)
        beam_results = torch.zeros((B, K, T)).to(prev_top_k_index)

        beam_results[:, :, 0] = prev_top_k_index
        for step in range(1, T):
            next_step_scores = output_logits[:, step, :]
            combined_scores = prev_top_k_scores.unsqueeze(-1) + next_step_scores.unsqueeze(1)
            top_k_scores, top_k_index = torch.topk(combined_scores.view(B, -1), k=beam_size, dim=1)
            beams_buf = top_k_index // V
            indices_buf = top_k_index.fmod(V)
            # combined_scores = beam_results[beams_buf]
            prev_path = beam_results[:, :, : step]
            beam_results[:, :, : step] = prev_path.gather(1, beams_buf.unsqueeze(2).repeat(1, 1, step))
            beam_results[:, :, step] = indices_buf
            prev_top_k_scores = top_k_scores

        return beam_results, top_k_scores

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


class NATransformerDecoder(FairseqNATSharedDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)
        self.softcopy = getattr(args, "softcopy", False)
        if self.softcopy:
            self.softcopy_learnable = self.args.softcopy_temp == 0
            if self.softcopy_learnable:
                self.para_softcopy_temp = torch.nn.Parameter(torch.tensor(1.0))
        else:
            self.softcopy_learnable =False
        self.concat_yhat = getattr(args, 'concat_yhat', False)
        if self.concat_yhat:
            if self.share_attn and self.share_ffn:
                self.reduce_concat = torch.nn.ModuleList()
                first_concat = torch.nn.Linear(self.args.decoder_embed_dim*2, self.args.decoder_embed_dim, bias=False)
                self.reduce_concat.append(first_concat)
                second_concat = torch.nn.Linear(self.args.decoder_embed_dim*2, self.args.decoder_embed_dim, bias=False)
                self.reduce_concat.append(second_concat)
                for _ in range(self.num_layers-2):
                    self.reduce_concat.append(second_concat)
            else:
                self.reduce_concat = torch.nn.ModuleList(
                    [torch.nn.Linear(self.args.decoder_embed_dim*2, self.args.decoder_embed_dim, bias=False)
                                                          for _ in range(self.args.decoder_layers - 1)])
            if self.args.concat_dropout > 0:
                self.concat_dropout = torch.nn.Dropout(self.args.concat_dropout)
            else:
                self.concat_dropout = None

        self.all_layer_drop = getattr(args, 'all_layer_drop', False)
        self.layer_drop_ratio = getattr(args, 'layer_drop_ratio', 0.0)
        if self.layer_drop_ratio > 0:
            self.first_layer = self.layers[0]
            self.reset_layers = ModifiedLayerDropModuleList(self.layer_drop_ratio, self.layers[1:])
        self.all_layers = ModifiedLayerDropModuleList(self.layer_drop_ratio, self.layers)

        self.yhat_posemb = getattr(args, 'yhat_posemb', False)

        self.length_dropout = getattr(args, 'length_dropout', 0.0)
        # self.repeat_layer = getattr(args, 'repeat_layer', 0)

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, train_ratio=None, **unused):
        _, all_features = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            train_ratio=train_ratio
        )
        all_layer_output_logits = all_features['all_layer_output_logits']
        return [F.log_softmax(x.transpose(0, 1), -1) if normalize else x.transpose(0, 1)
                for x in all_layer_output_logits]

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        if self.length_dropout != 0:
            length_out = F.linear(enc_feats, torch.nn.Dropout(self.length_dropout)(self.embed_length.weight))
        else:
            length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            train_ratio=None,
            **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy or self.softcopy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            if not self.softcopy:
                x, decoder_padding_mask = self.forward_embedding(
                    prev_output_tokens,
                    self.forward_copying_source(
                        src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                    ),
                )
            else:
                x = self.forward_softcopying_source(src_embd, src_mask, prev_output_tokens.ne(self.padding_idx))
                decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)

        else:
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]
        # layer_output_list = []
        # layer_out = torch.zeros(x.size()).to(x)
        # decoder layers
        all_layer_output_logits = []
        # temperature = self.args.yhat_temp

        # NOTE: the first layer is special
        if not self.all_layer_drop and self.layer_drop_ratio > 0:
            x, attn, _ = self.first_layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)
            iterated_layers = self.reset_layers
            is_first_layer_separately_calculated = True
            reduce_linear_index_offset = 1
        else:
            iterated_layers = self.all_layers
            is_first_layer_separately_calculated = False
            reduce_linear_index_offset = 0

        for i, layer in iterated_layers:
            # print(i, ', ', end='')
            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break
            # x_size = x.size()
            layer_out_logits = self.output_layer(x)
            # layer_out_logits =  torch.matmul(x.reshape(-1, x_size[-1]), self.output_projection.weight.detach().transpose(0, 1)).view(x_size[0], x_size[1], -1)

            # Note: training time
            if self.training:
                if train_ratio is not None and self.args.temp_anneal:
                    softmax_temp = train_ratio * 10 + 1
                else:
                    softmax_temp = self.args.softmax_temp

                if self.args.sample_option == 'softmax_sample':
                    with torch.no_grad():
                        samples = torch.multinomial(torch.softmax(layer_out_logits.detach() *
                                                                  softmax_temp, dim=-1).
                                                    view(-1, layer_out_logits.size(-1)), 1)
                    layer_out = self.embed_tokens(samples.view(-1)).view(x.size())
                elif self.args.sample_option == 'gumbel_st':
                    samples = _gumbel_softmax(layer_out_logits, tau=1 / softmax_temp,
                                              hard=True).view(-1, layer_out_logits.size(-1))
                    # samples = torch.softmax(layer_out_logits.detach() * self.args.softmax_temp, dim=-1).view(-1, layer_out_logits.size(-1))

                    layer_out = torch.matmul(samples, self.embed_tokens.weight).view(x.size())

                elif self.args.sample_option == 'gumbel_sm':
                    samples = _gumbel_softmax(layer_out_logits, tau=1 / softmax_temp,
                                              hard=False).view(-1, layer_out_logits.size(-1))
                    layer_out = torch.matmul(samples, self.embed_tokens.weight).view(x.size())
                # elif self.args.sample_option == 'softmax_sample':
                #     layer_out = self.embed_tokens(layer_out_logits.argmax(dim=-1))
                elif self.args.sample_option == 'topk':
                    bsz = x.size(0) * x.size(1)
                    with torch.no_grad():
                        topk_val, topk_idx = torch.topk(layer_out_logits, self.args.num_topk, sorted=False, dim=-1)
                        topk_k_weight = torch.softmax(topk_val * softmax_temp, dim=-1)
                    layer_out = torch.bmm(topk_k_weight.view(bsz, 1, self.args.num_topk),
                                          self.embed_tokens(topk_idx).view(bsz, self.args.num_topk, -1)).view(
                        x.size())
                elif self.args.sample_option == 'softmax_ss':
                    if self.args.force_detach:
                        with torch.no_grad():
                            weights = torch.softmax(layer_out_logits.detach() * softmax_temp,
                                                    dim=-1).view(-1, layer_out_logits.size(-1))
                        layer_out = torch.matmul(weights, self.embed_tokens.weight).view(x.size())
                    else:
                        layer_out = torch.matmul(
                            torch.softmax(layer_out_logits * softmax_temp,
                                          dim=-1).view(-1, layer_out_logits.size(-1)),
                            self.embed_tokens.weight).view(x.size())
                elif self.args.sample_option == 'hard':
                    if self.yhat_posemb:
                        layer_out = self.forward_embedding(layer_out_logits.argmax(dim=-1).transpose(0, 1))[0].transpose(0, 1)
                    else:
                        layer_out = self.embed_tokens(layer_out_logits.argmax(dim=-1))
                else:
                    raise NotImplementedError
            # NOTE: inference time
            else:
                if self.args.temp_anneal or self.args.sample_option in ['hard', 'gumbel_st', 'gumbel_sm', 'softmax_sample']:
                    if self.yhat_posemb:
                        layer_out = self.forward_embedding(layer_out_logits.argmax(dim=-1).transpose(0, 1))[0].transpose(0, 1)
                    else:
                        layer_out = self.embed_tokens(layer_out_logits.argmax(dim=-1))
                elif self.args.sample_option == 'topk':
                    bsz = x.size(0) * x.size(1)
                    topk_val, topk_idx = torch.topk(layer_out_logits, self.args.num_topk, sorted=False, dim=-1)
                    topk_k_weight = torch.softmax(topk_val * self.args.softmax_temp, dim=-1)
                    layer_out = torch.bmm(topk_k_weight.view(bsz, 1, self.args.num_topk),
                                          self.embed_tokens(topk_idx).view(bsz, self.args.num_topk, -1)).view(
                        x.size())
                elif self.args.sample_option == 'softmax_ss':
                    weights = torch.softmax(layer_out_logits * self.args.softmax_temp,
                                            dim=-1).view(-1, layer_out_logits.size(-1))
                    layer_out = torch.matmul(weights, self.embed_tokens.weight).view(x.size())
                else:
                    raise NotImplementedError

            if not is_first_layer_separately_calculated and i == 0:
                new_x = x
            else:
                all_layer_output_logits.append(layer_out_logits)
                if not self.concat_yhat:
                    new_x = (x + layer_out) / torch.sqrt(torch.tensor(2.))
                else:
                    new_x = torch.cat((x, layer_out), dim=-1)
                    if self.concat_dropout is not None:
                        new_x = self.concat_dropout(self.reduce_concat[i - 1 + reduce_linear_index_offset](new_x))
                    else:
                        new_x = self.reduce_concat[i - 1 + reduce_linear_index_offset](new_x)


            x, attn, _ = layer(
                new_x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)
        # print('\n---------')

        all_layer_output_logits.append(self.output_layer(x))

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states, "all_layer_output_logits": all_layer_output_logits}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions

        if self.dropout_anneal:
            x = self.dropout_module(x, self.train_ratio)
        else:
            x = self.dropout_module(x)

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_softcopying_source(self, src_embeds, src_masks, tgt_masks):
        # length_sources = torch.randint(1, 26, (src_embeds.size(0), )).to(src_embeds) # src_masks.sum(1)
        # length_targets = torch.randint(1, 52, (src_embeds.size(0), )).to(src_embeds) # tgt_masks.sum(1)
        length_sources = src_masks.sum(1)
        length_targets =  tgt_masks.sum(1)
        src_len_mat = torch.div(
            (torch.arange(src_embeds.size(1), device=src_embeds.device, dtype=src_embeds.dtype) ).unsqueeze(
                0).repeat(src_embeds.size(0), 1), length_sources.unsqueeze(1))
        tgt_len_mat = torch.div(
            (torch.arange(tgt_masks.size(1), device=src_embeds.device, dtype=src_embeds.dtype)).unsqueeze(
                0).repeat(src_embeds.size(0), 1), length_targets.unsqueeze(1))
        # test_sum = torch.relu(torch.einsum('km,kn->kmn', tgt_len_mat, -src_len_mat))
        # k = src_len_mat.size(0)
        m = src_len_mat.size(1)
        n = tgt_len_mat.size(1)
        # test_sum2 = torch.zeros(k, n, m)
        # for _k in range(k):
        #     for _n in range(n):
        #         for _m in range(m):
        #             test_sum2[_k, _n, _m] = torch.abs(tgt_len_mat[_k, _n] - src_len_mat[_k, _m])
        test_sum3 = - torch.abs(tgt_len_mat.unsqueeze(2).repeat(1, 1, m) - src_len_mat.unsqueeze(1).repeat(1, n, 1))
        # src_mask_2 = torch.arange(src_embeds.size(1)).expand(src_embeds.size(0), src_embeds.size(1)).to(length_sources) < length_sources.unsqueeze(1)
        test_sum3_2 = test_sum3.masked_fill(~src_masks.unsqueeze(1), -float("Inf"))
        if not self.softcopy_learnable:
            src_weight = torch.softmax(test_sum3_2 * self.args.softcopy_temp, dim=2)
        else:
            src_weight = torch.softmax(test_sum3_2 * self.para_softcopy_temp, dim=2)
        copied_embedding = torch.bmm(src_weight, src_embeds)

        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt


@register_model_architecture(
    "nat_sd_shared", "nat_sd_shared"
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

