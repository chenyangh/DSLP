# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange
import sacrebleu
from copy import deepcopy
EVAL_BLEU_ORDER = 4
from multiprocessing import Pool
import itertools
from argparse import Namespace


class MultiprocessingEncoder(object):
    def __init__(self, tgt_dict, tokenizer):
        self.tgt_dict = tgt_dict
        self.initializer()
        self.tokenizer = tokenizer

    def initializer(self):
        # global bpe, bleu
        self.bpe = self.tgt_dict.string
        self.bleu = sacrebleu.sentence_bleu

    def decode(self, tokens):
        # global bpe
        tokens = tokens[tokens!=1]
        tokens = self.bpe(tokens, '@@ ', unk_string=("UNKNOWNTOKENINREF"))
        tokens = self.tokenizer.decode(tokens)
        return tokens

    def get_bleu(self, pair):
        # global bleu
        hyp, ref = pair
        # return self.bleu([hyp], [[ref]], tokenize='none').score
        return self.bleu(hyp, [ref], smooth_method='exp').score

@register_task("translation_imitation_glat")
class TranslationImitationGlatTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        # global self_to_string
        # self_to_string = self.tgt_dict.string
        # global self_tokenizer
        # self_tokenizer = self.tokenizer

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        # fmt: on

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.args.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.args.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.args.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        self.bpe_deocde = MultiprocessingEncoder(self.tgt_dict, self.tokenizer)

        from fairseq.iterative_refinement_generator import IterativeImitationRefinementGenerator

        return IterativeImitationRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 0),  # NOTE;
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        sample["prev_target"] = self.inject_noise(sample["target"])

        #  1. Sample from model by searching
        # search_results.size() = batch * beam * length
        # search_scores.size() = batch * beam
        model.eval()  # Note: both train() and eval() make some sense
        with torch.no_grad():
            search_results, search_scores = self.sequence_generator.search_results(model, sample,
                                                        beam_size=self.args.ctc_beam_size_train)

        # 2. Rank results by bleu
        # 2.1 Get bleu scores per sentence: batch * beam
        model.train()
        search_results, search_results_bleu = self.get_per_sent_bleu_batch_by_beam(sample, search_results)

        # 2.2 Sort batch
        sorted_search_results, sorted_indices = torch.sort(search_results_bleu, dim=1, descending=True)

        all_loss = 0
        # 3 Training positive samples
        # positive_samples = torch.stack([search_results[i, x, :] for i, x in enumerate(sorted_indices[:, 0])], dim=0)
        positive_samples_ids = torch.gather(search_results, 1, sorted_indices[:, 0].unsqueeze(1).repeat(1, search_results.size(2)).unsqueeze(1)).squeeze(1)
        positive_samples = deepcopy(sample)
        positive_samples_target = positive_samples_ids.to(positive_samples['target'])

        # 3.1 Deal with the special case where BLUEs are all zero
        none_zero_bleu_cond = torch.sum(search_results_bleu, dim=1) != 0
        if not torch.all(none_zero_bleu_cond):
            original_target = deepcopy(positive_samples['target'])
            num_sample = original_target.size(0)
            max_len = max(positive_samples_target.size(1), original_target.size(1))

            if positive_samples_target.size(1) < max_len:
                target_fill_mat = torch.full((num_sample, max_len), self.tgt_dict.pad_index).to(positive_samples_target)
                target_fill_mat[:, :positive_samples_target.size(1)] = positive_samples_target
                positive_samples_target = target_fill_mat
            elif original_target.size(1) < max_len:
                target_fill_mat = torch.full((num_sample, max_len), self.tgt_dict.pad_index).to(original_target)
                target_fill_mat[:, :original_target.size(1)] = original_target
                original_target = target_fill_mat

            original_target[none_zero_bleu_cond] = positive_samples_target[none_zero_bleu_cond]
            positive_samples_target = original_target

        positive_samples['target'] = positive_samples_target
        positive_samples["prev_target"] = self.inject_noise(positive_samples["target"])  # Note this is unnecessary as we will regenerate pre_target anyways

        positive_loss, sample_size, logging_output = criterion(model, positive_samples)

        gt_sample_loss, sample_size, logging_output = criterion(model, sample)

        combined_loss = (positive_loss + gt_sample_loss) / 2
        optimizer.backward(combined_loss)

        # del positive_samples

        # # 4 Training negative samples
        # # Done by gradient accumulation
        # # negative_samples = sorted_indices[:, 1:]
        # negative_loss = torch.zeros(search_results.size(1) - 1)
        # for i in range(1, search_results.size(1)):
        #     per_beam_negative_sample_ids = torch.gather(search_results, 1, sorted_indices[:, i].unsqueeze(1).
        #                                                 repeat(1, search_results.size(2)).unsqueeze(1)).squeeze()
        #     per_beam_negative_samples = deepcopy(sample)
        #     per_beam_negative_samples['target'] = per_beam_negative_sample_ids.to(positive_samples_target)
        #     loss, sample_size, logging_output = criterion(model, per_beam_negative_samples)
        #     negative_loss[i-1] = loss
        #     del per_beam_negative_samples
        #
        # negative_loss = torch.logsumexp(negative_loss, 0)
        # optimizer.backward(-negative_loss)
        # # 3. Contrastive training
        # all_loss = positive_loss - negative_loss
        # # print(all_loss)
        return combined_loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def get_per_sent_bleu_batch_by_beam(self, sample, search_results):
        batch_size, beam_size, _ = search_results.size()
        from tqdm import tqdm

        # Note: this for loop can be accelerated.
        bleu_score_list = []
        ref_token_list = []
        hyp_token_list = []
        for batch_idx in range(batch_size):
            for beam_idx in range(beam_size):
                hyp_tokens = search_results[batch_idx, beam_idx]
                # hyp_str = decode(utils.strip_pad(hyp_tokens, self.tgt_dict.pad()))
                # ref_str = decode(utils.strip_pad(ref_tokens, self.tgt_dict.pad()), escape_unk=True)
                hyp_token_list.append(hyp_tokens)
            ref_tokens = sample['target'][batch_idx]
            ref_token_list.append(ref_tokens.to(search_results))

        bpe_deocde = self.bpe_deocde
        # import time
        # start_time = time.time()
        ref_str_list = [bpe_deocde.decode(x) for x in ref_token_list]
        hyp_str_list = [bpe_deocde.decode(x) for x in hyp_token_list]
        ref_str_list = list(itertools.chain.from_iterable(itertools.repeat(x, self.args.ctc_beam_size_train) for x in ref_str_list))
        bleu_score_list = [bpe_deocde.get_bleu(x) for x in zip(hyp_str_list, ref_str_list)]
        # print("--- %s seconds ---" % (time.time() - start_time))

        return search_results, torch.tensor(bleu_score_list).view(batch_size, beam_size).to(search_results.device)


