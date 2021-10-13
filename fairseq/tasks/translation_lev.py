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
import logging
logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@register_task("translation_lev")
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        # parser.add_argument(
        #     '--iter-decode-max-iter',
        #     default=0,
        #     type=int
        # )
        parser.add_argument(
            '--plain-ctc',
            action='store_true',
            default=False
        )
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
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
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
        model.train()
        train_ratio = max(0, min(1, update_num / self.args.max_update))
        sample["prev_target"] = self.inject_noise(sample["target"])
        sample["train_ratio"] = train_ratio
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

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

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        original_size = len(indices)
        if ignore_invalid_inputs and "ctc" in getattr(self.args, "arch"):
            max_positions = (
                (dataset.src_sizes[indices]).tolist(),
                (dataset.src_sizes[indices] * self.args.src_upsample_scale).tolist(), # (dataset.tgt_sizes[indices] * self.args.src_upsample_scale).tolist(),
            )
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            # logger.warning(
            #     (
            #         "{:,} samples have invalid sizes and will be skipped, "
            #         "max_positions={}, first few sample ids={}"
            #     ).format(len(ignored), max_positions, ignored[:10])
            # )
            logger.info(f"Dataset original size: {original_size}, filtered size: {len(indices)}")
        return indices

    # def _inference_with_bleu(self, generator, sample, model, direction="forward"):
    #     import sacrebleu
    #
    #     def decode(toks, dict, escape_unk=False):
    #         extra_symbols_to_ignore = []
    #         if hasattr(dict, "blank_index"): extra_symbols_to_ignore.append(dict.blank_index)
    #         if hasattr(dict, "mask_index"): extra_symbols_to_ignore.append(dict.mask_index)
    #         s = dict.string(
    #             toks.int().cpu(),
    #             self.args.eval_bleu_remove_bpe,
    #             # The default unknown string in fairseq is `<unk>`, but
    #             # this is tokenized by sacrebleu as `< unk >`, inflating
    #             # BLEU scores. Instead, we use a somewhat more verbose
    #             # alternative that is unlikely to appear in the real
    #             # reference, but doesn't get split into multiple tokens.
    #             unk_string=(
    #                 "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
    #             ),
    #             extra_symbols_to_ignore=extra_symbols_to_ignore or None
    #         )
    #         if self.tokenizer:
    #             s = self.tokenizer.decode(s)
    #         return s
    #
    #     gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
    #     hyps, refs, srcs = [], [], []
    #     for i in range(len(gen_out)):
    #         hyp = gen_out[i][0]['tokens']
    #         if hasattr(self.args, "ctc_loss"):
    #             _toks = hyp.int().tolist()
    #             hyp = hyp.new_tensor([v for i, v in enumerate(_toks) if i == 0 or v != _toks[i-1]])
    #
    #         hyps.append(decode(hyp, self.tgt_dict))
    #         refs.append(decode(
    #             utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
    #             # sample['target'][i],
    #             self.tgt_dict,
    #             escape_unk=True,  # don't count <unk> as matches to the hypo
    #         ))
    #         srcs.append(decode(
    #             utils.strip_pad(sample["net_input"]["src_tokens"][i], self.src_dict.pad()),
    #             # sample["net_input"]["src_tokens"][i],
    #             self.src_dict,
    #             escape_unk=True,  # don't count <unk> as matches to the hypo
    #         ))
    #
    #     lang_pair = self.direction_lang_pairs[direction]
    #     if self.args.eval_bleu_print_samples:
    #         logger.info(f'[{lang_pair}] example source    : ' + srcs[0])
    #         logger.info(f'[{lang_pair}] example reference : ' + refs[0])
    #         logger.info(f'[{lang_pair}] example hypothesis: ' + hyps[0])
    #
    #     if self.args.eval_tokenized_bleu:
    #         return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
    #     else:
    #         return sacrebleu.corpus_bleu(hyps, [refs])


