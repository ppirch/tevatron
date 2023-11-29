import logging
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from dataclasses import dataclass, field
from tevatron.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.data import TrainDataset, QPCollator
from tevatron.modeling import AggretrieverModel
from tevatron.trainer import TevatronTrainer
from tevatron.datasets import HFTrainDataset

logger = logging.getLogger(__name__)


@dataclass
class AggretrieverModelArguments(ModelArguments):
    agg_dim: int = field(default=640)
    semi_aggregate: bool = field(default=False)
    skip_mlm: bool = field(default=False)
    vocab_size: int = field(default=30522)


class AggretrieverTrainer(TevatronTrainer):
    def __init__(self, train_n_passages, *args, **kwargs):
        super(AggretrieverTrainer, self).__init__(*args, **kwargs)
        self.world_size = 1
        if self.args.negatives_x_device:
            self.world_size = torch.distributed.get_world_size()
        self.train_n_passages = train_n_passages
        self.effective_bsz = self.args.per_device_train_batch_size * self.world_size \
            if self.args.negatives_x_device \
            else self.args.per_device_train_batch_size 
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def listwise_scores(self, q_reps, p_reps):
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        scores = scores.view(self.effective_bsz, self.train_n_passages, -1)
        scores = scores.view(self.effective_bsz, -1)
        return scores
    
    def compute_loss(self, model, inputs):
        query, passage = inputs
        output = model(query=query, passage=passage)
        q_lexical_reps = output.q_lexical_reps
        q_semantic_reps = output.q_semantic_reps
        p_lexical_reps = output.p_lexical_reps
        p_semantic_reps = output.p_semantic_reps

        lexical_scores = self.listwise_scores(q_lexical_reps, p_lexical_reps)

        if q_semantic_reps is not None:
            semantic_scores = self.listwise_scores(q_semantic_reps, p_semantic_reps)
        else:
            semantic_scores = 0

        scores = lexical_scores + semantic_scores
        loss = 0

        hard_label_scores = torch.arange(
                lexical_scores.size(0),
                device=lexical_scores.device,
                dtype=torch.long
        )

        hard_label_scores = hard_label_scores * self.train_n_passages
        hard_label_scores = torch.nn.functional.one_hot(hard_label_scores, num_classes=lexical_scores.size(1)).float()
        if q_semantic_reps is not None:
            loss += self.kl_loss(nn.functional.log_softmax(scores, dim=-1), hard_label_scores) + \
                    0.5 * self.kl_loss(nn.functional.log_softmax(lexical_scores, dim=-1), hard_label_scores) + \
                    0.5 * self.kl_loss(nn.functional.log_softmax(semantic_scores, dim=-1), hard_label_scores)
        else:
            loss += self.kl_loss(nn.functional.log_softmax(scores, dim=-1), hard_label_scores)
        if self.args.negatives_x_device:
            loss = loss * self.world_size
        return loss


def main():
    parser = HfArgumentParser((AggretrieverModelArguments, DataArguments, TevatronTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: AggretrieverModelArguments
        data_args: DataArguments
        training_args: TevatronTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    if training_args.cont:
        logger.info("Loading model from checkpoint")
        model = AggretrieverModel.load(
            model_args=model_args,
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AggretrieverModel.build(
            model_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    train_dataset = train_dataset.process()
    train_dataset = train_dataset.filter(lambda x: len(x["negatives"]) >= data_args.train_n_passages)
    train_dataset = TrainDataset(data_args, train_dataset, tokenizer)

    trainer = AggretrieverTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        train_n_passages=data_args.train_n_passages,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
