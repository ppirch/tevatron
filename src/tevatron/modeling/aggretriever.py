# Modified from https://github.com/castorini/dhr 

import os
import copy
import json
import logging

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel

from ..arguments import ModelArguments, TrainingArguments
from .encoder import EncoderOutput, EncoderModel


logger = logging.getLogger(__name__)


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True, 
            name='pooler'
    ):
        super(LinearPooler, self).__init__()
        self.name = name
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, '{}.pt'.format(self.name))
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, '{}.pt'.format(self.name)), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training {} from scratch".format(self.name))
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, '{}.pt'.format(self.name)))
        with open(os.path.join(save_path, '{}_config.json').format(self.name), 'w') as f:
            json.dump(self._config, f)


@dataclass
class AggretrieverModelArguments(ModelArguments):
    agg_dim: int = field(default=640)
    semi_aggregate: bool = field(default=False)
    skip_mlm: bool = field(default=False)
    vocab_size: int = field(default=30522)

@dataclass
class AggretrieverOutput(EncoderOutput):
    q_lexical_reps: Optional[Tensor] = None
    q_semantic_reps: Optional[Tensor] = None
    p_lexical_reps: Optional[Tensor] = None
    p_semantic_reps: Optional[Tensor] = None


class AggretrieverModel(EncoderModel):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            untie_encoder: bool = False,
            negatives_x_device: bool = False,
            agg_dim: int = 640,
            semi_aggregate: bool = False,
            term_weight_trans: nn.Module = None,
            vocab_size: int = 30522,
            skip_mlm: bool = False,
    ):
        super(AggretrieverModel, self).__init__(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder,
            negatives_x_device=negatives_x_device,
        )
        self.agg_dim = agg_dim
        self.semi_aggregate = semi_aggregate
        self.term_weight_trans = term_weight_trans
        self.vocab_size = vocab_size
        self.skip_mlm = skip_mlm
        self.softmax = nn.Softmax(dim=-1)

    def cal_remove_dim(self, dims, vocab_size=30522):
        remove_dims = vocab_size % dims
        if remove_dims > 1000:  # the first 1000 tokens in BERT are useless
            remove_dims -= dims
        return remove_dims

    def aggregate(self,
                  lexical_reps: Tensor,
                  dims: int = 640,
                  remove_dims: int = -198,
                  full: bool = True,
                  ):
        if full:
            remove_dims = self.cal_remove_dim(dims*2, self.vocab_size)
            batch_size = lexical_reps.shape[0]
            if remove_dims >= 0:
                lexical_reps = lexical_reps[:, remove_dims:].view(batch_size, -1, dims*2)
            else:
                lexical_reps = torch.nn.functional.pad(lexical_reps, (0, -remove_dims), "constant", 0).view(batch_size, -1, dims*2)

            tok_reps, _ = lexical_reps.max(1)

            positive_tok_reps = tok_reps[:, 0:2*dims:2]
            negative_tok_reps = tok_reps[:, 1:2*dims:2]

            positive_mask = positive_tok_reps > negative_tok_reps
            negative_mask = positive_tok_reps <= negative_tok_reps
            tok_reps = positive_tok_reps * positive_mask - negative_tok_reps * negative_mask
        else:
            remove_dims = self.cal_remove_dim(dims, self.vocab_size)
            batch_size = lexical_reps.shape[0]
            lexical_reps = lexical_reps[:, remove_dims:].view(batch_size, -1, dims)
            tok_reps, index_reps = lexical_reps.max(1)

        return tok_reps

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, **kwargs):
        q_lexical_reps, q_semantic_reps = self.encode_query(query)
        p_lexical_reps, p_semantic_reps = self.encode_passage(passage)

        q_reps, p_reps = None, None
        if query is not None:
            q_lexical_reps = self.aggregate(q_lexical_reps, self.agg_dim, full=not self.semi_aggregate)
            if q_semantic_reps is not None:
                q_reps = self.merge_reps(q_lexical_reps, q_semantic_reps)
            else:
                q_reps = q_lexical_reps
        if passage is not None:
            p_lexical_reps = self.aggregate(p_lexical_reps, self.agg_dim, full=not self.semi_aggregate)
            if p_semantic_reps is not None:
                p_reps = self.merge_reps(p_lexical_reps, p_semantic_reps)
            else:
                p_reps = p_lexical_reps
        
        if self.training and self.negatives_x_device:
            q_lexical_reps = self.dist_gather_tensor(q_lexical_reps)
            p_lexical_reps = self.dist_gather_tensor(p_lexical_reps)
            q_semantic_reps = self.dist_gather_tensor(q_semantic_reps)
            p_semantic_reps = self.dist_gather_tensor(p_semantic_reps)

        return AggretrieverOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            q_lexical_reps=q_lexical_reps,
            q_semantic_reps=q_semantic_reps,
            p_lexical_reps=p_lexical_reps,
            p_semantic_reps=p_semantic_reps,
        )

    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True, output_hidden_states=True)
        p_seq_hidden = psg_out.hidden_states[-1]
        p_cls_hidden = p_seq_hidden[:, 0]  # get [CLS] embeddings
        p_term_weights = self.term_weight_trans(p_seq_hidden[:, 1:])  # batch, seq, 1

        if not self.skip_mlm:
            p_logits = psg_out.logits[:, 1:]  # batch, seq, vocab
            p_logits = self.softmax(p_logits)
            attention_mask = psg['attention_mask'][:, 1:].unsqueeze(-1)
            p_lexical_reps = torch.max((p_logits * p_term_weights) * attention_mask, dim=-2).values
        else:
            # w/o MLM
            # p_term_weights = torch.relu(p_term_weights)
            p_lexical_reps = torch.zeros(p_seq_hidden.shape[0], p_seq_hidden.shape[1], self.vocab_size, dtype=p_seq_hidden.dtype, device=p_seq_hidden.device)  # (batch, seq, vocab)
            p_lexical_reps = torch.scatter(p_lexical_reps, dim=-1, index=psg.input_ids[:, 1:, None], src=p_term_weights)
            p_lexical_reps = p_lexical_reps.max(-2).values

        if self.pooler is not None:
            p_semantic_reps = self.pooler(p=p_cls_hidden)  # D * d
        else:
            p_semantic_reps = None

        return p_lexical_reps, p_semantic_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None

        qry_out = self.lm_q(**qry, return_dict=True, output_hidden_states=True)
        q_seq_hidden = qry_out.hidden_states[-1]
        q_cls_hidden = q_seq_hidden[:, 0]  # get [CLS] embeddings

        q_term_weights = self.term_weight_trans(q_seq_hidden[:, 1:])  # batch, seq, 1

        if not self.skip_mlm:
            q_logits = qry_out.logits[:, 1:]  # batch, seq-1, vocab
            q_logits = self.softmax(q_logits)
            attention_mask = qry['attention_mask'][:, 1:].unsqueeze(-1)
            q_lexical_reps = torch.max((q_logits * q_term_weights) * attention_mask, dim=-2).values
        else:
            # w/o MLM
            # q_term_weights = torch.relu(q_term_weights)
            q_lexical_reps = torch.zeros(q_seq_hidden.shape[0], q_seq_hidden.shape[1], 30522, dtype=q_seq_hidden.dtype, device=q_seq_hidden.device)  # (batch, len, vocab)
            q_lexical_reps = torch.scatter(q_lexical_reps, dim=-1, index=qry.input_ids[:, 1:, None], src=q_term_weights)
            q_lexical_reps = q_lexical_reps.max(-2).values

        if self.pooler is not None:
            q_semantic_reps = self.pooler(q=q_cls_hidden)
        else:
            q_semantic_reps = None

        return q_lexical_reps, q_semantic_reps

    @staticmethod
    def merge_reps(lexical_reps, semantic_reps):
        dim = lexical_reps.shape[1] + semantic_reps.shape[1]
        merged_reps = torch.zeros(lexical_reps.shape[0], dim, dtype=lexical_reps.dtype, device=lexical_reps.device)
        merged_reps[:, :lexical_reps.shape[1]] = lexical_reps
        merged_reps[:, lexical_reps.shape[1]:] = semantic_reps
        return merged_reps

    @staticmethod
    def build_term_weight_transform(model_args):
        term_weight_trans = LinearPooler(
            model_args.projection_in_dim,
            1,
            tied=not model_args.untie_encoder,
            name='term_weight_trans'
        )
        term_weight_trans.load(model_args.model_name_or_path)
        return term_weight_trans

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder,
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
    
    @classmethod
    def build(
            cls,
            model_args: AggretrieverModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_q = AutoModel.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                if not model_args.skip_mlm:
                    lm_p = AutoModelForMaskedLM.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_p = AutoModel.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
            else:
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                else:
                    lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            if not model_args.skip_mlm:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            else:
                lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler and not (model_args.projection_out_dim == 0):
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        term_weight_trans = cls.build_term_weight_transform(model_args)

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            agg_dim=model_args.agg_dim,
            semi_aggregate=model_args.semi_aggregate,
            term_weight_trans=term_weight_trans,
            skip_mlm=model_args.skip_mlm,
            vocab_size=model_args.vocab_size,
            untie_encoder=model_args.untie_encoder,
            negatives_x_device=train_args.negatives_x_device,
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path: str,
            model_args: AggretrieverModelArguments,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        model_name_or_path = model_args.model_name_or_path
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_q = AutoModel.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                if not model_args.skip_mlm:
                    lm_p = AutoModelForMaskedLM.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_p = AutoModel.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **hf_kwargs)
                else:
                    lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            if not model_args.skip_mlm:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **hf_kwargs)
            else:
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = LinearPooler(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        term_weight_trans_weights = os.path.join(model_name_or_path, 'term_weight_trans.pt')
        term_weight_trans_config = os.path.join(model_name_or_path, 'term_weight_trans_config.json')
        if os.path.exists(term_weight_trans_weights) and os.path.exists(term_weight_trans_config):
            logger.info(f'found term_weight_trans weight and configuration')
            with open(term_weight_trans_config) as f:
                term_weight_trans_config_dict = json.load(f)
            # Todo: add name to config
            term_weight_trans_config_dict['name'] = 'term_weight_trans'
            term_weight_trans = LinearPooler(**term_weight_trans_config_dict)
            term_weight_trans.load(model_name_or_path)
        else:
            term_weight_trans = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            term_weight_trans=term_weight_trans,
            agg_dim=model_args.agg_dim,
            semi_aggregate=model_args.semi_aggregate,
            skip_mlm=model_args.skip_mlm,
            vocab_size=model_args.vocab_size,
            untie_encoder=model_args.untie_encoder,
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.pooler is not None:
            self.pooler.save_pooler(output_dir)
        self.term_weight_trans.save_pooler(output_dir)
