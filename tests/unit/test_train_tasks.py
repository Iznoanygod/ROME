"""The two shipped trainer tasks, as far as they go without GPUs."""

import json
import os

import pytest

from rome.train.base import FunctionTrainer, TrainTask


# -- TrainTask contract ----------------------------------------------------

def test_base_train_is_abstract():
    with pytest.raises(NotImplementedError):
        TrainTask().train([], "/tmp")


def test_validate_rejects_an_empty_dataset():
    with pytest.raises(ValueError, match="empty dataset"):
        TrainTask(name="t").validate([])


def test_checkpoint_layout_is_name_and_version(tmp_path):
    path = TrainTask.prepare_output_dir(str(tmp_path), 3, "grpo")
    assert path == os.path.join(str(tmp_path), "grpo", "v3")
    assert os.path.isdir(path)


def test_function_trainer_defaults_to_the_output_dir(tmp_path):
    trainer = FunctionTrainer(lambda dataset, output_dir, **kw: None)
    assert trainer.train([1], str(tmp_path)) == str(tmp_path)


def test_function_trainer_takes_the_functions_name():
    def my_finetune(dataset, output_dir, **kwargs):
        return output_dir

    assert FunctionTrainer(my_finetune).name == "my_finetune"


# -- ProteinMPNN -----------------------------------------------------------

def test_mpnn_writes_a_jsonl_shard_of_the_campaign_designs(tmp_path):
    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(
        ProteinMPNNConfig(backend="custom", train_func=lambda *a: "ck",
                          shard_format="jsonl")
    )
    records = [
        {"sequence": "MKV", "pdb_path": "/a.pdb", "pLDDT": 91.0, "pTM": 0.9,
         "pAE": 3.0, "junk": "dropped"},
        {"sequence": "MKW", "pdb_path": "/b.pdb", "pLDDT": 88.0, "pTM": 0.85,
         "pAE": 4.0},
    ]
    shard = trainer.write_shard(records, str(tmp_path))

    rows = [json.loads(line) for line in open(shard)]
    assert [r["sequence"] for r in rows] == ["MKV", "MKW"]
    assert "junk" not in rows[0]


def test_mpnn_custom_backend_gets_the_shard(tmp_path):
    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    seen = {}

    def my_train(shard_path, output_dir, config):
        seen["shard"] = shard_path
        return output_dir

    trainer = ProteinMPNNTrainer(
        ProteinMPNNConfig(backend="custom", train_func=my_train, shard_format="jsonl")
    )
    result = trainer.train(
        [{"sequence": "MKV", "pdb_path": "/a.pdb"}], str(tmp_path)
    )
    assert result == str(tmp_path)
    assert os.path.exists(seen["shard"])


def test_mpnn_needs_a_sequence_and_a_structure():
    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(
        ProteinMPNNConfig(backend="custom", train_func=lambda *a: "x")
    )
    with pytest.raises(ValueError, match="pdb_path"):
        trainer.validate([{"sequence": "MKV"}])


def test_mpnn_custom_backend_requires_a_train_func():
    from rome.train.mpnn import ProteinMPNNConfig

    with pytest.raises(ValueError, match="train_func is required"):
        ProteinMPNNConfig(backend="custom").validate()


def test_unknown_mpnn_backend_is_rejected():
    from rome.train.mpnn import ProteinMPNNConfig

    with pytest.raises(ValueError, match="unknown ProteinMPNN backend"):
        ProteinMPNNConfig(backend="vibes").validate()


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"pLDDT": 90.0, "pTM": 0.9, "pAE": 3.0}, True),
        ({"pLDDT": 79.9, "pTM": 0.9, "pAE": 3.0}, False),   # pLDDT too low
        ({"pLDDT": 90.0, "pTM": 0.7, "pAE": 3.0}, False),   # pTM too low
        ({"pLDDT": 90.0, "pTM": 0.9, "pAE": 5.1}, False),   # pAE too high
        ({"score": 90.0}, True),                            # score stands in for pLDDT
        ({}, False),                                        # nothing to judge on
    ],
)
def test_impress_corpus_filter(record, expected):
    from rome.train.mpnn import impress_corpus_filter

    assert impress_corpus_filter()(record) is expected


# -- LLM / GRPO ------------------------------------------------------------

def test_model_config_resolves_weights_and_tokenizer():
    from rome.train.llm import ModelConfig

    cfg = ModelConfig(base_model_name="base", model_name="finetuned")
    assert cfg.resolved_model_name() == "finetuned"    # prefer the fine-tune
    assert cfg.tokenizer_name() == "base"              # tokenizer from the base

    only_base = ModelConfig(base_model_name="base")
    assert only_base.resolved_model_name() == "base"


def test_model_config_needs_a_model():
    from rome.train.llm import ModelConfig

    with pytest.raises(ValueError, match="base_model_name or model_name"):
        ModelConfig().resolved_model_name()


def test_grpo_trainer_needs_a_model_config():
    from rome.train.llm import GRPOConfig, GRPOTrainer

    with pytest.raises(ValueError, match="model_config must be set"):
        GRPOTrainer(GRPOConfig())


def test_grpo_takes_its_gpus_from_the_model_config():
    from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

    task = GRPOTrainer(
        GRPOConfig(model_config=ModelConfig(base_model_name="m", required_gpus=8))
    )
    assert task.gpus == 8
    assert task.wants_hf_dataset is True


def test_grpo_rejects_a_corpus_without_prompts():
    from datasets import Dataset

    from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

    task = GRPOTrainer(GRPOConfig(model_config=ModelConfig(base_model_name="m")))
    dataset = Dataset.from_list([{"sequence": "MKV", "score": 1.0}])
    with pytest.raises(ValueError, match="needs a 'prompt' column"):
        task.validate(dataset)


def test_grpo_accepts_a_corpus_with_prompts():
    from datasets import Dataset

    from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

    task = GRPOTrainer(GRPOConfig(model_config=ModelConfig(base_model_name="m")))
    task.validate(Dataset.from_list([{"prompt": "2+2?", "score": 1.0}]))


def test_grpo_builds_a_trl_config_pointing_at_the_round_output(tmp_path):
    from rome.train.llm import GRPOConfig, ModelConfig

    config = GRPOConfig(
        model_config=ModelConfig(base_model_name="m"),
        learning_rate=1e-5,
        extra_args={"seed": 7},
    )
    trl_config = config.build_trl_config(str(tmp_path))
    assert trl_config.output_dir == str(tmp_path)
    assert trl_config.learning_rate == 1e-5
    assert trl_config.seed == 7
