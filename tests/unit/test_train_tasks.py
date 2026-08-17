"""The two shipped trainer tasks, as far as they go without GPUs."""

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

def _design(uid, backbone, path, n=40):
    return {
        "uid": uid,
        "backbone_id": backbone,
        "path": path,
        "sequence": "A" * n,
        "pLDDT": 91.0,
        "pTM": 0.9,
        "pAE": 3.0,
    }


def test_training_dataframe_points_at_structures_not_sequences():
    """The label is the sequence inside the file at `path`, not a column."""
    from rome.train.mpnn import build_training_dataframe

    frame = build_training_dataframe([_design("d1", "bb1", "/designs/d1.pdb")])
    row = frame.iloc[0]
    assert row["path"] == "/designs/d1.pdb"
    assert row["example_id"] == "{['impress_r']}{d1}{1}{['A_1']}"
    assert row["assembly_id"] == "1"


def test_training_dataframe_has_every_column_the_weighting_reads():
    """n_peptide especially: foundry reads it unguarded, so a miss is a KeyError."""
    from rome.train.mpnn import build_training_dataframe

    frame = build_training_dataframe([_design("d1", "bb1", "/designs/d1.pdb")])
    for column in ("n_prot", "n_peptide", "n_nuc", "n_ligand", "q_pn_unit_is_loi",
                   "cluster", "n_non_atomized_tokens"):
        assert column in frame.columns, column
    assert frame.iloc[0]["n_prot"] == 1
    assert frame.iloc[0]["n_nuc"] == 0


def test_token_count_comes_from_the_sequence():
    from rome.train.mpnn import build_training_dataframe

    frame = build_training_dataframe([_design("d1", "bb1", "/d1.pdb", n=137)])
    assert frame.iloc[0]["n_non_atomized_tokens"] == 137


def test_short_designs_count_as_peptides():
    from rome.train.mpnn import PEPTIDE_MAX_RESIDUES, build_training_dataframe

    frame = build_training_dataframe([
        _design("short", "bb1", "/a.pdb", n=PEPTIDE_MAX_RESIDUES),
        _design("long", "bb1", "/b.pdb", n=PEPTIDE_MAX_RESIDUES + 1),
    ])
    assert list(frame["n_peptide"]) == [1, 0]


def test_designs_cluster_by_backbone_so_one_backbone_cannot_dominate():
    """Weight is beta/cluster_size, so clustering is the anti-domination knob."""
    from rome.train.mpnn import build_training_dataframe

    frame = build_training_dataframe([
        _design("d1", "bb1", "/1.pdb"),
        _design("d2", "bb1", "/2.pdb"),
        _design("d3", "bb2", "/3.pdb"),
    ])
    assert list(frame["cluster"]) == ["bb1", "bb1", "bb2"]


def test_a_design_without_a_backbone_becomes_its_own_cluster():
    from rome.train.mpnn import build_training_dataframe

    record = _design("d1", None, "/1.pdb")
    frame = build_training_dataframe([record])
    assert frame.iloc[0]["cluster"] == "d1"


def test_provenance_is_carried_into_the_shard():
    """The shard is the audit trail for what a round trained on."""
    from rome.train.mpnn import build_training_dataframe

    record = _design("d1", "bb1", "/1.pdb")
    record["model_version"] = 3
    frame = build_training_dataframe([record])
    assert frame.iloc[0]["pLDDT"] == 91.0
    assert frame.iloc[0]["produced_under_version"] == 3


def test_write_shard_produces_a_readable_parquet(tmp_path):
    import pandas as pd

    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "ck"))
    path = trainer.write_shard(
        [_design("d1", "bb1", "/1.pdb"), _design("d2", "bb2", "/2.pdb")],
        str(tmp_path),
    )
    assert list(pd.read_parquet(path)["design_id"]) == ["d1", "d2"]


def test_custom_train_func_receives_the_shard(tmp_path):
    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    seen = {}

    def my_train(shard_path, output_dir, config):
        seen["shard"] = shard_path
        return output_dir

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=my_train))
    result = trainer.train([_design("d1", "bb1", "/1.pdb")], str(tmp_path))
    assert result == str(tmp_path)
    assert os.path.exists(seen["shard"])


def test_mpnn_needs_a_structure_path_not_just_a_sequence():
    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    with pytest.raises(ValueError, match="path"):
        trainer.validate([{"sequence": "MKV"}])


def test_unknown_mpnn_model_type_is_rejected():
    from rome.train.mpnn import ProteinMPNNConfig

    with pytest.raises(ValueError, match="unknown model_type"):
        ProteinMPNNConfig(model_type="vibes").validate()


def test_latest_checkpoint_ignores_non_checkpoint_files(tmp_path):
    """foundry's own helper returns sorted(iterdir())[-1] — any stray file wins."""
    from rome.train.mpnn import latest_checkpoint_file

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch-0001.ckpt").write_text("a")
    (ckpt_dir / "epoch-0002.ckpt").write_text("b")
    (ckpt_dir / "zzz-stray.log").write_text("noise")

    assert latest_checkpoint_file(str(tmp_path)).endswith("epoch-0002.ckpt")


def test_latest_checkpoint_is_none_before_the_first_round(tmp_path):
    from rome.train.mpnn import latest_checkpoint_file

    assert latest_checkpoint_file(str(tmp_path)) is None


def test_resume_prefers_the_published_checkpoint_over_the_initial_one(tmp_path):
    from dataclasses import dataclass

    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    @dataclass
    class FakeCheckpointConfig:
        path: str
        weight_loading_config: object = None
        reset_optimizer: bool = False

    trainer = ProteinMPNNTrainer(
        ProteinMPNNConfig(initial_checkpoint="/legacy/converted.ckpt")
    )

    # Round 1: no published checkpoint -> the converted legacy weights, which
    # carry no optimizer state.
    first = trainer._resume_config(FakeCheckpointConfig)
    assert first.path == "/legacy/converted.ckpt"
    assert first.reset_optimizer is True

    # Round 2: resume the published checkpoint, keeping the Noam schedule.
    second = trainer._resume_config(FakeCheckpointConfig, model_path="/ckpt/epoch-0003.ckpt")
    assert second.path == "/ckpt/epoch-0003.ckpt"
    assert second.reset_optimizer is False


def test_resume_is_none_with_nothing_to_resume_from():
    from rome.train.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig())
    assert trainer._resume_config(object) is None


def test_noam_schedule_warms_up_then_decays():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "optim"):
        pytest.skip("torch is stubbed in this environment")

    from rome.train.mpnn import create_noam_scheduler

    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=1.0)
    scheduler = create_noam_scheduler(optimizer, d_model=128, warmup_steps=10)

    lrs = []
    for _ in range(30):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    peak = lrs.index(max(lrs))
    assert 0 < peak < len(lrs) - 1        # rises then falls
    assert lrs[0] < lrs[peak] > lrs[-1]


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"pLDDT": 95.0, "pTM": 0.92, "pAE": 3.0}, True),
        ({"pLDDT": 92.9, "pTM": 0.92, "pAE": 3.0}, False),  # pLDDT too low
        ({"pLDDT": 95.0, "pTM": 0.89, "pAE": 3.0}, False),  # pTM too low
        ({"pLDDT": 95.0, "pTM": 0.92, "pAE": 4.1}, False),  # pAE too high
        ({"score": 95.0}, True),                            # score stands in for pLDDT
        ({}, False),                                        # nothing to judge on
        # A design IMPRESS itself keeps is not automatically training material:
        # this clears IMPRESS's own bar and is still rejected.
        ({"pLDDT": 97.3, "pTM": 0.81, "pAE": 4.9}, False),
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
