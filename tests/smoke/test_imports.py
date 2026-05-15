"""Import-smoke tests.

Catches the class of bug that motivated this branch in the first place:
broken/typo'd imports, missing modules, or names not exported by their
declared parent.
"""


def test_rome_package_imports():
    import rome
    assert "Workflow" in rome.__all__
    assert "grpo" in rome.__all__
    assert "sft" in rome.__all__


def test_train_submodule_exposes_trainers():
    from rome.train import GRPO, SFT
    assert GRPO.__name__ == "GRPO"
    assert SFT.__name__ == "SFT"


def test_flows_submodule_imports():
    from rome.flows.sequentialflow import SequentialFlow, SequentialFlowConfig
    assert SequentialFlow.__name__ == "SequentialFlow"
    assert SequentialFlowConfig.__name__ == "SequentialFlowConfig"


def test_config_classes_importable():
    from rome.config import LoRAConfig, ModelConfig
    assert LoRAConfig.__name__ == "LoRAConfig"
    assert ModelConfig.__name__ == "ModelConfig"


def test_trainer_workflow_importable():
    from rome.trainer import Trainer
    from rome.workflow import Workflow
    assert Trainer.__name__ == "Trainer"
    assert Workflow.__name__ == "Workflow"
