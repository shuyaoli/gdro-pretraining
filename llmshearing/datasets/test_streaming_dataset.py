import pytest
from collections import defaultdict
import numpy as np
from streaming.base.dataset import StreamingDataset
from llmshearing.datasets.streaming_dataset import (
    TextDynamicStreamingDataset, TextStreamingDataset
)
from llmshearing.callbacks.dynamic_loading_callback import DynamicLoadingCallback

@pytest.fixture
def text_dynamic_dataset():
    # Use the original parameters for realistic tests
    return TextDynamicStreamingDataset(
        local="/home/shuyaoli/LLM-Shearing/llm_data/LLM-Shearing/for_prune", 
        set_names=["cc", "github", "wiki"],
        proportion=[0.5, 0.25, 0.25],
        shuffle=True,
        is_uint16=True,
        max_seq_len=4096
    )


def test_update_proportion(text_dynamic_dataset):
    dataset = text_dynamic_dataset
    iter_dataset = iter(dataset)
    sets_before = defaultdict(int)
    for _ in range(1000):
        sample = next(iter_dataset)
        sets_before[sample["set"]] += 1
    # Initial distribution check (loose bounds due to randomness)
    assert abs(sets_before['cc'] / 1000 - 0.5) < 0.15
    assert abs(sets_before['github'] / 1000 - 0.25) < 0.15
    assert abs(sets_before['wiki'] / 1000 - 0.25) < 0.15

    # Update proportion and test again
    dataset.update_proportion([0.1, 0.2, 0.7], dataset.lambdas)
    sets_after = defaultdict(int)
    for _ in range(1000):
        sample = next(iter_dataset)
        sets_after[sample["set"]] += 1
    assert abs(sets_after['wiki'] / 1000 - 0.7) < 0.15


def test_load_state_dict(text_dynamic_dataset):
    dataset = text_dynamic_dataset
    used_sample_ids = [
        np.random.randint(0, 100, 10).tolist(),
        np.random.randint(100, 200, 10).tolist(),
        np.random.randint(200, 300, 10).tolist()
    ]
    state_dict = {
        "epoch": 0,
        "num_canonical_nodes": 128,
        "shuffle_seed": 9176,
        "proportion": [0.3, 0.2, 0.5],
        "lambdas": [0.3, 0.3, 0.4],
        "used_sample_ids": used_sample_ids
    }
    dataset.load_state_dict(state_dict)
    # There is no dataset.epoch, but we can check shuffle_seed and proportion
    assert dataset.shuffle_seed == 9176
    np.testing.assert_allclose(dataset.proportion, [0.3, 0.2, 0.5])
    np.testing.assert_allclose(dataset.lambdas, [0.3, 0.3, 0.4])


def test_dynamic_loading_callback_update_proportion():
    set_names = ["cc", "github", "wiki"]
    target_loss = [1.0, 2.0, 3.0]
    initial_prop = [0.5, 0.25, 0.25]
    callback = DynamicLoadingCallback(
        target_loss=target_loss,
        proportion=initial_prop,
        set_names=set_names,
        update_type="doremi"
    )
    # Simulate losses
    losses = [1.5, 2.0, 2.5]
    current_prop = [0.5, 0.25, 0.25]
    current_lambdas = [1/3, 1/3, 1/3]
    new_prop, new_lambdas = callback.update_proportion(current_prop, current_lambdas, losses)
    # Check that the new proportions sum to 1 and are different from the original
    assert pytest.approx(sum(new_prop), 0.01) == 1.0
    assert any(abs(p - cp) > 1e-3 for p, cp in zip(new_prop, current_prop))

    # Test for update_type="bandit"
    callback_bandit = DynamicLoadingCallback(
        target_loss=target_loss,
        proportion=initial_prop,
        set_names=set_names,
        update_type="bandit"
    )
    new_prop_bandit, new_lambdas_bandit = callback_bandit.update_proportion(current_prop, current_lambdas, losses)
    assert pytest.approx(sum(new_prop_bandit), 0.01) == 1.0
    assert any(abs(p - cp) > 1e-3 for p, cp in zip(new_prop_bandit, current_prop))

    # Test for update_type="pd-kl"
    callback_pdkl = DynamicLoadingCallback(
        target_loss=target_loss,
        proportion=initial_prop,
        set_names=set_names,
        update_type="pd-kl"
    )
    new_prop_pdkl, new_lambdas_pdkl = callback_pdkl.update_proportion(current_prop, current_lambdas, losses)
    assert pytest.approx(sum(new_prop_pdkl), 0.01) == 1.0
    assert any(abs(p - cp) > 1e-3 for p, cp in zip(new_prop_pdkl, current_prop))

    # Test for update_type="pd-chi-square"
    callback_pdchi = DynamicLoadingCallback(
        target_loss=target_loss,
        proportion=initial_prop,
        set_names=set_names,
        update_type="pd-chi-square"
    )
    # pd-chi-square expects initial_proportion attribute
    callback_pdchi.initial_proportion = initial_prop
    new_prop_pdchi, new_lambdas_pdchi = callback_pdchi.update_proportion(current_prop, current_lambdas, losses)
    assert pytest.approx(sum(new_prop_pdchi), 0.01) == 1.0
    assert any(abs(p - cp) > 1e-3 for p, cp in zip(new_prop_pdchi, current_prop))

    # Test for update_type="constant"
    callback_constant = DynamicLoadingCallback(
        target_loss=target_loss,
        proportion=initial_prop,
        set_names=set_names,
        update_type="constant"
    )
    new_prop_const, new_lambdas_const = callback_constant.update_proportion(current_prop, current_lambdas, losses)
    assert new_prop_const == current_prop
    assert new_lambdas_const == current_lambdas
