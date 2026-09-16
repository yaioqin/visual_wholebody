"""Three-process CPU tests; run without importing Isaac Gym.

python -m unittest legged_gym.tests.test_distributed_training -v
"""

import contextlib
import copy
from datetime import timedelta
import io
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from legged_gym.distributed_ppo import (
    DistributedPPO,
    average_gradients,
    global_mean,
    normalize_global_advantages,
)


WORLD_SIZE = 3
NUM_PROP = 3
NUM_PRIV = 2
NUM_HISTORY = 10
NUM_OBS = NUM_PROP + NUM_PRIV + NUM_HISTORY * NUM_PROP
NUM_ACTIONS = 18


def _assert_equal_across_ranks(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, tensor.detach().contiguous())
    for actual in gathered[1:]:
        torch.testing.assert_close(actual, gathered[0], rtol=0, atol=0)


def _assert_tree_close(actual, expected, *, exact=False):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=0 if exact else 2e-4,
            atol=0 if exact else 2e-6,
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_close(actual[key], expected[key], exact=exact)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_tree_close(value, reference, exact=exact)
    else:
        assert actual == expected, (actual, expected)


def _assert_optimizer_equal_across_ranks(optimizer):
    states = [None] * WORLD_SIZE
    dist.all_gather_object(states, optimizer.state_dict())
    for actual in states[1:]:
        _assert_tree_close(actual, states[0], exact=True)


def _make_model(seed):
    torch.manual_seed(seed)
    # The upstream model prints its architecture on construction.
    with contextlib.redirect_stdout(io.StringIO()):
        model = ActorCritic(
            num_actor_obs=NUM_PROP,
            num_critic_obs=NUM_PROP,
            num_actions=NUM_ACTIONS,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            priv_encoder_dims=[4],
            leg_control_head_hidden_dims=[8],
            arm_control_head_hidden_dims=[8],
            num_leg_actions=12,
            num_arm_actions=6,
            adaptive_arm_gains=False,
            adaptive_arm_gains_scale=1.0,
            num_priv=NUM_PRIV,
            num_hist=NUM_HISTORY,
            num_prop=NUM_PROP,
            output_tanh=False,
            init_std=[0.6] * NUM_ACTIONS,
        )
    # The real model currently has no registered buffers. Add one to verify
    # that synchronization also covers non-parameter model state.
    model.register_buffer("test_initial_seed", torch.tensor([seed]))
    return model


def _make_algorithm(algorithm_class, model):
    return algorithm_class(
        model,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=3e-4,
        max_grad_norm=0.05,
        schedule="adaptive",
        desired_kl=0.01,
        torque_supervision=False,
        adaptive_arm_gains=False,
        min_policy_std=[0.05] * NUM_ACTIONS,
        priv_reg_coef_schedual=[0.03, 0.03, 0, 1],
        mixing_schedule=[0.3, 0, 1],
        device="cpu",
    )


def _normalization_case(rank):
    # Unequal counts distinguish a sample-weighted reduction from averaging
    # the ranks' means. Both reward heads share the upstream scalar normalizer.
    def raw_values(worker):
        return torch.arange((worker + 2) * 2, dtype=torch.float32).view(-1, 2) + 11 * worker

    raw = raw_values(rank)
    combined = torch.cat([raw_values(worker) for worker in range(WORLD_SIZE)])
    expected = (raw - combined.mean()) / (combined.std() + 1e-8)
    torch.testing.assert_close(normalize_global_advantages(raw.clone()), expected)
    torch.testing.assert_close(global_mean(raw), combined.mean())

    # Each rank alone has zero variance, but the global batch does not.
    raw = torch.full((2, 3, 2), float(rank * 5))
    combined = torch.cat([torch.full_like(raw, float(worker * 5)) for worker in range(WORLD_SIZE)])
    expected = (raw - combined.mean()) / (combined.std() + 1e-8)
    torch.testing.assert_close(normalize_global_advantages(raw), expected)


def _gradients_case(rank):
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, -2.0])),
        torch.nn.Parameter(torch.tensor([3.0])),
        torch.nn.Parameter(torch.tensor([4.0])),
        torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False),
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=0.1)
    # Give the subsequently unused parameter existing Adam momentum. Turning
    # None into a zero gradient would then change both its value and state.
    for parameter in parameters[:3]:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_unused = parameters[2].detach().clone()
    before_unused_state = copy.deepcopy(optimizer.state[parameters[2]])

    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad)
        for parameter in parameters
    ]
    reference_optimizer = torch.optim.Adam(reference_parameters, lr=0.01, weight_decay=0.1)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    # Large, opposing local gradients expose clipping before averaging.
    parameters[0].grad = torch.tensor([[100.0, 0.0], [0.0, 1.0], [-97.0, 2.0]][rank])
    parameters[1].grad = torch.tensor([6.0]) if rank == 0 else None
    average_gradients(iter(parameters))
    torch.testing.assert_close(parameters[0].grad, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([2.0]))
    assert parameters[2].grad is None
    assert parameters[3].grad is None

    reference_parameters[0].grad = torch.tensor([1.0, 1.0])
    reference_parameters[1].grad = torch.tensor([2.0])
    torch.nn.utils.clip_grad_norm_(parameters, 0.5)
    torch.nn.utils.clip_grad_norm_(reference_parameters, 0.5)
    optimizer.step()
    reference_optimizer.step()

    for actual, expected in zip(parameters, reference_parameters):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_equal_across_ranks(actual)
    torch.testing.assert_close(parameters[2], before_unused, rtol=0, atol=0)
    _assert_tree_close(optimizer.state[parameters[2]], before_unused_state, exact=True)
    _assert_tree_close(optimizer.state_dict(), reference_optimizer.state_dict(), exact=True)
    _assert_optimizer_equal_across_ranks(optimizer)


def _collect_rollout(algorithm, rank, *, hist_encoding=False):
    generator = torch.Generator().manual_seed(800 + rank)
    with torch.inference_mode():
        for step in range(algorithm.storage.num_transitions_per_env):
            obs = torch.randn(2, NUM_OBS, generator=generator) + rank * 0.7
            algorithm.act(obs, obs, hist_encoding=hist_encoding)
            leg_rewards = torch.randn(2, generator=generator) + rank
            arm_rewards = torch.randn(2, generator=generator) - rank * 0.5
            dones = torch.tensor([step == 1, step == 2 and rank == 1])
            algorithm.process_env_step(leg_rewards, arm_rewards, dones, {})
    # Rank 0 has nearly zero KL while the others exceed the target. A local
    # adaptive schedule would give the replicas different learning rates.
    algorithm.storage.mu.add_(rank * 0.2)
    return torch.randn(2, NUM_OBS, generator=generator) + rank * 0.7


def _concatenate_storage(local, reference):
    for name, value in vars(local).items():
        if not isinstance(value, torch.Tensor):
            continue
        gathered = [torch.empty_like(value) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, value)
        # compute_returns replaces advantages with an inference tensor.
        with torch.inference_mode():
            getattr(reference, name).copy_(torch.cat(gathered, dim=1))
    reference.step = local.step


def _training_case(rank):
    model = _make_model(seed=13 + rank)
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    initial_models = [torch.empty_like(initial) for _ in range(WORLD_SIZE)]
    dist.all_gather(initial_models, initial)
    assert not torch.equal(initial_models[0], initial_models[1])

    algorithm = _make_algorithm(DistributedPPO, model)
    for value in model.state_dict().values():
        _assert_equal_across_ranks(value)
    assert model.test_initial_seed.item() == 13

    reference_model = _make_model(seed=100)
    reference_model.load_state_dict(model.state_dict())
    reference = _make_algorithm(PPO, reference_model)
    algorithm.counter = reference.counter = 2
    algorithm.init_storage(2, 3, [NUM_OBS], [NUM_OBS], [NUM_ACTIONS])
    reference.init_storage(2 * WORLD_SIZE, 3, [NUM_OBS], [NUM_OBS], [NUM_ACTIONS])
    last_obs = _collect_rollout(algorithm, rank)
    _concatenate_storage(algorithm.storage, reference.storage)
    gathered_last_obs = [torch.empty_like(last_obs) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_last_obs, last_obs)

    # The production runner keeps both rollout collection and return/advantage
    # computation inside inference_mode; updates run outside that context.
    with torch.inference_mode():
        algorithm.compute_returns(last_obs)
        reference.compute_returns(torch.cat(gathered_last_obs))
    torch.testing.assert_close(
        algorithm.storage.advantages,
        reference.storage.advantages[:, rank * 2:(rank + 1) * 2],
    )
    original_history = copy.deepcopy(model.actor.history_encoder.state_dict())
    algorithm.update()
    reference.update()
    assert algorithm.learning_rate == reference.learning_rate == 3e-4 / 1.5
    _assert_tree_close(model.state_dict(), reference_model.state_dict())
    _assert_tree_close(algorithm.optimizer.state_dict(), reference.optimizer.state_dict())
    for parameter in model.parameters():
        _assert_equal_across_ranks(parameter)
    _assert_optimizer_equal_across_ranks(algorithm.optimizer)
    _assert_tree_close(model.actor.history_encoder.state_dict(), original_history, exact=True)
    assert all(parameter not in algorithm.optimizer.state for parameter in model.actor.history_encoder.parameters())

    # The runner alternates PPO with history-policy rollouts and DAgger. Fill
    # the cleared storage with a fresh rollout before testing that second path.
    last_obs = _collect_rollout(algorithm, rank, hist_encoding=True)
    _concatenate_storage(algorithm.storage, reference.storage)
    dist.all_gather(gathered_last_obs, last_obs)
    with torch.inference_mode():
        algorithm.compute_returns(last_obs)
        reference.compute_returns(torch.cat(gathered_last_obs))
    algorithm.update_dagger()
    reference.update_dagger()
    _assert_tree_close(model.state_dict(), reference_model.state_dict())
    _assert_tree_close(
        algorithm.hist_encoder_optimizer.state_dict(),
        reference.hist_encoder_optimizer.state_dict(),
    )
    assert any(
        not torch.equal(value, original_history[name])
        for name, value in model.actor.history_encoder.state_dict().items()
    )
    for parameter in model.parameters():
        _assert_equal_across_ranks(parameter)
    _assert_optimizer_equal_across_ranks(algorithm.hist_encoder_optimizer)


def _worker(rank, rendezvous, case):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
        world_size=WORLD_SIZE, timeout=timedelta(seconds=45),
    )
    try:
        {"normalization": _normalization_case, "gradients": _gradients_case,
         "training": _training_case}[case](rank)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
class DistributedTrainingTest(unittest.TestCase):
    def _run_case(self, case):
        with tempfile.TemporaryDirectory(prefix="distributed-ppo-test-") as directory:
            mp.spawn(_worker, args=(str(Path(directory) / "rendezvous"), case), nprocs=WORLD_SIZE, join=True)

    def test_global_normalization_matches_concatenated_samples(self):
        self._run_case("normalization")

    def test_gradient_average_clip_and_unused_adam_parameters(self):
        self._run_case("gradients")

    def test_ppo_and_dagger_match_single_process_combined_batch(self):
        self._run_case("training")


if __name__ == "__main__":
    unittest.main()
