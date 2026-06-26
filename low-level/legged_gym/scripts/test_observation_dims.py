def expected_num_observations(use_5d_base_command, use_arm_base_message):
    base_num_proprio = 66
    history_len = 10
    num_priv = 18
    delta_proprio = (2 if use_5d_base_command else 0) + (5 if use_arm_base_message else 0)
    num_proprio = base_num_proprio + delta_proprio
    return num_proprio * (history_len + 1) + num_priv


def main():
    history_len = 10
    original_obs_dim = 66 * (history_len + 1) + 18

    assert expected_num_observations(False, False) == original_obs_dim
    assert expected_num_observations(True, False) == original_obs_dim + 2 * (history_len + 1)
    assert expected_num_observations(False, True) == original_obs_dim + 5 * (history_len + 1)
    assert expected_num_observations(True, True) == original_obs_dim + 7 * (history_len + 1)
    print("observation dimension smoke test passed")


if __name__ == "__main__":
    main()
