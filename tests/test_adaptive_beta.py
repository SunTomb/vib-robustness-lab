from vib_project.adaptive_beta import beta_warmup_value, update_beta_for_target_kl


def test_beta_warmup_value_reaches_target():
    assert beta_warmup_value(epoch=0, warmup_epochs=5, beta_target=0.1) == 0.0
    assert beta_warmup_value(epoch=5, warmup_epochs=5, beta_target=0.1) == 0.1
    assert beta_warmup_value(epoch=10, warmup_epochs=5, beta_target=0.1) == 0.1


def test_update_beta_for_target_kl_clamps():
    assert update_beta_for_target_kl(0.1, kl=30.0, target_low=5.0, target_high=20.0) > 0.1
    assert update_beta_for_target_kl(0.1, kl=1.0, target_low=5.0, target_high=20.0) < 0.1
    assert update_beta_for_target_kl(1.0, kl=30.0, target_low=5.0, target_high=20.0) == 1.0
