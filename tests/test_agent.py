from agents.local_agent import investigate


def test_local_agent_cites_policy_and_does_not_act():
    result = investigate(
        {"amount": 400, "hour": 1, "velocity_24h": 1, "distance_km": 160, "new_device": 1}
    )
    assert result["action"] == "none"
    assert result["recommend_review"] is True
    assert result["evidence"][0]["source"] == "review.md"
    assert len(result["steps"]) <= 4
    assert "freeze" not in result["report"].lower()


def test_quiet_transaction_is_not_pushed_to_review():
    result = investigate(
        {"amount": 20, "hour": 14, "velocity_24h": 1, "distance_km": 4, "new_device": 0}
    )
    assert result["recommend_review"] is False
    assert result["action"] == "none"
