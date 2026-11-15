"""Primitives that keep credentials safe."""

from __future__ import annotations

from app.security import hash_password, hash_token, new_api_token, verify_password


def test_password_hashes_are_salted_and_verifiable():
    first = hash_password("same-password")
    second = hash_password("same-password")
    assert first != second
    assert verify_password(first, "same-password")
    assert not verify_password(first, "other-password")
    assert not verify_password("not-a-hash", "same-password")


def test_api_tokens_are_random_and_only_the_hash_is_kept():
    plain_a, prefix_a, hash_a = new_api_token()
    plain_b, prefix_b, hash_b = new_api_token()

    assert plain_a != plain_b
    assert plain_a.startswith("rag_")
    assert prefix_a == plain_a[:12]
    assert hash_a == hash_token(plain_a)
    assert hash_a != plain_a
    assert len(hash_a) == 64
    assert plain_a not in hash_a
