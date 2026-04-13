from reflective_research.tools.evidence_quality import is_commerce_or_social_host


def test_blocks_amazon_in_shopping() -> None:
    assert is_commerce_or_social_host("https://www.amazon.in/foo") is True
    assert is_commerce_or_social_host("https://amazon.in/women") is True


def test_allows_aws_docs() -> None:
    assert is_commerce_or_social_host("https://docs.aws.amazon.com/sagemaker/latest/dg/x.html") is False


def test_noncommerce() -> None:
    assert is_commerce_or_social_host("https://arxiv.org/abs/2401.00001") is False
