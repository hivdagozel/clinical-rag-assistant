from pathlib import Path

from src.collection_targets import TargetSelector, canonical_product_fields, normalize


TARGETS = Path(__file__).resolve().parent.parent / "config" / "medicine_collection_targets.yaml"


def test_target_selector_matches_official_ingredient_before_product_name():
    selector = TargetSelector(TARGETS, ("diabetes",))
    match = selector.match({
        "name": "ÖRNEK 1000 MG TABLET",
        "activeIngredient": "METFORMİN HİDROKLORÜR 1000 mg",
        "atcCode": "A10BA02",
    })
    assert match.selected
    assert match.categories == ("diabetes",)
    assert "metformin" in match.active_ingredients
    assert any(reason.startswith("atc_prefix:A10") for reason in match.reasons)


def test_target_selector_records_all_matching_categories_and_combination():
    selector = TargetSelector(TARGETS, ("analgesic", "common"))
    match = selector.match({
        "name": "ÖRNEK PLUS 500 MG TABLET",
        "activeIngredients": ["parasetamol", "asetilsalisilik asit"],
        "atcCode": "N02BE",
    })
    assert match.categories == ("analgesic",)
    assert match.combination_product is True
    assert len(match.active_ingredients) == 2


def test_canonical_key_separates_strength_form_and_release_type():
    normal = canonical_product_fields("İLAÇ 500 MG FİLM KAPLI TABLET", ("parasetamol",))
    extended = canonical_product_fields("İLAÇ 500 MG XR TABLET", ("parasetamol",))
    assert normal["canonical_product_key"] != extended["canonical_product_key"]
    assert normal["strength"] == "500 MG"
    assert extended["release_type"] == "xr"


def test_normalization_handles_turkish_case_and_spacing():
    assert normalize("  İNSÜLİN--GLARJİN ") == "insulin glarjin"
