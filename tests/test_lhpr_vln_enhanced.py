"""Tests for LHPRVLNEnhancedSFTPromptBuilder — fantasy-vln display_env parity."""
import base64
import io

import pytest
from PIL import Image, ImageEnhance

from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory
from easi.tasks.lhpr_vln.prompts.enhanced_sft import (
    LHPRVLNEnhancedSFTPromptBuilder,
    _encode_image_enhanced,
)
from easi.tasks.lhpr_vln.prompts.sft import LHPRVLNSFTPromptBuilder


ACTION_SPACE = ["move_forward", "turn_left", "turn_right", "stop"]


def _make_image(path, size=(512, 512), color=(64, 64, 64), mode="RGB"):
    img = Image.new(mode, size, color=color)
    img.save(path, format="PNG")


def _decode_data_url(data_url: str) -> Image.Image:
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


@pytest.fixture
def builder():
    b = LHPRVLNEnhancedSFTPromptBuilder(
        window_size=5,
        max_history_images=20,
        enhance_contrast=1.5,
        resize_to=366,
    )
    b.set_action_space(ACTION_SPACE)
    return b


@pytest.fixture
def memory():
    return AgentMemory(
        task_description="go to the kitchen",
        action_space=ACTION_SPACE,
        current_observation=Observation(rgb_path="/dev/null"),
    )


# ---- encoder unit tests ----

def test_enhance_missing_file_returns_none(tmp_path):
    assert _encode_image_enhanced(str(tmp_path / "nope.png")) is None


def test_enhance_noop_contrast_and_no_resize(tmp_path):
    p = tmp_path / "img.png"
    _make_image(p, size=(64, 64), color=(100, 150, 200))

    data_url = _encode_image_enhanced(str(p), contrast=1.0, resize_to=0)
    out = _decode_data_url(data_url)
    assert out.size == (64, 64)
    # PIL re-encode preserves pixels exactly for lossless PNG.
    assert out.getpixel((10, 10)) == (100, 150, 200)


def test_enhance_resize_only(tmp_path):
    p = tmp_path / "img.png"
    _make_image(p, size=(512, 512), color=(128, 128, 128))

    data_url = _encode_image_enhanced(str(p), contrast=1.0, resize_to=366)
    out = _decode_data_url(data_url)
    assert out.size == (366, 366)


def _make_split_image(path, size=(32, 32), low=64, high=192):
    """Half low / half high image — gives a known mean for contrast math."""
    img = Image.new("RGB", size, color=(low, low, low))
    for x in range(size[0] // 2, size[0]):
        for y in range(size[1]):
            img.putpixel((x, y), (high, high, high))
    img.save(path, format="PNG")


def test_enhance_contrast_only(tmp_path):
    """Contrast blends towards the image mean; half-64 / half-192 has mean 128.

    PIL ImageEnhance.Contrast(factor=c): out = grey * (1-c) + src * c,
    where grey = constant image of mean(src). For this fixture:
      mean = 128; c = 1.5
      low 64  → 128*(−0.5) + 64*1.5  = −64 + 96  = 32
      high 192 → 128*(−0.5) + 192*1.5 = −64 + 288 = 224
    """
    p = tmp_path / "img.png"
    _make_split_image(p, size=(32, 32), low=64, high=192)

    out = _decode_data_url(_encode_image_enhanced(str(p), contrast=1.5, resize_to=0))
    assert out.size == (32, 32)
    assert out.getpixel((5, 5)) == (32, 32, 32)      # on the low half
    assert out.getpixel((25, 5)) == (224, 224, 224)  # on the high half


def test_enhance_combined_matches_direct_pil(tmp_path):
    p = tmp_path / "img.png"
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    # Gradient so resize + contrast have visible effect.
    for x in range(100):
        for y in range(100):
            img.putpixel((x, y), (x * 2, y * 2, 128))
    img.save(p, format="PNG")

    # Direct PIL transform as reference truth.
    ref = Image.open(p)
    ref = ImageEnhance.Contrast(ref).enhance(1.5)
    ref = ref.resize((50, 50))

    out = _decode_data_url(_encode_image_enhanced(str(p), contrast=1.5, resize_to=50))
    assert out.size == ref.size
    # PNG round-trip must preserve pixels byte-for-byte.
    assert list(out.getdata()) == list(ref.getdata())


def test_enhance_rgba_passthrough(tmp_path):
    """Alpha channel survives the contrast + resize chain."""
    p = tmp_path / "img.png"
    # Split RGBA: half 64 / half 192 on RGB, constant alpha 200.
    img = Image.new("RGBA", (64, 64), color=(64, 64, 64, 200))
    for x in range(32, 64):
        for y in range(64):
            img.putpixel((x, y), (192, 192, 192, 200))
    img.save(p, format="PNG")

    out = _decode_data_url(_encode_image_enhanced(str(p), contrast=1.5, resize_to=32))
    assert out.mode == "RGBA"
    assert out.size == (32, 32)
    # Alpha unchanged by the contrast enhancer.
    assert out.getpixel((10, 10))[3] == 200
    # Low half pushed away from mean(=128) toward 32. Allow ±1 for bicubic
    # resize edge pixels crossing into the split boundary.
    r, g, b, _ = out.getpixel((5, 5))
    assert abs(r - 32) <= 1 and abs(g - 32) <= 1 and abs(b - 32) <= 1


# ---- builder integration ----

def test_parse_response_unchanged(builder, memory):
    actions = builder.parse_response(
        "<action><|left|><|forward|><|right|><|stop|></action>", memory,
    )
    assert [a.action_name for a in actions] == [
        "turn_left", "move_forward", "turn_right", "stop",
    ]


def test_build_content_uses_enhanced_encoder(tmp_path, builder):
    # Half-64 / half-192 so the contrast shift is measurable (uniform fills
    # would be a no-op under PIL's mean-anchored contrast).
    for name in ("left", "front", "right"):
        _make_split_image(tmp_path / f"{name}.png", size=(512, 512), low=64, high=192)
    current_views = [str(tmp_path / f"{n}.png") for n in ("left", "front", "right")]

    content = builder._build_content(
        instruction="test",
        history_paths=[],
        current_views=current_views,
    )

    urls = [b["image_url"]["url"] for b in content if b.get("type") == "image_url"]
    assert len(urls) == 3
    for url in urls:
        img = _decode_data_url(url)
        assert img.size == (366, 366)
        # Left half → 32, right half → 224. Sample from each half.
        assert img.getpixel((10, 10)) == (32, 32, 32)
        assert img.getpixel((350, 10)) == (224, 224, 224)


def test_build_content_structural_parity_with_parent(tmp_path, builder):
    """Enhanced builder's block layout matches the baseline SFT builder."""
    paths = []
    for name in ("left", "front", "right"):
        p = tmp_path / f"{name}.png"
        _make_image(p, size=(32, 32), color=(100, 100, 100))
        paths.append(str(p))

    parent = LHPRVLNSFTPromptBuilder(window_size=5, max_history_images=20)
    parent.set_action_space(ACTION_SPACE)

    parent_blocks = parent._build_content("test", [], paths)
    child_blocks = builder._build_content("test", [], paths)

    def _shape(blocks):
        return [b["type"] for b in blocks]

    assert _shape(parent_blocks) == _shape(child_blocks)
