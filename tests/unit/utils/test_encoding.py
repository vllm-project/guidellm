from __future__ import annotations

import uuid
from typing import Any, Generic, TypeVar

import pytest
from pydantic import BaseModel, Field

from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
)
from guidellm.schemas.request import GenerationRequestArguments
from guidellm.utils.encoding import Encoder, MessageEncoding, Serializer


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str = Field(description="Name field for testing")
    value: int = Field(description="Value field for testing")


class SampleModelSubclass(SampleModel):
    """Subclass of SampleModel for testing."""

    extra_field: str


SampleModelT = TypeVar("SampleModelT", bound=SampleModel)


class ComplexModel(BaseModel, Generic[SampleModelT]):
    """Complex Pydantic model for testing."""

    items: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    nested: SampleModelT | None = Field(default=None)


class GenricModelWrapper(Generic[SampleModelT]):
    """Simulates a layered generic type."""

    def method(self, **kwargs) -> ComplexModel[SampleModelT]:
        return ComplexModel[SampleModelT](**kwargs)


class TestMessageEncoding:
    """Test suite for MessageEncoding class."""

    @pytest.fixture(
        params=[
            {"serialization": None, "encoding": None},
            {"serialization": "dict", "encoding": None},
            {"serialization": "sequence", "encoding": None},
            {"serialization": None, "encoding": "msgpack"},
            {"serialization": "dict", "encoding": "msgpack"},
            {"serialization": "sequence", "encoding": "msgpack"},
            {"serialization": None, "encoding": "msgspec"},
            {"serialization": "dict", "encoding": "msgspec"},
            {"serialization": "sequence", "encoding": "msgspec"},
            {"serialization": None, "encoding": ["msgspec", "msgpack"]},
            {"serialization": "dict", "encoding": ["msgspec", "msgpack"]},
        ],
        ids=[
            "no_serialization_no_encoding",
            "dict_serialization_no_encoding",
            "str_serialization_no_encoding",
            "no_serialization_msgpack",
            "dict_serialization_msgpack",
            "str_serialization_msgpack",
            "no_serialization_msgspec",
            "dict_serialization_msgspec",
            "str_serialization_msgspec",
            "no_serialization_encoding_list",
            "dict_serialization_encoding_list",
        ],
    )
    def valid_instances(self, request):
        """Fixture providing test data for MessageEncoding."""
        constructor_args = request.param
        try:
            instance = MessageEncoding(**constructor_args)
            return instance, constructor_args
        except ImportError:
            pytest.skip("Required encoding library not available")

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test MessageEncoding inheritance and type relationships."""
        assert issubclass(MessageEncoding, Generic)
        assert hasattr(MessageEncoding, "DEFAULT_ENCODING_PREFERENCE")
        assert isinstance(MessageEncoding.DEFAULT_ENCODING_PREFERENCE, list)
        assert MessageEncoding.DEFAULT_ENCODING_PREFERENCE == ["msgspec", "msgpack"]

        # Check classmethods
        assert hasattr(MessageEncoding, "encode_message")
        assert callable(MessageEncoding.encode_message)
        assert hasattr(MessageEncoding, "decode_message")
        assert callable(MessageEncoding.decode_message)

        # Check instance methods
        assert hasattr(MessageEncoding, "__init__")
        assert hasattr(MessageEncoding, "register_pydantic")
        assert hasattr(MessageEncoding, "encode")
        assert hasattr(MessageEncoding, "decode")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test MessageEncoding initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, MessageEncoding)
        assert hasattr(instance, "serializer")
        assert isinstance(instance.serializer, Serializer)
        assert instance.serializer.serialization == constructor_args["serialization"]
        assert hasattr(instance, "encoder")
        assert isinstance(instance.encoder, Encoder)

        expected_encoding = constructor_args["encoding"]
        if isinstance(expected_encoding, list):
            assert instance.encoder.encoding in expected_encoding
        else:
            assert instance.encoder.encoding == expected_encoding

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "obj",
        [
            None,
            0,
            0.0,
            "0.1.2.3",
            [0, 0.0, "0.1.2.3", None],
            (0, 0.0, "0.1.2.3", None),
            {"key1": 0, "key2": 0.0, "key3": "0.1.2.3", "key4": None},
        ],
    )
    def test_encode_decode_python(self, valid_instances, obj: Any):
        """Test MessageEncoding encode/decode with comprehensive data types."""
        instance, constructor_args = valid_instances

        message = instance.encode(obj)
        decoded = instance.decode(message)

        if isinstance(obj, tuple):
            assert list(decoded) == list(obj)
        else:
            assert decoded == obj

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "obj",
        [
            SampleModel(name="sample", value=123),
            ComplexModel(
                items=["item1", "item2"],
                metadata={"key": "value"},
                nested=SampleModel(name="sample", value=123),
            ),
            (
                SampleModel(name="sample", value=123),
                None,
                ComplexModel(
                    items=["item1", "item2"],
                    metadata={"key": "value"},
                    nested=SampleModel(name="sample", value=123),
                ),
            ),
            {
                "key1": SampleModel(name="sample", value=123),
                "key2": None,
                "key3": ComplexModel(
                    items=["item1", "item2"],
                    metadata={"key": "value"},
                    nested=SampleModel(name="sample", value=123),
                ),
            },
        ],
    )
    def test_encode_decode_pydantic(self, valid_instances, obj: Any):
        """Test MessageEncoding encode/decode with Pydantic models."""
        instance, constructor_args = valid_instances

        if (
            constructor_args["serialization"] is None
            and constructor_args["encoding"] is not None
        ):
            # msgpack/msgspec don't support Pydantic models natively
            pytest.skip("Skipping unsupported Pydantic serialization/encoding combo")

        # Register Pydantic models for proper serialization
        instance.register_pydantic(SampleModel)
        instance.register_pydantic(ComplexModel)

        message = instance.encode(obj)
        decoded = instance.decode(message)

        if isinstance(obj, tuple):
            assert list(decoded) == list(obj)
        else:
            assert decoded == obj

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "obj",
        [
            (
                None,
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(
                    timings=RequestTimings(
                        targeted_start=1.0,
                        queued=0.1,
                        dequeued=0.2,
                        scheduled_at=0.3,
                        resolve_start=1.1,
                        resolve_end=1.5,
                        finalized=1.6,
                    )
                ),
            ),
            (
                GenerationResponse(
                    request_id=str(uuid.uuid4()),
                    request_args=None,
                    text="test response",
                ),
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(
                    timings=RequestTimings(
                        targeted_start=1.0,
                        queued=0.1,
                        dequeued=0.2,
                        scheduled_at=0.3,
                        resolve_start=1.1,
                        resolve_end=1.5,
                        finalized=1.6,
                    )
                ),
            ),
        ],
    )
    def test_encode_decode_generative(self, valid_instances, obj: Any):
        """Test MessageEncoding encode/decode with generative models."""
        instance, constructor_args = valid_instances

        if (
            constructor_args["serialization"] is None
            and constructor_args["encoding"] is not None
        ):
            # msgpack/msgspec don't support Pydantic models natively
            pytest.skip("Skipping unsupported Pydantic serialization/encoding combo")

        instance.register_pydantic(GenerationRequest)
        instance.register_pydantic(GenerationResponse)
        instance.register_pydantic(RequestInfo)

        message = instance.encode(obj)
        decoded = instance.decode(message)

        assert list(decoded) == list(obj)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "serialization",
        [
            None,
            "dict",
            "sequence",
        ],
    )
    @pytest.mark.parametrize(
        "encoding",
        [None, "msgpack", "msgspec"],
    )
    @pytest.mark.parametrize(
        "obj",
        [
            "0.1.2.3",
            [0, 0.0, "0.1.2.3", None, SampleModel(name="sample", value=123)],
            {
                "key1": 0,
                "key2": 0.0,
                "key3": "0.1.2.3",
                "key4": None,
                "key5": ComplexModel(
                    items=["item1", "item2"],
                    metadata={"key": "value"},
                    nested=SampleModel(name="sample", value=123),
                ),
            },
        ],
    )
    def test_encode_decode_message(self, serialization, encoding, obj):
        """Test MessageEncoding.encode_message and decode_message class methods."""
        if encoding is not None and serialization is None and obj != "0.1.2.3":
            pytest.skip("Skipping unsupported serialization/encoding combo")

        try:
            serializer = Serializer(serialization) if serialization else None
            encoder = Encoder(encoding) if encoding else None

            message = MessageEncoding.encode_message(obj, serializer, encoder)
            decoded = MessageEncoding.decode_message(message, serializer, encoder)

            if isinstance(obj, tuple):
                assert list(decoded) == list(obj)
            else:
                assert decoded == obj
        except ImportError:
            pytest.skip("Required encoding library not available")

    @pytest.mark.smoke
    def test_register_pydantic(self):
        """Test MessageEncoding.register_pydantic functionality."""
        instance = MessageEncoding(serialization="dict", encoding=None)
        assert len(instance.serializer.pydantic_registry) == 0
        instance.register_pydantic(SampleModel)
        assert len(instance.serializer.pydantic_registry) == 1
        assert (
            instance.serializer.pydantic_registry.values().__iter__().__next__()
            is SampleModel
        )

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test invalid initialization (unsupported encoding)."""
        inst = MessageEncoding(serialization="dict", encoding=["invalid_encoding"])  # type: ignore[arg-type]
        assert inst.encoder.encoding is None
        with pytest.raises(ImportError):
            MessageEncoding(serialization="dict", encoding="invalid")  # type: ignore[arg-type]


class TestEncoder:
    """Test suite for Encoder class."""

    @pytest.fixture(
        params=[
            None,
            "msgpack",
            "msgspec",
            ["msgspec", "msgpack"],
            ["msgpack", "msgspec"],
        ],
        ids=[
            "none",
            "msgpack",
            "msgspec",
            "list_pref_msgspec_first",
            "list_pref_msgpack_first",
        ],
    )
    def valid_instances(self, request):
        args = request.param
        try:
            inst = Encoder(args)
        except ImportError:
            pytest.skip("Encoding backend missing")
        return inst, args

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert hasattr(Encoder, "encode")
        assert hasattr(Encoder, "decode")
        assert hasattr(Encoder, "_resolve_encoding")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, args = valid_instances
        assert isinstance(inst, Encoder)
        if isinstance(args, list):
            assert inst.encoding in args or inst.encoding is None
        else:
            assert inst.encoding == args

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        with pytest.raises(ImportError):
            Encoder("invalid")  # type: ignore[arg-type]

    @pytest.mark.smoke
    @pytest.mark.parametrize("obj", [None, 0, 1.2, "text", [1, 2], {"a": 1}])
    def test_encode_decode(self, valid_instances, obj):
        inst, _ = valid_instances
        msg = inst.encode(obj)
        out = inst.decode(msg)
        assert out == obj


class TestSerializer:
    """Test suite for Serializer class."""

    @pytest.fixture(params=[None, "dict", "sequence"], ids=["none", "dict", "sequence"])
    def valid_instances(self, request):
        inst = Serializer(request.param)
        return inst, request.param

    @pytest.mark.smoke
    def test_class_signatures(self):
        assert hasattr(Serializer, "serialize")
        assert hasattr(Serializer, "deserialize")
        assert hasattr(Serializer, "register_pydantic")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        inst, mode = valid_instances
        assert isinstance(inst, Serializer)
        assert inst.serialization == mode

    @pytest.mark.smoke
    def test_register_pydantic(self, valid_instances):
        inst, _ = valid_instances
        assert len(inst.pydantic_registry) == 0
        inst.register_pydantic(SampleModel)
        assert len(inst.pydantic_registry) == 1

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "obj",
        [
            1,
            "str_val",
            [1, 2, 3],
            SampleModel(name="x", value=1),
            {"k": SampleModel(name="y", value=2)},
        ],
    )
    def test_serialize_deserialize(self, valid_instances, obj):
        inst, mode = valid_instances
        inst.register_pydantic(SampleModel)
        msg = inst.serialize(obj)
        out = inst.deserialize(msg)
        if isinstance(obj, list):
            assert list(out) == obj
        else:
            assert out == obj

    @pytest.mark.regression
    def test_sequence_mapping_roundtrip(self):
        inst = Serializer("sequence")
        inst.register_pydantic(SampleModel)
        data = {
            "a": SampleModel(name="a", value=1),
            "b": SampleModel(name="b", value=2),
        }
        msg = inst.serialize(data)
        out = inst.deserialize(msg)
        assert out == data

    @pytest.mark.sanity
    def test_to_from_dict_variations(self):
        inst = Serializer("dict")
        inst.register_pydantic(SampleModel)
        model = SampleModel(name="n", value=3)
        lst = [model, 5]
        mp = {"k1": model, "k2": 9}
        assert inst.from_dict(inst.to_dict(model)) == model
        assert inst.from_dict(inst.to_dict(lst)) == lst
        assert inst.from_dict(inst.to_dict(mp)) == mp

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "collection",
        [
            [SampleModel(name="x", value=1), 2, 3],
            (SampleModel(name="y", value=2), None),
        ],
    )
    def test_to_from_sequence_collections(self, collection):
        inst = Serializer("sequence")
        inst.register_pydantic(SampleModel)
        seq = inst.to_sequence(collection)
        out = inst.from_sequence(seq)
        assert len(out) == len(collection)
        assert all(a == b for a, b in zip(out, list(collection), strict=False))

    @pytest.mark.sanity
    def test_to_from_sequence_mapping(self):
        inst = Serializer("sequence")
        inst.register_pydantic(SampleModel)
        data = {"k": SampleModel(name="z", value=7), "j": 1}
        seq = inst.to_sequence(data)
        out = inst.from_sequence(seq)
        assert out == data

    @pytest.mark.sanity
    def test_sequence_multiple_root_raises(self):
        inst = Serializer("sequence")
        part1 = inst.pack_next_sequence("python", inst.to_sequence_python(1), None)
        part2 = inst.pack_next_sequence("python", inst.to_sequence_python(2), None)
        with pytest.raises(ValueError):
            inst.from_sequence(part1 + part2)  # type: ignore[operator]

    @pytest.mark.sanity
    def test_pack_next_sequence_type_mismatch(self):
        inst = Serializer("sequence")
        first_payload = inst.to_sequence_python(1)
        first = inst.pack_next_sequence("python", first_payload, None)
        bad_payload: Any = (
            first_payload.decode() if isinstance(first_payload, bytes) else b"1"
        )
        with pytest.raises(ValueError):
            inst.pack_next_sequence("python", bad_payload, first)

    @pytest.mark.sanity
    def test_unpack_invalid(self):
        inst = Serializer("sequence")
        with pytest.raises(ValueError):
            inst.unpack_next_sequence("X|3|abc")
        with pytest.raises(ValueError):
            inst.unpack_next_sequence("p?bad")

    @pytest.mark.sanity
    def test_dynamic_import_load_pydantic(self, monkeypatch):
        inst = Serializer("dict")
        inst.pydantic_registry.clear()
        sample = SampleModel(name="dyn", value=5)
        dumped = inst.to_dict(sample)
        inst.pydantic_registry.clear()
        restored = inst.from_dict(dumped)
        assert restored == sample

    @pytest.mark.sanity
    def test_generic_model(self):
        inst = Serializer("dict")
        inst.register_pydantic(ComplexModel[SampleModelSubclass])
        nested = ComplexModel[SampleModelSubclass](
            items=["i1", "i2"],
            metadata={"m": 1},
            nested=SampleModelSubclass(name="nested", value=10, extra_field="extra"),
        )
        dumped = inst.to_dict(nested)
        restored = inst.from_dict(dumped)
        assert restored == nested

    @pytest.mark.sanity
    @pytest.mark.xfail(
        reason="A generic object returned by a generic method loses its type args"
    )
    def test_generic_emitted_type(self):
        generic_instance = GenricModelWrapper[SampleModelSubclass]()

        inst = Serializer("dict")
        inst.register_pydantic(ComplexModel[SampleModelSubclass])
        nested = generic_instance.method(
            items=["i1", "i2"],
            metadata={"m": 1},
            nested=SampleModelSubclass(name="nested", value=10, extra_field="extra"),
        )
        dumped = inst.to_dict(nested)
        restored = inst.from_dict(dumped)
        assert restored == nested
