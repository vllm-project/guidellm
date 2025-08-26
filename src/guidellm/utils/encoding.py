"""
Message encoding utilities for multiprocess communication with Pydantic model support.

Provides binary serialization and deserialization of Python objects using various
serialization formats and encoding packages to enable performance configurations
for distributed scheduler operations. Supports configurable two-stage processing
pipeline: object serialization (to dict/sequence) followed by binary encoding
(msgpack/msgspec) with specialized Pydantic model handling for type preservation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Generic, Literal, Optional, TypeVar, Union

try:
    import msgpack
    from msgpack import Packer, Unpacker

    HAS_MSGPACK = True
except ImportError:
    msgpack = Packer = Unpacker = None
    HAS_MSGPACK = False

try:
    from msgspec.msgpack import Decoder as MsgspecDecoder
    from msgspec.msgpack import Encoder as MsgspecEncoder

    HAS_MSGSPEC = True
except ImportError:
    MsgspecDecoder = MsgspecEncoder = None
    HAS_MSGSPEC = False

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False

from pydantic import BaseModel
from typing_extensions import TypeAlias

__all__ = [
    "Encoder",
    "EncodingTypesAlias",
    "MessageEncoding",
    "SerializationTypesAlias",
    "Serializer",
]

ObjT = TypeVar("ObjT")
MsgT = TypeVar("MsgT")

SerializationTypesAlias: TypeAlias = Annotated[
    Optional[Literal["dict", "sequence"]],
    "Type alias for available serialization strategies",
]
EncodingTypesAlias: TypeAlias = Annotated[
    Optional[Literal["msgpack", "msgspec"]],
    "Type alias for available binary encoding formats",
]


class MessageEncoding(Generic[ObjT, MsgT]):
    """
    High-performance message encoding and decoding for multiprocessing communication.

    Supports configurable object serialization and binary encoding with specialized
    handling for Pydantic models. Provides a two-stage pipeline of serialization
    (object to dict/str) followed by encoding (dict/str to binary) for optimal
    performance and compatibility across different transport mechanisms used in
    distributed scheduler operations.

    Example:
    ::
        from guidellm.utils.encoding import MessageEncoding
        from pydantic import BaseModel

        class DataModel(BaseModel):
            name: str
            value: int

        # Configure with dict serialization and msgpack encoding
        encoding = MessageEncoding(serialization="dict", encoding="msgpack")
        encoding.register_pydantic(DataModel)

        # Encode and decode objects
        data = DataModel(name="test", value=42)
        encoded_msg = encoding.encode(data)
        decoded_data = encoding.decode(encoded_msg)

    :cvar DEFAULT_ENCODING_PREFERENCE: Preferred encoding formats in priority order
    """

    DEFAULT_ENCODING_PREFERENCE: ClassVar[list[str]] = ["msgspec", "msgpack"]

    @classmethod
    def encode_message(
        cls,
        obj: ObjT,
        serializer: Serializer | None,
        encoder: Encoder | None,
    ) -> MsgT:
        """
        Encode object using specified serializer and encoder.

        :param obj: Object to encode
        :param serializer: Serializer for object conversion, None for no serialization
        :param encoder: Encoder for binary conversion, None for no encoding
        :return: Encoded message ready for transport
        """
        serialized = serializer.serialize(obj) if serializer else obj

        return encoder.encode(serialized) if encoder else serialized

    @classmethod
    def decode_message(
        cls,
        message: MsgT,
        serializer: Serializer | None,
        encoder: Encoder | None,
    ) -> ObjT:
        """
        Decode message using specified serializer and encoder.
        Must match the encoding configuration originally used.

        :param message: Encoded message to decode
        :param serializer: Serializer for object reconstruction, None for no
            serialization
        :param encoder: Encoder for binary decoding, None for no encoding
        :return: Reconstructed object
        """
        serialized = encoder.decode(message) if encoder else message

        return serializer.deserialize(serialized) if serializer else serialized

    def __init__(
        self,
        serialization: SerializationTypesAlias = None,
        encoding: EncodingTypesAlias | list[EncodingTypesAlias] = None,
        pydantic_models: list[type[BaseModel]] | None = None,
    ) -> None:
        """
        Initialize MessageEncoding with serialization and encoding strategies.

        :param serialization: Serialization strategy (None, "dict", or "sequence")
        :param encoding: Encoding strategy (None, "msgpack", "msgspec", or
            preference list)
        """
        self.serializer = Serializer(serialization, pydantic_models)
        self.encoder = Encoder(encoding)

    def register_pydantic(self, model: type[BaseModel]) -> None:
        """
        Register Pydantic model for specialized serialization handling.

        :param model: Pydantic model class to register for type preservation
        """
        self.serializer.register_pydantic(model)

    def encode(self, obj: ObjT) -> MsgT:
        """
        Encode object using instance configuration.

        :param obj: Object to encode using configured serialization and encoding
        :return: Encoded message ready for transport
        """
        return self.encode_message(
            obj=obj,
            serializer=self.serializer,
            encoder=self.encoder,
        )

    def decode(self, message: MsgT) -> ObjT:
        """
        Decode message using instance configuration.

        :param message: Encoded message to decode using configured strategies
        :return: Reconstructed object
        """
        return self.decode_message(
            message=message,
            serializer=self.serializer,
            encoder=self.encoder,
        )


class Encoder:
    """
    Binary encoding and decoding using MessagePack or msgspec formats.

    Handles binary serialization of Python objects using configurable encoding
    strategies with automatic fallback when dependencies are unavailable. Supports
    both standalone instances and pooled encoder/decoder pairs for performance
    optimization in high-throughput scenarios.
    """

    def __init__(
        self, encoding: EncodingTypesAlias | list[EncodingTypesAlias] = None
    ) -> None:
        """
        Initialize encoder with specified encoding strategy.

        :param encoding: Encoding format preference (None, "msgpack", "msgspec", or
            preference list)
        """
        self.encoding, self.encoder, self.decoder = self._resolve_encoding(encoding)

    def encode(self, obj: Any) -> bytes | Any:
        """
        Encode object to binary format using configured encoding strategy.

        :param obj: Object to encode (must be serializable by chosen format)
        :return: Encoded bytes or original object if no encoding configured
        :raises ImportError: If required encoding library is not available
        """
        if self.encoding == "msgpack":
            if not HAS_MSGPACK:
                raise ImportError("msgpack is not available")

            return self.encoder.pack(obj) if self.encoder else msgpack.packb(obj)

        if self.encoding == "msgspec":
            if not HAS_MSGSPEC:
                raise ImportError("msgspec is not available")

            return (
                self.encoder.encode(obj)
                if self.encoder
                else MsgspecEncoder().encode(obj)
            )

        return obj

    def decode(self, data: bytes | Any) -> Any:
        """
        Decode binary data using configured encoding strategy.

        :param data: Binary data to decode or object if no encoding configured
        :return: Decoded Python object
        :raises ImportError: If required encoding library is not available
        """
        if self.encoding == "msgpack":
            if not HAS_MSGPACK:
                raise ImportError("msgpack is not available")

            if self.decoder is not None:
                self.decoder.feed(data)
                return self.decoder.unpack()

            return msgpack.unpackb(data, raw=False)

        if self.encoding == "msgspec":
            if not HAS_MSGSPEC:
                raise ImportError("msgspec is not available")

            if self.decoder is not None:
                return self.decoder.decode(data)

            return MsgspecDecoder().decode(data)

        return data

    def _resolve_encoding(
        self, encoding: EncodingTypesAlias | list[EncodingTypesAlias] | None
    ) -> tuple[EncodingTypesAlias, Any, Any]:
        def _get_available_encoder_decoder(
            encoding: EncodingTypesAlias,
        ) -> tuple[Any, Any]:
            if encoding == "msgpack" and HAS_MSGPACK:
                return Packer(), Unpacker(raw=False)
            if encoding == "msgspec" and HAS_MSGSPEC:
                return MsgspecEncoder(), MsgspecDecoder()
            return None, None

        if not isinstance(encoding, list):
            if encoding is None:
                return None, None, None

            encoder, decoder = _get_available_encoder_decoder(encoding)
            if encoder is None or decoder is None:
                raise ImportError(f"Encoding '{encoding}' is not available.")

            return encoding, encoder, decoder

        for test_encoding in encoding:
            encoder, decoder = _get_available_encoder_decoder(test_encoding)
            if encoder is not None and decoder is not None:
                return test_encoding, encoder, decoder

        return None, None, None


class Serializer:
    """
    Object serialization with specialized Pydantic model support.

    Converts Python objects to serializable formats (dict/sequence) with type
    preservation for Pydantic models. Maintains object integrity through
    encoding/decoding cycles by storing class metadata and enabling proper
    reconstruction of complex objects. Supports both dictionary-based and
    sequence-based serialization strategies for different use cases.
    """

    def __init__(
        self,
        serialization: SerializationTypesAlias = None,
        pydantic_models: list[type[BaseModel]] | None = None,
    ):
        """
        Initialize serializer with strategy and Pydantic registry.

        :param serialization: Default serialization strategy for this instance
        """
        self.serialization = serialization
        self.pydantic_registry: dict[tuple[str, str], type[BaseModel]] = {}
        if pydantic_models:
            for model in pydantic_models:
                self.register_pydantic(model)

    def register_pydantic(self, model: type[BaseModel]) -> None:
        """
        Register Pydantic model for specialized serialization handling.

        :param model: Pydantic model class to register for type preservation
        """
        key = f"{model.__module__}:{model.__name__}"
        self.pydantic_registry[key] = model

    def load_pydantic(self, type_name: str, module_name: str) -> type[BaseModel]:
        """
        Load Pydantic class by name with registry fallback to dynamic import.

        :param type_name: Class name to load
        :param module_name: Module containing the class
        :return: Loaded Pydantic model class
        """
        key = f"{module_name}:{type_name}"

        if key in self.pydantic_registry:
            return self.pydantic_registry[key]

        # Dynamic import fallback; need to update to better handle generics
        module = __import__(module_name, fromlist=[type_name])
        pydantic_class = getattr(module, type_name)
        self.pydantic_registry[key] = pydantic_class

        return pydantic_class

    def serialize(self, obj: Any) -> Any:
        """
        Serialize object using specified or configured strategy.

        :param obj: Object to serialize
        :return: Serialized representation (dict, str, or original object)
        """
        if self.serialization == "dict":
            return self.to_dict(obj)
        elif self.serialization == "sequence":
            return self.to_sequence(obj)

        return obj

    def deserialize(self, msg: Any) -> Any:
        """
        Deserialize object using specified or configured strategy.

        :param msg: Serialized message to deserialize
        :return: Reconstructed object
        """
        if self.serialization == "dict":
            return self.from_dict(msg)
        elif self.serialization == "sequence":
            return self.from_sequence(msg)

        return msg

    def to_dict(self, obj: Any) -> Any:
        """
        Convert object to dictionary with Pydantic model type preservation.

        :param obj: Object to convert (BaseModel, collections, or primitive)
        :return: Dictionary representation with type metadata for Pydantic models
        """
        if isinstance(obj, BaseModel):
            return self.to_dict_pydantic(obj)

        if isinstance(obj, (list, tuple)) and any(
            isinstance(item, BaseModel) for item in obj
        ):
            return [
                self.to_dict_pydantic(item) if isinstance(item, BaseModel) else item
                for item in obj
            ]

        if isinstance(obj, dict) and any(
            isinstance(value, BaseModel) for value in obj.values()
        ):
            return {
                key: self.to_dict_pydantic(value)
                if isinstance(value, BaseModel)
                else value
                for key, value in obj.items()
            }

        return obj

    def from_dict(self, data: Any) -> Any:
        """
        Reconstruct object from dictionary with Pydantic model type restoration.

        :param data: Dictionary representation possibly containing type metadata
        :return: Reconstructed object with proper types restored
        """
        if isinstance(data, (list, tuple)):
            return [
                self.from_dict_pydantic(item)
                if isinstance(item, dict) and "*PYD*" in item
                else item
                for item in data
            ]
        elif isinstance(data, dict) and data:
            if "*PYD*" in data:
                return self.from_dict_pydantic(data)

            return {
                key: self.from_dict_pydantic(value)
                if isinstance(value, dict) and "*PYD*" in value
                else value
                for key, value in data.items()
            }

        return data

    def to_dict_pydantic(self, item: Any) -> Any:
        """
        Convert item to dictionary with Pydantic type metadata.

        :param item: Item to convert (may or may not be a Pydantic model)
        :return: Dictionary with type preservation metadata
        """
        return {
            "*PYD*": True,
            "typ": item.__class__.__name__,
            "mod": item.__class__.__module__,
            "dat": item.model_dump(mode="python"),
        }

    def from_dict_pydantic(self, item: dict[str, Any]) -> Any:
        """
        Reconstruct object from dictionary with Pydantic type metadata.

        :param item: Dictionary containing type metadata and data
        :return: Reconstructed Pydantic model or original data
        """
        type_name = item["typ"]
        module_name = item["mod"]
        model_class = self.load_pydantic(type_name, module_name)

        return model_class.model_validate(item["dat"])

    def to_sequence(self, obj: Any) -> str | Any:
        """
        Convert object to sequence format with type-aware serialization.

        Handles Pydantic models, collections, and mappings with proper type
        preservation through structured sequence encoding.

        :param obj: Object to serialize to sequence format
        :return: Serialized sequence string or bytes
        """
        if isinstance(obj, BaseModel):
            payload_type = "pydantic"
            payload = self.to_sequence_pydantic(obj)
        elif isinstance(obj, (list, tuple)) and any(
            isinstance(item, BaseModel) for item in obj
        ):
            payload_type = "collection_sequence"
            payload = None

            for item in obj:
                is_pydantic = isinstance(item, BaseModel)
                payload = self.pack_next_sequence(
                    type_="pydantic" if is_pydantic else "python",
                    payload=(
                        self.to_sequence_pydantic(item)
                        if is_pydantic
                        else self.to_sequence_python(item)
                    ),
                    current=payload,
                )
        elif isinstance(obj, Mapping) and any(
            isinstance(value, BaseModel) for value in obj.values()
        ):
            payload_type = "collection_mapping"
            keys = ",".join(str(key) for key in obj)
            payload = keys.encode() + b"|" if HAS_ORJSON else keys + "|"
            for item in obj.values():
                is_pydantic = isinstance(item, BaseModel)
                payload = self.pack_next_sequence(
                    type_="pydantic" if is_pydantic else "python",
                    payload=(
                        self.to_sequence_pydantic(item)
                        if is_pydantic
                        else self.to_sequence_python(item)
                    ),
                    current=payload,
                )
        else:
            payload_type = "python"
            payload = self.to_sequence_python(obj)

        return self.pack_next_sequence(payload_type, payload, None)

    def from_sequence(self, data: str | Any) -> Any:  # noqa: C901, PLR0912
        """
        Reconstruct object from sequence format with type restoration.

        Handles deserialization of objects encoded with to_sequence, properly
        restoring Pydantic models and collection structures.

        :param data: Serialized sequence data to reconstruct
        :return: Reconstructed object with proper types
        :raises ValueError: If sequence format is invalid or contains multiple
            packed sequences
        """
        type_, payload, remaining = self.unpack_next_sequence(data)
        if remaining:
            raise ValueError("Data contains multiple packed sequences; expected one.")

        if type_ == "pydantic":
            return self.from_sequence_pydantic(payload)

        if type_ == "python":
            return self.from_sequence_python(payload)

        if type_ in {"collection_sequence", "collection_tuple"}:
            items = []
            while payload:
                type_, item_payload, payload = self.unpack_next_sequence(payload)
                if type_ == "pydantic":
                    items.append(self.from_sequence_pydantic(item_payload))
                elif type_ == "python":
                    items.append(self.from_sequence_python(item_payload))
                else:
                    raise ValueError("Invalid type in collection sequence")
            return items

        if type_ != "collection_mapping":
            raise ValueError(f"Invalid type for mapping sequence: {type_}")

        if isinstance(payload, bytes):
            keys_end = payload.index(b"|")
            keys = payload[:keys_end].decode().split(",")
            payload = payload[keys_end + 1 :]
        else:
            keys_end = payload.index("|")
            keys = payload[:keys_end].split(",")
            payload = payload[keys_end + 1 :]

        items = {}
        index = 0
        while payload:
            type_, item_payload, payload = self.unpack_next_sequence(payload)
            if type_ == "pydantic":
                items[keys[index]] = self.from_sequence_pydantic(item_payload)
            elif type_ == "python":
                items[keys[index]] = self.from_sequence_python(item_payload)
            else:
                raise ValueError("Invalid type in mapping sequence")
            index += 1
        return items

    def to_sequence_pydantic(self, obj: BaseModel) -> str | bytes:
        """
        Serialize Pydantic model to sequence format with class metadata.

        :param obj: Pydantic model instance to serialize
        :return: Sequence string or bytes containing class info and JSON data
        """
        class_name: str = obj.__class__.__name__
        class_module: str = obj.__class__.__module__
        json_data = obj.__pydantic_serializer__.to_json(obj)

        return (
            (class_name.encode() + b"|" + class_module.encode() + b"|" + json_data)
            if HAS_ORJSON
            else (
                class_name + "|" + class_module + "|" + json_data.decode()
                if isinstance(json_data, bytes)
                else json_data
            )
        )

    def from_sequence_pydantic(self, data: str | bytes) -> BaseModel:
        """
        Reconstruct Pydantic model from sequence format.

        :param data: Sequence data containing class metadata and JSON
        :return: Reconstructed Pydantic model instance
        """
        if isinstance(data, bytes):
            class_name_end = data.index(b"|")
            class_name = data[:class_name_end].decode()
            module_name_end = data.index(b"|", class_name_end + 1)
            module_name = data[class_name_end + 1 : module_name_end].decode()
            json_data = data[module_name_end + 1 :]
        else:
            class_name_end = data.index("|")
            class_name = data[:class_name_end]
            module_name_end = data.index("|", class_name_end + 1)
            module_name = data[class_name_end + 1 : module_name_end]
            json_data = data[module_name_end + 1 :]

        model_class = self.load_pydantic(class_name, module_name)

        return model_class.model_validate_json(json_data)

    def to_sequence_python(self, obj: Any) -> str | bytes:
        """
        Serialize Python object to JSON format.

        :param obj: Python object to serialize
        :return: JSON string or bytes representation
        """
        return orjson.dumps(obj) if HAS_ORJSON else json.dumps(obj)

    def from_sequence_python(self, data: str | bytes) -> Any:
        """
        Deserialize Python object from JSON format.

        :param data: JSON string or bytes to deserialize
        :return: Reconstructed Python object
        :raises ImportError: If orjson is required but not available
        """
        if isinstance(data, bytes):
            if not HAS_ORJSON:
                raise ImportError("orjson is not available, cannot deserialize bytes")
            return orjson.loads(data)

        return json.loads(data)

    def pack_next_sequence(  # noqa: C901, PLR0912
        self,
        type_: Literal[
            "pydantic",
            "python",
            "collection_tuple",
            "collection_sequence",
            "collection_mapping",
        ],
        payload: str | bytes,
        current: str | bytes | None,
    ) -> str | bytes:
        """
        Pack payload into sequence format with type and length metadata.

        :param type_: Type identifier for the payload
        :param payload: Data to pack into sequence
        :param current: Current sequence data to append to (unused but maintained
            for signature compatibility)
        :return: Packed sequence with type, length, and payload
        :raises ValueError: If payload type doesn't match current type or unknown
            type specified
        """
        if current is not None and type(payload) is not type(current):
            raise ValueError("Payload and current must be of the same type")

        payload_len = len(payload)

        if isinstance(payload, bytes):
            payload_len = payload_len.to_bytes(
                length=(payload_len.bit_length() + 7) // 8 if payload_len > 0 else 1,
                byteorder="big",
            )
            if type_ == "pydantic":
                payload_type = b"P"
            elif type_ == "python":
                payload_type = b"p"
            elif type_ == "collection_tuple":
                payload_type = b"T"
            elif type_ == "collection_sequence":
                payload_type = b"S"
            elif type_ == "collection_mapping":
                payload_type = b"M"
            else:
                raise ValueError(f"Unknown type for packing: {type_}")
            delimiter = b"|"
        else:
            payload_len = str(payload_len)
            if type_ == "pydantic":
                payload_type = "P"
            elif type_ == "python":
                payload_type = "p"
            elif type_ == "collection_tuple":
                payload_type = "T"
            elif type_ == "collection_sequence":
                payload_type = "S"
            elif type_ == "collection_mapping":
                payload_type = "M"
            else:
                raise ValueError(f"Unknown type for packing: {type_}")
            delimiter = "|"

        next_sequence = payload_type + delimiter + payload_len + delimiter + payload

        return current + next_sequence if current else next_sequence

    def unpack_next_sequence(  # noqa: C901, PLR0912
        self, data: str | bytes
    ) -> tuple[
        Literal[
            "pydantic",
            "python",
            "collection_tuple",
            "collection_sequence",
            "collection_mapping",
        ],
        str | bytes,
        str | bytes | None,
    ]:
        """
        Unpack sequence format to extract type, payload, and remaining data.

        :param data: Packed sequence data to unpack
        :return: Tuple of (type, payload, remaining_data)
        :raises ValueError: If sequence format is invalid or unknown type character
        """
        if isinstance(data, bytes):
            if len(data) < len(b"T|N") or data[1:2] != b"|":
                raise ValueError("Invalid packed data format")

            type_char = data[0:1]
            if type_char == b"P":
                type_ = "pydantic"
            elif type_char == b"p":
                type_ = "python"
            elif type_char == b"T":
                type_ = "collection_tuple"
            elif type_char == b"S":
                type_ = "collection_sequence"
            elif type_char == b"M":
                type_ = "collection_mapping"
            else:
                raise ValueError("Unknown type character in packed data")

            len_end = data.index(b"|", 2)
            payload_len = int.from_bytes(data[2:len_end], "big")
            payload = data[len_end + 1 : len_end + 1 + payload_len]
            remaining = (
                data[len_end + 1 + payload_len :]
                if len_end + 1 + payload_len < len(data)
                else None
            )

            return type_, payload, remaining

        if len(data) < len("T|N") or data[1] != "|":
            raise ValueError("Invalid packed data format")

        type_char = data[0]
        if type_char == "P":
            type_ = "pydantic"
        elif type_char == "p":
            type_ = "python"
        elif type_char == "S":
            type_ = "collection_sequence"
        elif type_char == "M":
            type_ = "collection_mapping"
        else:
            raise ValueError("Unknown type character in packed data")

        len_end = data.index("|", 2)
        payload_len = int(data[2:len_end])
        payload = data[len_end + 1 : len_end + 1 + payload_len]
        remaining = (
            data[len_end + 1 + payload_len :]
            if len_end + 1 + payload_len < len(data)
            else None
        )

        return type_, payload, remaining
