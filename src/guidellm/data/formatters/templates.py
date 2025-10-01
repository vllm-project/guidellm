import textwrap
from typing import Union

from jinja2 import Template

from guidellm.utils import RegistryMixin

__all__ = [
    "DEFAULT_AUDIO_TRANSCRIPTIONS_TEMPLATE",
    "DEFAULT_AUDIO_TRANSLATIONS_TEMPLATE",
    "DEFAULT_CHAT_COMPLETIONS_TEMPLATE",
    "DEFAULT_TEXT_COMPLETIONS_TEMPLATE",
    "JinjaTemplatesRegistry",
]


class JinjaTemplatesRegistry(RegistryMixin[Union[Template, str]]):
    pass


DEFAULT_TEXT_COMPLETIONS_TEMPLATE = JinjaTemplatesRegistry.register("text_completions")(
    textwrap.dedent("""
        {% set obj = {
            "json_body": {
                "prompt": (
                    text_column[0]
                    if text_column and text_column|length == 1
                    else text_column
                )
            }
        } %}

        {% if output_tokens_count is defined and output_tokens_count is not none %}
            {% do obj["json_body"].update({
                "max_tokens": output_tokens_count,
                "max_completion_tokens": output_tokens_count,
                "stop": None,
                "ignore_eos": True
            }) %}
        {% elif max_tokens is defined and max_tokens is not none %}
            {% do obj["json_body"].update({"max_tokens": max_tokens}) %}
        {% elif max_completion_tokens is defined and max_completion_tokens is not none %}
            {% do obj["json_body"].update({"max_completion_tokens": max_completion_tokens}) %}
        {% endif %}

        {{ obj }}
    """).strip()  # noqa: E501
)

DEFAULT_CHAT_COMPLETIONS_TEMPLATE = JinjaTemplatesRegistry.register("chat_completions")(
    textwrap.dedent("""
        {% set obj = {
            "json_body": {
                "messages": [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
            }
        } %}

        {%- for item in text_column or [] %}
            {% do obj["json_body"].messages[0].content.append({"type": "text", "text": item}) %}
        {%- endfor %}

        {%- for item in image_column or [] %}
            {% do obj["json_body"].messages[0].content.append({
                "type": "image_url",
                "image_url": encode_image(
                    item,
                    max_size=max_size|default(None),
                    max_width=max_width|default(None),
                    max_height=max_height|default(None),
                    encode_type=image_encode_type|default(encode_type|default(None))
                )
            }) %}
        {%- endfor %}

        {%- for item in video_column or [] %}
            {% do obj["json_body"].messages[0].content.append({
                "type": "video_url",
                "video_url": encode_video(
                    item,
                    encode_type=video_encode_type|default(encode_type|default(None))
                )
            }) %}
        {%- endfor %}

        {%- for item in audio_column or [] %}
            {%- set audio_type, audio_val = encode_audio(
                item,
                sample_rate=sample_rate|default(None),
                max_duration=max_duration|default(None),
                encode_type=audio_encode_type|default(encode_type|default(None))
            ) -%}
            {% do content_list.append({"type": audio_type, audio_type: audio_val}) %}
        {%- endfor %}

        {% if output_tokens_count is defined and output_tokens_count is not none %}
            {% do obj["json_body"].update({
                "max_completion_tokens": output_tokens_count,
                "stop": None,
                "ignore_eos": True
            }) %}
        {% elif max_tokens is defined and max_tokens is not none %}
            {% do obj["json_body"].update({"max_completion_tokens": max_tokens}) %}
        {% elif max_completion_tokens is defined and max_completion_tokens is not none %}
            {% do obj["json_body"].update({"max_completion_tokens": max_completion_tokens}) %}
        {% endif %}

        {{ obj }}
    """).strip()  # noqa: E501
)

DEFAULT_AUDIO_TRANSCRIPTIONS_TEMPLATE = JinjaTemplatesRegistry.register(
    "audio_transcriptions"
)(
    textwrap.dedent("""
        {
            {%- if output_tokens_count_column is defined and output_tokens_count_column is not none -%}
                "max_tokens": {{ output_tokens_count_column }},
                "max_completion_tokens": {{ output_tokens_count_column }},
                "stop": None,
                "ignore_eos": True,
            {%- else -%}
                {%- if max_tokens is defined and max_tokens is not none -%}
                    "max_tokens": {{ max_tokens }},
                {%- endif -%}
                {%- if max_completion_tokens is defined and max_completion_tokens is not none -%}
                    "max_completion_tokens": {{ max_completion_tokens }},
                {%- endif -%}
            {%- endif -%}
            "files": {
                "file": {{ encode_audio_file(
                    audio_column[0],
                    encode_type=audio_encode_type|default(encode_type|default(None))
                ) }}
            }
            {%- if text_column and text_column|length > 0 -%}
            ,
            "json": {
                "prompt": {{ text_column[0] }}
            }
            {%- endif -%}
        }
    """).strip()  # noqa: E501
)

DEFAULT_AUDIO_TRANSLATIONS_TEMPLATE = JinjaTemplatesRegistry.register(
    "audio_translations"
)(
    textwrap.dedent("""
        {
            {%- if output_tokens_count_column is defined and output_tokens_count_column is not none -%}
                "max_tokens": {{ output_tokens_count_column }},
                "max_completion_tokens": {{ output_tokens_count_column }},
                "stop": None,
                "ignore_eos": True,
            {%- else -%}
                {%- if max_tokens is defined and max_tokens is not none -%}
                    "max_tokens": {{ max_tokens }},
                {%- endif -%}
                {%- if max_completion_tokens is defined and max_completion_tokens is not none -%}
                    "max_completion_tokens": {{ max_completion_tokens }},
                {%- endif -%}
            {%- endif -%}
            "files": {
                "file": {{ encode_audio_file(
                    audio_column[0],
                    encode_type=audio_encode_type|default(encode_type|default(None))
                ) }}
            }
            {%- if text_column and text_column|length > 0 -%}
            ,
            "json": {
                "prompt": {{ text_column[0] }}
            }
            {%- endif -%}
        }
    """).strip()  # noqa: E501
)
