import re
from pathlib import Path

from loguru import logger

from guidellm.settings import settings
from guidellm.utils.text import load_text


def create_report(js_data: dict, output_path: str | Path) -> Path:
    """
    Creates a report from the dictionary and saves it to the output path.

    :param js_data: dict with match str and json data to inject
    :type js_data: dict
    :param output_path: the file to save the report to.
    :type output_path: str
    :return: the path to the saved report
    :rtype: str
    """

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    html_content = load_text(settings.report_generation.source)
    report_content = inject_data(
        js_data,
        html_content,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content)
    return output_path


def inject_data(
    js_data: dict,
    html: str,
) -> str:
    """
    Injects the json data into the HTML,
    replacing placeholders only within the <head> section.

    :param js_data: the json data to inject
    :type js_data: dict
    :param html: the html to inject the data into
    :type html: str
    :return: the html with the json data injected
    :rtype: str
    """
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.DOTALL | re.IGNORECASE)
    if not head_match:
        logger.warning("<head> section missing, returning original HTML.")

        return html

    head_content = head_match.group(1)

    # Replace placeholders only inside the <head> content
    for placeholder, script in js_data.items():
        head_content = head_content.replace(placeholder, script)

    # Rebuild the HTML
    new_head = f"<head>{head_content}</head>"
    return html[: head_match.start()] + new_head + html[head_match.end() :]
