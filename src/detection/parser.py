import json
import re


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _fix_missing_key_quote(text: str) -> str:
    # "key: value → "key": value （キーの閉じクォート抜け）
    return re.sub(r'"(\w+): ', r'"\1": ', text)


def parse_connectors(text: str) -> list:
    text = _strip_markdown(text)

    start = text.find('{"connectors"')
    if start == -1:
        start = text.find('{ "connectors"')
    if start == -1:
        return []

    depth = 0
    for i, char in enumerate(text[start:]):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:start + i + 1]
                try:
                    data = json.loads(json_str)
                    return data.get("connectors", [])
                except json.JSONDecodeError:
                    fixed = _fix_missing_key_quote(json_str)
                    try:
                        data = json.loads(fixed)
                        return data.get("connectors", [])
                    except json.JSONDecodeError:
                        return []
    return []
