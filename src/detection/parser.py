import json


def parse_connectors(text: str) -> list:
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
                    return []
    return []
