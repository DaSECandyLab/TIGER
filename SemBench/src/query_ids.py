"""Canonical numeric and suffixed benchmark query identifiers."""

import re


def parse_query_id(value: str | int) -> int | str:
    match = re.fullmatch(r"[qQ]?(\d+)([a-z]*)", str(value).strip())
    if match is None:
        raise ValueError(f"Invalid query ID: {value!r}")
    number, suffix = match.groups()
    return f"{int(number)}{suffix}" if suffix else int(number)


def query_id_sort_key(value: str | int) -> tuple[int, str]:
    query = str(parse_query_id(value))
    match = re.fullmatch(r"(\d+)([a-z]*)", query)
    return int(match[1]), match[2]
