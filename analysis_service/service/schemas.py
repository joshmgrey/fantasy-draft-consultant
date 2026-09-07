"""Glue between Flask request/response objects and analysis_core.contract.

The wire dataclasses live in analysis_core so both sides share them; this
module only turns them into Flask responses and pulls a request body out.
"""

from typing import Optional

from flask import jsonify, request

from analysis_core.contract import (
    AnalyzeRequest,
    AnalyzeResponse,
    ContractError,
    ErrorResponse,
)


def parse_analyze_request() -> AnalyzeRequest:
    data = request.get_json(silent=True)
    if data is None:
        raise ContractError("request body must be a JSON object")
    return AnalyzeRequest.from_wire(data)


def analyze_response(resp: AnalyzeResponse):
    out = jsonify(resp.to_wire())
    out.status_code = 200
    return out


def error_response(
    code: str,
    detail: str = "",
    *,
    request_id: Optional[str] = None,
    status: Optional[int] = None,
):
    body = ErrorResponse(error=code, detail=detail, request_id=request_id)
    out = jsonify(body.to_wire())
    out.status_code = status if status is not None else body.status_code
    return out
