"""
basic_function.py

제10회 2026 미래에셋증권 AI Festival
Task 2: 다중 조회 및 비교·연산용 deterministic Python functions

설계 원칙
---------
1. 검색/LLM이 공시에서 필요한 수치와 단위를 추출한 뒤 이 모듈에 전달한다.
2. 이 모듈은 "계산"만 담당한다. 공시 내용 추론이나 임의 보완은 하지 않는다.
3. 0 나눗셈, 누락값, 단위 불일치 등은 명시적으로 예외 처리한다.
4. 최종 답변 생성 LLM이 계산 결과와 근거 공시를 함께 사용하도록 한다.

권장 입력 형태 예시
-------------------
{
    "operation": "percent_change",
    "args": {
        "old": 1000,
        "new": 1200
    }
}

또는

{
    "operation": "argmax",
    "args": {
        "values": {"A사": 1200, "B사": 1500}
    }
}
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Number = Union[int, float]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BasicFunctionError(ValueError):
    """basic_function 연산 과정에서 발생하는 공통 예외."""


class InvalidNumberError(BasicFunctionError):
    """숫자가 아니거나 유한하지 않은 값이 들어온 경우."""


class DivisionByZeroError(BasicFunctionError):
    """분모가 0인 연산을 시도한 경우."""


class UnitConversionError(BasicFunctionError):
    """지원하지 않는 단위 또는 단위 변환 오류."""


# ---------------------------------------------------------------------------
# Numeric validation / helpers
# ---------------------------------------------------------------------------

def _to_float(value: Number, name: str = "value") -> float:
    """입력을 유한한 float로 검증/변환한다."""
    if isinstance(value, bool):
        raise InvalidNumberError(f"{name} must be numeric, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidNumberError(f"{name} is not a valid number: {value!r}") from exc

    if not isfinite(result):
        raise InvalidNumberError(f"{name} must be finite: {value!r}")
    return result


def _to_float_list(values: Iterable[Number], name: str = "values") -> List[float]:
    result = [_to_float(v, f"{name}[{i}]") for i, v in enumerate(values)]
    if not result:
        raise BasicFunctionError(f"{name} must not be empty.")
    return result


def safe_round(value: Number, digits: int = 4) -> float:
    """일관된 결과 출력을 위한 반올림."""
    return round(_to_float(value), int(digits))


# ---------------------------------------------------------------------------
# 1. 기본 산술
# ---------------------------------------------------------------------------

def add(a: Number, b: Number) -> float:
    return _to_float(a, "a") + _to_float(b, "b")


def subtract(a: Number, b: Number) -> float:
    return _to_float(a, "a") - _to_float(b, "b")


def multiply(a: Number, b: Number) -> float:
    return _to_float(a, "a") * _to_float(b, "b")


def divide(a: Number, b: Number) -> float:
    denominator = _to_float(b, "b")
    if denominator == 0:
        raise DivisionByZeroError("Cannot divide by zero.")
    return _to_float(a, "a") / denominator


def total(values: Sequence[Number]) -> float:
    """합계."""
    return sum(_to_float_list(values))


def average(values: Sequence[Number]) -> float:
    """산술평균."""
    nums = _to_float_list(values)
    return sum(nums) / len(nums)


def median_value(values: Sequence[Number]) -> float:
    """중앙값."""
    return float(median(_to_float_list(values)))


def weighted_average(values: Sequence[Number], weights: Sequence[Number]) -> float:
    """가중평균."""
    nums = _to_float_list(values, "values")
    w = _to_float_list(weights, "weights")

    if len(nums) != len(w):
        raise BasicFunctionError("values and weights must have the same length.")

    weight_sum = sum(w)
    if weight_sum == 0:
        raise DivisionByZeroError("Sum of weights cannot be zero.")

    return sum(v * weight for v, weight in zip(nums, w)) / weight_sum


# ---------------------------------------------------------------------------
# 2. 증감 / 성장률
# ---------------------------------------------------------------------------

def absolute_change(old: Number, new: Number) -> float:
    """
    절대 증감액 = new - old
    예: 매출 100 -> 120 => +20
    """
    return _to_float(new, "new") - _to_float(old, "old")


def percent_change(old: Number, new: Number) -> float:
    """
    증감률(%) = (new - old) / old * 100

    예:
        old=100, new=120 -> 20.0
        old=100, new=80  -> -20.0
    """
    old_v = _to_float(old, "old")
    new_v = _to_float(new, "new")

    if old_v == 0:
        raise DivisionByZeroError(
            "Percent change is undefined when the old value is zero."
        )
    return (new_v - old_v) / old_v * 100.0


def growth_rate(old: Number, new: Number) -> float:
    """percent_change의 별칭."""
    return percent_change(old, new)


def percentage_point_change(old_percent: Number, new_percent: Number) -> float:
    """
    퍼센트포인트 변화(%p) = new_percent - old_percent

    예: 12% -> 15% = +3%p
    """
    return _to_float(new_percent, "new_percent") - _to_float(old_percent, "old_percent")


def cagr(start_value: Number, end_value: Number, periods: Number) -> float:
    """
    연평균성장률(CAGR, %) = ((end/start)^(1/periods) - 1) * 100

    periods:
        연 단위 비교면 연수,
        분기 단위면 기간 수를 그대로 쓰기보다 호출부에서 연환산 여부를 명확히 결정할 것.
    """
    start = _to_float(start_value, "start_value")
    end = _to_float(end_value, "end_value")
    n = _to_float(periods, "periods")

    if start <= 0:
        raise BasicFunctionError("CAGR requires start_value > 0.")
    if end < 0:
        raise BasicFunctionError("CAGR requires end_value >= 0.")
    if n <= 0:
        raise BasicFunctionError("periods must be > 0.")

    return ((end / start) ** (1.0 / n) - 1.0) * 100.0


def period_changes(values: Sequence[Number]) -> List[Dict[str, Optional[float]]]:
    """
    시계열 값의 전기 대비 증감액/증감률 계산.

    반환 예:
    [
        {"index": 1, "old": 100, "new": 120, "change": 20, "change_pct": 20},
        ...
    ]

    이전 값이 0이면 change_pct는 None.
    """
    nums = _to_float_list(values)
    if len(nums) < 2:
        return []

    result = []
    for i in range(1, len(nums)):
        old, new = nums[i - 1], nums[i]
        pct = None if old == 0 else (new - old) / old * 100.0
        result.append({
            "index": i,
            "old": old,
            "new": new,
            "change": new - old,
            "change_pct": pct,
        })
    return result


# ---------------------------------------------------------------------------
# 3. 비율 / 비중 / 구성
# ---------------------------------------------------------------------------

def ratio(numerator: Number, denominator: Number, as_percent: bool = False) -> float:
    """
    비율 = numerator / denominator
    as_percent=True이면 100을 곱해 백분율로 반환.
    """
    denominator_v = _to_float(denominator, "denominator")
    if denominator_v == 0:
        raise DivisionByZeroError("denominator cannot be zero.")

    result = _to_float(numerator, "numerator") / denominator_v
    return result * 100.0 if as_percent else result


def share(part: Number, whole: Number) -> float:
    """
    비중(%) = part / whole * 100

    예: 설비투자 300 / 총투자 1000 = 30%
    """
    return ratio(part, whole, as_percent=True)


def composition_shares(values: Mapping[str, Number]) -> Dict[str, float]:
    """
    여러 항목의 구성비(%) 계산.

    예:
        {"유상증자": 300, "CB": 200, "BW": 500}
    ->
        {"유상증자": 30.0, "CB": 20.0, "BW": 50.0}
    """
    if not values:
        raise BasicFunctionError("values must not be empty.")

    nums = {k: _to_float(v, f"values[{k!r}]") for k, v in values.items()}
    s = sum(nums.values())

    if s == 0:
        raise DivisionByZeroError("Sum of values cannot be zero.")

    return {k: v / s * 100.0 for k, v in nums.items()}


# ---------------------------------------------------------------------------
# 4. 기업/연도/항목 간 비교
# ---------------------------------------------------------------------------

def difference(a: Number, b: Number) -> float:
    """
    단순 차이 = a - b

    비교 방향이 중요하므로 호출부에서
    'A - B'인지 'B - A'인지 명확하게 지정할 것.
    """
    return _to_float(a, "a") - _to_float(b, "b")


def relative_difference(base: Number, compare: Number) -> float:
    """
    base 대비 compare의 상대 차이(%)
    = (compare - base) / base * 100

    percent_change(base, compare)와 동일한 의미.
    """
    return percent_change(base, compare)


def compare_two(
    name_a: str,
    value_a: Number,
    name_b: str,
    value_b: Number,
) -> Dict[str, Any]:
    """
    두 대상의 값을 비교해 큰/작은 대상과 차이를 반환.
    동률도 명시적으로 처리한다.
    """
    a = _to_float(value_a, "value_a")
    b = _to_float(value_b, "value_b")

    if a > b:
        larger, smaller = name_a, name_b
    elif b > a:
        larger, smaller = name_b, name_a
    else:
        larger = smaller = None

    return {
        "name_a": name_a,
        "value_a": a,
        "name_b": name_b,
        "value_b": b,
        "difference_a_minus_b": a - b,
        "absolute_difference": abs(a - b),
        "larger": larger,
        "smaller": smaller,
        "is_tie": a == b,
    }


def argmax(values: Mapping[str, Number]) -> Dict[str, Any]:
    """가장 큰 값을 가진 항목과 값을 반환. 공동 1위 지원."""
    if not values:
        raise BasicFunctionError("values must not be empty.")

    nums = {k: _to_float(v, f"values[{k!r}]") for k, v in values.items()}
    max_value = max(nums.values())
    winners = [k for k, v in nums.items() if v == max_value]

    return {
        "keys": winners,
        "value": max_value,
        "is_tie": len(winners) > 1,
    }


def argmin(values: Mapping[str, Number]) -> Dict[str, Any]:
    """가장 작은 값을 가진 항목과 값을 반환. 공동 최저 지원."""
    if not values:
        raise BasicFunctionError("values must not be empty.")

    nums = {k: _to_float(v, f"values[{k!r}]") for k, v in values.items()}
    min_value = min(nums.values())
    winners = [k for k, v in nums.items() if v == min_value]

    return {
        "keys": winners,
        "value": min_value,
        "is_tie": len(winners) > 1,
    }


def rank_values(
    values: Mapping[str, Number],
    descending: bool = True,
) -> List[Dict[str, Any]]:
    """
    값 순위.
    동일 값은 같은 rank를 부여하는 competition ranking 방식.
    예: 100, 100, 80 -> 1, 1, 3
    """
    if not values:
        raise BasicFunctionError("values must not be empty.")

    items = [(k, _to_float(v, f"values[{k!r}]")) for k, v in values.items()]
    items.sort(key=lambda x: x[1], reverse=descending)

    result: List[Dict[str, Any]] = []
    previous_value: Optional[float] = None
    previous_rank = 0

    for i, (key, value) in enumerate(items, start=1):
        if previous_value is None or value != previous_value:
            rank = i
            previous_rank = rank
        else:
            rank = previous_rank

        result.append({"rank": rank, "key": key, "value": value})
        previous_value = value

    return result


# ---------------------------------------------------------------------------
# 5. 누적 / 유형별 집계
# ---------------------------------------------------------------------------

def cumulative_sum(values: Sequence[Number]) -> List[float]:
    """누적합."""
    nums = _to_float_list(values)
    result = []
    running = 0.0
    for v in nums:
        running += v
        result.append(running)
    return result


def sum_by_group(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_key: str,
) -> Dict[str, float]:
    """
    유형별/기업별/연도별 합계.

    예:
    records = [
        {"type": "CB", "amount": 100},
        {"type": "CB", "amount": 50},
        {"type": "BW", "amount": 70},
    ]

    sum_by_group(records, "type", "amount")
    -> {"CB": 150, "BW": 70}
    """
    if not records:
        return {}

    result: Dict[str, float] = {}

    for i, row in enumerate(records):
        if group_key not in row:
            raise BasicFunctionError(f"records[{i}] missing group_key={group_key!r}")
        if value_key not in row:
            raise BasicFunctionError(f"records[{i}] missing value_key={value_key!r}")

        group = str(row[group_key])
        value = _to_float(row[value_key], f"records[{i}][{value_key!r}]")
        result[group] = result.get(group, 0.0) + value

    return result


def average_by_group(
    records: Sequence[Mapping[str, Any]],
    group_key: str,
    value_key: str,
) -> Dict[str, float]:
    """유형별/기업별/연도별 평균."""
    if not records:
        return {}

    buckets: Dict[str, List[float]] = {}

    for i, row in enumerate(records):
        if group_key not in row or value_key not in row:
            raise BasicFunctionError(
                f"records[{i}] must contain {group_key!r} and {value_key!r}"
            )

        group = str(row[group_key])
        value = _to_float(row[value_key], f"records[{i}][{value_key!r}]")
        buckets.setdefault(group, []).append(value)

    return {k: sum(vs) / len(vs) for k, vs in buckets.items()}


# ---------------------------------------------------------------------------
# 6. 단위 변환
# ---------------------------------------------------------------------------

# 원화 기준 배수.
KRW_UNIT_MULTIPLIERS: Dict[str, float] = {
    "원": 1.0,
    "천원": 1_000.0,
    "만원": 10_000.0,
    "백만원": 1_000_000.0,
    "억원": 100_000_000.0,
    "조원": 1_000_000_000_000.0,
}


def convert_krw_unit(value: Number, from_unit: str, to_unit: str = "원") -> float:
    """
    한국 원화 단위 변환.

    지원:
        원, 천원, 만원, 백만원, 억원, 조원

    예:
        convert_krw_unit(12.3, "억원", "백만원") -> 1230.0
    """
    if from_unit not in KRW_UNIT_MULTIPLIERS:
        raise UnitConversionError(f"Unsupported from_unit: {from_unit}")
    if to_unit not in KRW_UNIT_MULTIPLIERS:
        raise UnitConversionError(f"Unsupported to_unit: {to_unit}")

    value_v = _to_float(value, "value")
    value_in_won = value_v * KRW_UNIT_MULTIPLIERS[from_unit]
    return value_in_won / KRW_UNIT_MULTIPLIERS[to_unit]


def normalize_percent(value: Number, unit: str = "%") -> float:
    """
    백분율 값을 % 기준 숫자로 정규화.

    unit="%": 12.5 -> 12.5
    unit="ratio": 0.125 -> 12.5
    """
    v = _to_float(value, "value")

    if unit == "%":
        return v
    if unit == "ratio":
        return v * 100.0

    raise UnitConversionError(f"Unsupported percent unit: {unit}")


# ---------------------------------------------------------------------------
# 7. 검증 보조
# ---------------------------------------------------------------------------

def all_same_unit(units: Sequence[str]) -> bool:
    """비교 대상 단위가 모두 같은지 확인."""
    if not units:
        return True
    first = units[0]
    return all(unit == first for unit in units)


def require_same_unit(units: Sequence[str]) -> None:
    """
    단위 불일치 시 예외.
    금액 비교 전에 호출하는 것을 권장.
    """
    if not all_same_unit(units):
        raise UnitConversionError(f"Units are not identical: {list(units)}")


# ---------------------------------------------------------------------------
# 8. Generic dispatcher
# ---------------------------------------------------------------------------

_OPERATION_MAP = {
    # arithmetic
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "sum": total,
    "total": total,
    "average": average,
    "mean": average,
    "median": median_value,
    "weighted_average": weighted_average,

    # change / growth
    "absolute_change": absolute_change,
    "difference": difference,
    "percent_change": percent_change,
    "growth_rate": growth_rate,
    "relative_difference": relative_difference,
    "percentage_point_change": percentage_point_change,
    "cagr": cagr,
    "period_changes": period_changes,

    # ratio / share
    "ratio": ratio,
    "share": share,
    "composition_shares": composition_shares,

    # compare / rank
    "compare_two": compare_two,
    "argmax": argmax,
    "max": argmax,
    "argmin": argmin,
    "min": argmin,
    "rank": rank_values,
    "rank_values": rank_values,

    # aggregate
    "cumulative_sum": cumulative_sum,
    "sum_by_group": sum_by_group,
    "average_by_group": average_by_group,

    # units
    "convert_krw_unit": convert_krw_unit,
    "normalize_percent": normalize_percent,
}

OPERATION_SPECS = {
    "add": {
        "description": "두 값을 더합니다.",
        "arguments": {
            "a": "number",
            "b": "number"
        }
    },
    "subtract": {
        "description": "a에서 b를 뺍니다.",
        "arguments": {
            "a": "number",
            "b": "number"
        }
    },
    "multiply": {
        "description": "두 값을 곱합니다.",
        "arguments": {
            "a": "number",
            "b": "number"
        }
    },
    "divide": {
        "description": "a를 b로 나눕니다.",
        "arguments": {
            "a": "number",
            "b": "number"
        }
    },
    "sum": {
        "description": "여러 값의 합계를 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },
    "total": {
        "description": "여러 값의 합계를 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },
    "average": {
        "description": "여러 값의 평균을 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },
    "mean": {
        "description": "여러 값의 평균을 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },
    "median": {
        "description": "여러 값의 중앙값을 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },
    "weighted_average": {
        "description": "가중평균을 계산합니다.",
        "arguments": {
            "values": "list[number]",
            "weights": "list[number]"
        }
    },

    "absolute_change": {
        "description": "이전 값에서 현재 값으로 얼마나 증가하거나 감소했는지 절대 증감액을 계산합니다. new-old",
        "arguments": {
            "old": "number",
            "new": "number"
        }
    },
    "percent_change": {
        "description": "이전 값 대비 현재 값의 증감률(%)을 계산합니다.",
        "arguments": {
            "old": "number",
            "new": "number"
        }
    },
    "growth_rate": {
        "description": "이전 값 대비 현재 값의 증감률(%)을 계산합니다.",
        "arguments": {
            "old": "number",
            "new": "number"
        }
    },
    "relative_difference": {
        "description": "기준값 대비 비교값이 몇 % 증가하거나 감소했는지 계산합니다.",
        "arguments": {
            "base": "number",
            "compare": "number"
        }
    },
    "percentage_point_change": {
        "description": "두 백분율 값 사이의 변화폭을 퍼센트포인트(%p)로 계산합니다.",
        "arguments": {
            "old_percent": "number",
            "new_percent": "number"
        }
    },
    "cagr": {
        "description": "시작값과 종료값 사이의 연평균성장률(CAGR)을 계산합니다.",
        "arguments": {
            "start_value": "number",
            "end_value": "number",
            "periods": "number"
        }
    },
    "period_changes": {
        "description": "연도별 또는 분기별 연속 값들의 전기 대비 증감액과 증감률을 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },

    "ratio": {
        "description": "분자와 분모의 비율을 계산합니다.",
        "arguments": {
            "numerator": "number",
            "denominator": "number",
            "as_percent": "bool(optional)"
        }
    },
    "share": {
        "description": "전체 값에서 특정 값이 차지하는 비중(%)을 계산합니다.",
        "arguments": {
            "part": "number",
            "whole": "number"
        }
    },
    "composition_shares": {
        "description": "여러 항목 각각이 전체에서 차지하는 구성비(%)를 계산합니다.",
        "arguments": {
            "values": "object[str, number]"
        }
    },

    "difference": {
        "description": "두 값의 차이 a-b를 계산합니다.",
        "arguments": {
            "a": "number",
            "b": "number"
        }
    },
    "compare_two": {
        "description": "두 대상을 비교해 더 큰 대상, 더 작은 대상, 차이, 동률 여부를 반환합니다.",
        "arguments": {
            "name_a": "string",
            "value_a": "number",
            "name_b": "string",
            "value_b": "number"
        }
    },
    "argmax": {
        "description": "여러 기업, 연도, 항목 중 가장 큰 값을 가진 대상을 찾습니다.",
        "arguments": {
            "values": "object[str, number]"
        }
    },
    "max": {
        "description": "여러 기업, 연도, 항목 중 가장 큰 값을 가진 대상을 찾습니다.",
        "arguments": {
            "values": "object[str, number]"
        }
    },
    "argmin": {
        "description": "여러 기업, 연도, 항목 중 가장 작은 값을 가진 대상을 찾습니다.",
        "arguments": {
            "values": "object[str, number]"
        }
    },
    "min": {
        "description": "여러 기업, 연도, 항목 중 가장 작은 값을 가진 대상을 찾습니다.",
        "arguments": {
            "values": "object[str, number]"
        }
    },
    "rank_values": {
        "description": "여러 기업, 연도, 항목의 값을 기준으로 순위를 계산합니다.",
        "arguments": {
            "values": "object[str, number]",
            "descending": "bool(optional)"
        }
    },
    "rank": {
        "description": "여러 기업, 연도, 항목의 값을 기준으로 순위를 계산합니다.",
        "arguments": {
            "values": "object[str, number]",
            "descending": "bool(optional)"
        }
    },

    "cumulative_sum": {
        "description": "연속된 값들의 누적합을 계산합니다.",
        "arguments": {
            "values": "list[number]"
        }
    },
    "sum_by_group": {
        "description": "여러 레코드를 기업, 유형, 연도 등 특정 기준으로 묶어 합계를 계산합니다.",
        "arguments": {
            "records": "list[object]",
            "group_key": "string",
            "value_key": "string"
        }
    },
    "average_by_group": {
        "description": "여러 레코드를 기업, 유형, 연도 등 특정 기준으로 묶어 평균을 계산합니다.",
        "arguments": {
            "records": "list[object]",
            "group_key": "string",
            "value_key": "string"
        }
    },

    "convert_krw_unit": {
        "description": "원, 천원, 만원, 백만원, 억원, 조원 사이의 금액 단위를 변환합니다.",
        "arguments": {
            "value": "number",
            "from_unit": "string",
            "to_unit": "string"
        }
    },
    "normalize_percent": {
        "description": "ratio 또는 % 형식의 비율 값을 % 기준 숫자로 정규화합니다.",
        "arguments": {
            "value": "number",
            "unit": "string"
        }
    }
}

def supported_operations() -> List[str]:
    """지원 연산명 목록."""
    return sorted(_OPERATION_MAP.keys())

def get_operation_specs():
    return OPERATION_SPECS

def execute_operation(operation: str, **kwargs: Any) -> Any:
    """
    문자열 연산명을 실제 함수에 연결.

    예:
        execute_operation("percent_change", old=100, new=120)
        execute_operation("argmax", values={"A": 100, "B": 120})

    검색계획/연산계획 LLM이 operation + args JSON을 만들고,
    Python executor가 이 함수를 호출하는 구조를 권장한다.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise BasicFunctionError("operation must be a non-empty string.")

    op = operation.strip().lower()

    if op not in _OPERATION_MAP:
        raise BasicFunctionError(
            f"Unsupported operation: {operation!r}. "
            f"Supported operations: {supported_operations()}"
        )

    return _OPERATION_MAP[op](**kwargs)


# ---------------------------------------------------------------------------
# Optional: result container for executor integration
# ---------------------------------------------------------------------------

@dataclass
class OperationResult:
    operation: str
    success: bool
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }


def safe_execute_operation(operation: str, **kwargs: Any) -> Dict[str, Any]:
    """
    API/Agent 실행기에서 예외 때문에 전체 파이프라인이 중단되지 않도록
    success/error 형태로 감싼 실행 함수.

    예:
        {
            "operation": "percent_change",
            "success": True,
            "result": 20.0,
            "error": None
        }
    """
    try:
        result = execute_operation(operation, **kwargs)
        return OperationResult(
            operation=operation,
            success=True,
            result=result,
            error=None,
        ).to_dict()
    except (BasicFunctionError, TypeError, KeyError) as exc:
        return OperationResult(
            operation=operation,
            success=False,
            result=None,
            error=str(exc),
        ).to_dict()


if __name__ == "__main__":
    # 간단한 smoke test
    print("supported_operations =", supported_operations())
    print("percent_change =", execute_operation("percent_change", old=100, new=120))
    print(
        "argmax =",
        execute_operation(
            "argmax",
            values={"A사": 1300, "B사": 1700, "C사": 1200},
        ),
    )
    print(
        "sum_by_group =",
        execute_operation(
            "sum_by_group",
            records=[
                {"type": "유상증자", "amount": 300},
                {"type": "CB", "amount": 200},
                {"type": "CB", "amount": 100},
            ],
            group_key="type",
            value_key="amount",
        ),
    )

    
