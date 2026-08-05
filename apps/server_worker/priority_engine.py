# This file is part of DT_SERVER.
#
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

"""Online synchronization-priority assignment for tracked persons.

    priority = predicted cumulative urgency + AoI penalty

Input format::

    {
        "persons": [
            {
                "track_id": 101,
                "position": {"x": 1.2, "y": 0.8, "z": 0.1}
            }
        ]
    }

Output format::

    {
        "persons": [
            {
                "track_id": 101,
                "position": {"x": 1.2, "y": 0.8, "z": 0.1},
                "hazard_score": 1.25,
                "aoi_score": 0.10,
                "rank": 1
            }
        ]
    }

Every input person is returned in the original list order. ``rank=1`` means
the highest synchronization priority. ``hazard_score`` is the predicted
cumulative spatial urgency, and ``aoi_score`` is the weighted AoI penalty.
The total ranking score is ``hazard_score + aoi_score``. Existing input fields
are preserved in the output.

Only a track ID and a world position are required from the inference stage.
Velocity is estimated from the previous observation of the same track ID.
The elapsed time between consecutive calls becomes the unit time for velocity,
future-state prediction, urgency integration, and AoI.  The public
``assign_priority`` function keeps its engine alive between calls, so it can
be inserted directly after ``DLInferencer.process_tensor()``.

Hazards must be supplied only as ground-plane point coordinates through
``PriorityConfig(hazards=...)``. All coordinates use the two position keys in
``plane_axes`` (``x`` and ``y`` by default) and must use the same coordinate
system and unit as each person's position. Example::

    PriorityConfig(
        hazards=(
            (1.5, 0.8),
        ),
    )

Each ``(x, y)`` pair is one exact hazard point. The urgency calculation uses
the straight-line distance and movement direction from a person to each point.
There is no radius, line, polygon, or three-dimensional hazard calculation.

The default parameters are initially tuned for the following deployment
scenario:

* a 12 m (x-axis) by 4 m (y-axis) corridor,
* the lower-left corner is the origin ``(0, 0)``,
* positions and distances are measured in metres,
* observations normally arrive once per second, and
* the exact hazard point is ``(2 m, 4 m)``.

The map boundary itself is not used by the ranking calculation. These values
are an initial corridor-specific setting and should be retuned if the physical
space, coordinate unit, observation interval, or typical object speed changes.

"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Optional, Tuple


Point2D = Tuple[float, float]


DEFAULT_HAZARDS: Tuple[Point2D, ...] = (
    (2.0, 4.0),
)


@dataclass(frozen=True)
class PriorityConfig:
    """PBUS-A configuration initially tuned for a 12 m x 4 m corridor."""

    # 객체 위치에서 지면 좌표로 사용할 두 축입니다.
    # 기본값은 position["x"], position["y"]이며, 선택한 축이 거리·속도·
    # 이동 방향 계산에 모두 사용되므로 축을 바꾸면 랭킹도 달라집니다.
    plane_axes: Tuple[str, str] = ("x", "y")

    # 현재 시점 이후 몇 스텝까지 위치를 예측할지 정합니다.
    # 4이면 t, t+dt, t+2dt, t+3dt, t+4dt의 총 5개 시점을 평가합니다.
    # 값이 클수록 현재 위치보다 미래 이동 경로가 우선순위에 더 반영됩니다.
    prediction_steps: int = 4

    # 첫 호출에는 이전 timestamp가 없으므로 이 값을 단위시간 dt로 사용합니다.
    # 두 번째 호출부터는 실제 호출 간격을 사용합니다. dt는 속도, 미래 예측
    # 시간 범위, 공간 긴급도 적분 및 AoI 증가량에 함께 영향을 줍니다.
    initial_unit_time_s: float = 1.0

    # 위험 지점들의 (x, y) 좌표입니다. 객체 위치와 동일한 좌표계 및 단위를
    # 사용해야 합니다. 객체가 위험 지점에 가깝거나 그 방향으로 이동할수록
    # 공간 긴급도가 높아지므로 가장 직접적으로 랭킹에 영향을 줍니다.
    # 반경이나 면적은 적용하지 않고 각 좌표를 정확한 한 점으로 취급합니다.
    hazards: Tuple[Point2D, ...] = DEFAULT_HAZARDS

    # 위험 지점까지의 거리를 공간 긴급도로 변환하는 방식입니다.
    # "rbf": exp(-distance / sigma_d), 논문의 대표 수식과 같은 완만한 감쇠.
    # "inv": exp(inverse_length_scale / (distance + sigma_d)), 근거리 강조.
    # 12 m x 4 m 복도 기본 설정은 거리 차이를 안정적으로 반영하는 rbf입니다.
    kernel_mode: str = "rbf"

    # 거리 민감도입니다.
    # "rbf"에서는 클수록 먼 객체도 높은 위험도 영향을 받습니다.
    # "inv"에서는 클수록 지수값이 작아져 근거리 점수가 낮고 완만해집니다.
    # 기본값 2 m에서는 위험 지점에서 2 m 떨어졌을 때 거리 가중치가
    # exp(-1), 약 0.368이 됩니다.
    sigma_d: float = 2.0

    # 이동 방향 민감도입니다. 위험 지점을 향하는 속도의 내적을
    # tanh(dot / sigma_v)로 반영합니다. 작을수록 방향 영향이 빠르게
    # 강해지고, 클수록 이동 방향에 따른 점수 차이가 완만해집니다.
    # 기본값은 약 1 m/s의 복도 보행 속도를 기준으로 합니다.
    sigma_v: float = 1.0

    # "inv" 거리 커널에서만 사용하는 증폭 계수입니다.
    # 클수록, 특히 위험 지점과 가까운 객체의 공간 긴급도가 크게 증가합니다.
    # 기본 rbf 모드에서는 이 값이 계산에 사용되지 않습니다.
    inverse_length_scale: float = 2.0

    # 미래 스텝의 할인율입니다. step i의 점수에 delta**i가 곱해집니다.
    # 1.0이면 모든 예측 시점을 동일하게 보고, 0~1 사이에서 작아질수록
    # 먼 미래보다 현재와 가까운 미래를 더 중요하게 봅니다.
    delta: float = 0.9

    # 한 예측 시점에서 모든 위험 지점으로부터 얻는 공간 긴급도의 상한입니다.
    # 값이 너무 작으면 고위험 객체들의 점수가 같은 상한에 묶여 랭킹 차이가
    # 줄어듭니다. None으로 설정하면 상한을 적용하지 않습니다.
    # 현재 단일 위험 지점 복도 설정에서는 점수 차이를 보존하도록 비활성화합니다.
    w_clip: Optional[float] = None

    # AoI 증가 민감도입니다. AoI 항은 exp(aoi / sigma_aoi) - 1입니다.
    # 작을수록 오래 동기화되지 않은 객체의 점수가 더 빠르게 상승합니다.
    # 1초 단위 테스트에서 수 초 내에 AoI 효과를 확인하도록 4초로 설정합니다.
    sigma_aoi: float = 4.0

    # 전체 우선순위에서 AoI 항에 곱하는 가중치입니다.
    # 클수록 위험 지역과의 거리보다 동기화되지 않은 시간이 중요해집니다.
    # 0이면 AoI는 랭킹에 영향을 주지 않습니다.
    lambda_aoi: float = 0.1

    # AoI 지수 계산값의 최대치로, 지나치게 큰 수와 overflow를 방지합니다.
    # 작게 설정하면 매우 오래 미동기화된 객체의 AoI 점수가 일찍 포화됩니다.
    aoi_exp_clip: float = 60.0

    # 호출 시 selected_count를 생략했을 때 사용할 기본 동기화 객체 수입니다.
    # 현재 스텝의 rank 계산 자체는 바꾸지 않지만, 선택된 상위 N개의 AoI를
    # 0으로 초기화하므로 다음 스텝 이후의 랭킹에 간접적으로 영향을 줍니다.
    selected_count: int = 3

    # 측정 속도에 적용하는 평활화 계수 alpha입니다.
    # 1.0이면 직전 두 위치로 구한 속도를 그대로 사용합니다. 작게 설정하면
    # 이전 속도를 더 유지하여 위치 노이즈로 인한 랭킹 변동을 줄입니다.
    velocity_smoothing: float = 1.0

    # timestamp 차이가 이 값보다 작으면 새 속도를 계산하지 않고 이전 속도를
    # 유지합니다. 단위시간도 최소 이 값으로 제한하여 0으로 나누는 것을 막습니다.
    minimum_dt_s: float = 1e-6

    # 객체가 이 시간보다 오래 관측되지 않으면 저장한 위치·속도·AoI 상태를
    # 삭제합니다. 이후 같은 ID가 다시 들어오면 신규 객체처럼 시작합니다.
    track_timeout_s: float = 10.0

    # hazard_score와 aoi_score는 항상 출력됩니다. True이면 두 점수의 합인
    # priority_score도 추가합니다. 계산과 rank에는 영향을 주지 않습니다.
    include_priority_score: bool = False

    def __post_init__(self) -> None:
        if len(self.plane_axes) != 2 or self.plane_axes[0] == self.plane_axes[1]:
            raise ValueError("plane_axes must contain two different position keys")
        if self.prediction_steps < 0:
            raise ValueError("prediction_steps must not be negative")
        if self.initial_unit_time_s <= 0.0:
            raise ValueError("initial_unit_time_s must be positive")
        if self.kernel_mode not in {"rbf", "inv"}:
            raise ValueError("kernel_mode must be 'rbf' or 'inv'")
        if self.sigma_d <= 0.0 or self.sigma_v <= 0.0:
            raise ValueError("sigma_d and sigma_v must be positive")
        if self.sigma_aoi <= 0.0:
            raise ValueError("sigma_aoi must be positive")
        if self.selected_count < 0:
            raise ValueError("selected_count must not be negative")
        if not 0.0 < self.velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in (0, 1]")
        if self.minimum_dt_s <= 0.0:
            raise ValueError("minimum_dt_s must be positive")
        if self.track_timeout_s <= 0.0:
            raise ValueError("track_timeout_s must be positive")
        for index, hazard in enumerate(self.hazards):
            if len(hazard) != 2:
                raise ValueError(f"hazards[{index}] must be one (x, y) point")
            try:
                hazard_x = float(hazard[0])
                hazard_y = float(hazard[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"hazards[{index}] coordinates must be numeric"
                ) from exc
            if not math.isfinite(hazard_x) or not math.isfinite(hazard_y):
                raise ValueError(f"hazards[{index}] coordinates must be finite")


@dataclass
class _TrackState:
    plane_x: float
    plane_y: float
    timestamp: float
    velocity_x: float
    velocity_y: float
    aoi_seconds: float
    selected_last_step: bool


@dataclass
class _Candidate:
    input_index: int
    track_id: Hashable
    plane_x: float
    plane_y: float
    velocity_x: float
    velocity_y: float
    previous_aoi: float
    hazard_score: float
    aoi_score: float
    score: float


class PriorityEngine:
    """Stateful online engine with one public ranking operation."""

    def __init__(self, config: Optional[PriorityConfig] = None) -> None:
        self.config = config or PriorityConfig()
        self._tracks: Dict[Hashable, _TrackState] = {}
        self._last_decision_timestamp: Optional[float] = None
        self._lock = threading.RLock()

    def reset(self) -> None:
        """Clear velocity and AoI history."""

        with self._lock:
            self._tracks.clear()
            self._last_decision_timestamp = None

    def assign_priority(
        self,
        inference_result: Mapping,
        timestamp: Optional[float] = None,
        selected_count: Optional[int] = None,
    ) -> dict:
        """Return a copy of ``inference_result`` with a 1-based rank per person.

        Args:
            inference_result:
                Input in the following form::

                    {
                        "persons": [
                            {
                                "track_id": 101,
                                "position": {
                                    "x": 1.2,
                                    "y": 0.8,
                                    "z": 0.1
                                }
                            }
                        ]
                    }

                ``track_id`` must be unique in one input frame. The configured
                two ground-plane coordinates (``x`` and ``y`` by default) must
                be numeric. Other position keys and person fields are kept.
            timestamp:
                Observation time in seconds. ROS clock time is recommended.
                ``time.monotonic()`` is used when omitted.
            selected_count:
                Number of highest-ranked persons considered synchronized in
                this step. When omitted, ``PriorityConfig.selected_count`` is
                used (3 by default). If fewer persons are present, all of them
                are considered synchronized.

        The persons remain in their original list order. ``hazard_score``,
        ``aoi_score``, and ``rank`` are added. Rank 1 is the highest
        synchronization priority. For example::

            {
                "persons": [
                    {
                        "track_id": 101,
                        "position": {"x": 1.2, "y": 0.8, "z": 0.1},
                        "hazard_score": 1.25,
                        "aoi_score": 0.10,
                        "rank": 1
                    }
                ]
            }

        ``hazard_score`` contains the predicted cumulative spatial urgency.
        ``aoi_score`` already includes ``lambda_aoi``. Their sum is the score
        used for ranking. When ``include_priority_score=True``, that sum is
        also returned as ``priority_score``.
        """

        if not isinstance(inference_result, Mapping):
            raise TypeError("inference_result must be a mapping")

        raw_persons = inference_result.get("persons", [])
        if raw_persons is None:
            raw_persons = []
        if not isinstance(raw_persons, list):
            raise ValueError("inference_result['persons'] must be a list")

        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("timestamp must be finite")
        effective_selected_count = self._resolve_selected_count(selected_count)

        with self._lock:
            self._prune_stale_tracks(now)
            unit_dt = self._unit_time(now)

            output_persons = []
            candidates = []
            seen_track_ids = set()

            for input_index, raw_person in enumerate(raw_persons):
                if not isinstance(raw_person, Mapping):
                    raise ValueError(f"persons[{input_index}] must be a mapping")

                person = dict(raw_person)
                position = raw_person.get("position")
                if not isinstance(position, Mapping):
                    raise ValueError(
                        f"persons[{input_index}]['position'] must be a mapping"
                    )

                track_id = raw_person.get("track_id")
                if track_id is None:
                    raise ValueError(f"persons[{input_index}] has no track_id")
                if not isinstance(track_id, Hashable):
                    raise ValueError(
                        f"persons[{input_index}]['track_id'] must be hashable"
                    )
                if track_id in seen_track_ids:
                    raise ValueError(f"duplicate track_id in one frame: {track_id!r}")
                seen_track_ids.add(track_id)

                plane_x, plane_y = self._read_ground_position(position, input_index)
                previous = self._tracks.get(track_id)
                velocity_x, velocity_y = self._estimate_velocity(
                    previous,
                    plane_x,
                    plane_y,
                    now,
                )
                previous_aoi = self._current_aoi(previous, now)

                hazard_score = self._predicted_cumulative_urgency(
                    plane_x,
                    plane_y,
                    velocity_x,
                    velocity_y,
                    unit_dt,
                )
                aoi_score = self._aoi_penalty(previous_aoi)
                score = hazard_score + aoi_score

                # Preserve the original nested position data and any extra fields.
                person["position"] = dict(position)
                output_persons.append(person)
                candidates.append(
                    _Candidate(
                        input_index=input_index,
                        track_id=track_id,
                        plane_x=plane_x,
                        plane_y=plane_y,
                        velocity_x=velocity_x,
                        velocity_y=velocity_y,
                        previous_aoi=previous_aoi,
                        hazard_score=hazard_score,
                        aoi_score=aoi_score,
                        score=score,
                    )
                )

            # Python's sort is stable, so equal scores preserve inference order.
            ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
            for rank, candidate in enumerate(ranked, start=1):
                output_person = output_persons[candidate.input_index]
                output_person["hazard_score"] = float(candidate.hazard_score)
                output_person["aoi_score"] = float(candidate.aoi_score)
                output_person["rank"] = rank
                if self.config.include_priority_score:
                    output_person["priority_score"] = float(candidate.score)

            synchronized_count = min(effective_selected_count, len(ranked))
            selected_ids = {
                candidate.track_id for candidate in ranked[:synchronized_count]
            }
            for candidate in candidates:
                is_selected = candidate.track_id in selected_ids
                next_aoi = 0.0 if is_selected else candidate.previous_aoi
                self._tracks[candidate.track_id] = _TrackState(
                    plane_x=candidate.plane_x,
                    plane_y=candidate.plane_y,
                    timestamp=now,
                    velocity_x=candidate.velocity_x,
                    velocity_y=candidate.velocity_y,
                    aoi_seconds=next_aoi,
                    selected_last_step=is_selected,
                )

            self._last_decision_timestamp = now
            output = dict(inference_result)
            output["persons"] = output_persons
            return output

    def _resolve_selected_count(self, selected_count: Optional[int]) -> int:
        if selected_count is None:
            return self.config.selected_count
        if isinstance(selected_count, bool) or not isinstance(selected_count, int):
            raise TypeError("selected_count must be an integer")
        if selected_count < 0:
            raise ValueError("selected_count must not be negative")
        return selected_count

    def _read_ground_position(
        self,
        position: Mapping,
        input_index: int,
    ) -> Point2D:
        axis_x, axis_y = self.config.plane_axes
        try:
            plane_x = float(position[axis_x])
            plane_y = float(position[axis_y])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"persons[{input_index}]['position'] must contain numeric "
                f"{axis_x!r} and {axis_y!r}"
            ) from exc
        if not math.isfinite(plane_x) or not math.isfinite(plane_y):
            raise ValueError(f"persons[{input_index}] position must be finite")
        return plane_x, plane_y

    def _estimate_velocity(
        self,
        previous: Optional[_TrackState],
        plane_x: float,
        plane_y: float,
        now: float,
    ) -> Point2D:
        if previous is None:
            return 0.0, 0.0

        dt = now - previous.timestamp
        if dt < self.config.minimum_dt_s:
            return previous.velocity_x, previous.velocity_y

        measured_x = (plane_x - previous.plane_x) / dt
        measured_y = (plane_y - previous.plane_y) / dt
        alpha = self.config.velocity_smoothing
        velocity_x = alpha * measured_x + (1.0 - alpha) * previous.velocity_x
        velocity_y = alpha * measured_y + (1.0 - alpha) * previous.velocity_y
        return velocity_x, velocity_y

    def _unit_time(self, now: float) -> float:
        if self._last_decision_timestamp is None:
            return self.config.initial_unit_time_s
        dt = now - self._last_decision_timestamp
        if dt < self.config.minimum_dt_s:
            return self.config.minimum_dt_s
        return dt

    @staticmethod
    def _current_aoi(
        previous: Optional[_TrackState],
        now: float,
    ) -> float:
        if previous is None or previous.selected_last_step:
            return 0.0
        elapsed = max(0.0, now - previous.timestamp)
        return previous.aoi_seconds + elapsed

    def _prune_stale_tracks(self, now: float) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.timestamp > self.config.track_timeout_s
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

    def _predicted_cumulative_urgency(
        self,
        plane_x: float,
        plane_y: float,
        velocity_x: float,
        velocity_y: float,
        unit_dt: float,
    ) -> float:
        total = 0.0
        for step_index in range(self.config.prediction_steps + 1):
            offset = step_index * unit_dt
            predicted_position = (
                plane_x + velocity_x * offset,
                plane_y + velocity_y * offset,
            )
            urgency = self._instant_urgency(
                predicted_position,
                (velocity_x, velocity_y),
            )
            total += (
                (self.config.delta ** step_index)
                * urgency
                * unit_dt
            )
        return total

    def _instant_urgency(
        self,
        position: Point2D,
        velocity: Point2D,
    ) -> float:
        total = 0.0
        for hazard in self.config.hazards:
            distance = math.hypot(
                position[0] - hazard[0],
                position[1] - hazard[1],
            )
            kernel = self._distance_kernel(distance)
            alignment = self._alignment_score(position, velocity, hazard)
            total += kernel * alignment

        if self.config.w_clip is not None:
            total = min(total, self.config.w_clip)
        return total

    def _distance_kernel(self, distance: float) -> float:
        if self.config.kernel_mode == "rbf":
            return math.exp(-max(distance, 0.0) / self.config.sigma_d)

        exponent = self.config.inverse_length_scale / (
            max(distance, 0.0) + self.config.sigma_d
        )
        return math.exp(min(exponent, 200.0))

    def _alignment_score(
        self,
        position: Point2D,
        velocity: Point2D,
        hazard: Point2D,
    ) -> float:
        direction_x, direction_y = self._unit_vector(
            hazard[0] - position[0],
            hazard[1] - position[1],
        )
        dot = direction_x * velocity[0] + direction_y * velocity[1]
        return 1.0 + math.tanh(dot / self.config.sigma_v)

    @staticmethod
    def _unit_vector(x: float, y: float) -> Point2D:
        magnitude = math.hypot(x, y)
        if magnitude <= 1e-12:
            return 0.0, 0.0
        return x / magnitude, y / magnitude

    def _aoi_penalty(self, aoi_seconds: float) -> float:
        exponent = max(0.0, aoi_seconds) / self.config.sigma_aoi
        exponent = min(exponent, self.config.aoi_exp_clip)
        return self.config.lambda_aoi * (math.exp(exponent) - 1.0)


_DEFAULT_ENGINE = PriorityEngine()


def assign_priority(
    inference_result: Mapping,
    timestamp: Optional[float] = None,
    selected_count: Optional[int] = None,
) -> dict:
    """Return the input result with a synchronization ``rank`` per person.

    This is the one-function integration entry point intended for
    ``ros2_main.py``:

        inference_result = assign_priority(
            inference_result,
            timestamp=timestamp,
            selected_count=3,
        )

    The default engine retains per-track position, velocity, and AoI state
    between calls. ``selected_count`` may change on every call; when omitted,
    its configured default value is 3. See the module documentation above for
    the complete input/output schemas and hazard-coordinate configuration.
    """

    return _DEFAULT_ENGINE.assign_priority(
        inference_result,
        timestamp,
        selected_count,
    )
