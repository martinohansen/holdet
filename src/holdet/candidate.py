from typing import Protocol


class Candidate(Protocol):
    captain: bool

    @property
    def id(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def team(self) -> str:
        ...

    @property
    def keeper(self) -> bool:
        ...

    @property
    def defense(self) -> bool:
        ...

    @property
    def midfielder(self) -> bool:
        ...

    @property
    def forward(self) -> bool:
        ...

    @property
    def price(self) -> float:
        ...

    @property
    def value(self) -> float:
        ...

    @property
    def xValue(self) -> float:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...
