from dataclasses import dataclass, field


@dataclass
class DailyRecord:
    day: str
    datetime: str = "~"
    _2xx: int = 0
    _3xx: int = 0
    _4xx: int = 0
    _5xx: int = 0
    all: int = field(init=False)
    specimen: str = "~"
    filled: bool = field(init=False)
    resilience: float = 0.0
    digest: str = "~"
    content: str = "Unknown"
    fixity: float = 0.0
    chaos: float = 0.0
    chaosn: float = 0.0

    @property
    def all(self) -> int:
        return self._2xx + self._3xx + self._4xx + self._5xx

    @all.setter
    def all(self, _):
        pass

    @property
    def specimen(self) -> str:
        if self._specimen != "~":
            return self._specimen
        for k in ("_2xx", "_4xx", "_5xx", "_3xx"):
            if getattr(self, k):
                return k[1:]
        return self._specimen

    @specimen.setter
    def specimen(self, v):
        self._specimen = v if isinstance(v, str) else "~"

    @property
    def filled(self) -> bool:
        return self.specimen != "~" and not self.all

    @filled.setter
    def filled(self, _):
        pass

    def incr(self, status, count=1):
        k = "_" + status
        try:
            setattr(self, k, getattr(self, k) + count)
        except AttributeError as e:
            pass
