class PeriodicSamples():
    PERIODS = {
        "Second": 14,
        "Minute": 12,
        "Hour": 10,
        "Day": 8,
        "Month": 6,
        "Year": 4
    }


    def __init__(self):
        self.count = 0
        self.sample = {p: 0 for p in self.PERIODS}
        self._prev = {p: "~" for p in self.PERIODS}


    def __call__(self, dt):
        self.count += 1
        for k, v in self.PERIODS.items():
            if dt[:v] == self._prev[k]:
                break
            self._prev[k] = dt[:v]
            self.sample[k] += 1


    def __str__(self):
        return "\t".join([str(self.count)] + [str(v) for v in self.sample.values()])
