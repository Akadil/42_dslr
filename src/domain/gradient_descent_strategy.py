class GradientDescentStrategy:
    def __init__(self, batch_size: int | None = None, name: str = "Batch") -> None:
        self.batch_size = batch_size
        self.name = name

    def __str__(self):
        return f"{self.name} (batch_size={self.batch_size})"

    def __repr__(self) -> str:
        return f"GradientDescentStrategy(name={self.name}, batch_size={self.batch_size})"

    @classmethod
    def BATCH(cls):
        return cls(batch_size=None, name="Batch")

    @classmethod
    def STOCHASTIC(cls):
        return cls(batch_size=1, name="Stochastic")

    @classmethod
    def MINI_BATCH(cls, batch_size: int):
        return cls(batch_size=batch_size, name="Mini-Batch")
    

# Personal tester code
if __name__ == "__main__":
    print(GradientDescentStrategy.BATCH())  # Output: Batch (batch_size=None)
    print(GradientDescentStrategy.STOCHASTIC())  # Output: Stochastic (batch_size=1)
    print(GradientDescentStrategy.MINI_BATCH(32))  # Output: Mini-Batch (batch_size=32)
    print(GradientDescentStrategy.MINI_BATCH(16))  # Output: Mini-Batch (batch_size=16)