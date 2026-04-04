class GradientDescentStrategy:
    def __init__(self, batch_size: int | None = None, name: str = "Batch") -> None:
        self.batch_size = batch_size
        self.name = name

    @classmethod
    def BATCH(cls):
        return cls(batch_size=None, name="Batch")

    @classmethod
    def STOCHASTIC(cls):
        return cls(batch_size=1, name="Stochastic")

    @classmethod
    def MINI_BATCH(cls, batch_size: int):
        return cls(batch_size=batch_size, name="Mini-Batch")

def main():
    batch_strategy = GradientDescentStrategy.BATCH()
    print(batch_strategy.name, batch_strategy.batch_size)

    sgd_strategy = GradientDescentStrategy.STOCHASTIC()
    print(sgd_strategy.name, sgd_strategy.batch_size)

    mini_batch_strategy = GradientDescentStrategy.MINI_BATCH(batch_size=32)
    print(mini_batch_strategy.name, mini_batch_strategy.batch_size)






if __name__ == "__main__": 
    main()   