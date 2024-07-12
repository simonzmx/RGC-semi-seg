from torch import nn
from torch import optim


class CustomPolynomialLR(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1.0, warmup_epochs=0, last_epoch=-1, verbose=False):
        super(CustomPolynomialLR, self).__init__(optimizer, last_epoch, verbose)
        self.power = power
        self.warmup_epochs = warmup_epochs
        self.total_iters = total_iters
        self.actual_total_iters = total_iters - warmup_epochs

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch < self.warmup_epochs or self.actual_total_iters <= 0:
            factor = 1.0
        else:
            epoch_actual = self.last_epoch - self.warmup_epochs  # [0, self.total_iters]
            factor = (1.0 - epoch_actual / self.actual_total_iters) ** self.power

        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]


if __name__ == '__main__':
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=1.)
    steps = 100
    epochs = 100
    # scheduler = CustomPolynomialLR(optimizer, total_iters=epochs, power=0.9, warmup_epochs=10)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)

    for epoch in range(epochs):
        for idx in range(steps):
            pass
        print(scheduler.get_lr())
        scheduler.step()
