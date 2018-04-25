class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class AccuracyMeter(object):
    def __init__(self):
        self.correct = 0
        self.count = 0
        self.accuracy = None

    def update(self, correct, number):
        self.correct += correct
        self.count += number
        self.accuracy = self.correct / self.count
