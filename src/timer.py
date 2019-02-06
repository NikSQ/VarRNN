import time


class Timer:
    def __init__(self, enabled):
        self.enabled = enabled
        self.begin = None

    def start(self):
        if self.enabled:
            self.begin = time.clock()

    def restart(self, module):
        if self.enabled:
            print('[\t{:^15}\t] {:1.5f} seconds'.format(module, time.clock() - self.begin))
            self.begin = time.clock()


