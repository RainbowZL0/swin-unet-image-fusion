class StateRecorder:
    def __init__(self):
        self.record_list = []
        self.record_stack = []

    def record(self, new_item):
        self.record_list.append(new_item)
        self.record_stack.append(new_item)

    def read(self):
        return self.record_stack.pop()
