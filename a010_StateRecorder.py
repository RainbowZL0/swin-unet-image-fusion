class StateRecorder:
    def __init__(self):
        self.record_stack = []

    def record(self, new_item):
        self.record_stack.append(new_item)

    def read(self):
        return self.record_stack.pop()

    def delete_all(self):
        self.record_stack.clear()

    def peek(self):
        if self.record_stack:
            return self.record_stack[-1]
        else:
            return None
