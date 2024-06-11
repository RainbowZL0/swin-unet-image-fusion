import sys


class NewLineWrap:
    def write(self, x):
        # Wrap sys.stdout.write to add a newline on each write
        sys.stdout.write(x + '\n')
        return self

    def flush(self):
        # Flush the output (important for Jupyter notebooks, interactive output)
        sys.stdout.flush()
        return self
