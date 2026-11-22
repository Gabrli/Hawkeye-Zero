import csv


class CsvLogger:
    """Write rows to a CSV file after writing its header."""

    def __init__(self, path: str, header):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def write(self, row):
        self._writer.writerow(row)

    def close(self):
        self._file.close()
