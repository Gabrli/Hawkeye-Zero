import csv


class CsvLogger:
    """Write a header and rows to a CSV file."""

    def __init__(self, path: str, header):
        """Open a CSV file for writing and emit its header.

        Args:
            path: Destination file path. Existing contents are overwritten.
            header: Column names to write as the first row.
        """
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def write(self, row):
        """Write one row to the CSV file.

        Args:
            row: Sequence of values to serialize as a CSV record.
        """
        self._writer.writerow(row)

    def close(self):
        """Close the underlying file."""
        self._file.close()