class PrettyTable:
    def __init__(self, field_names):
        self.field_names = [str(x) for x in field_names]
        self.rows = []
        self.float_format = ".3"

    def add_row(self, row):
        self.rows.append(list(row))

    def _format_cell(self, value):
        if isinstance(value, float):
            try:
                return format(value, self.float_format)
            except ValueError:
                return f"{value:.3f}"
        return str(value)

    def __str__(self):
        rows = [[self._format_cell(value) for value in row] for row in self.rows]
        table = [self.field_names] + rows
        widths = [max(len(row[i]) for row in table) for i in range(len(self.field_names))]

        def render(row):
            return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

        sep = "-+-".join("-" * width for width in widths)
        lines = [render(self.field_names), sep]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def __repr__(self):
        return str(self)
