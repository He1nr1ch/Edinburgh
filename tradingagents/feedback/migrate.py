"""Move the feedback store from one database to another: SQLite to Postgres, once.

Every table is copied with its ids, in one transaction on the destination, so
a copy that fails leaves the destination as it was. The destination must be
empty: this is a move into a new database, not a merge.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.engine import Engine

from tradingagents.feedback.schema import decisions, metadata

CHUNK = 500


def engine_for(url: str) -> Engine:
    """An engine for ``url``; a SQLite path may start with ``~``."""
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        if path and path != ":memory:":
            url = "sqlite:///" + str(Path(path).expanduser())
    return create_engine(url, future=True)


def copy_store(source_url: str, target_url: str) -> dict[str, dict[str, int]]:
    """Copy every table from ``source_url`` to ``target_url``; rows copied and skipped, by table.

    A row whose decision is missing from the source (SQLite does not enforce
    the link, Postgres does) is skipped and counted. Raises ``ValueError``
    when the two are the same database, the source file is missing, or the
    destination already holds rows.
    """
    source, target = engine_for(source_url), engine_for(target_url)
    if source.url == target.url:
        raise ValueError("the source and the destination are the same database")
    if source.url.get_backend_name() == "sqlite" and source.url.database not in (None, "", ":memory:") \
            and not Path(source.url.database).is_file():
        raise ValueError(f"no database at {source.url.database}")
    metadata.create_all(target)
    present = inspect(source)
    report: dict[str, dict[str, int]] = {}
    with source.connect() as src, target.begin() as dst:
        for table in metadata.sorted_tables:
            if dst.execute(select(func.count()).select_from(table)).scalar():
                raise ValueError(f"the destination's {table.name} table already holds rows; copy into an empty database")
        kept_decisions: set[int] = set()
        for table in metadata.sorted_tables:            # parents before the tables that point at them
            report[table.name] = {"copied": 0, "skipped": 0}
            if not present.has_table(table.name):
                continue                                # a store from before this table existed
            had = {c["name"] for c in present.get_columns(table.name)}
            columns = [c for c in table.columns if c.name in had]
            batch: list[dict] = []
            for row in src.execute(select(*columns)).mappings():
                if "decision_id" in row and row["decision_id"] not in kept_decisions:
                    report[table.name]["skipped"] += 1
                    continue
                if table is decisions:
                    kept_decisions.add(row["id"])
                batch.append(dict(row))
                if len(batch) >= CHUNK:
                    dst.execute(table.insert(), batch)
                    report[table.name]["copied"] += len(batch)
                    batch = []
            if batch:
                dst.execute(table.insert(), batch)
                report[table.name]["copied"] += len(batch)
            # The ids came with the rows, so Postgres's own counter must start after them.
            if dst.dialect.name == "postgresql" and "id" in table.columns:
                dst.execute(text(f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                                 f"COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table.name}"))
    return report
