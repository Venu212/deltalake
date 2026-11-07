"""Generate sample orders as a single Parquet file: data/orders.parquet (next to this script).

Run from the project root:
    .venv\\Scripts\\python.exe src\\deltalake\\make_orders.py
"""
import glob
import os
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path

# Windows: Spark needs Hadoop's winutils.exe/hadoop.dll. Must be set before Spark starts.
os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["HADOOP_HOME"], "bin")

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DecimalType, IntegerType, StringType, StructField, StructType

TARGET = Path(__file__).resolve().parent / "data" / "orders.parquet"

# (order_id, quantity, units, unit_cost, customer)
ROWS = [
    (1001, 10, "pcs", "2.50", "Alice"),
    (1002, 3, "box", "15.00", "Bob"),
    (1003, 25, "kg", "4.20", "Cathy"),
    (1004, 1, "pcs", "199.99", "David"),
    (1005, 8, "l", "3.75", "Alice"),
    (1006, 12, "pcs", "7.10", "Eva"),
    (1007, 5, "box", "22.40", "Frank"),
    (1008, 40, "kg", "1.95", "Bob"),
    (1009, 2, "pcs", "89.00", "Grace"),
    (1010, 15, "l", "2.60", "Cathy"),
    (1011, 6, "box", "18.25", "Henry"),
    (1012, 30, "pcs", "0.99", "Alice"),
    (1013, 4, "kg", "12.50", "Eva"),
    (1014, 9, "pcs", "14.30", "David"),
    (1015, 20, "l", "1.80", "Irene"),
    (1016, 7, "box", "9.90", "Frank"),
    (1017, 50, "pcs", "0.45", "Grace"),
    (1018, 3, "kg", "27.00", "Henry"),
    (1019, 11, "pcs", "5.55", "Bob"),
    (1020, 2, "box", "64.00", "Irene"),
]

SCHEMA = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("quantity", IntegerType(), False),
    StructField("units", StringType(), False),
    StructField("unit_cost", DecimalType(10, 2), False),
    StructField("customer", StringType(), False),
])


def main() -> None:
    spark = SparkSession.builder.appName("make-orders").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark.createDataFrame(
            [(o, q, u, Decimal(c), cu) for o, q, u, c, cu in ROWS], SCHEMA
        )
        # Decimals (not floats) so money values are exact.
        df = df.withColumn(
            "total_cost", (F.col("quantity") * F.col("unit_cost")).cast(DecimalType(12, 2))
        ).select("order_id", "quantity", "units", "unit_cost", "total_cost", "customer")

        # Spark writes a folder of part files; write one part, then move it to a single file.
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="orders-")
        try:
            out = os.path.join(tmp, "out")
            df.coalesce(1).write.mode("overwrite").parquet(out)
            part = glob.glob(os.path.join(out, "part-*.parquet"))[0]
            shutil.move(part, TARGET)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        check = spark.read.parquet(str(TARGET))
        check.orderBy("order_id").show(25)
        print(f"Wrote {check.count()} rows to {TARGET}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
