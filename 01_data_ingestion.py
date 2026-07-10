from pathlib import Path
import logging
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DATA_DIR / "olist_analytics.db"
REVIEW_SAMPLE_SIZE = 5_000

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def setup_directories() -> None:
    """Create output directories required by the pipeline."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate_source_files() -> None:
    """Raise an error when one or more required source files are missing."""
    missing = [name for name in FILES.values() if not (RAW_DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing))


def load_source_data() -> dict[str, pd.DataFrame]:
    """Load the required Olist CSV files into DataFrames."""
    logging.info("Loading source CSV files.")

    orders = pd.read_csv(
        RAW_DATA_DIR / FILES["orders"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    reviews = pd.read_csv(
        RAW_DATA_DIR / FILES["reviews"],
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )

    return {
        "orders": orders,
        "items": pd.read_csv(RAW_DATA_DIR / FILES["items"]),
        "products": pd.read_csv(RAW_DATA_DIR / FILES["products"]),
        "customers": pd.read_csv(RAW_DATA_DIR / FILES["customers"]),
        "reviews": reviews,
    }


def build_tables(source: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Transform source data into a small analytical dimensional model."""
    orders = source["orders"]

    logging.info("Building Fact_OrderItem.")
    fact_order_item = source["items"].merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )
    fact_order_item = fact_order_item[
        [
            "order_id", "order_item_id", "customer_id", "product_id",
            "seller_id", "order_purchase_timestamp", "price", "freight_value",
        ]
    ].rename(columns={"order_purchase_timestamp": "purchase_timestamp"})

    logging.info("Building Dim_Order.")
    dim_order = orders[
        [
            "order_id", "order_status", "order_purchase_timestamp",
            "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date",
        ]
    ].copy()

    logging.info("Building Dim_Customer.")
    dim_customer = source["customers"]
    dim_customer = dim_customer[
        [
            "customer_id", "customer_unique_id", "customer_city",
            "customer_state", "customer_zip_code_prefix",
        ]
    ].copy()

    logging.info("Building Dim_Product.")
    dim_product = source["products"]
    dim_product = dim_product[
        [
            "product_id", "product_category_name", "product_name_lenght",
            "product_description_lenght", "product_photos_qty",
            "product_weight_g", "product_length_cm", "product_height_cm",
            "product_width_cm",
        ]
    ].rename(
        columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        }
    )
    dim_product["product_category_name"] = (
        dim_product["product_category_name"].fillna("Unknown").astype("string")
    )

    logging.info("Building Fact_Review.")
    fact_review = source["reviews"]
    fact_review = fact_review[
        [
            "review_id", "order_id", "review_score", "review_comment_message",
            "review_creation_date", "review_answer_timestamp",
        ]
    ].copy()
    fact_review["review_comment_message"] = (
        fact_review["review_comment_message"]
        .astype("string")
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    fact_review = fact_review[
        fact_review["review_comment_message"].notna()
        & fact_review["review_comment_message"].ne("")
    ]

    sample_size = min(REVIEW_SAMPLE_SIZE, len(fact_review))
    if sample_size < REVIEW_SAMPLE_SIZE:
        logging.warning("Only %s reviews contain comments; using all of them.", sample_size)
    else:
        logging.info("Sampling %s reviews for AI enrichment.", sample_size)

    fact_review = fact_review.sample(n=sample_size, random_state=42).reset_index(drop=True)
    fact_review.insert(0, "review_key", fact_review.index + 1)
    queue_index = fact_review.index
    fact_review["ai_sentiment"] = pd.Series(pd.NA, index=queue_index, dtype="Int64")
    fact_review["ai_driver"] = pd.Series(pd.NA, index=queue_index, dtype="string")
    fact_review["processing_status"] = "pending"
    fact_review["processed_at"] = pd.Series(
        pd.NaT, index=queue_index, dtype="datetime64[ns]"
    )
    fact_review["processing_error"] = pd.Series(pd.NA, index=queue_index, dtype="string")

    return {
        "Fact_OrderItem": fact_order_item,
        "Dim_Order": dim_order,
        "Dim_Customer": dim_customer,
        "Dim_Product": dim_product,
        "Fact_Review": fact_review,
    }


def load_database(tables: dict[str, pd.DataFrame]) -> None:
    """Load transformed tables into SQLite and create analytical indexes."""
    if DB_PATH.exists():
        logging.warning("Rebuilding %s; existing enrichment will be removed.", DB_PATH)

    logging.info("Loading analytical tables into %s.", DB_PATH)
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for table_name, dataframe in tables.items():
            dataframe.to_sql(table_name, conn, if_exists="replace", index=False)
            logging.info("Loaded %s rows into %s.", len(dataframe), table_name)

        conn.executescript(
            """
            CREATE UNIQUE INDEX ux_order_item
                ON Fact_OrderItem(order_id, order_item_id);
            CREATE UNIQUE INDEX ux_order ON Dim_Order(order_id);
            CREATE UNIQUE INDEX ux_customer ON Dim_Customer(customer_id);
            CREATE UNIQUE INDEX ux_product ON Dim_Product(product_id);
            CREATE UNIQUE INDEX ux_review_key ON Fact_Review(review_key);
            CREATE INDEX idx_order_item_customer ON Fact_OrderItem(customer_id);
            CREATE INDEX idx_order_item_product ON Fact_OrderItem(product_id);
            CREATE INDEX idx_review_order ON Fact_Review(order_id);
            CREATE INDEX idx_review_status ON Fact_Review(processing_status);
            """
        )

        pending = conn.execute(
            "SELECT COUNT(*) FROM Fact_Review WHERE processing_status = 'pending'"
        ).fetchone()[0]
        logging.info("%s reviews are ready for AI enrichment.", pending)


def main() -> None:
    """Execute the complete ingestion pipeline."""
    setup_directories()
    validate_source_files()
    source_data = load_source_data()
    load_database(build_tables(source_data))
    logging.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Data ingestion pipeline failed.")
        raise
