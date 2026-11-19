"""Autonomous Databricks Infrastructure Setup Script for Sales AI.

Configures:
1. Unity Catalog (`sales_ai`)
2. Medallion Schemas (`bronze`, `silver`, `gold`)
3. Unity Catalog Volume (`sales_ai.bronze.raw_files` and folders)
4. Uploads dataset CSVs to Volume
5. Uploads notebooks to Databricks Workspace
6. Inspects / configures SQL Warehouse
7. Creates Databricks Workflow Job (Sales_ai)
8. Updates backend/.env with live connection parameters
"""

import base64
import os
import sys
from pathlib import Path
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.service.jobs import (
    JobSettings,
    NotebookTask,
    Source,
    Task,
    TaskDependency,
)
from databricks.sdk.service.workspace import ImportFormat, Language

WORKSPACE_DIR = "/Workspace/Sales_ai/src"
CATALOG = "sales_ai"
SCHEMAS = ["bronze", "silver", "gold"]
VOLUME_NAME = "raw_files"


def setup_databricks(host=None, token=None):
    print("=" * 60)
    print("SALES AI - AUTONOMOUS DATABRICKS INFRASTRUCTURE SETUP")
    print("=" * 60)

    # 1. Initialize client
    kwargs = {}
    if host:
        kwargs["host"] = host
    if token:
        kwargs["token"] = token

    try:
        w = WorkspaceClient(**kwargs)
        user = w.current_user.me()
        print(f"[OK] Authenticated to Databricks as: {user.user_name}")
    except Exception as exc:
        print(f"[ERROR] Failed to authenticate to Databricks: {exc}")
        return False

    # 2. Check Metastore and Catalogs
    print("\n--- Phase 1: Metastore & Catalogs ---")
    try:
        existing_catalogs = [c.name for c in w.catalogs.list()]
        print(f"Existing catalogs: {', '.join(existing_catalogs)}")

        if CATALOG in existing_catalogs:
            print(f"[OK] Catalog '{CATALOG}' already exists.")
        else:
            print(f"[*] Creating catalog '{CATALOG}'...")
            w.catalogs.create(name=CATALOG, comment="Sales AI Retail Forecasting Catalog")
            print(f"[OK] Catalog '{CATALOG}' created successfully.")
    except Exception as exc:
        print(f"[!] Catalog check/creation notice: {exc}")

    # 3. Create Schemas
    print("\n--- Phase 2: Schemas ---")
    try:
        existing_schemas = [s.name for s in w.schemas.list(catalog_name=CATALOG)]
        print(f"Existing schemas in '{CATALOG}': {', '.join(existing_schemas)}")

        for schema in SCHEMAS:
            if schema in existing_schemas:
                print(f"[OK] Schema '{CATALOG}.{schema}' already exists.")
            else:
                print(f"[*] Creating schema '{CATALOG}.{schema}'...")
                w.schemas.create(name=schema, catalog_name=CATALOG, comment=f"Sales AI {schema.capitalize()} layer")
                print(f"[OK] Schema '{CATALOG}.{schema}' created.")
    except Exception as exc:
        print(f"[!] Schema check/creation notice: {exc}")

    # 4. Create Volume
    print("\n--- Phase 3: Unity Catalog Volume ---")
    volume_path = f"/Volumes/{CATALOG}/bronze/{VOLUME_NAME}"
    try:
        existing_volumes = [v.name for v in w.volumes.list(catalog_name=CATALOG, schema_name="bronze")]
        if VOLUME_NAME in existing_volumes:
            print(f"[OK] Volume '{volume_path}' already exists.")
        else:
            print(f"[*] Creating volume '{volume_path}'...")
            w.volumes.create(catalog_name=CATALOG, schema_name="bronze", name=VOLUME_NAME, volume_type=VolumeType.MANAGED)
            print(f"[OK] Volume '{volume_path}' created successfully.")
    except Exception as exc:
        print(f"[!] Volume check/creation notice: {exc}")

    # 5. Upload Dataset CSVs to Volume
    print("\n--- Phase 4: Uploading Project Datasets to Volume ---")
    data_dir = Path(__file__).resolve().parent.parent / "data" / "Initial-Training-Data"
    dataset_mapping = {
        "products.csv": f"{volume_path}/products.csv",
        "stores.csv": f"{volume_path}/stores.csv",
        "sales_transactions.csv": f"{volume_path}/incoming/sales_transactions.csv",
        "calendar_marketing_external.csv": f"{volume_path}/incoming/calendar_marketing_external.csv",
    }

    for filename, target_vol_path in dataset_mapping.items():
        local_file = data_dir / filename
        if not local_file.exists():
            print(f"[SKIP] Local file not found: {local_file}")
            continue

        size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"[*] Uploading {filename} ({size_mb:.2f} MB) -> {target_vol_path}...")
        try:
            with open(local_file, "rb") as f:
                w.files.upload(target_vol_path, f, overwrite=True)
            print(f"[OK] Uploaded {filename} successfully.")
        except Exception as exc:
            print(f"[!] Failed to upload {filename}: {exc}")

    # 6. Upload Notebooks to Workspace
    print("\n--- Phase 5: Uploading Notebooks to Workspace ---")
    notebooks_dir = Path(__file__).resolve().parent.parent / "Databricks" / "Sales_ai" / "src"
    notebook_files = [
        "01_setup_sales_ai.py",
        "02_bronze_ingestion.py",
        "03_silver_transformation.py",
        "04_exploratory_data_analysis.py",
        "05_gold_transformation.py",
        "06_model_training.py",
        "07_batch_inference.py",
    ]

    try:
        w.workspace.mkdirs(WORKSPACE_DIR)
        print(f"[OK] Workspace directory ensured: {WORKSPACE_DIR}")
    except Exception as exc:
        print(f"[!] mkdirs notice: {exc}")

    for nb in notebook_files:
        nb_path = notebooks_dir / nb
        if not nb_path.exists():
            print(f"[SKIP] Notebook not found: {nb_path}")
            continue

        dest_path = f"{WORKSPACE_DIR}/{nb_path.stem}"
        print(f"[*] Uploading {nb} -> {dest_path}...")
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                content = f.read()
            encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
            w.workspace.import_(
                path=dest_path,
                content=encoded_content,
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                overwrite=True,
            )
            print(f"[OK] Uploaded {nb}.")
        except Exception as exc:
            print(f"[!] Failed to upload notebook {nb}: {exc}")

    # 7. Inspect SQL Warehouses
    print("\n--- Phase 6: Inspect SQL Warehouses ---")
    warehouse_info = None
    try:
        warehouses = list(w.warehouses.list())
        print(f"Found {len(warehouses)} SQL Warehouse(s):")
        for wh in warehouses:
            print(f" - [{wh.id}] {wh.name} (State: {wh.state}, Auto-stop: {wh.auto_stop_mins} mins)")
            if warehouse_info is None and wh.state.value in ("RUNNING", "STOPPED", "STARTING"):
                warehouse_info = wh

        if warehouse_info:
            print(f"\n[Selected Warehouse]: {warehouse_info.name} (ID: {warehouse_info.id})")
        else:
            print("\n[!] No active SQL warehouse found. You may need to create or start one in the Databricks UI.")
    except Exception as exc:
        print(f"[!] Error listing warehouses: {exc}")

    # 8. Create or Update Databricks Job
    print("\n--- Phase 7: Configure Databricks Workflow Job ---")
    job_id = None
    job_name = "Sales_ai"
    try:
        existing_jobs = [j for j in w.jobs.list(name=job_name)]
        if existing_jobs:
            job_id = existing_jobs[0].job_id
            print(f"[OK] Existing job found: {job_name} (ID: {job_id})")
        else:
            print(f"[*] Creating new job: {job_name}...")
            tasks = [
                Task(
                    task_key="Initial_Setup",
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/01_setup_sales_ai", source=Source.WORKSPACE),
                ),
                Task(
                    task_key="Bronze_Transformation",
                    depends_on=[TaskDependency(task_key="Initial_Setup")],
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/02_bronze_ingestion", source=Source.WORKSPACE),
                ),
                Task(
                    task_key="Silver_Transformation",
                    depends_on=[TaskDependency(task_key="Bronze_Transformation")],
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/03_silver_transformation", source=Source.WORKSPACE),
                ),
                Task(
                    task_key="EDA",
                    depends_on=[TaskDependency(task_key="Silver_Transformation")],
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/04_exploratory_data_analysis", source=Source.WORKSPACE),
                ),
                Task(
                    task_key="Gold_Transformation",
                    depends_on=[TaskDependency(task_key="Silver_Transformation")],
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/05_gold_transformation", source=Source.WORKSPACE),
                ),
                Task(
                    task_key="Model_Training",
                    depends_on=[TaskDependency(task_key="Gold_Transformation")],
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/06_model_training", source=Source.WORKSPACE),
                ),
                Task(
                    task_key="Inferencing",
                    depends_on=[TaskDependency(task_key="Model_Training")],
                    notebook_task=NotebookTask(notebook_path=f"{WORKSPACE_DIR}/07_batch_inference", source=Source.WORKSPACE),
                ),
            ]
            new_job = w.jobs.create(name=job_name, tasks=tasks)
            job_id = new_job.job_id
            print(f"[OK] Created job {job_name} with ID: {job_id}")
    except Exception as exc:
        print(f"[!] Job creation notice: {exc}")

    # 9. Update Configuration in backend/.env
    print("\n--- Phase 8: Updating Local Configuration ---")
    backend_env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
    root_env_path = Path(__file__).resolve().parent.parent / ".env"

    server_host = None
    http_path = None
    if warehouse_info and warehouse_info.odbc_params:
        server_host = warehouse_info.odbc_params.hostname
        http_path = warehouse_info.odbc_params.path
    elif host:
        server_host = host.replace("https://", "").replace("http://", "").split("/")[0]

    incoming_path = f"/Volumes/{CATALOG}/bronze/{VOLUME_NAME}/incoming"

    print("Discovered configuration values:")
    print(f" - Host: {server_host}")
    print(f" - HTTP Path: {http_path}")
    print(f" - Job ID: {job_id}")
    print(f" - Incoming Volume Path: {incoming_path}")

    # Update backend/.env preserving existing keys
    for target_env in [backend_env_path, root_env_path]:
        if not target_env.exists():
            continue
        with open(target_env, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith("DATABRICKS_SERVER_HOSTNAME=") and server_host:
                new_lines.append(f"DATABRICKS_SERVER_HOSTNAME={server_host}\n")
            elif line.startswith("DATABRICKS_HTTP_PATH=") and http_path:
                new_lines.append(f"DATABRICKS_HTTP_PATH={http_path}\n")
            elif line.startswith("DATABRICKS_TOKEN=") and token:
                new_lines.append(f"DATABRICKS_TOKEN={token}\n")
            elif line.startswith("DATABRICKS_JOB_ID=") and job_id:
                new_lines.append(f"DATABRICKS_JOB_ID={job_id}\n")
            elif line.startswith("DATABRICKS_VOLUME_INCOMING_PATH="):
                new_lines.append(f"DATABRICKS_VOLUME_INCOMING_PATH={incoming_path}\n")
            else:
                new_lines.append(line)

        with open(target_env, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        print(f"[OK] Updated configuration in: {target_env}")

    print("\n" + "=" * 60)
    print("DATABRICKS INFRASTRUCTURE AUTOMATION COMPLETE!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    cli_host = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATABRICKS_HOST") or os.getenv("DATABRICKS_SERVER_HOSTNAME")
    cli_token = sys.argv[2] if len(sys.argv) > 2 else os.getenv("DATABRICKS_TOKEN")
    setup_databricks(host=cli_host, token=cli_token)

