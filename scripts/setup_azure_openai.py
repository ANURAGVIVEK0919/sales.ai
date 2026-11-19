"""Autonomous Azure OpenAI Setup Script for Sales AI.

Inspects:
1. Azure CLI authentication (`az account show`)
2. Existing Azure OpenAI / Cognitive Services accounts (`az cognitiveservices account list`)
3. Existing model deployments (`az cognitiveservices account deployment list`)
4. Extracts endpoint and keys, and updates backend/.env
"""

import json
import subprocess
import sys
from pathlib import Path


def run_az_command(args):
    cmd = ["az"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        return None, result.stderr
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return result.stdout.strip(), None


def setup_azure_openai(resource_group=None, account_name=None, deployment_name="gpt-4o"):
    print("=" * 60)
    print("SALES AI - AUTONOMOUS AZURE OPENAI DISCOVERY & SETUP")
    print("=" * 60)

    # 1. Check Azure Login
    print("\n--- Phase 1: Checking Azure CLI Authentication ---")
    account, err = run_az_command(["account", "show"])
    if err:
        print("[!] Azure CLI is not signed in. Run 'az login' first.")
        print(f"Details: {err.strip()}")
        return False

    print(f"[OK] Signed into Azure Subscription: {account.get('name')} (ID: {account.get('id')})")

    # 2. Inspect Existing Cognitive Services / OpenAI Accounts
    print("\n--- Phase 2: Inspecting Azure OpenAI Resources ---")
    accounts, err = run_az_command(["cognitiveservices", "account", "list"])
    if err or not accounts:
        print("[!] No Cognitive Services / Azure OpenAI accounts found or failed to list.")
        return False

    openai_accounts = [a for a in accounts if a.get("kind") == "OpenAI"]
    print(f"Found {len(openai_accounts)} Azure OpenAI account(s):")
    for a in openai_accounts:
        print(f" - {a.get('name')} (Resource Group: {a.get('resourceGroup')}, Location: {a.get('location')})")

    selected_account = None
    if account_name:
        selected_account = next((a for a in openai_accounts if a.get("name") == account_name), None)
    elif openai_accounts:
        selected_account = openai_accounts[0]

    if not selected_account:
        print("[!] No matching Azure OpenAI account selected or available.")
        return False

    rg = selected_account.get("resourceGroup")
    name = selected_account.get("name")
    endpoint = selected_account.get("properties", {}).get("endpoint")
    print(f"\n[Selected Account]: {name} in {rg}")
    print(f"Endpoint: {endpoint}")

    # 3. Inspect Deployments
    print("\n--- Phase 3: Inspecting Model Deployments ---")
    deployments, err = run_az_command(["cognitiveservices", "account", "deployment", "list", "-g", rg, "-n", name])
    selected_deployment = None
    if deployments:
        print(f"Found {len(deployments)} deployment(s):")
        for d in deployments:
            model_info = d.get("properties", {}).get("model", {})
            model_name = model_info.get("name", "unknown")
            model_version = model_info.get("version", "")
            d_name = d.get("name")
            print(f" - Deployment: '{d_name}' | Model: {model_name} (version {model_version})")
            if d_name == deployment_name or model_name in ("gpt-4o", "gpt-4", "gpt-35-turbo"):
                if not selected_deployment:
                    selected_deployment = d_name

    target_deployment = selected_deployment or deployment_name
    print(f"[Target Deployment Name]: {target_deployment}")

    # 4. Fetch Keys
    print("\n--- Phase 4: Fetching API Key ---")
    keys, err = run_az_command(["cognitiveservices", "account", "keys", "list", "-g", rg, "-n", name])
    if err or not keys:
        print(f"[!] Failed to fetch keys: {err}")
        return False

    api_key = keys.get("key1") or keys.get("key2")
    if not api_key:
        print("[!] No valid key found.")
        return False
    print("[OK] Successfully retrieved API Key.")

    # 5. Update backend/.env
    print("\n--- Phase 5: Updating Local Environment Files ---")
    backend_env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
    root_env_path = Path(__file__).resolve().parent.parent / ".env"

    for env_file in [backend_env_path, root_env_path]:
        if not env_file.exists():
            continue
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith("AZURE_OPENAI_ENDPOINT="):
                new_lines.append(f"AZURE_OPENAI_ENDPOINT={endpoint}\n")
            elif line.startswith("AZURE_OPENAI_API_KEY="):
                new_lines.append(f"AZURE_OPENAI_API_KEY={api_key}\n")
            elif line.startswith("AZURE_OPENAI_DEPLOYMENT_NAME="):
                new_lines.append(f"AZURE_OPENAI_DEPLOYMENT_NAME={target_deployment}\n")
            else:
                new_lines.append(line)

        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"[OK] Updated Azure OpenAI settings in: {env_file}")

    print("\n" + "=" * 60)
    print("AZURE OPENAI CONFIGURATION COMPLETE!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    rg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    acc_arg = sys.argv[2] if len(sys.argv) > 2 else None
    dep_arg = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o"
    setup_azure_openai(resource_group=rg_arg, account_name=acc_arg, deployment_name=dep_arg)

