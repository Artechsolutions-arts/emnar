from onyx.configs.app_configs import POSTGRES_PASSWORD, POSTGRES_DB
import os
print(f"Env var in OS: {os.environ.get('POSTGRES_PASSWORD')}")
print(f"App config: {POSTGRES_PASSWORD}")
print(f"App config DB: {POSTGRES_DB}")
