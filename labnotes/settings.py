from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["labnotes/settings/base_settings.yaml", "labnotes/settings/settings.yaml"],
    environments=True,
    load_dotenv=True,
    env_switcher="ENV_FOR_DYNACONF",
    env="default",
    merge_enabled=True,
)