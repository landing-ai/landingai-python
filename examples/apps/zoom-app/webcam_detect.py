from sys import platform

if platform == "win32":
    import winreg


class WebcamDetect:
    REG_KEY = winreg.HKEY_CURRENT_USER
    WEBCAM_REG_SUBKEY = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam\\NonPackaged"
    WECAM_TIMESTAMP_VALUE_NAME = "LastUsedTimeStop"

    def __init__(self):
        self._reg_key = winreg.OpenKey(self.REG_KEY, self.WEBCAM_REG_SUBKEY)

    def get_active_apps(self):
        active_apps = []

        subkey_cnt, value_cnt, last_mod = winreg.QueryInfoKey(self._reg_key)
        for i in range(subkey_cnt):
            subkey_name = winreg.EnumKey(self._reg_key, i)
            subkey_full_name = f"{self.WEBCAM_REG_SUBKEY}\\{subkey_name}"

            subkey = winreg.OpenKey(self.REG_KEY, subkey_full_name)
            stopped_timestamp, _ = winreg.QueryValueEx(
                subkey, self.WECAM_TIMESTAMP_VALUE_NAME
            )
            if stopped_timestamp == 0:
                active_apps.append(subkey_name.replace("#", "/"))

        return active_apps

    def is_active(self):
        return len(self.get_active_apps()) > 0

    def is_active_app(self, app_name):
        return (
            len(
                [
                    app
                    for app in self.get_active_apps()
                    if app_name.lower() in app.lower()
                ]
            )
            > 0
        )

